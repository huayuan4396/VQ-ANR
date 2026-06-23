import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def voxel_shuffle(inputs, upscale_factor):
    batch_size, channels, in_height, in_width, in_depth = inputs.size()
    channels //= upscale_factor**3

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    out_depth = in_depth * upscale_factor

    inputs = inputs.reshape(
        batch_size,
        channels,
        upscale_factor,
        upscale_factor,
        upscale_factor,
        in_height,
        in_width,
        in_depth,
    )
    return inputs.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(
        batch_size, channels, out_height, out_width, out_depth
    )


class VoxelShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv3d(
            in_channels, out_channels * (upscale_factor**3), 3, 1, 1
        )

    def forward(self, x):
        return voxel_shuffle(self.conv(x), self.upscale_factor)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class VectorQuantizer(nn.Module):
    """
    Reference:
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.permute(0, 2, 3, 4, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)

        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        )

        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)

        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view(latents_shape)

        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss

        quantized_latents = latents + (quantized_latents - latents).detach()

        e_mean = torch.mean(encoding_one_hot, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return (
            quantized_latents.permute(0, 4, 1, 2, 3).contiguous(),
            vq_loss,
            [perplexity, encoding_one_hot, encoding_inds],
        )

    def get_codebook_entry(self, indices, shape):
        min_encodings = torch.zeros(indices.shape[0], self.K).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, inputs):
        return inputs + self.resblock(inputs)


class Encoder(nn.Module):
    def __init__(self, in_channels, embedding_dim, hidden_dims):
        super().__init__()
        hidden_dims = list(hidden_dims or [64, 128, 128])
        modules = []

        for hidden_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    Downsample(in_channels, hidden_dim),
                    nn.LeakyReLU(),
                    ResidualLayer(hidden_dim, hidden_dim),
                )
            )
            in_channels = hidden_dim

        modules.append(
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        )
        modules.append(
            nn.Sequential(
                nn.Conv3d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU(),
            )
        )

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, out_channels, embedding_dim, hidden_dims):
        super().__init__()
        hidden_dims = list(hidden_dims or [64, 128, 128])
        decoder_dims = list(reversed(hidden_dims))
        modules = [
            nn.Sequential(
                nn.Conv3d(embedding_dim, decoder_dims[0], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        ]

        for i in range(len(decoder_dims) - 1):
            modules.append(
                nn.Sequential(
                    ResidualLayer(decoder_dims[i], decoder_dims[i]),
                    VoxelShuffle(decoder_dims[i], decoder_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                ResidualLayer(decoder_dims[-1], decoder_dims[-1]),
                VoxelShuffle(decoder_dims[-1], 32),
                nn.LeakyReLU(),
                nn.Conv3d(32, out_channels, kernel_size=3, stride=1, padding=1),
            )
        )

        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder(x)


class VQModel(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim, n_embed, hidden_dims=None, beta=0.25):
        super().__init__()
        self.encoder = Encoder(in_ch, embed_dim, hidden_dims)
        self.decoder = Decoder(out_ch, embed_dim, hidden_dims)
        self.vq = VectorQuantizer(n_embed, embed_dim, beta=beta)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def vector_quantization(self, x):
        return self.vq(x)

    def forward(self, x):
        encoding = self.encode(x)
        quantized_inputs, vq_loss, _ = self.vq(encoding)
        return self.decode(quantized_inputs), vq_loss

    def get_codebook(self):
        return self.vq.embedding


class ResBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, num_features // 4),
            nn.ReLU(),
            nn.Linear(num_features // 4, num_features),
            nn.ReLU(),
        )

    def forward(self, features):
        return 0.5 * (self.net(features) + features)


class SignalAttn(nn.Module):
    def __init__(self, dim, n_head, head_dim, att_threshold=None):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        inner_dim = n_head * head_dim
        self.inner_dim = inner_dim
        self.dim = dim

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.scale = head_dim**-0.5
        self.att_threshold = att_threshold

    def forward(self, fr, dt):
        q = self.to_q(fr)
        k = self.to_k(dt)
        v = self.to_v(dt)
        b, n, h_d = q.shape
        dn = k.shape[1]

        q = q.reshape(b, n, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, dn, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, dn, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        if self.att_threshold is not None:
            attn = attn.clip(min=self.att_threshold) - self.att_threshold
            attn = attn / torch.sum(attn, dim=-1, keepdim=True)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, h_d)
        return self.to_out(out)


class ANR(nn.Module):
    def __init__(
        self,
        spatial_dim,
        ensemble_dim,
        out_dim,
        hidden_dim,
        embed_dim,
        n_embed,
        n_head=6,
        head_dim=64,
        use_pe=True,
        pe_dim=128,
        pe_sigma=1024,
        adaptive_idim=False,
        att_threshold=None,
        mode="att",
    ):
        super().__init__()
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.pos_dim = spatial_dim * pe_dim
        self.spatial_dim = spatial_dim
        self.ensemble_dim = ensemble_dim

        self.ensemble_transform = nn.Linear(ensemble_dim, embed_dim)
        self.attn = SignalAttn(embed_dim, n_head, head_dim, att_threshold=att_threshold)

        mlp = []
        if mode == "att":
            mlp.append(
                nn.Sequential(nn.Linear(embed_dim + self.pos_dim, hidden_dim), nn.ReLU())
            )
            mlp.append(nn.Linear(hidden_dim, out_dim))
        elif mode == "balance":
            mlp.append(
                nn.Sequential(nn.Linear(embed_dim + self.pos_dim, hidden_dim), nn.ReLU())
            )
            mlp.append(nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU()))
            mlp.append(ResBlock(2 * hidden_dim))
            mlp.append(nn.Linear(2 * hidden_dim, out_dim))
        elif mode == "mlp":
            mlp.append(
                nn.Sequential(nn.Linear(embed_dim + self.pos_dim, hidden_dim), nn.ReLU())
            )
            mlp.append(nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.ReLU()))
            mlp.append(nn.Sequential(nn.Linear(2 * hidden_dim, 4 * hidden_dim), nn.ReLU()))
            mlp.append(ResBlock(4 * hidden_dim))
            mlp.append(nn.Linear(4 * hidden_dim, out_dim))
        else:
            raise ValueError(f"Unsupported ANR mode: {mode}")

        self.mlp = nn.Sequential(*mlp)
        self.out_dim = out_dim

    def convert_posenc(self, x):
        w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device))
        x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
        return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1)

    def forward(self, coords, values):
        xyz = self.convert_posenc(coords[:, :, :3])
        e = self.ensemble_transform(coords[:, :, 3:])
        e = self.attn(e, values)
        x = torch.cat((xyz, e), dim=-1)
        return self.mlp(x)
