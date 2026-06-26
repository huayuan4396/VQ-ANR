from pathlib import Path
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
from utils import count_parameters, save_loss_info


def train_vq(vol_dl, model, cfg, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.vq.learning_rate)
    info = {"MSE Loss": [], "VQ Loss": [], "Time": 0}
    best_loss = 1e5

    for epoch in range(1, cfg.model.vq.epochs + 1):
        epoch_loss = 0
        vq_loss_epoch = 0
        start_time = time.time()

        for data in vol_dl:
            v = data["data"].to(memory_format=torch.contiguous_format)
            v = v.float().to(device)
            if len(v.shape) == 4:
                v = v[None, ...]

            vrec, vq_loss = model(v)
            rec_loss = F.mse_loss(vrec.view(-1), v.view(-1))
            loss = rec_loss + vq_loss

            epoch_loss += rec_loss.item()
            vq_loss_epoch += vq_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[{epoch:02d}/{cfg.model.vq.epochs:d}] MSE Loss: {epoch_loss:.4f}")

        info["MSE Loss"].append(epoch_loss)
        info["VQ Loss"].append(vq_loss_epoch)
        info["Time"] += time.time() - start_time

        loss_path = Path(cfg.model.model_path) / (
            f"{cfg.model.vq.params.embed_dim:d}_{cfg.model.vq.params.n_embed:d}_vq_loss.yaml"
        )
        save_loss_info(loss_path, info)

        if best_loss > info["MSE Loss"][-1]:
            best_loss = info["MSE Loss"][-1]
            prefix = f"{cfg.model.vq.params.embed_dim:d}_{cfg.model.vq.params.n_embed:d}_vq"
            torch.save(model.state_dict(), Path(cfg.model.model_path) / f"{prefix}_best_model.pth")
            torch.save(model.vq.state_dict(), Path(cfg.model.model_path) / f"{prefix}_best_codebook.pth")


def train_anr(vol_dl, model, codebook, cfg, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.anr.learning_rate)
    info = {
        "MSE Loss": [],
        "Att Params": 0,
        "MLP Params": 0,
        "CodeBook Params": 0,
        "Total Params": 0,
        "Time": 0,
    }
    best_loss = 1e5

    num_params = count_parameters(model)
    num_params_att = count_parameters(model.attn)
    num_params_mlp = count_parameters(model.mlp)

    print(f"Number of Parameters in Attention is {num_params_att:d}, {num_params_att / num_params:f}")
    print(f"Number of Parameters in MLP is {num_params_mlp:d}, {num_params_mlp / num_params:f}")

    info["Att Params"] = num_params_att
    info["MLP Params"] = num_params_mlp
    info["CodeBook Params"] = cfg.model.anr.params.embed_dim * cfg.model.anr.params.n_embed
    info["Total Params"] = num_params_att + num_params_mlp + info["CodeBook Params"]

    print(f"Model Size is {info['Total Params'] * 2 / 1024 / 1024:f} MB")

    for epoch in range(1, cfg.model.anr.epochs + 1):
        epoch_loss = 0
        start_time = time.time()

        for data in vol_dl:
            coords = data["coords"].to(memory_format=torch.contiguous_format)
            coords = coords.float().to(device)

            values = data["values"].to(memory_format=torch.contiguous_format)
            values = values.float().to(device)
            if len(values.shape) == 2:
                values = values[None, ...]

            recon = model(coords, codebook)
            rec_loss = F.mse_loss(recon.view(-1), values.view(-1))
            epoch_loss += rec_loss.item()

            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()

        print(f"[{epoch:02d}/{cfg.model.anr.epochs:d}] MSE Loss: {epoch_loss:.4f}")

        info["MSE Loss"].append(epoch_loss)
        info["Time"] += time.time() - start_time

        name = (
            f"anr_{cfg.model.anr.params.embed_dim:d}_{cfg.model.anr.params.n_embed:d}_"
            f"{cfg.model.anr.params.n_head:d}_{cfg.model.anr.params.head_dim:d}_"
            f"{cfg.model.anr.params.hidden_dim:d}_att_{cfg.model.anr.params.att_threshold:.6f}"
        )
        save_loss_info(Path(cfg.model.model_path) / f"{name}_loss.yaml", info)

        if best_loss > info["MSE Loss"][-1]:
            best_loss = info["MSE Loss"][-1]
            torch.save(model.state_dict(), Path(cfg.model.model_path) / f"{name}_best_model.pth")
            
        prefix = f"{cfg.model.vq.params.embed_dim:d}_{cfg.model.vq.params.n_embed:d}_vq"
    torch.save(codebook.half().state_dict(), Path(cfg.model.model_path) / f"{prefix}_best_codebook.pth")
