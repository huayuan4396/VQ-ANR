from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_mgrid, load_ensemble_params


class VQDataSet(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = Path(cfg.data.path)
        self.res = cfg.data.res if cfg.data.dataset == "Nyx" else [128, 128, 128]
        self.num = cfg.data.num_ensemble_training
        self.crop_size = cfg.data.crop_size
        self.factor = cfg.data.factor
        self.selected_idx = [int(i) for i in np.arange(0, self.num)]
        self.time_steps = self.cfg.data.time_steps
        self.probe = 0

    def _volume_path(self, idx, time_step):
        if self.cfg.data.dataset == "Nyx":
            return self.path / f"{idx:04d}" / f"{time_step:04d}.dat"
        if self.cfg.data.dataset in ["Castro", "Colver"]:
            return self.path.parent / f"{self.cfg.data.dataset}_reduced" / f"{idx:04d}" / f"{time_step:04d}.dat"
        raise ValueError(f"Unsupported dataset for VQ training: {self.cfg.data.dataset}")

    def get_samples(self):
        if self.probe != 0:
            return

        self.data = []
        for idx in self.selected_idx:
            selected_t = list(torch.randperm(self.time_steps)[:1])
            for t in selected_t:
                d = np.fromfile(self._volume_path(idx, int(t)), dtype="<f")
                d = d.reshape((self.res[2], self.res[1], self.res[0])).transpose()
                d = 2 * (d - d.min()) / (d.max() - d.min()) - 1
                self.data.append(torch.from_numpy(d).float()[None, ...])

    def __len__(self):
        return len(self.selected_idx) * self.factor

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.get_samples()
        idx = idx // self.factor

        x = 0 if self.res[0] == self.crop_size[0] else np.random.randint(0, self.res[0] - self.crop_size[0])
        y = 0 if self.res[1] == self.crop_size[1] else np.random.randint(0, self.res[1] - self.crop_size[1])
        z = 0 if self.res[2] == self.crop_size[2] else np.random.randint(0, self.res[2] - self.crop_size[2])

        d = self.data[idx][
            :,
            x : x + self.crop_size[0],
            y : y + self.crop_size[1],
            z : z + self.crop_size[2],
        ]

        self.probe += 1
        if self.probe == len(self.selected_idx) * self.factor:
            self.probe = 0

        return {"data": d}


class EnsembleTData(Dataset):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.path = Path(self.cfg.data.path)
        self.res = self.cfg.data.res
        self.device = device
        self.factor = self.cfg.data.factor

        if self.cfg.data.dataset in ["Nyx", "Castro"]:
            self.sampling_ratio = 0.2
        elif self.cfg.data.dataset == "Colver":
            self.sampling_ratio = 0.12
        elif self.cfg.data.dataset == "MPAS-Ocean":
            self.sampling_ratio = 0.15
        else:
            raise ValueError(f"Unsupported dataset for ANR training: {self.cfg.data.dataset}")

        self.ensemble_parms = load_ensemble_params(
            self.cfg.data.dataset,
            params_dir=getattr(self.cfg.data, "params_path", None),
            device=self.device,
        )
        self.selected_idx = [int(i) for i in np.arange(0, self.cfg.data.num_ensemble_training)]
        self.time_steps = self.cfg.data.time_steps

        print(f"Total Number of Training Ensembles are {len(self.selected_idx):d}")
        print(self.selected_idx)
        print(
            "Total Training Samples are {:d}".format(
                int(self.factor * len(self.selected_idx) * int(self.sampling_ratio * self.time_steps))
            )
        )

        if self.cfg.data.dataset in ["Nyx", "Castro", "Colver"]:
            self.coords = torch.from_numpy(get_mgrid(self.res, dim=3)).float().to(self.device)
            self.samples = np.prod(self.res)
        else:
            self.coords = np.load(self.path / "coord.npy")
            self.coords = self.coords.reshape(-1, 3)
            self.coords[:, 0] = 2.0 * (self.coords[:, 0] - self.coords[:, 0].min()) / (
                self.coords[:, 0].max() - self.coords[:, 0].min()
            ) - 1.0
            self.coords[:, 1] = 2.0 * (self.coords[:, 1] - self.coords[:, 1].min()) / (
                self.coords[:, 1].max() - self.coords[:, 1].min()
            ) - 1.0
            self.coords[:, 2] = 2.0 * (self.coords[:, 2] - self.coords[:, 2].min()) / (
                self.coords[:, 2].max() - self.coords[:, 2].min()
            ) - 1.0

            self.coords = torch.from_numpy(self.coords).float()
            self.samples = 11845146
            d = np.fromfile(self.path / "0000" / "0000.dat", dtype="<f")
            self.index = np.where(d != -1e34)
            self.coords = self.coords[self.index].to(self.device)

        self.batch_size = self.cfg.data.batch_size
        self.samples_per_vol = 2**15
        self.probe = 0
        self.num = (
            int(self.factor * len(self.selected_idx) * int(self.sampling_ratio * self.time_steps) * self.samples_per_vol)
            // self.batch_size
        )

    def _volume_path(self, idx, time_step):
        return self.path / f"{idx:04d}" / f"{time_step:04d}.dat"

    def get_random_points(self):
        if self.probe != 0:
            return

        if hasattr(self, "selected_coords"):
            del self.selected_coords
        if hasattr(self, "vol"):
            del self.vol
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        selected_coords_ = []
        vol_ = []
        for i in range(0, len(self.selected_idx)):
            selected_t = list(
                torch.randperm(self.time_steps - 2)[: int(self.sampling_ratio * self.time_steps) - 2].numpy()
                + [1]
            ) + [0, self.time_steps - 1]

            for t in selected_t:
                idx = self.selected_idx[i]
                d = np.memmap(self._volume_path(idx, int(t)), dtype="<f", mode="r")
                if self.cfg.data.dataset in ["MPAS-Ocean"]:
                    d = d[self.index]
                d = 2 * (d - d.min()) / (d.max() - d.min()) - 1
                vol = torch.from_numpy(d).float().to(self.device)

                sample_idx = torch.randperm(self.samples, device=self.device)[: int(self.factor * self.samples_per_vol)]
                coords = self.coords[sample_idx, ...]
                vol = vol[sample_idx]

                ensemble_param = self.ensemble_parms[i][None, ...].float()
                time_chunks = torch.zeros((ensemble_param.size(0), 1), device=self.device)
                time = t / (self.time_steps - 1)
                time -= 0.5
                time *= 2.0
                time_chunks.fill_(time)

                params = torch.cat((ensemble_param, time_chunks), dim=-1)
                params = params.expand(coords.size(0), -1)
                selected_coords_.append(torch.cat((coords, params), dim=-1))
                vol_.append(vol)

                del coords, vol, sample_idx

            if self.device.type == "cuda":
                torch.cuda.synchronize()

        self.selected_coords = torch.cat(selected_coords_, dim=0)
        self.vol = torch.cat(vol_, dim=0)

        idx = torch.randperm(self.vol.size(0), device=self.device)
        self.selected_coords = self.selected_coords[idx]
        self.vol = self.vol[idx]

        del selected_coords_, vol_, idx
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.get_random_points()

        selected_coords = self.selected_coords[idx * self.batch_size : (idx + 1) * self.batch_size, :]
        vol = self.vol[idx * self.batch_size : (idx + 1) * self.batch_size, :]

        self.probe += 1
        if self.probe == self.num:
            self.probe = 0
        return {"coords": selected_coords, "values": vol}
