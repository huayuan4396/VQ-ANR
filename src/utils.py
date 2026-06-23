import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def save_loss_info(filepath, loss):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as f:
        yaml.dump(loss, f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_path(path, base_dir=SRC_DIR):
    path = Path(str(path)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def get_mgrid(sidelen, dim=2, s=1, t=0):
    """Generate flattened coordinates in [-1, 1]."""
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(
            np.mgrid[: sidelen[1] : s, : sidelen[0] : s], axis=-1
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[1] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[0] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[: sidelen[2] : s, : sidelen[1] : s, : sidelen[0] : s],
            axis=-1,
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[2] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[0] - 1)
    elif dim == 4:
        pixel_coords = np.stack(
            np.mgrid[
                : sidelen[0] : (t + 1),
                : sidelen[3] : s,
                : sidelen[2] : s,
                : sidelen[1] : s,
            ],
            axis=-1,
        )[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[3] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[1] - 1)
    else:
        raise NotImplementedError(f"Not implemented for dim={dim}")

    pixel_coords -= 0.5
    pixel_coords *= 2.0
    return np.reshape(pixel_coords, (-1, dim))


def PSNR(vol, preds):
    mse = np.mean((vol - preds) ** 2)
    diff = vol.max() - vol.min()
    return 20 * np.log10(diff) - 10 * np.log10(mse)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def _normalize(values, ranges):
    normalized = []
    for value, (low, high) in zip(values, ranges):
        normalized.append(2.0 * (value - low) / (high - low) - 1.0)
    return normalized


def load_ensemble_params(dataset, params_dir=None, device=None, colver_d1_min=0.505):
    """Load and normalize ensemble parameters used by ANR training/inference."""
    params_dir = Path(params_dir) if params_dir else PROJECT_DIR / "Params"

    specs = {
        "Nyx": (
            "nyx.csv",
            [(0.0215, 0.0235), (0.1200, 0.1550), (0.5500, 0.8500)],
        ),
        "Castro": (
            "castro.csv",
            [(0.8, 0.95), (0.8, 0.95)],
        ),
        "Colver": (
            "colverleaf3d.csv",
            [
                (colver_d1_min, 1.0),
                (1.0, 2.0),
                (1.5, 3.0),
                (0.75, 2.0),
                (1.5, 3.5),
                (4.0, 7.0),
            ],
        ),
        "MPAS-Ocean": (
            "MPAS-Ocean.csv",
            [(0.0, 5.0), (0.25, 1.0), (600.0, 1500.0), (100.0, 300.0)],
        ),
    }

    if dataset not in specs:
        raise ValueError(f"Unsupported dataset: {dataset}")

    filename, ranges = specs[dataset]
    csv_path = params_dir / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {csv_path}")

    ensemble_params = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            values = [float(row[i]) for i in range(len(ranges))]
            values = _normalize(values, ranges)
            ensemble_params.append(torch.FloatTensor(values).float())

    if not ensemble_params:
        raise ValueError(f"No ensemble parameters found in {csv_path}")

    ensemble_params = torch.stack(ensemble_params)
    if device is not None:
        ensemble_params = ensemble_params.to(device)
    return ensemble_params
