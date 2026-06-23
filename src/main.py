import argparse
import sys
import time as timer
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataio import EnsembleTData, VQDataSet
from model import ANR, VQModel, VectorQuantizer
from train import train_anr, train_vq
from utils import (
    PSNR,
    SRC_DIR,
    count_parameters,
    get_mgrid,
    load_config,
    load_ensemble_params,
    resolve_path,
    save_loss_info,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="Castro.yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inf"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--n_embed", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=32)
    parser.add_argument("--att_threshold", type=float, default=0.0015)
    parser.add_argument("--params_dir", type=str, default="../Params")
    return parser.parse_args()


def checkpoint_prefix(cfg):
    params = cfg.model.anr.params
    return (
        f"{params.embed_dim:d}_{params.n_embed:d}_{params.n_head:d}_"
        f"{params.head_dim:d}_{params.hidden_dim:d}"
    )


def anr_name(cfg):
    return f"anr_{checkpoint_prefix(cfg)}_att_{cfg.model.anr.params.att_threshold:.6f}"


def load_codebook(cfg, device):
    params = cfg.model.anr.params
    vq = VectorQuantizer(params.n_embed, params.embed_dim)
    codebook_path = Path(cfg.model.model_path) / f"{params.embed_dim:d}_{params.n_embed:d}_vq_best_codebook.pth"
    vq.load_state_dict(torch.load(codebook_path, map_location="cpu"))
    codebook = vq.embedding.weight.data
    if len(codebook.shape) == 2:
        codebook = codebook[None, ...]
    return codebook.float().to(device)


def prepare_config(args):
    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = SRC_DIR / "configs" / config_path
    config = load_config(config_path, True)

    config.device = args.device
    config.model.model_path = str(resolve_path(config.model.model_path))
    config.model.result_path = str(resolve_path(config.model.result_path))
    config.data.params_path = str(resolve_path(args.params_dir))

    Path(config.model.model_path).mkdir(parents=True, exist_ok=True)
    Path(config.model.result_path).mkdir(parents=True, exist_ok=True)

    config.model.vq.params.n_embed = args.n_embed
    config.model.vq.params.embed_dim = args.embed_dim
    config.model.anr.params.n_embed = args.n_embed
    config.model.anr.params.embed_dim = args.embed_dim
    config.model.anr.params.n_head = args.n_head
    config.model.anr.params.head_dim = args.head_dim
    config.model.anr.params.att_threshold = args.att_threshold
    return config


def seed_everything(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_id):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return torch.device("cuda")
    return torch.device("cpu")


def run_train(config, device):
    params = config.model.anr.params
    codebook_path = Path(config.model.model_path) / f"{params.embed_dim:d}_{params.n_embed:d}_vq_best_codebook.pth"

    if not codebook_path.exists():
        volumes = VQDataSet(config)
        model = VQModel(**config.model.vq.params).to(device)
        train_vol = DataLoader(volumes, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
        print("Begin to Train VQModel!")
        train_vq(train_vol, model, config, device)

    codebook = load_codebook(config, device)
    volumes = EnsembleTData(config, device)
    print("Read Ensemble Data Successfully!")
    train_vol = DataLoader(volumes, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    model = ANR(**config.model.anr.params).to(device)
    train_anr(train_vol, model, codebook, config, device)


def build_inference_coords(config, ensemble_param, time_step):
    coords = torch.from_numpy(get_mgrid(config.data.res, dim=3)).float()
    time_chunks = torch.zeros((1))
    time_value = time_step / (config.data.time_steps - 1)
    time_value -= 0.5
    time_value *= 2.0
    time_chunks.fill_(time_value)

    params = torch.cat((ensemble_param, time_chunks), dim=-1).float()
    params = params.expand(coords.size(0), -1)
    return torch.cat((coords, params), dim=-1)


def infer_volume(model, codebook, xyzet_coords, device, max_points=2**16):
    values = []
    with torch.no_grad():
        for start in range(0, xyzet_coords.shape[0], max_points):
            end = min(start + max_points, xyzet_coords.shape[0])
            pred = model(xyzet_coords[start:end, :][None, ...].to(device), codebook)
            values += list(pred.squeeze().detach().cpu().numpy())

    values = np.asarray(values, dtype="<f")
    values = np.clip(values, -1.0, 1.0)
    return values.flatten("F")


def run_inf(config, device):
    print(f"Begin to Infer the {config.data.dataset} Data Set!")
    codebook = load_codebook(config, device)

    model = ANR(**config.model.anr.params).to(device)
    model_path = Path(config.model.model_path) / f"{anr_name(config)}_best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.float()
    model.eval()

    num_params = count_parameters(model) + config.model.anr.params.n_embed * config.model.anr.params.embed_dim
    print(f"Model Size is {num_params * 2 / 1024 / 1024:.2f} MB")

    colver_d1_min = 0.2575 if config.data.dataset == "Colver" else 0.505
    ensemble_params = load_ensemble_params(
        config.data.dataset,
        params_dir=config.data.params_path,
        colver_d1_min=colver_d1_min,
    )
    print(ensemble_params.shape)

    info = {"PSNR": [], "Avg": 0, "Time": 0}
    num = 0
    result_dir = Path(config.model.result_path) / checkpoint_prefix(config) / f"{config.model.anr.params.att_threshold:.6f}"

    for i in range(config.data.num_ensemble_training, len(ensemble_params)):
        ensemble_dir = result_dir / f"{i:04d}"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        for t in range(0, config.data.time_steps):
            pred_path = ensemble_dir / f"{t:04d}.dat"
            if pred_path.exists():
                v = np.fromfile(pred_path, dtype="<f")
            else:
                start_time = timer.time()
                xyzet_coords = build_inference_coords(config, ensemble_params[i], t)
                v = infer_volume(model, codebook, xyzet_coords, device)
                info["Time"] += timer.time() - start_time

            gt_path = Path(config.data.path) / f"{i:04d}" / f"{t:04d}.dat"
            gt = np.fromfile(gt_path, dtype="<f")
            gt = 2.0 * (gt - gt.min()) / (gt.max() - gt.min()) - 1.0

            psnr = PSNR(gt, v)
            info["PSNR"].append(f"PSNR is {psnr:f} under Ensemble ID {i:04d} at Time Step {t:04d}")
            info["Avg"] += float(psnr)
            num += 1

            print(f"PSNR is {psnr:.2f} under Ensemble ID {i:04d} at Time Step {t:04d}")
            save_loss_info(
                Path(config.model.result_path) / f"{anr_name(config)}_PSNR.yaml",
                info,
            )

    info["Avg"] /= num
    info["Time"] /= num
    print(f"Avg is {info['Avg']:.2f}")
    save_loss_info(Path(config.model.result_path) / f"{anr_name(config)}_PSNR.yaml", info)


def main():
    args = parse_args()
    config = prepare_config(args)
    print(config)

    seed_everything(0)
    device = get_device(config.device)

    if args.mode == "train":
        run_train(config, device)
    elif args.mode == "inf":
        run_inf(config, device)
    else:
        raise NotImplementedError("Not Implemented!")


if __name__ == "__main__":
    sys.exit(main())
