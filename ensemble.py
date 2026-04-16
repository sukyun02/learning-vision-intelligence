"""
Ensemble Inference — Weighted Soft Voting
=========================================
모델 추가 방법:
  MODELS 리스트에 항목 하나 추가하면 끝.

  {
      "name"    : "모델이름",
      "factory" : lambda: 모델생성함수(num_classes=100),
      "ckpt"    : "체크포인트/경로.pth",
      "ckpt_key": None,   # state dict가 dict 안에 있으면 키 이름 (예: "model_state")
                          # raw state dict이면 None
      "weight"  : 0.4,    # 소프트 보팅 가중치 (합이 1이 되도록)
  }

사용법:
  python ensemble.py
  python ensemble.py --data_root ./data --batch_size 128
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pyramidnet import pyramidnet272
from data.cifar100 import get_val_transform, FINE_TO_COARSE
import torchvision

# 팀원 모델 import (경로 추가)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrn/시각지능wrn"))
from wideresnet import wrn_28_10

# ---------------------------------------------------------------------------
# 모델 목록 — 여기에 추가/수정만 하면 됨
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name"    : "PyramidNet-272",
        "factory" : lambda: pyramidnet272(num_classes=100),
        "ckpt"    : "checkpoints_sukyun02_300ep/checkpoints_sukyun02/best_seed42.pth",
        "ckpt_key": "model_state",   # best_seed42.pth 내 키
        "weight"  : 0.5,
    },
    {
        "name"    : "WRN-28-10",
        "factory" : lambda: wrn_28_10(num_classes=100),
        "ckpt"    : "wrn/시각지능wrn/best_wrn_28_10.pth",
        "ckpt_key": None,
        "weight"  : 0.5,
    },
    # 나중에 DHVT 추가 시 아래 주석 해제 + 경로/가중치 수정
    # {
    #     "name"    : "DHVT",
    #     "factory" : lambda: dhvt_small(num_classes=100),
    #     "ckpt"    : "path/to/dhvt_best.pth",
    #     "ckpt_key": None,
    #     "weight"  : 0.33,
    # },
]

# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def load_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    """체크포인트에서 모델을 로드한다. 키 유무 / module. prefix 자동 처리."""
    model = cfg["factory"]()

    ckpt = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)

    # state dict 추출
    if isinstance(ckpt, dict):
        key = cfg.get("ckpt_key")
        if key and key in ckpt:
            state = ckpt[key]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt  # dict 자체가 state dict
    else:
        state = ckpt

    # AveragedModel의 "module." prefix 제거
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def get_logits(model: torch.nn.Module, loader: DataLoader,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """모델의 전체 val set logits와 labels를 반환한다."""
    all_logits, all_labels = [], []
    for imgs, labels in tqdm(loader, ncols=80, leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def compute_metrics(logits: torch.Tensor,
                    labels: torch.Tensor) -> tuple[float, float]:
    """Top-1 Accuracy + SC Density (Top-5 기반) 반환."""
    mapping = torch.tensor(FINE_TO_COARSE, dtype=torch.long)

    # Top-1
    top1 = (logits.argmax(dim=1) == labels).float().mean().item()

    # SC Density
    _, top5 = logits.topk(5, dim=1)
    top5_coarse = mapping[top5]
    gt_coarse   = mapping[labels]
    matches     = top5_coarse.eq(gt_coarse.unsqueeze(1))
    sc_density  = matches.float().sum(dim=1).div(5).mean().item()

    return top1, sc_density


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # val loader
    val_set = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True,
        transform=get_val_transform(),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # 가중치 정규화
    total_w = sum(m["weight"] for m in MODELS)
    weights  = [m["weight"] / total_w for m in MODELS]

    # 모델별 logits 수집
    all_logits = []
    labels     = None

    for cfg, w in zip(MODELS, weights):
        print(f"[{cfg['name']}] 로딩 중... (weight={w:.3f})")
        model  = load_model(cfg, device)
        logits, lbl = get_logits(model, val_loader, device)

        top1, sc = compute_metrics(logits, lbl)
        print(f"  단독 성능 → Top-1: {top1*100:.2f}%  SC Density: {sc*100:.2f}%")

        all_logits.append(logits * w)
        labels = lbl
        del model
        torch.cuda.empty_cache()

    # 소프트 보팅
    print("\n[앙상블] Weighted Soft Voting")
    ensemble_logits = torch.stack(all_logits).sum(dim=0)
    ens_top1, ens_sc = compute_metrics(ensemble_logits, labels)
    print(f"  앙상블 성능 → Top-1: {ens_top1*100:.2f}%  SC Density: {ens_sc*100:.2f}%")

    # 개별 vs 앙상블 요약
    print("\n" + "="*50)
    print(f"{'모델':<20} {'Top-1':>8} {'SC Density':>12}")
    print("-"*50)
    for cfg, w, logits in zip(MODELS, weights, all_logits):
        t1, sc = compute_metrics(logits / w, labels)
        print(f"{cfg['name']:<20} {t1*100:>7.2f}%  {sc*100:>10.2f}%")
    print("-"*50)
    print(f"{'Ensemble':<20} {ens_top1*100:>7.2f}%  {ens_sc*100:>10.2f}%")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    main(args)
