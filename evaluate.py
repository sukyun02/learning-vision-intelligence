"""
Evaluation Script — CIFAR-100 Classification Challenge

Metrics (per PDF specification):
  1. Top-1 Accuracy
  2. Super-Class Accuracy: Top-5 predictions 중 정답 super-class에 속하는 비율

Usage:
    python evaluate.py --ckpt checkpoints/swa_final_seed42.pth
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pyramidnet import pyramidnet272
from data.cifar100 import (CIFAR100_MEAN, CIFAR100_STD,
                            fine_to_coarse_tensor, FINE_TO_COARSE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataset, batch_size=100):
    model.eval()

    val_transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    dataset_val = torchvision.datasets.CIFAR100(
        root=dataset.root, train=False, download=False, transform=val_transform)
    loader = DataLoader(dataset_val, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    all_logits = []
    all_labels = []

    for imgs, labels in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(imgs)
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = F.softmax(all_logits, dim=-1)

    # Top-1 Accuracy
    preds = probs.argmax(dim=1)
    top1_acc = (preds == all_labels).float().mean().item()

    # Super-Class Accuracy (PDF 규정):
    # Top-5 예측 중 정답과 같은 super-class에 속하는 비율
    mapping = torch.tensor(FINE_TO_COARSE, device=DEVICE)
    _, top5_preds = probs.topk(5, dim=1)                    # (N, 5)
    true_coarse = mapping[all_labels]                        # (N,)
    top5_coarse = mapping[top5_preds]                        # (N, 5)
    sc_acc = (top5_coarse == true_coarse.unsqueeze(1)).float().mean(dim=1).mean().item()

    return top1_acc, sc_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    import torchvision
    val_set = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=True,
        transform=T.Compose([T.ToTensor(),
                             T.Normalize(CIFAR100_MEAN, CIFAR100_STD)]))

    model = pyramidnet272(num_classes=100).to(DEVICE)

    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    if isinstance(ckpt, dict) and "swa_state" in ckpt:
        state = ckpt["swa_state"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    # SWA AveragedModel wraps keys with "module." prefix — strip it
    state = {k.replace("module.", ""): v for k, v in state.items()
             if k != "n_averaged"}
    model.load_state_dict(state, strict=False)

    print(f"Checkpoint loaded: {args.ckpt}")

    top1_acc, sc_acc = evaluate(model, val_set)
    total = top1_acc * 100 + sc_acc * 100

    print(f"\nTop-1 Accuracy  : {top1_acc*100:.2f}%")
    print(f"SC Density      : {sc_acc*100:.2f}%")
    print(f"Total Score     : {total:.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
