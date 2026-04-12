"""
plot_log.py — CSV 로그를 읽어 학습 곡선 그래프를 저장

Usage:
    # 특정 seed 하나
    python plot_log.py --seed 42

    # 여러 seed 비교 (3-seed 실험 후)
    python plot_log.py --seed 42 0 1 --compare

    # 저장 경로 지정
    python plot_log.py --seed 42 --out_dir plots
"""

import argparse
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI 없는 환경에서도 동작
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# 스타일 설정
# ---------------------------------------------------------------------------

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
plt.rcParams.update({
    "figure.dpi"      : 150,
    "font.size"       : 11,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "axes.spines.top" : False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# CSV 로드
# ---------------------------------------------------------------------------

def load_log(log_path: str) -> dict:
    """CSV를 읽어 컬럼별 list 딕셔너리 반환."""
    data = {k: [] for k in
            ["epoch", "lr", "train_loss", "train_acc", "val_top1", "val_superclass"]}

    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in data:
                data[k].append(float(row[k]))

    return {k: np.array(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# 단일 seed 그래프 (4-panel)
# ---------------------------------------------------------------------------

def plot_single(data: dict, seed: int, swa_start: int, out_dir: str):
    epochs = data["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"PyramidNet-272 α200 — CIFAR-100  (seed={seed})",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── 1. Training Loss ──────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, data["train_loss"], color=COLORS[0], linewidth=1.2, label="Train Loss")
    ax.axvline(swa_start, color="gray", linestyle="--", alpha=0.6, label=f"SWA start ({swa_start})")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)

    # ── 2. Top-1 Accuracy ─────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, data["val_top1"] * 100, color=COLORS[1],
            linewidth=1.5, label="Val Top-1")
    ax.plot(epochs, data["train_acc"] * 100, color=COLORS[0],
            linewidth=1.0, linestyle="--", alpha=0.7, label="Train Acc")
    ax.axvline(swa_start, color="gray", linestyle="--", alpha=0.6)
    ax.axhline(84, color=COLORS[1], linestyle=":", alpha=0.5, label="Target 84%")
    ax.set_title("Top-1 Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    # ── 3. Super-Class Accuracy ───────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(epochs, data["val_superclass"] * 100, color=COLORS[2],
            linewidth=1.5, label="Super-Class Acc")
    ax.axvline(swa_start, color="gray", linestyle="--", alpha=0.6)
    ax.axhline(93, color=COLORS[2], linestyle=":", alpha=0.5, label="Target 93%")
    ax.set_title("Super-Class Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    # ── 4. Learning Rate ──────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(epochs, data["lr"], color=COLORS[3], linewidth=1.2)
    ax.axvline(swa_start, color="gray", linestyle="--", alpha=0.6, label=f"SWA start")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"training_curves_seed{seed}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")

    # ── Top-1 / Super-Class 비교 (세로로 나란히) ──────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, data["val_top1"] * 100,      color=COLORS[1], label="Top-1 Acc")
    ax.plot(epochs, data["val_superclass"] * 100, color=COLORS[2], label="Super-Class Acc")
    ax.axhline(84, color=COLORS[1], linestyle=":", alpha=0.5, label="Target 84%")
    ax.axhline(93, color=COLORS[2], linestyle=":", alpha=0.5, label="Target 93%")
    ax.axvline(swa_start, color="gray", linestyle="--", alpha=0.5, label=f"SWA start")
    ax.set_title(f"Validation Accuracy — seed {seed}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path2 = os.path.join(out_dir, f"val_accuracy_seed{seed}.png")
    plt.savefig(out_path2, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path2}")


# ---------------------------------------------------------------------------
# 다중 seed 비교 그래프
# ---------------------------------------------------------------------------

def plot_compare(all_data: dict, swa_start: int, out_dir: str):
    """seeds별 val_top1 / val_superclass를 한 그래프에 오버레이."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PyramidNet-272 α200 — 3-Seed Comparison", fontsize=13, fontweight="bold")

    for idx, (seed, data) in enumerate(all_data.items()):
        color = COLORS[idx % len(COLORS)]
        epochs = data["epoch"]

        axes[0].plot(epochs, data["val_top1"] * 100,
                     color=color, linewidth=1.4, label=f"seed {seed}")
        axes[1].plot(epochs, data["val_superclass"] * 100,
                     color=color, linewidth=1.4, label=f"seed {seed}")

    for ax, title, target in zip(axes,
                                  ["Top-1 Accuracy", "Super-Class Accuracy"],
                                  [84, 93]):
        ax.axhline(target, color="black", linestyle=":", alpha=0.4, label=f"Target {target}%")
        ax.axvline(swa_start, color="gray", linestyle="--", alpha=0.4, label="SWA start")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "compare_seeds.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {out_path}")

    print("\n── Final Metrics Summary ──")
    top1_vals, super_vals = [], []
    for seed, data in all_data.items():
        t1 = data["val_top1"][-1] * 100
        sc = data["val_superclass"][-1] * 100
        top1_vals.append(t1)
        super_vals.append(sc)
        print(f"  seed {seed}  Top-1: {t1:.2f}%  Super: {sc:.2f}%")
    print(f"  mean   Top-1: {np.mean(top1_vals):.2f}% ± {np.std(top1_vals):.2f}%")
    print(f"  mean   Super: {np.mean(super_vals):.2f}% ± {np.std(super_vals):.2f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    all_data = {}
    for seed in args.seed:
        log_path = os.path.join(args.ckpt_dir, f"log_seed{seed}.csv")
        if not os.path.exists(log_path):
            print(f"[WARN] 로그 파일 없음: {log_path}")
            continue
        print(f"Loading: {log_path}")
        all_data[seed] = load_log(log_path)

    if not all_data:
        print("로그 파일을 찾지 못했습니다. --ckpt_dir 경로를 확인하세요.")
        return

    swa_start = args.total_epochs - args.swa_epochs

    for seed, data in all_data.items():
        print(f"\n[seed {seed}] 그래프 생성 중...")
        plot_single(data, seed, swa_start, args.out_dir)

    if args.compare and len(all_data) > 1:
        print("\n[비교 그래프] 생성 중...")
        plot_compare(all_data, swa_start, args.out_dir)

    print(f"\n모든 그래프가 '{args.out_dir}/' 폴더에 저장되었습니다.")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training log CSV")
    parser.add_argument("--seed",          type=int, nargs="+", default=[42],
                        help="시각화할 seed 번호 (여러 개 가능: --seed 42 0 1)")
    parser.add_argument("--ckpt_dir",      type=str, default="./checkpoints",
                        help="CSV 로그 파일이 있는 폴더")
    parser.add_argument("--out_dir",       type=str, default="./plots",
                        help="그래프 저장 폴더")
    parser.add_argument("--total_epochs",  type=int, default=1800)
    parser.add_argument("--swa_epochs",    type=int, default=450)
    parser.add_argument("--compare",       action="store_true",
                        help="여러 seed를 한 그래프에 비교")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
