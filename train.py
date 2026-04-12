"""
Training Script — PyramidNet-272 α200 + ShakeDrop on CIFAR-100

Key settings (from proposal):
  - 1800 epochs total
  - batch size 128
  - SGD + Cosine Annealing + warmup 5 ep
  - AutoAugment + Cutout + CutMix
  - SWA applied at last 450 epochs
  - Super-class hierarchical loss (λ=0.8)
  - Top-1 and Super-Class accuracy tracked
  - 3-seed reproducible

Usage:
    python train.py --seed 42
    python train.py --seed 0  --epochs 1800
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm
from dotenv import load_dotenv
import wandb

def _load_env():
    """--env-file 인자가 있으면 해당 파일을, 없으면 기본 .env를 로드"""
    import sys as _sys
    env_file = ".env"
    for i, arg in enumerate(_sys.argv):
        if arg == "--env-file" and i + 1 < len(_sys.argv):
            env_file = _sys.argv[i + 1]
            break
    load_dotenv(env_file, override=True)

_load_env()

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pyramidnet import pyramidnet272
from data.cifar100 import get_dataloaders, fine_to_coarse_tensor
from losses.hierarchical_loss import HierarchicalLoss


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + CosineAnnealingLR (PyTorch 내장)
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, epochs, warmup_epochs, eta_min=1e-4):
    # 1단계: linear warmup
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1e-3,
        end_factor   = 1.0,
        total_iters  = warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = epochs - warmup_epochs,
        eta_min = eta_min,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers  = [warmup, cosine],
        milestones  = [warmup_epochs],
    )


# ---------------------------------------------------------------------------
# Train / Eval helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    epoch=0, total_epochs=0):
    model.train()
    total_loss = total_correct = total_n = 0

    pbar = tqdm(loader, ncols=100, leave=False,
                desc=f"  Train {epoch:4d}/{total_epochs}")

    for batch in pbar:
        if len(batch) == 4:
            imgs, la, lb, lam = batch
            imgs, la, lb = imgs.to(device), la.to(device), lb.to(device)
            lam = lam.to(device)
        else:
            imgs, la = batch
            imgs, la = imgs.to(device), la.to(device)
            lb, lam  = la, torch.ones(1, device=device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss   = criterion(logits, la, lb, lam)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, la, lb, lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == la).sum().item()
        total_loss    += loss.item() * imgs.size(0)
        total_n       += imgs.size(0)

        pbar.set_postfix({
            "loss": f"{total_loss / total_n:.4f}",
            "acc" : f"{total_correct / total_n * 100:.1f}%",
        })

    pbar.close()
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    fine_correct = coarse_correct = total_n = 0

    pbar = tqdm(loader, ncols=100, leave=False, desc="  Eval ")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds  = logits.argmax(dim=1)

        fine_correct   += (preds == labels).sum().item()
        coarse_correct += (fine_to_coarse_tensor(preds) ==
                           fine_to_coarse_tensor(labels)).sum().item()
        total_n        += imgs.size(0)

        pbar.set_postfix({
            "top1" : f"{fine_correct / total_n * 100:.1f}%",
            "super": f"{coarse_correct / total_n * 100:.1f}%",
        })

    pbar.close()
    return fine_correct / total_n, coarse_correct / total_n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {args.seed}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---------------------------------------------------------------------------
    # wandb 초기화
    # ---------------------------------------------------------------------------
    # wandb 로컬 로그 디렉토리: .env → ./wandb, .env.xxx → ./wandb_xxx
    env_file = args.env_file
    basename = os.path.basename(env_file)          # e.g. ".env.sukyun02"
    if basename.startswith(".env.") and len(basename) > 5:
        suffix = basename[5:]                      # e.g. "sukyun02"
        wandb_dir = f"./wandb_{suffix}"
    else:
        wandb_dir = "./wandb"
    os.makedirs(wandb_dir, exist_ok=True)

    run = wandb.init(
        entity  = os.getenv("WANDB_ENTITY"),
        project = os.getenv("WANDB_PROJECT", "pyramidnet-cifar100"),
        name    = f"pyramidnet272_seed{args.seed}",
        dir     = wandb_dir,
        config  = {
            "model"        : "PyramidNet-272 α200 + ShakeDrop",
            "dataset"      : "CIFAR-100",
            "seed"         : args.seed,
            "epochs"       : args.epochs,
            "batch_size"   : args.batch_size,
            "lr"           : args.lr,
            "weight_decay" : args.weight_decay,
            "swa_epochs"   : args.swa_epochs,
            "optimizer"    : "SGD + Nesterov",
            "scheduler"    : "CosineAnnealing + warmup 5ep",
            "augmentation" : "AutoAugment + Cutout + CutMix",
            "loss"         : "HierarchicalLoss λ=0.8",
        },
    )

    # Data
    train_loader, val_loader = get_dataloaders(
        data_root   = args.data_root,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        use_cutmix  = True,
        seed        = args.seed,
    )

    # Model
    model = pyramidnet272(num_classes=100).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"PyramidNet-272 α200  |  Parameters: {param_count:.2f} M")

    wandb.watch(model, log="all", log_freq=100)

    # Loss
    criterion = HierarchicalLoss(lam_coarse=0.8)

    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr           = args.lr,
        momentum     = 0.9,
        weight_decay = args.weight_decay,
        nesterov     = True,
    )

    # LR Scheduler: warmup 5ep → CosineAnnealingLR
    scheduler = build_scheduler(optimizer, args.epochs, warmup_epochs=5)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # SWA
    swa_model   = AveragedModel(model)
    swa_start   = args.epochs - args.swa_epochs          # e.g. 1800 - 450 = 1350
    swa_lr      = args.lr * 0.1
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr,
                          anneal_epochs=10, anneal_strategy='cos')

    best_top1 = 0.0
    val_top1  = 0.0
    val_super = 0.0
    log_path  = os.path.join(args.ckpt_dir, f"log_seed{args.seed}.csv")

    with open(log_path, "w") as f:
        f.write("epoch,lr,train_loss,train_acc,val_top1,val_superclass\n")

    t0 = time.time()

    epoch_bar = tqdm(range(1, args.epochs + 1), ncols=110,
                     desc=f"Seed {args.seed}")

    for epoch in epoch_bar:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch=epoch, total_epochs=args.epochs)

        if epoch < swa_start:
            scheduler.step()
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        cur_lr  = optimizer.param_groups[0]['lr']
        elapsed = (time.time() - t0) / 3600
        eta_h   = elapsed / epoch * (args.epochs - epoch) if epoch > 0 else 0

        if epoch % 20 == 0 or epoch == args.epochs:
            eval_model = swa_model if epoch >= swa_start else model
            if epoch >= swa_start:
                update_bn(train_loader, swa_model, device=device)
            val_top1, val_super = evaluate(eval_model, val_loader, device)

            with open(log_path, "a") as f:
                f.write(f"{epoch},{cur_lr:.6f},{train_loss:.4f},"
                        f"{train_acc:.4f},{val_top1:.4f},{val_super:.4f}\n")

            # Save best checkpoint
            if val_top1 > best_top1:
                best_top1 = val_top1
                ckpt_path = os.path.join(
                    args.ckpt_dir, f"best_seed{args.seed}.pth")
                torch.save({
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "swa_state"  : swa_model.state_dict(),
                    "optimizer"  : optimizer.state_dict(),
                    "val_top1"   : val_top1,
                    "val_super"  : val_super,
                    "seed"       : args.seed,
                }, ckpt_path)
                wandb.save(ckpt_path)
                wandb.run.summary["best_top1"]  = val_top1  * 100
                wandb.run.summary["best_super"] = val_super * 100

            wandb.log({
                "val/fineclass_acc" : val_top1  * 100,
                "val/superclass_acc": val_super * 100,
            }, step=epoch)

        wandb.log({
            "epoch"       : epoch,
            "train/loss"  : train_loss,
            "train/acc"   : train_acc * 100,
            "train/lr"    : cur_lr,
            "swa_active"  : int(epoch >= swa_start),
        }, step=epoch)

        swa_flag = "SWA✓" if epoch >= swa_start else "    "
        epoch_bar.set_postfix({
            "loss"  : f"{train_loss:.4f}",
            "acc"   : f"{train_acc*100:.1f}%",
            "top1"  : f"{val_top1*100:.2f}%",
            "super" : f"{val_super*100:.2f}%",
            "lr"    : f"{cur_lr:.5f}",
            "eta"   : f"{eta_h:.1f}h",
            swa_flag: "",
        })

    epoch_bar.close()

    update_bn(train_loader, swa_model, device=device)
    final_top1, final_super = evaluate(swa_model, val_loader, device)
    print(f"\n===== Final SWA Results (seed={args.seed}) =====")
    print(f"  Top-1 Accuracy   : {final_top1*100:.2f}%")
    print(f"  Super-Class Acc. : {final_super*100:.2f}%")

    torch.save(swa_model.state_dict(),
               os.path.join(args.ckpt_dir, f"swa_final_seed{args.seed}.pth"))

    wandb.run.summary["final_top1"]  = final_top1 * 100
    wandb.run.summary["final_super"] = final_super * 100
    wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train PyramidNet-272 on CIFAR-100")
    parser.add_argument("--seed",         type=int,   default=int(os.getenv("SEED",         42)))
    parser.add_argument("--epochs",       type=int,   default=int(os.getenv("EPOCHS",       1800)))
    parser.add_argument("--batch_size",   type=int,   default=int(os.getenv("BATCH_SIZE",   128)))
    parser.add_argument("--lr",           type=float, default=float(os.getenv("LR",         0.1)))
    parser.add_argument("--weight_decay", type=float, default=float(os.getenv("WEIGHT_DECAY", 5e-4)))
    parser.add_argument("--swa_epochs",   type=int,   default=int(os.getenv("SWA_EPOCHS",   450)))
    parser.add_argument("--data_root",    type=str,   default=os.getenv("DATA_ROOT",        "./data"))
    parser.add_argument("--ckpt_dir",     type=str,   default=os.getenv("CKPT_DIR",         "./checkpoints"))
    parser.add_argument("--num_workers",  type=int,   default=8)
    parser.add_argument("--env-file",     type=str,   default=".env",
                        help="Path to .env file (e.g. .env.sukyun)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
