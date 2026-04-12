"""
3-Seed Reproducibility Runner
Runs train.py for seeds [42, 0, 1] and reports mean ± std.

Usage:
    python run_seeds.py --epochs 1800
    python run_seeds.py --epochs 50 --quick   # quick smoke test
"""

import argparse
import subprocess
import sys
import re


def run_seed(seed: int, extra_args: list):
    cmd = [
        sys.executable, "train.py",
        "--seed",    str(seed),
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"  Running seed {seed}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"[ERROR] seed {seed} failed with return code {result.returncode}")
        return None, None

    # Parse final line from stdout
    return None, None   # actual values read from log files


def main(args):
    import os, csv
    import numpy as np
    from dotenv import dotenv_values

    seeds     = [42, 0, 1]
    top1_list = []
    super_list = []

    # env 파일에서 CKPT_DIR 읽기
    if args.env_file:
        env_vals = dotenv_values(args.env_file)
    else:
        env_vals = dotenv_values(".env")
    ckpt_dir = env_vals.get("CKPT_DIR", "./checkpoints")

    extra = []
    if args.env_file:
        extra += ["--env-file", args.env_file]
    if args.epochs:
        extra += ["--epochs", str(args.epochs)]
    if args.batch_size:
        extra += ["--batch_size", str(args.batch_size)]

    for seed in seeds:
        run_seed(seed, extra)

        log_path = os.path.join(ckpt_dir, f"log_seed{seed}.csv")
        if not os.path.exists(log_path):
            print(f"[WARN] Log not found for seed {seed}")
            continue

        with open(log_path) as f:
            rows = list(csv.DictReader(f))

        if rows:
            last = rows[-1]
            top1  = float(last["val_top1"])
            super_ = float(last["val_superclass"])
            top1_list.append(top1)
            super_list.append(super_)
            print(f"Seed {seed}  →  Top-1: {top1*100:.2f}%  "
                  f"Super: {super_*100:.2f}%")

    print(f"\n{'='*60}")
    if top1_list:
        print(f"Top-1  : {np.mean(top1_list)*100:.2f}% ± {np.std(top1_list)*100:.2f}%")
        print(f"Super  : {np.mean(super_list)*100:.2f}% ± {np.std(super_list)*100:.2f}%")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file",   type=str, default=None,
                        help="Path to .env file (e.g. .env.sukyun)")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    main(parser.parse_args())
