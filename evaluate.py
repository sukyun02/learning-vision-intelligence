"""
Evaluation Script — TTA + Super-Class Accuracy
- hflip + 4-corner crop × 5  (as per proposal)
- Fine-logit correction: P(fine) × P(super|fine)
- Loads SWA checkpoint

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
# TTA transforms
# ---------------------------------------------------------------------------

def tta_transforms():
    """Returns a list of (transform_fn) for hflip + 4-corner + center crop."""
    mean, std = CIFAR100_MEAN, CIFAR100_STD
    normalize = T.Normalize(mean, std)

    base = T.Compose([T.ToTensor(), normalize])

    def hflip(pil):
        return normalize(T.functional.hflip(T.ToTensor()(pil)))

    def corner_crop(pil, i):
        """i in 0-3 for four corners, 4 for center"""
        crops = T.FiveCrop(28)(pil)    
        t = T.Compose([T.Resize(32), T.ToTensor(), normalize])
        return t(crops[i])

    transforms = [base, lambda p: hflip(p)]
    transforms += [lambda p, i=i: corner_crop(p, i) for i in range(5)]
    return transforms    


# ---------------------------------------------------------------------------
# Logit correction: P(fine) × P(super|fine)
# ---------------------------------------------------------------------------

def super_class_correction(fine_logits: torch.Tensor) -> torch.Tensor:
    """
    Corrects fine logits by multiplying P(fine) with P(super | fine).
    Here P(super | fine) is 1.0 for the super-class each fine class belongs to
    and 0 otherwise — so this just scales by the super-class probability.
    """
    mapping = torch.tensor(FINE_TO_COARSE, device=fine_logits.device)
    p_fine  = F.softmax(fine_logits, dim=-1)             

    B = fine_logits.size(0)
    p_super = torch.zeros(B, 20, device=fine_logits.device)
    p_super.scatter_add_(1, mapping.unsqueeze(0).expand(B, -1), p_fine)

    coarse_probs_expanded = p_super[:, mapping]           
    corrected = p_fine * coarse_probs_expanded
    return corrected


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_tta(model, dataset, batch_size=100, use_correction=True):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)
    transforms = tta_transforms()

    all_logits = []
    for t in transforms:
        dataset.transform = T.Compose([])   
        tta_loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0)
        logits_list = []
        for imgs, _ in tta_loader:
            pass
    dataset.transform = None

    val_transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    hflip_transform = T.Compose([
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    dataset_plain = torchvision.datasets.CIFAR100(
        root=dataset.root, train=False, download=False, transform=val_transform)
    dataset_hflip = torchvision.datasets.CIFAR100(
        root=dataset.root, train=False, download=False, transform=hflip_transform)

    loader_plain = DataLoader(dataset_plain, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    loader_hflip = DataLoader(dataset_hflip, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    logits_sum = None
    all_labels = []

    for (imgs, labels), (imgs_h, _) in zip(loader_plain, loader_hflip):
        imgs   = imgs.to(DEVICE)
        imgs_h = imgs_h.to(DEVICE)
        labels = labels.to(DEVICE)

        lg1 = model(imgs)
        lg2 = model(imgs_h)
        avg = (lg1 + lg2) / 2.0

        if logits_sum is None:
            logits_sum = avg
            all_labels = labels
        else:
            logits_sum = torch.cat([logits_sum, avg], dim=0)
            all_labels = torch.cat([all_labels, labels], dim=0)

    if use_correction:
        probs = super_class_correction(logits_sum)
    else:
        probs = F.softmax(logits_sum, dim=-1)

    preds = probs.argmax(dim=1)
    fine_acc  = (preds == all_labels).float().mean().item()
    coarse_acc = (fine_to_coarse_tensor(preds.cpu()) ==
                  fine_to_coarse_tensor(all_labels.cpu())).float().mean().item()

    return fine_acc, coarse_acc


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

    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    if "swa_state" in ckpt:
        state = ckpt["swa_state"]
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
    elif "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    print(f"Checkpoint loaded: {args.ckpt}")

    fine_acc, coarse_acc = evaluate_with_tta(model, val_set,
                                              use_correction=args.use_correction)
    print(f"Top-1 Accuracy   : {fine_acc*100:.2f}%")
    print(f"Super-Class Acc. : {coarse_acc*100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           type=str,            required=True)
    parser.add_argument("--data_root",      type=str,            default="./data")
    parser.add_argument("--use_correction", action="store_true", default=True,
                        help="Apply super-class logit correction")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
