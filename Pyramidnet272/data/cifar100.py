"""
CIFAR-100 Data Preparation
- AutoAugment + Cutout + CutMix (as per proposal)
- Super-class mapping helper
"""

import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.transforms import AutoAugment, AutoAugmentPolicy


# ---------------------------------------------------------------------------
# CIFAR-100 super-class mapping (fine-class index → coarse-class index)
# ---------------------------------------------------------------------------

# fmt: off
FINE_TO_COARSE = [
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
   16,  4, 17,  4,  2,  0, 17,  4, 18, 17,
   10,  3,  2, 12, 12, 16, 12,  1,  9, 19,
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
   16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
   18,  1,  2, 15,  6,  0, 17,  8, 14, 13,
]
# fmt: on

COARSE_NAMES = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor", "large_natural_outdoor", "large_omnivores_herbivores",
    "medium_mammals", "non-insect_invertebrates", "people", "reptiles",
    "small_mammals", "trees", "vehicles_1", "vehicles_2",
]


def fine_to_coarse_tensor(fine_labels: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(FINE_TO_COARSE, dtype=torch.long, device=fine_labels.device)
    return mapping[fine_labels]


# ---------------------------------------------------------------------------
# Cutout transform
# ---------------------------------------------------------------------------

class Cutout:
    """Randomly masks a square patch."""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, img):
        # img: PIL image → tensor handled after ToTensor
        return img   # applied after ToTensor via Lambda below


class CutoutTensor:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length  = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        mask = torch.ones(h, w)
        for _ in range(self.n_holes):
            cy = np.random.randint(h)
            cx = np.random.randint(w)
            y1, y2 = max(0, cy - self.length // 2), min(h, cy + self.length // 2)
            x1, x2 = max(0, cx - self.length // 2), min(w, cx + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


# ---------------------------------------------------------------------------
# CutMix collate function
# ---------------------------------------------------------------------------

def cutmix_collate(batch, alpha=1.0):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)

    lam = np.random.beta(alpha, alpha)
    B, C, H, W = images.shape
    perm = torch.randperm(B)

    cut_rat = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)

    images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    lam_real = 1 - (x2 - x1) * (y2 - y1) / (H * W)

    return images, labels, labels[perm], torch.tensor(lam_real, dtype=torch.float32)


class CutMixCollator:
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob  = prob

    def __call__(self, batch):
        if np.random.rand() < self.prob:
            imgs, la, lb, lam = cutmix_collate(batch, self.alpha)
            return imgs, la, lb, lam
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        return images, labels, labels, torch.ones(1)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def get_train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        AutoAugment(AutoAugmentPolicy.CIFAR10),   # closest policy for CIFAR
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        CutoutTensor(n_holes=1, length=16),
    ])


def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def get_dataloaders(data_root="./data", batch_size=128, num_workers=4,
                    use_cutmix=True, cutmix_alpha=1.0, cutmix_prob=0.5,
                    seed=42):
    torch.manual_seed(seed)

    train_set = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True,
        transform=get_train_transform(),
    )
    val_set = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True,
        transform=get_val_transform(),
    )

    collator = CutMixCollator(alpha=cutmix_alpha, prob=cutmix_prob) if use_cutmix else None

    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    base_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        base_loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collator, drop_last=True,
        generator=generator,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        **base_loader_kwargs,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        **base_loader_kwargs,
    )
    return train_loader, val_loader
