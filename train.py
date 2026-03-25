"""
Step 3 — Train the galaxy morphology classifier.

Fine-tunes a ResNet18 with two classification heads:
  Q1  "Galaxy shape?"    →  smooth / features or disk / star or artifact
  Q2  "Edge-on galaxy?"  →  edge-on / not edge-on

Usage
-----
    python train.py                        # defaults (8 epochs, batch 32)
    python train.py --epochs 12 --batch-size 64

The trained weights are saved to  data/artifacts/galaxy_classifier.pth
"""

import argparse
import os
import random

import pandas as pd
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models

from config import (
    LABELS_PATH, IMAGES_DIR, ARTIFACTS_DIR, MODEL_WEIGHTS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, VAL_SPLIT, SEED, DEVICE,
    train_transform, val_transform,
)


# ──────────────────────────────────────────────
#  Reproducibility — fix every random seed so
#  that runs are deterministic.
# ──────────────────────────────────────────────
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────
class GalaxyDataset(Dataset):
    """Load galaxy images paired with Q1 + Q2 labels.

    Parameters
    ----------
    csv_path   : path to the labels CSV (needs columns id, q1_label, q2_label)
    images_dir : folder with {galaxy_id}.jpg files
    transform  : torchvision transform to apply to each image
    """

    def __init__(self, csv_path: str, images_dir: str, transform=None):
        self.transform = transform

        # Galaxy IDs are 18-digit integers — read as strings so they
        # match the JPEG filenames exactly (no float truncation).
        df = pd.read_csv(csv_path, dtype={"id": "string"})

        # Build paths and keep only rows whose image exists on disk.
        df["img_path"] = df["id"].apply(
            lambda gid: os.path.join(images_dir, f"{gid}.jpg")
        )
        self.df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                f"No JPEGs found in {images_dir}. Run download_images.py first."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["q1_label"]), int(row["q2_label"])


# ──────────────────────────────────────────────
#  Model
#
#  ResNet18 backbone shared between two heads:
#
#     image → [ResNet18 backbone] → 512-D features
#                                       │
#                                  ┌────┴────┐
#                                head_q1   head_q2
#                                (512→3)   (512→2)
# ──────────────────────────────────────────────
class GalaxyClassifier(nn.Module):
    """ResNet18 with two classification heads (Q1 + Q2)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        num_features = backbone.fc.in_features      # 512
        backbone.fc = nn.Identity()                  # remove original classifier

        self.backbone = backbone
        self.head_q1 = nn.Linear(num_features, 3)   # smooth / features / star
        self.head_q2 = nn.Linear(num_features, 2)   # edge-on / not edge-on

    def forward(self, x):
        """Return raw logits: (batch×3, batch×2)."""
        features = self.backbone(x)
        return self.head_q1(features), self.head_q2(features)


# ──────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────
def make_loaders(batch_size: int):
    """Build train / val DataLoaders with separate transforms.

    We create two independent Dataset instances (one with augmentation,
    one without) and split them using the same random indices.  This
    avoids a known PyTorch pitfall where changing the transform on a
    ``random_split`` subset leaks into both splits.
    """
    train_ds = GalaxyDataset(LABELS_PATH, IMAGES_DIR, transform=train_transform)
    val_ds   = GalaxyDataset(LABELS_PATH, IMAGES_DIR, transform=val_transform)

    n_total = len(train_ds)
    n_val   = max(1, int(n_total * VAL_SPLIT))
    n_train = n_total - n_val

    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(
        Subset(train_ds, indices[:n_train].tolist()),
        batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_ds, indices[n_train:].tolist()),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader, n_train, n_val


# ──────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────
def train(epochs: int, batch_size: int):
    """Train the galaxy classifier and save weights."""

    train_loader, val_loader, n_train, n_val = make_loaders(batch_size)
    print(f"Device: {DEVICE}  |  train: {n_train}  val: {n_val}\n")

    model = GalaxyClassifier(pretrained=True).to(DEVICE)
    criterion_q1 = nn.CrossEntropyLoss()
    criterion_q2 = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, epochs + 1):

        # ── Train ────────────────────────────────
        model.train()
        running_loss, total = 0.0, 0
        ok_q1, ok_q2 = 0, 0

        for images, lbl_q1, lbl_q2 in train_loader:
            images = images.to(DEVICE)
            lbl_q1 = lbl_q1.to(DEVICE)
            lbl_q2 = lbl_q2.to(DEVICE)

            optimizer.zero_grad()
            logits_q1, logits_q2 = model(images)

            loss_q1 = criterion_q1(logits_q1, lbl_q1)
            loss_q2 = criterion_q2(logits_q2, lbl_q2)
            # Both questions contribute equally to the gradient.
            loss = loss_q1 + loss_q2

            loss.backward()
            optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            total += bs
            ok_q1 += (logits_q1.argmax(1) == lbl_q1).sum().item()
            ok_q2 += (logits_q2.argmax(1) == lbl_q2).sum().item()

        # ── Validate ─────────────────────────────
        model.eval()
        v_ok_q1, v_ok_q2, v_total = 0, 0, 0

        with torch.inference_mode():
            for images, lbl_q1, lbl_q2 in val_loader:
                images = images.to(DEVICE)
                lbl_q1 = lbl_q1.to(DEVICE)
                lbl_q2 = lbl_q2.to(DEVICE)

                logits_q1, logits_q2 = model(images)
                bs = images.size(0)
                v_total += bs
                v_ok_q1 += (logits_q1.argmax(1) == lbl_q1).sum().item()
                v_ok_q2 += (logits_q2.argmax(1) == lbl_q2).sum().item()

        print(
            f"  Epoch {epoch}/{epochs}  |  "
            f"loss {running_loss / total:.4f}  |  "
            f"train Q1 {ok_q1/total:.3f}  Q2 {ok_q2/total:.3f}  |  "
            f"val Q1 {v_ok_q1/v_total:.3f}  Q2 {v_ok_q2/v_total:.3f}"
        )

    # ── Save ─────────────────────────────────────
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, MODEL_WEIGHTS)
    torch.save(model.state_dict(), path)
    print(f"\nModel saved → {path}")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the galaxy classifier")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print("=" * 50)
    print("  Galaxy Classifier — Training")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}")
    print("=" * 50 + "\n")

    train(epochs=args.epochs, batch_size=args.batch_size)
