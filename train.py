"""
train.py — Training script for the DeepfakeDetector.

Usage:
    python train.py --data_dir /path/to/dataset \\
                    --epochs 20 \\
                    --batch_size 32 \\
                    --lr 1e-4 \\
                    --output_dir checkpoints/

Dataset directory layout expected:
    data_dir/
        train/
            real/   *.jpg / *.png
            fake/   *.jpg / *.png
        val/
            real/
            fake/

Training features:
  • AdamW optimiser with cosine-annealing LR schedule
  • Cross-entropy loss with optional label smoothing
  • Mixed-precision (torch.cuda.amp) when CUDA is available
  • Best-checkpoint saving (by val accuracy)
  • TensorBoard-compatible CSV metric logging
"""

import os
import csv
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import DeepfakeDetector
from src.inference import get_device

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── ImageNet normalisation ────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def build_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=build_transforms(train=True),
    )
    val_ds = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=build_transforms(train=False),
    )

    # Ensure label mapping: 'fake' -> 1, 'real' -> 0
    log.info("Class -> index mapping: %s", train_ds.class_to_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, epoch
) -> dict:
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits, _, _ = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc  += accuracy(logits.detach(), labels)

        if (step + 1) % 20 == 0:
            log.info(
                "Epoch %d  step %d/%d  loss=%.4f  acc=%.4f",
                epoch, step + 1, len(loader),
                total_loss / (step + 1),
                total_acc  / (step + 1),
            )

    n = len(loader)
    return {"loss": total_loss / n, "acc": total_acc / n}


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _, _ = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

    n = len(loader)
    return {"loss": total_loss / n, "acc": total_acc / n}


def save_checkpoint(model, optimizer, epoch, val_acc, path: str):
    torch.save({
        "epoch": epoch,
        "val_acc": val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    log.info("Checkpoint saved -> %s  (val_acc=%.4f)", path, val_acc)


def run_training(args):
    device = get_device()
    log.info("Using device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    log.info("Train batches: %d  |  Val batches: %d",
             len(train_loader), len(val_loader))

    # ── Model ─────────────────────────────────────────────────────────────
    model = DeepfakeDetector(pretrained=True, freeze_stem=True).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    # ── Output dir ────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Metric log (CSV) ──────────────────────────────────────────────────
    csv_path = out_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr"])

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        log.info(
            "Epoch %d/%d  [%.1fs]  "
            "train_loss=%.4f  train_acc=%.4f  |  "
            "val_loss=%.4f  val_acc=%.4f  lr=%.2e",
            epoch, args.epochs, elapsed,
            train_m["loss"], train_m["acc"],
            val_m["loss"],   val_m["acc"],
            current_lr,
        )

        csv_writer.writerow([
            epoch, train_m["loss"], train_m["acc"],
            val_m["loss"], val_m["acc"], current_lr,
        ])
        csv_file.flush()

        # Save best checkpoint
        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            save_checkpoint(
                model, optimizer, epoch, best_val_acc,
                str(out_dir / "best_model.pth"),
            )

        # Periodic checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_m["acc"],
                str(out_dir / f"checkpoint_epoch{epoch:03d}.pth"),
            )

    csv_file.close()
    log.info("Training complete. Best val_acc: %.4f", best_val_acc)
    log.info("Metrics saved to: %s", csv_path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train DeepfakeDetector")
    p.add_argument("--data_dir",       required=True,        help="Root dataset directory")
    p.add_argument("--output_dir",     default="checkpoints",help="Where to save checkpoints")
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--label_smoothing",type=float, default=0.1)
    p.add_argument("--num_workers",    type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
