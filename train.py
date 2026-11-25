"""
Kullanım:
    python train.py --data_dir /yol/fashionmnist --epochs 10

data_dir içinde şu iki dosya olmalı:
    fashion-mnist_train.csv
    fashion-mnist_test.csv

Çıktılar (--out_dir klasörüne):
    best_resnet18.pt      en iyi doğrulama doğruluğundaki model ağırlıkları
    metrics.json          rapor şablonu için tüm sayısal sonuçlar
    classification_report.txt
    confusion_matrix.png
    training_curves.png
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
IMG_SIZE = 64
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet istatistikleri


class FashionCSV(Dataset):
    """CSV'den gelen 28x28 gri görüntüyü 3 kanallı, 64x64 tensöre çevirir."""

    def __init__(self, images, labels, train):
        self.images, self.labels = images, labels
        steps = [transforms.Resize(IMG_SIZE, antialias=True)]
        if train:
            steps += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        steps += [transforms.Normalize(MEAN, STD)]
        self.tf = transforms.Compose(steps)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).float().div(255.0)  # 28x28
        x = x.unsqueeze(0).repeat(3, 1, 1)                        # 3x28x28
        return self.tf(x), int(self.labels[i])


def load_csv(path):
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].to_numpy()
    images = df.iloc[:, 1:].to_numpy(dtype=np.uint8).reshape(-1, 28, 28)
    return images, labels


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, targets, total_loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            total_loss += criterion(out, y).item() * x.size(0)
        preds.append(out.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    acc = float((preds == targets).mean())
    return total_loss / len(targets), acc, preds, targets


@torch.no_grad()
def measure_inference_ms(model, device, n_warmup=20, n_runs=200):
    """Tek görüntü (batch=1) için model ileri geçiş süresi, ms."""
    model.eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    for _ in range(n_warmup):
        model(x)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    sync(device)
    return (time.perf_counter() - t0) / n_runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # Veri: 60.000 eğitim -> 54.000 eğitim + 6.000 doğrulama; 10.000 test
    X, y = load_csv(os.path.join(args.data_dir, "fashion-mnist_train.csv"))
    X_test, y_test = load_csv(os.path.join(args.data_dir, "fashion-mnist_test.csv"))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=args.seed
    )
    print(f"Eğitim: {len(y_tr)}  Doğrulama: {len(y_val)}  Test: {len(y_test)}")

    pin = device.type == "cuda"
    mk = lambda ds, shuffle: DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.workers, pin_memory=pin,
    )
    train_loader = mk(FashionCSV(X_tr, y_tr, train=True), True)
    val_loader = mk(FashionCSV(X_val, y_val, train=False), False)
    test_loader = mk(FashionCSV(X_test, y_test, train=False), False)

    # Model, Loss, Optimizer
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Eğitim
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_path = 0.0, os.path.join(args.out_dir, "best_resnet18.pt")
    sync(device)
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, yb in train_loader:
            x, yb = x.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        scheduler.step()

        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        hist["train_loss"].append(running / seen)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss {running/seen:.4f} "
              f"| val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
    sync(device)
    train_seconds = time.perf_counter() - t_start

    # Test (en iyi ağırlıklarla)
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_acc, preds, targets = evaluate(model, test_loader, device)
    report_txt = classification_report(targets, preds, target_names=CLASS_NAMES, digits=4)
    report = classification_report(targets, preds, target_names=CLASS_NAMES, output_dict=True)
    print("\n" + report_txt)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    infer_ms = measure_inference_ms(model, device)

    metrics = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "input_size": f"3x{IMG_SIZE}x{IMG_SIZE}",
        "best_val_acc": best_acc,
        "test_acc": test_acc,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "train_time_seconds": train_seconds,
        "inference_ms_per_image_batch1": infer_ms,
        "torch": torch.__version__,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Görseller
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek"); ax.set_title("Karışıklık matrisi (test)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(hist["train_loss"], label="train"); axs[0].plot(hist["val_loss"], label="val")
    axs[0].set_title("Loss"); axs[0].set_xlabel("Epoch"); axs[0].legend()
    axs[1].plot(hist["val_acc"]); axs[1].set_title("Doğrulama doğruluğu"); axs[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
