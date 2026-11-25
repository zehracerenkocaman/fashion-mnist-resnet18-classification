"""
Kullanım:
    python3 predict_photos.py --images photos --weights outputs/best_resnet18.pt

Her fotoğraf için iki varyant denenir (raporda ön işlemenin etkisini göstermek için):
    ham   : gri tonlama + kırpma + kareye tamamlama + 28x28 (ters çevirme / kontrast YOK)
    onisl : ham + arka plana göre fark alma (koyu giysi/açık duvar -> açık giysi/koyu zemin)
            + eşikleme + kontrast germe  => Fashion-MNIST görünümü

Çıktılar (--out_dir):
    photo_results.png   giriş / ham 28x28 / ön işlenmiş 28x28 ve tahminler
    photo_results.csv   gerçek etiket, tahminler, güven, süre
"""
import argparse
import csv
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
IMG_SIZE = 64
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Gerçek etiketler, None = Fashion-MNIST'te karşılığı yok.
TRUE_LABELS = {
    "img1.jpeg": "Bag",
    "img2.jpeg": None ,  # katlı mini etek: sınıf yok 
    "img3.jpeg": "Ankle boot",
    "img4.jpeg": "Coat",
    "img5.jpeg": "Sneaker",       
}

# O durumda dosya adı için elle kutu oluşturma: (sol, üst, sağ, alt), 0-1 arası oranlar.
# Örnek: "img.jpeg": (0.10, 0.20, 0.90, 0.95)
CROPS = {}


def border_bg(arr):
    """Kenar şeritlerinin medyanı = arka plan parlaklığı."""
    h, w = arr.shape
    m = max(2, int(0.05 * min(h, w)))
    strips = np.concatenate([
        arr[:m].ravel(), arr[-m:].ravel(), arr[:, :m].ravel(), arr[:, -m:].ravel()
    ])
    return float(np.median(strips))


def crop_box(arr, name):
    h, w = arr.shape
    if name in CROPS:
        l, t, r, b = CROPS[name]
        return int(l * w), int(t * h), int(r * w), int(b * h)
    bg = border_bg(arr)
    mask = np.abs(arr.astype(np.int16) - bg) > 30
    cols = np.where(mask.mean(axis=0) > 0.02)[0]
    rows = np.where(mask.mean(axis=1) > 0.02)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def square_pad(img, bg):
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), int(bg))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def make_variants(path):
    name = os.path.basename(path)
    rgb = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # telefon yönü (EXIF)
    gray = rgb.convert("L")
    arr = np.asarray(gray)
    bg = border_bg(arr)
    box = crop_box(arr, name)
    sq = square_pad(gray.crop(box), bg).resize((28, 28), Image.LANCZOS)

    ham = np.asarray(sq, dtype=np.float32)

    # Arka plana göre mutlak fark: koyu-açık kutupluluktan bağımsız,
    # arka plan ~0, nesne parlak (Fashion-MNIST görünümü).
    p = np.abs(ham - bg)
    p[p < 15] = 0
    if p.max() > 0:
        p = p / p.max() * 255.0
    return rgb, ham, p.astype(np.float32), bg


TF = transforms.Compose([
    transforms.Resize(IMG_SIZE, antialias=True),
    transforms.Normalize(MEAN, STD),
])


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict(model, arr28, device):
    x = torch.from_numpy(arr28).float().div(255.0).unsqueeze(0).repeat(3, 1, 1)
    x = TF(x).unsqueeze(0).to(device)
    sync(device)
    t0 = time.perf_counter()
    logits = model(x)
    sync(device)
    ms = (time.perf_counter() - t0) * 1000
    prob = torch.softmax(logits, dim=1)[0].cpu()
    top = prob.topk(3)
    return [(CLASS_NAMES[i], float(v)) for v, i in zip(top.values, top.indices)], ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="photos")
    ap.add_argument("--weights", default="outputs/best_resnet18.pt")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()
    with torch.no_grad():  # ısınma
        model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device))

    paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        for p in glob.glob(os.path.join(args.images, ext))
    )
    if not paths:
        raise SystemExit(f"{args.images} içinde görsel bulunamadı.")

    rows = []
    fig, axs = plt.subplots(len(paths), 3, figsize=(9, 3.2 * len(paths)), squeeze=False)
    for r, path in enumerate(paths):
        name = os.path.basename(path)
        truth = TRUE_LABELS.get(name)
        rgb, ham, onisl, bg = make_variants(path)
        res = {}
        for tag, arr in (("ham", ham), ("onisl", onisl)):
            top3, ms = predict(model, arr, device)
            res[tag] = (top3, ms)

        def ok(tag):
            if truth is None:
                return "-"
            return "evet" if res[tag][0][0][0] == truth else "hayır"

        rows.append({
            "dosya": name,
            "gercek": truth if truth else "(sınıf yok)",
            "arka_plan_parlakligi": round(bg, 1),
            "ham_tahmin": res["ham"][0][0][0],
            "ham_guven": round(res["ham"][0][0][1], 4),
            "ham_dogru": ok("ham"),
            "onisl_tahmin": res["onisl"][0][0][0],
            "onisl_guven": round(res["onisl"][0][0][1], 4),
            "onisl_ilk3": "; ".join(f"{c} {p:.2f}" for c, p in res["onisl"][0]),
            "onisl_dogru": ok("onisl"),
            "cikarim_ms": round(res["onisl"][1], 2),
        })
        print(f"{name:14s} gerçek={truth!s:12s} ham={res['ham'][0][0][0]:12s} "
              f"({res['ham'][0][0][1]:.2f})  onisl={res['onisl'][0][0][0]:12s} "
              f"({res['onisl'][0][0][1]:.2f})")

        axs[r][0].imshow(rgb); axs[r][0].set_title(f"{name}\n(gerçek: {truth or 'sınıf yok'})", fontsize=9)
        axs[r][1].imshow(ham, cmap="gray", vmin=0, vmax=255)
        axs[r][1].set_title(f"Ham 28x28\n{res['ham'][0][0][0]} ({res['ham'][0][0][1]:.2f})", fontsize=9)
        axs[r][2].imshow(onisl, cmap="gray", vmin=0, vmax=255)
        axs[r][2].set_title(f"Ön işlenmiş 28x28\n{res['onisl'][0][0][0]} ({res['onisl'][0][0][1]:.2f})", fontsize=9)
        for a in axs[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "photo_results.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(args.out_dir, "photo_results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nKaydedildi: {args.out_dir}/photo_results.png ve photo_results.csv")


if __name__ == "__main__":
    main()
