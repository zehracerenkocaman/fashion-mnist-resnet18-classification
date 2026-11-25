"""Fashion-MNIST ResNet-18 sınıflandırıcı - Gradio uygulaması.

Ön işleme ve model girişi, predict_photos.py / train.py ile birebir aynıdır.

Çalıştırma (mnist-resnet klasöründe, venv aktifken):
    pip install gradio
    python app.py
Ağırlık dosyası: outputs/best_resnet18.pt
"""
import time

import gradio as gr
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
WEIGHTS = "outputs/best_resnet18.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.to(DEVICE).eval()
with torch.no_grad():  # ısınma
    model(torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE))

TF = transforms.Compose([
    transforms.Resize(IMG_SIZE, antialias=True),
    transforms.Normalize(MEAN, STD),
])


# Ön işleme
def border_bg(arr):
    """Kenar şeritlerinin medyanı = arka plan parlaklığı."""
    h, w = arr.shape
    m = max(2, int(0.05 * min(h, w)))
    strips = np.concatenate([
        arr[:m].ravel(), arr[-m:].ravel(), arr[:, :m].ravel(), arr[:, -m:].ravel()
    ])
    return float(np.median(strips))


def crop_box(arr):
    h, w = arr.shape
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


def make_variants(img):
    rgb = ImageOps.exif_transpose(img).convert("RGB")
    gray = rgb.convert("L")
    arr = np.asarray(gray)
    bg = border_bg(arr)
    box = crop_box(arr)
    sq = square_pad(gray.crop(box), bg).resize((28, 28), Image.LANCZOS)
    ham = np.asarray(sq, dtype=np.float32)

    p = np.abs(ham - bg)
    p[p < 15] = 0
    if p.max() > 0:
        p = p / p.max() * 255.0
    return ham, p.astype(np.float32)


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict(img, mode):
    if img is None:
        return None, None, ""
    ham, onisl = make_variants(img)
    arr28 = onisl if mode == "Ön işlemli (önerilen)" else ham

    x = torch.from_numpy(arr28).float().div(255.0).unsqueeze(0).repeat(3, 1, 1)
    x = TF(x).unsqueeze(0).to(DEVICE)
    sync()
    t0 = time.perf_counter()
    logits = model(x)
    sync()
    ms = (time.perf_counter() - t0) * 1000
    prob = torch.softmax(logits, dim=1)[0].cpu()

    scores = {CLASS_NAMES[i]: float(prob[i]) for i in range(len(CLASS_NAMES))}
    seen = Image.fromarray(arr28.astype(np.uint8)).resize((224, 224), Image.NEAREST)
    return scores, seen, f"{ms:.2f} ms ({DEVICE.type})"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Fotoğraf yükle (kıyafet, ayakkabı veya çanta)"),
        gr.Radio(["Ön işlemli (önerilen)", "Ham"], value="Ön işlemli (önerilen)",
                 label="Ön işleme"),
    ],
    outputs=[
        gr.Label(num_top_classes=3, label="Tahmin (ilk 3 sınıf)"),
        gr.Image(label="Modelin gördüğü görüntü (28x28)"),
        gr.Textbox(label="Model tahmin süresi"),
    ],
    title="Fashion-MNIST ResNet-18 Sınıflandırıcı",
    description="Düz bir zemin üzerinde, tek bir nesne içeren fotoğraf yükle. "
                "Model yalnızca 10 Fashion-MNIST sınıfını tanır; listede olmayan "
                "bir nesnede de bir sınıf seçmek zorundadır.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
