"""
Step 4 — Classify a galaxy image.

Loads the trained model and prints the predicted morphology for both
Galaxy Zoo questions (Q1: shape, Q2: edge-on).

Usage
-----
    python predict.py galaxy.jpg       # classify a specific image
    python predict.py                   # picks the first image in the dataset

Example output
--------------
    Image : data/processed/train/587722982290686153.jpg

    Q1 – Galaxy shape
      Prediction : features or disk
        smooth           : 0.123
        features or disk : 0.845
        star or artifact : 0.032

    Q2 – Edge-on?
      Prediction : not edge-on
        edge-on     : 0.082
        not edge-on : 0.918
"""

import argparse
import os

import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import (
    ARTIFACTS_DIR, IMAGES_DIR, MODEL_WEIGHTS, DEVICE,
    Q1_CLASSES, Q2_CLASSES, val_transform,
)

# Import the model class from train.py so we can rebuild the same
# architecture and load the saved weights into it.
from train import GalaxyClassifier


def load_model():
    """Load trained weights into a GalaxyClassifier.

    We pass ``pretrained=False`` because we are loading our own trained
    weights, not the original ImageNet ones.
    """
    model = GalaxyClassifier(pretrained=False)
    path = os.path.join(ARTIFACTS_DIR, MODEL_WEIGHTS)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


def predict(image_path: str):
    """Classify a single galaxy image.

    Returns
    -------
    q1_label : str    e.g. "features or disk"
    q1_probs : dict   {class_name: probability}
    q2_label : str    e.g. "not edge-on"
    q2_probs : dict   {class_name: probability}
    """
    model = load_model()

    img = Image.open(image_path).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)

    with torch.inference_mode():
        logits_q1, logits_q2 = model(tensor)
        probs_q1 = torch.softmax(logits_q1, dim=1)[0].cpu().numpy()
        probs_q2 = torch.softmax(logits_q2, dim=1)[0].cpu().numpy()

    q1_label = Q1_CLASSES[int(probs_q1.argmax())]
    q2_label = Q2_CLASSES[int(probs_q2.argmax())]
    q1_probs = {Q1_CLASSES[i]: float(probs_q1[i]) for i in Q1_CLASSES}
    q2_probs = {Q2_CLASSES[i]: float(probs_q2[i]) for i in Q2_CLASSES}

    return q1_label, q1_probs, q2_label, q2_probs


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a galaxy image")
    parser.add_argument(
        "image", nargs="?", default=None,
        help="Path to a galaxy JPEG (default: first image in dataset)",
    )
    args = parser.parse_args()

    # Resolve image path
    if args.image:
        img_path = args.image
    else:
        files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")]
        if not files:
            raise RuntimeError(f"No images in {IMAGES_DIR}.")
        img_path = os.path.join(IMAGES_DIR, files[0])

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Run prediction
    q1_label, q1_probs, q2_label, q2_probs = predict(img_path)

    # Pretty-print results to console
    print(f"Image : {img_path}\n")

    print("Q1 – Galaxy shape")
    print(f"  Prediction : {q1_label}")
    for name, p in q1_probs.items():
        print(f"    {name:20s}: {p:.3f}")

    print("\nQ2 – Edge-on?")
    print(f"  Prediction : {q2_label}")
    for name, p in q2_probs.items():
        print(f"    {name:20s}: {p:.3f}")

    # Show the galaxy image with classification results
    img = Image.open(img_path)
    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(10, 4),
                                          gridspec_kw={"width_ratios": [1, 1]})

    # Left: galaxy image
    ax_img.imshow(img)
    ax_img.set_title(os.path.basename(img_path), fontsize=9)
    ax_img.axis("off")

    # Right: classification results as text
    lines = [
        f"Q1 – Galaxy shape",
        f"  Prediction: {q1_label}",
        "",
    ]
    for name, p in q1_probs.items():
        lines.append(f"  {name:20s} {p:.1%}")
    lines += [
        "",
        f"Q2 – Edge-on?",
        f"  Prediction: {q2_label}",
        "",
    ]
    for name, p in q2_probs.items():
        lines.append(f"  {name:20s} {p:.1%}")

    ax_txt.text(0.05, 0.95, "\n".join(lines), transform=ax_txt.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace")
    ax_txt.axis("off")

    fig.suptitle("Galaxy Classifier", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
