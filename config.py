"""
Shared configuration for the Galaxy Classifier project.

All constants, paths, hyper-parameters and image transforms live here
so that every other script can just ``from config import ...``.
"""

from torchvision import transforms

# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
RAW_CATALOG_PATH = "data/raw/gz2_hart16.csv.gz"
LABELS_PATH = "data/interim/labels_q1_q2.csv"
IMAGES_DIR = "data/processed/train"
ARTIFACTS_DIR = "data/artifacts"
MODEL_WEIGHTS = "galaxy_classifier.pth"       # single checkpoint name

# ──────────────────────────────────────────────
#  Training hyper-parameters
# ──────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 1e-4
IMG_SIZE = 224
VAL_SPLIT = 0.20        # 20 % of data used for validation
SEED = 42

# ──────────────────────────────────────────────
#  ImageNet normalisation values
#  (required because we fine-tune a pre-trained ResNet18)   https://www.geeksforgeeks.org/python/how-to-normalize-images-in-pytorch/
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────
#  Human-readable class labels
# ──────────────────────────────────────────────
Q1_CLASSES = {0: "smooth", 1: "features or disk", 2: "star or artifact"}
Q2_CLASSES = {0: "edge-on", 1: "not edge-on"}

# ──────────────────────────────────────────────
#  Device — CPU is fine for this project's model size
# ──────────────────────────────────────────────
DEVICE = "cpu"

# ──────────────────────────────────────────────
#  Image transforms
#
#  train_transform — random augmentation (flip + rotation) to reduce
#                    overfitting during training.
#  val_transform   — deterministic resize + normalisation for
#                    validation and inference.
# ──────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
