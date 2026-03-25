# Galaxy Zoo Model

This project uses PyTorch to classify galaxy images from the Galaxy Zoo dataset.
It focuses on two galaxy classification questions:

- What is the main galaxy shape?
- Is the galaxy edge-on or not?

The project includes a full workflow for preparing labels, downloading galaxy
images, training the model, and running predictions.

## Project Files

- `prepare_labels.py`
  Reads the raw Galaxy Zoo catalog and creates clean labels for training.

- `download_images.py`
  Downloads galaxy cutout images from SDSS using the galaxy coordinates in the
  labels file.

- `train.py`
  Trains the PyTorch model using a ResNet18 backbone with two output heads.

- `predict.py`
  Loads the trained model and predicts the class of a galaxy image.

- `config.py`
  Stores shared settings such as paths, class labels, transforms, and training
  hyperparameters.

- `index.html`
  Simple landing page for the project.

## Python Virtual Environment Setup

It is recommended to create a virtual environment before installing the
dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

After activating the virtual environment, install the required packages:

```bash
pip install torch torchvision pandas pillow matplotlib requests
```

## Data Structure

The project expects data in the following folders:

- `data/raw/`
- `data/interim/`
- `data/processed/train/`
- `data/artifacts/`

Large datasets, downloaded images, and trained model weights are not included in
the GitHub repository.

## How to Use the Project

### 1. Prepare Labels

This creates the cleaned label file from the raw Galaxy Zoo data.

```bash
python prepare_labels.py
```

### 2. Download Galaxy Images

This downloads galaxy cutout images from SDSS.

```bash
python download_images.py --limit 5000
```

You can change the limit if you want fewer or more images.

### 3. Train the Model

This trains the galaxy classifier and saves the model weights.

```bash
python train.py
```

You can also change the number of epochs or batch size:

```bash
python train.py --epochs 12 --batch-size 64
```

### 4. Run Prediction

This uses the trained model to classify a galaxy image.

```bash
python predict.py
```

Or predict a specific image:

```bash
python predict.py data/processed/train/example.jpg
```

## Model Summary

The model uses a shared ResNet18 feature extractor and two output heads:

- one head predicts the main galaxy shape
- one head predicts whether the galaxy is edge-on

This allows the model to learn both tasks from the same input image.

## Notes

- The project currently implements 2 of the 5 Galaxy Zoo classification
  questions.
- The trained weights file is expected in `data/artifacts/`.
- If prediction fails, make sure you have already trained the model or placed
  the correct weights file in the artifacts folder.
