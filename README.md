# CamoXpert

CamoXpert is a **dynamically adaptive neural network for camouflaged object detection**, implemented in PyTorch. It combines convolutional and attention-based modules to achieve high accuracy while remaining lightweight, making it suitable for resource-constrained devices.

---

## Features

- Lightweight and efficient backbone
- Multi-scale feature extraction
- Adaptive modules for accurate camouflaged object detection
- Easy to set up and test

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/emon210234/camoxpert.git
cd camoxpert
```
### 2. Install Python 3.12

Fedora:
```bash
sudo dnf install python3.12
```

Windows:
- Go to the official Python website: https://www.python.org/downloads/windows/
- Download the installer for Python 3.12
- Run the installer:
  - Check “Add Python 3.12 to PATH”
  - Choose “Install Now” or “Customize Installation” if needed


macOS:
```bash
brew install python@3.12
```

### 3. Create a virtual environment

Fedora:
```bash
python3.12 -m venv camox-env
source camox-env/bin/activate
```

Windows:
```bash
python -m venv camox-env
camox-env\Scripts\activate
```

### 4. Installing dependencies
```bash
pip install -r requirements.txt
```

### 5. Download the dataset

Download the COD10K dataset from Kaggle: [https://www.kaggle.com/datasets/boatshuai/cod10k-v3/data]
Place the dataset in the data directory of the cloned repository and rename folders as follows:
```bash
data/cod10k-v3/
├── train/
│   ├── Images/
│   ├── GT_Edge/
│   ├── GT_Instance/
│   └── GT_Object/
└── test/
    ├── Images/
    ├── GT_Edge/
    ├── GT_Instance/
    └── GT_Object/
```

Your final directory structure should look like:

```bash
camoxpert/
├── camox-env/
├── data/
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
├── main.py
└── requirements.txt
```

### 6. Quick test
Run the following command to test the model:
```bash
python3.12 main.py --test_model --img_size 256
```

Expected output:

```bash
EdgeNeXt-S model created successfully!
Number of parameters: 4.05M
Feature shapes:
Stage 1: torch.Size([1, 48, 64, 64])
Stage 2: torch.Size([1, 96, 32, 32])
Stage 3: torch.Size([1, 160, 16, 16])
Stage 4: torch.Size([1, 256, 8, 8])
Classification output shape: torch.Size([1, 1000])
```
### 7. Current connection status

- Connected:
  - Input Pipeline → Backbone
      - dataset.py loads and preprocesses images
      - main.py creates DataLoader feeding batches to the model
  - Backbone → Output
      - Multi-scale features and classification output verified with --test_model

- Not Yet Connected:
  - Backbone Features → Expert Modules
  - Expert Modules → Fusion Head
  - Fusion Head → Segmentation Output
