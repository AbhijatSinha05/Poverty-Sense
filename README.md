# 🌍 Poverty-Sense

**AI-Powered Poverty Detection and Analysis Using Satellite Imagery**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

> Leveraging machine learning and geospatial analysis to identify and understand poverty patterns through satellite imagery and geographic data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Technology Stack](#technology-stack)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Poverty-Sense** is an advanced machine learning project that uses satellite imagery and geographic data to detect and analyze poverty levels in different regions. By combining computer vision with geospatial analysis, this project aims to provide data-driven insights that can help policymakers and humanitarian organizations better understand and address poverty.

### Why This Matters

- 📡 **Scalable Analysis**: Analyze large geographic areas without ground surveys
- 🔍 **Data-Driven Insights**: Identify poverty patterns and vulnerable communities
- 🌐 **Global Potential**: Apply machine learning to satellite data from anywhere in the world
- 🤝 **Social Impact**: Support evidence-based policy decisions and resource allocation

---

## ✨ Features

- **Deep Learning Models**: State-of-the-art neural networks for image classification and analysis
- **Geospatial Integration**: OpenStreetMap data and geographic analysis for context
- **Multi-Modal Analysis**: Combines satellite imagery with geographic and infrastructure data
- **Web Interface**: Gradio-based interactive UI for easy model inference
- **Data Pipeline**: Robust data loading and preprocessing utilities
- **Model Training**: Comprehensive training scripts with best practices
- **Visualization**: Advanced plotting and analysis tools for results interpretation

---

## 📁 Project Structure

```
Poverty-Sense/
├── app/                    # Web interface and application logic
├── data/                   # Data loading and preprocessing
├── models/                 # Model definitions and architectures
├── training/               # Training scripts and utilities
├── utils/                  # Helper functions and utilities
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

### Directory Details

| Directory | Purpose |
|-----------|---------|
| `app/` | Gradio web application and inference endpoints |
| `data/` | Data loading, preprocessing, and augmentation |
| `models/` | PyTorch model architectures and components |
| `training/` | Training loops, evaluation, and checkpointing |
| `utils/` | Utility functions for geospatial analysis, visualization, and logging |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- GPU support recommended (CUDA 11.0+)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AbhijatSinha05/Poverty-Sense.git
cd Poverty-Sense
```

### Step 2: Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n poverty-sense python=3.10
conda activate poverty-sense
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

The project uses the following key dependencies:

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework |
| `timm` | Pre-trained model library |
| `torch-geometric` | Graph neural networks |
| `numpy` | Numerical computing |
| `osmnx` | OpenStreetMap data retrieval |
| `contextily` | Basemap tiles for visualization |
| `scikit-learn` | Machine learning utilities |
| `gradio` | Web interface framework |
| `matplotlib`, `Pillow` | Visualization and image processing |

For the complete list, see [requirements.txt](requirements.txt).

---

## 🏃 Quick Start

### 1. Run the Web Interface

```bash
python app/main.py
```

This will start a Gradio web server. Open your browser and navigate to the provided URL (typically `http://localhost:7860`).

### 2. Train a Model

```bash
python training/train.py --config config.yaml
```

### 3. Make Predictions

```python
from models import PovertyDetector
from data import load_satellite_image

# Initialize model
model = PovertyDetector(pretrained=True)

# Load satellite image
image = load_satellite_image("path/to/satellite/image.tif")

# Predict poverty level
prediction = model.predict(image)
print(f"Poverty Score: {prediction['score']:.2%}")
```

---

## 💡 Usage

### Loading Geographic Data

```python
import osmnx as ox
from utils.geo import get_area_features

# Get geographic data for a region
location = "Lagos, Nigeria"
area = ox.geocode_to_gdf(location)

# Extract relevant features
features = get_area_features(area)
```

### Visualizing Results

```python
from utils.visualization import plot_predictions, plot_geospatial

# Plot predictions on satellite imagery
plot_predictions(satellite_image, predictions)

# Create geospatial heat map
plot_geospatial(gdf, poverty_scores)
```

### Custom Model Training

```python
from training.trainer import Trainer
from models import PovertyNet

# Initialize trainer
trainer = Trainer(
    model=PovertyNet(),
    batch_size=32,
    learning_rate=1e-4,
    epochs=100
)

# Train model
trainer.train(train_dataloader, val_dataloader)
```

---

## 🛠️ Technology Stack

### Deep Learning
- **PyTorch**: Core deep learning framework
- **TorchVision**: Pre-trained models and image utilities
- **TIMM**: Timm library for state-of-the-art model architectures
- **Torch Geometric**: Graph neural networks for relational data

### Geospatial Analysis
- **OSMnx**: OpenStreetMap data retrieval and analysis
- **Contextily**: Basemap tiles and satellite imagery
- **XYZServices**: Access to tile services

### ML & Data Science
- **Scikit-learn**: Machine learning utilities and metrics
- **NumPy**: Numerical computing
- **Matplotlib & Pillow**: Visualization and image processing

### User Interface
- **Gradio**: Interactive web-based interface for model inference

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 conventions
- Add docstrings to all functions and classes
- Include type hints where possible
- Write unit tests for new functionality

---

## 📊 Results & Benchmarks

_Results and benchmarks will be added as the project develops._

---

## 🔮 Future Work

- [ ] Integration with real-time satellite data feeds
- [ ] Multi-modal data fusion (infrared, radar, optical)
- [ ] Real-time inference pipeline
- [ ] Mobile application for field verification
- [ ] Multi-country validation and deployment
- [ ] Explainability and interpretability tools
- [ ] Climate resilience indicators

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Abhijat Sinha**

- GitHub: [@AbhijatSinha05](https://github.com/AbhijatSinha05)

---

## 🙏 Acknowledgments

- OpenStreetMap community for geographic data
- PyTorch team for the excellent framework
- All contributors and collaborators

---

## 📞 Contact & Support

For questions, issues, or suggestions:

- Open an issue on [GitHub Issues](https://github.com/AbhijatSinha05/Poverty-Sense/issues)
- Reach out via GitHub discussions

---

## ⭐ Show Your Support

If this project was helpful, please consider giving it a star! Your support motivates continued development.

---

**Last Updated**: May 5, 2026
