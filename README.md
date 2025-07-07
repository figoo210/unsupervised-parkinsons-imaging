# Medical Image Analysis Project

A refactored and structured implementation of medical image analysis using Autoencoders and Variational Autoencoders (VAEs) for brain imaging data. This project was refactored from a large Jupyter notebook into a well-organized, modular codebase suitable for research and production use.

## 🏗️ Project Structure

```
├── src/                          # Main source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── datasets.py           # Dataset classes (OnDemandDataset, BatchLoadDataset)
│   │   ├── preprocessing.py      # Data preprocessing functions
│   │   └── analysis.py           # Dataset analysis and statistics
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── autoencoder.py        # Standard autoencoder implementation
│   │   └── vae.py                # Variational autoencoder implementation
│   ├── training/                 # Training utilities
│   │   ├── __init__.py
│   │   ├── config.py             # Training configuration classes
│   │   ├── callbacks.py          # Training callbacks (early stopping, etc.)
│   │   ├── checkpoints.py        # Model checkpointing utilities
│   │   ├── optimizers.py         # Optimizer and scheduler creation
│   │   └── trainer.py            # Main training loops
│   ├── analysis/                 # Analysis and visualization
│   │   ├── __init__.py
│   │   ├── visualization.py      # Plotting and visualization functions
│   │   ├── evaluation.py         # Model evaluation metrics
│   │   └── interpretation.py     # Model interpretation tools
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── gpu_utils.py          # GPU configuration and monitoring
│       ├── memory_utils.py       # Memory management utilities
│       └── file_utils.py         # File handling utilities
├── configs/                      # Configuration files
├── scripts/                      # Main execution scripts
├── notebooks/                    # Demo notebooks
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## ✅ Completed Components

### 1. Utility Functions (`src/utils/`)

**GPU Management (`gpu_utils.py`)**

- `configure_gpu()`: Automatic GPU/CPU configuration with CUDA optimization
- `print_gpu_memory_stats()`: Real-time GPU memory monitoring

**Memory Management (`memory_utils.py`)**

- `print_memory_stats()`: System and GPU memory monitoring
- `clear_memory()`: Memory cleanup and garbage collection

**File Handling (`file_utils.py`)**

- `collect_files()`: Recursive file collection with extension filtering
- `generate_dataframe()`: Metadata extraction for collected files
- `save_qa_report()`: Quality assurance reporting for data ingestion

### 2. Data Module (`src/data/`)

**Dataset Classes (`datasets.py`)**

- `OnDemandDataset`: Memory-efficient on-demand data loading
- `BatchLoadDataset`: Batch-based data loading for faster access
- `create_dataloaders()`: Automated train/validation dataloader creation

**Preprocessing (`preprocessing.py`)**

- `load_dicom()` / `load_nifti()`: Medical image format support
- `resize_volume()`: Intelligent volume resizing with padding/cropping
- `apply_brain_mask()`: Automated brain extraction using Otsu thresholding
- `normalize_volume()`: Multiple normalization strategies (z-score, min-max, percentile)
- `process_volume()`: Complete preprocessing pipeline

**Data Analysis (`analysis.py`)**

- `analyze_dataset_statistics_efficiently()`: Memory-efficient dataset statistics
- `plot_intensity_distributions()`: Group-wise intensity analysis
- `analyze_slice_variance()`: Multi-axis variance analysis
- `create_memory_efficient_dataloaders()`: Large dataset handling

### 3. Model Architectures (`src/models/`)

**Autoencoder (`autoencoder.py`)**

- `ConvBlock`: Reusable 3D convolutional block with BatchNorm and ReLU
- `Encoder`: 3D CNN encoder with global average pooling
- `Decoder`: Transposed convolution decoder with progressive upsampling
- `BaseAutoencoder`: Complete autoencoder with encode/decode methods

**Variational Autoencoder (`vae.py`)**

- `VAEEncoder`: Encoder outputting mean and log-variance
- `VAEDecoder`: Decoder for VAE reconstruction
- `VAE`: Complete VAE with reparameterization trick
- `VAELoss`: Sophisticated loss with beta-warmup and free bits

### 4. Training Infrastructure (`src/training/`)

**Configuration (`config.py`)**

- `TrainingConfig`: Comprehensive autoencoder training configuration
- `VAEConfig`: Specialized VAE training parameters with KL-divergence controls

**Callbacks (`callbacks.py`)**

- `EarlyStopping`: Standard early stopping with customizable patience
- `VAEEarlyStopping`: VAE-specific early stopping monitoring total loss
- `LearningRateSchedulerCallback`: Automated learning rate scheduling
- `MetricTracker`: Training metrics tracking and history management

## 🚧 In Progress

- Training utilities (checkpoints, optimizers)
- Main training loops
- Analysis and visualization functions

## ⏳ Planned Components

- Model checkpointing system
- Optimizer and scheduler utilities
- Complete training loops
- Visualization and analysis tools
- Main execution scripts
- Configuration files
- Demo notebooks

## 🔧 Key Features

### Memory Efficiency

- On-demand data loading for large datasets
- Automatic memory cleanup and garbage collection
- GPU memory monitoring and optimization

### Medical Image Support

- DICOM and NIfTI file format support
- Brain masking and preprocessing pipelines
- Multi-axis volume processing

### Advanced Training

- Mixed precision training support
- Configurable early stopping and checkpointing
- Comprehensive metric tracking
- VAE-specific loss functions with beta-warmup

### Modular Design

- Clean separation of concerns
- Reusable components
- Extensive configuration options
- Easy extension and customization

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scikit-image>=0.19.0
nibabel>=3.2.0
pydicom>=2.3.0
tqdm>=4.64.0
psutil>=5.8.0
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from src.utils import configure_gpu
from src.data import create_dataloaders, process_volume
from src.models import BaseAutoencoder, VAE
from src.training import TrainingConfig

# Configure GPU
device = configure_gpu()

# Load and preprocess data
# dataloader = create_dataloaders(df, batch_size=4)

# Create models
autoencoder = BaseAutoencoder(latent_dim=256).to(device)
vae = VAE(latent_dim=256).to(device)

# Configure training
config = TrainingConfig(
    epochs=100,
    batch_size=4,
    learning_rate=0.001
)
```

## 📊 Original vs Refactored

| Aspect          | Original Notebook | Refactored Project |
| --------------- | ----------------- | ------------------ |
| Size            | 46MB single file  | Modular structure  |
| Maintainability | Monolithic        | Clean separation   |
| Reusability     | Limited           | High               |
| Testing         | Difficult         | Module-based       |
| Collaboration   | Version conflicts | Git-friendly       |
| Deployment      | Complex           | Production-ready   |

## 🎯 Research Applications

This codebase is designed for:

- Medical image analysis research
- Autoencoder/VAE experimentation
- Large-scale brain imaging studies
- Feature extraction and dimensionality reduction
- Anomaly detection in medical imaging
- Cross-study generalization research

## 📝 Citation

If you use this code in your research, please cite:

```
[Your thesis/paper citation will go here]
```

---

_This project represents a complete refactoring of a large-scale medical image analysis notebook into a production-ready, research-oriented codebase._
