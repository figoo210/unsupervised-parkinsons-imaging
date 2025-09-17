# Autoencoder Architecture Optimization for Medical Imaging

This project implements a systematic approach to optimize autoencoder architectures for DAT-scan medical images. The implementation follows a structured methodology to compare different architectures, optimize latent space dimensions, and explore VAE parameters.

## Project Structure

- `data/`: Data loading and preprocessing modules
  - `data_ingestion.py`: Functions for loading DICOM files
  - `dataloader.py`: PyTorch DataLoader implementations
  - `data_visualization.py`: Visualization utilities for data exploration
- `models/`: Model implementations and experiment modules
  - `autoencoder/`: Autoencoder model implementations
    - `direct_ae.py`: Autoencoder without bottleneck
    - `model_variants.py`: Different architecture variants
  - `vae/`: Variational autoencoder implementations
    - `beta_experiments.py`: VAE parameter exploration
  - `experiment_tracker.py`: Experiment tracking utilities
  - `latent_optimization.py`: Latent space dimension experiments
- `visualization/`: Visualization utilities
  - `reconstruction_visualizer.py`: Tools for visualizing reconstructions
  - `experiment_plotter.py`: Tools for plotting experiment results
- `utils/`: Utility functions
- `experiment_runner.py`: Main script for running experiments

## Setup and Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

The project implements four types of experiments:

1. **Architecture Comparison**: Compare different autoencoder architectures
2. **Latent Space Optimization**: Explore different latent space dimensions
3. **Beta Parameter Exploration**: Test different beta values for VAE
4. **Advanced VAE Configurations**: Explore advanced VAE configurations

### Architecture Comparison

This experiment compares different autoencoder architectures:
- Direct: Autoencoder without bottleneck
- Light: Lightweight autoencoder with reduced channels
- Grouped: Autoencoder with grouped convolutions
- Efficient: Autoencoder with depthwise separable convolutions

```bash
python experiment_runner.py --experiment architecture --data_dir data/Images --mask_path data/masks/rmask_ICV.nii --epochs 10
```

### Latent Space Optimization

This experiment tests different latent space dimensions (512, 256, 128, 64, 32) to find the optimal size:

```bash
python experiment_runner.py --experiment latent --data_dir data/Images --mask_path data/masks/rmask_ICV.nii --epochs 10
```

### Beta Parameter Exploration

This experiment explores different beta values for VAE to balance reconstruction quality and latent space regularization:

```bash
python experiment_runner.py --experiment beta --data_dir data/Images --mask_path data/masks/rmask_ICV.nii --epochs 10
```

### Advanced VAE Configurations

This experiment tests advanced VAE configurations like cyclical annealing and free bits approach:

```bash
python experiment_runner.py --experiment advanced_vae --data_dir data/Images --mask_path data/masks/rmask_ICV.nii --epochs 10
```

## Experiment Results

All experiment results are saved in the `output/Experiments` directory, organized by experiment type:

- `output/Experiments/Architecture/`: Architecture comparison results
- `output/Experiments/LatentSpace/`: Latent space optimization results
- `output/Experiments/VAEBeta/`: Beta parameter exploration results
- `output/Experiments/AdvancedVAE/`: Advanced VAE configuration results

Each experiment directory contains:
- `metadata.json`: Experiment metadata
- `training_history.csv`: Training metrics
- `training_curves.png`: Training loss curves
- `reconstructions/`: Sample reconstructions
- Experiment-specific plots and visualizations

## Analyzing Results

To analyze experiment results, you can use the visualization utilities:

```python
from visualization.experiment_plotter import ExperimentPlotter
from visualization.reconstruction_visualizer import ReconstructionVisualizer

# Plot experiment comparison
plotter = ExperimentPlotter()
plotter.plot_experiment_comparison(
    ["output/Experiments/Architecture/AE_direct_20250917_022500",
     "output/Experiments/Architecture/AE_light_20250917_022600"],
    metric="val_loss",
    title="Architecture Comparison",
    save_path="output/Visualizations/architecture_comparison.png"
)

# Create summary report
plotter.create_summary_report(
    "output/Experiments/Architecture/AE_direct_20250917_022500",
    output_path="output/Reports/direct_ae_summary.md"
)
```

## Experiment Workflow

The recommended workflow for model optimization is:

1. **Architecture Comparison**: Start by testing different autoencoder architectures without a bottleneck to find the lightest design that achieves good reconstruction quality.

2. **Latent Space Optimization**: After finding the best architecture, run experiments with different latent space dimensions (512 → 256 → 128 → 64 → 32) to identify the threshold where error increases significantly.

3. **VAE Parameter Exploration**: Once the optimal architecture and latent size are determined, explore different beta values in the VAE to balance reconstruction quality and latent space regularization.

4. **Advanced VAE Configurations**: Finally, test advanced VAE configurations like cyclical annealing and free bits approach to further improve the model.

## Documentation

Each experiment generates comprehensive documentation including:
- Quantitative metrics (MSE, parameter count, training time)
- Visual examples (original vs. reconstructed images)
- Error maps and analysis
- Comparison plots

This documentation helps in understanding the trade-offs between different architectures and parameter settings.
