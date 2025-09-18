# Autoencoder Optimization Experiments

This document explains the optimizations implemented for the 3D autoencoder architecture.

## Optimization Goals

1. Reduce parameter count while maintaining reconstruction quality
2. Improve memory efficiency
3. Reduce training time
4. Explore different bottleneck architectures

## Implemented Optimizations

### 1. Optimized Encoder Architecture

- **Direct strided convolution start**: Begin with a strided convolution (1->4) instead of an initial non-strided layer
- **Channel reduction blocks**: Add 1x1 convolutions at the end of each block to reduce channels by a factor of 4
- **Progressive channel scaling**: Carefully balance channel counts at each resolution level

```
Input -> (1->4 strided) -> (4->32 non-strided) -> (32->8 reduction) -> ...
```

### 2. Grouped Convolution Bottleneck

Implemented the specific bottleneck architecture with grouped convolutions:

```
latent-space → Reshape(Nlat-space, 1, 1, 1) → Upsampling3D(Nlat-space, Nx, Ny, Nz) 
→ Conv3D(Nlat-space, Nlat-space, (Nx, Ny, Nz), groups = Nlat-space) 
→ Conv3D(Nlat-space, Nchan, 1) → Activation
```

This approach:
- Associates each latent space variable with a "basis image"
- Combines them by weighing them by the latent space intensity
- Significantly reduces parameter count in the bottleneck

### 3. Model Variants

Two new model variants were added:

1. **OptimizedAutoencoder**: Implements the optimized encoder with channel reduction blocks and a standard decoder
2. **GroupedLatentAutoencoder**: Combines the optimized encoder with the grouped convolution bottleneck architecture

## Running Experiments

To run experiments with the optimized architectures:

```bash
python run_optimized_experiment.py --epochs 20 --batch_size 4
```

## Expected Results

The optimized models should show:
- Significantly reduced parameter counts
- Comparable or better reconstruction quality (MSE)
- Faster training times
- Lower memory usage

## Comparison with Original Models

The experiment runner will automatically generate comparison plots showing:
- Validation loss comparison
- Parameter count comparison
- Training time comparison
- Reconstruction quality comparison

Results will be saved in the output directory.
