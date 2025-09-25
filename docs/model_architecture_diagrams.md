# Autoencoder Model Architecture Diagrams

## 1. Direct Autoencoder (`direct`)

```
Input (1×64×128×128)
    │
    ▼
Conv3D(1→16) + BN + ReLU
    │
    ▼
Conv3D(16→32, stride=2) + BN + ReLU → Conv3D(32→32) + BN + ReLU
    │
    ▼
Conv3D(32→64, stride=2) + BN + ReLU → Conv3D(64→64) + BN + ReLU
    │
    ▼
Conv3D(64→128, stride=2) + BN + ReLU → Conv3D(128→128) + BN + ReLU
    │
    ▼
Conv3D(128→256, stride=2) + BN + ReLU → Conv3D(256→256) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(256→128) + BN + ReLU → Conv3D(128→128) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(128→64) + BN + ReLU → Conv3D(64→64) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(64→32) + BN + ReLU → Conv3D(32→32) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(32→16) + BN + ReLU → Conv3D(16→16) + BN + ReLU
    │
    ▼
Conv3D(16→1, kernel_size=1)
    │
    ▼
Output (1×64×128×128)
```

## 2. Light Autoencoder (`light`)

```
Input (1×64×128×128)
    │
    ▼
Conv3D(1→8) + BN + ReLU
    │
    ▼
Conv3D(8→16, stride=2) + BN + ReLU → Conv3D(16→16) + BN + ReLU
    │
    ▼
Conv3D(16→32, stride=2) + BN + ReLU → Conv3D(32→32) + BN + ReLU
    │
    ▼
Conv3D(32→64, stride=2) + BN + ReLU → Conv3D(64→64) + BN + ReLU
    │
    ▼
Conv3D(64→128, stride=2) + BN + ReLU → Conv3D(128→128) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(128→64) + BN + ReLU → Conv3D(64→64) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(64→32) + BN + ReLU → Conv3D(32→32) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(32→16) + BN + ReLU → Conv3D(16→16) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(16→8) + BN + ReLU → Conv3D(8→8) + BN + ReLU
    │
    ▼
Conv3D(8→1, kernel_size=1)
    │
    ▼
Output (1×64×128×128)
```

## 3. Grouped Convolution Autoencoder (`grouped`)

```
Input (1×64×128×128)
    │
    ▼
Conv3D(1→16) + BN + ReLU  # Regular convolution
    │
    ▼
GroupedConv3D(16→32, stride=2, groups=4) + BN + ReLU → GroupedConv3D(32→32, groups=4) + BN + ReLU
    │
    ▼
GroupedConv3D(32→64, stride=2, groups=4) + BN + ReLU → GroupedConv3D(64→64, groups=4) + BN + ReLU
    │
    ▼
GroupedConv3D(64→128, stride=2, groups=4) + BN + ReLU → GroupedConv3D(128→128, groups=4) + BN + ReLU
    │
    ▼
GroupedConv3D(128→256, stride=2, groups=4) + BN + ReLU → GroupedConv3D(256→256, groups=4) + BN + ReLU
    │
    ▼
Upsample(2x) → GroupedConv3D(256→128, groups=4) + BN + ReLU → GroupedConv3D(128→128, groups=4) + BN + ReLU
    │
    ▼
Upsample(2x) → GroupedConv3D(128→64, groups=4) + BN + ReLU → GroupedConv3D(64→64, groups=4) + BN + ReLU
    │
    ▼
Upsample(2x) → GroupedConv3D(64→32, groups=4) + BN + ReLU → GroupedConv3D(32→32, groups=4) + BN + ReLU
    │
    ▼
Upsample(2x) → GroupedConv3D(32→16, groups=4) + BN + ReLU → GroupedConv3D(16→16, groups=4) + BN + ReLU
    │
    ▼
Conv3D(16→1, kernel_size=1)  # Regular convolution
    │
    ▼
Output (1×64×128×128)
```

## 4. Efficient Autoencoder (`efficient`)

```
Input (1×64×128×128)
    │
    ▼
Conv3D(1→16) + BN + ReLU  # Regular convolution
    │
    ▼
DepthwiseConv3D(16→16, stride=2, groups=16) + BN + ReLU → PointwiseConv3D(16→32, 1x1x1) + BN + ReLU →
DepthwiseConv3D(32→32, groups=32) + BN + ReLU → PointwiseConv3D(32→32, 1x1x1) + BN + ReLU
    │
    ▼
DepthwiseConv3D(32→32, stride=2, groups=32) + BN + ReLU → PointwiseConv3D(32→64, 1x1x1) + BN + ReLU →
DepthwiseConv3D(64→64, groups=64) + BN + ReLU → PointwiseConv3D(64→64, 1x1x1) + BN + ReLU
    │
    ▼
DepthwiseConv3D(64→64, stride=2, groups=64) + BN + ReLU → PointwiseConv3D(64→128, 1x1x1) + BN + ReLU →
DepthwiseConv3D(128→128, groups=128) + BN + ReLU → PointwiseConv3D(128→128, 1x1x1) + BN + ReLU
    │
    ▼
DepthwiseConv3D(128→128, stride=2, groups=128) + BN + ReLU → PointwiseConv3D(128→256, 1x1x1) + BN + ReLU →
DepthwiseConv3D(256→256, groups=256) + BN + ReLU → PointwiseConv3D(256→256, 1x1x1) + BN + ReLU
    │
    ▼
Upsample(2x) → DepthwiseSeparableConv3D(256→128) + BN + ReLU → DepthwiseSeparableConv3D(128→128) + BN + ReLU
    │
    ▼
Upsample(2x) → DepthwiseSeparableConv3D(128→64) + BN + ReLU → DepthwiseSeparableConv3D(64→64) + BN + ReLU
    │
    ▼
Upsample(2x) → DepthwiseSeparableConv3D(64→32) + BN + ReLU → DepthwiseSeparableConv3D(32→32) + BN + ReLU
    │
    ▼
Upsample(2x) → DepthwiseSeparableConv3D(32→16) + BN + ReLU → DepthwiseSeparableConv3D(16→16) + BN + ReLU
    │
    ▼
Conv3D(16→1, kernel_size=1)  # Regular convolution
    │
    ▼
Output (1×64×128×128)
```

## 5. Optimized Autoencoder (`optimized`)

```
Input (1×64×128×128)
    │
    ▼
Conv3D(1→4, stride=2) + BN + ReLU → Conv3D(4→32) + BN + ReLU → Conv3D(32→8, 1x1x1) + BN + ReLU
    │
    ▼
Conv3D(8→16, stride=2) + BN + ReLU → Conv3D(16→16) + BN + ReLU → Conv3D(16→4, 1x1x1) + BN + ReLU
    │
    ▼
Conv3D(4→32, stride=2) + BN + ReLU → Conv3D(32→32) + BN + ReLU → Conv3D(32→8, 1x1x1) + BN + ReLU
    │
    ▼
Conv3D(8→64, stride=2) + BN + ReLU → Conv3D(64→64) + BN + ReLU → Conv3D(64→16, 1x1x1) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(16→32) + BN + ReLU → Conv3D(32→32) + BN + ReLU → Conv3D(32→8, 1x1x1) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(8→16) + BN + ReLU → Conv3D(16→16) + BN + ReLU → Conv3D(16→4, 1x1x1) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(4→8) + BN + ReLU → Conv3D(8→8) + BN + ReLU → Conv3D(8→2, 1x1x1) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(2→4) + BN + ReLU → Conv3D(4→4) + BN + ReLU
    │
    ▼
Conv3D(4→1, kernel_size=1)
    │
    ▼
Output (1×64×128×128)
```

## 6. Grouped Latent Autoencoder (`grouped_latent`)

```
Input (1×64×128×128)
    │
    ▼
# Encoder (same as Optimized Autoencoder)
Conv3D(1→4, stride=2) + BN + ReLU → Conv3D(4→32) + BN + ReLU → Conv3D(32→8, 1x1x1) + BN + ReLU
    │
    ▼
Conv3D(8→16, stride=2) + BN + ReLU → Conv3D(16→16) + BN + ReLU → Conv3D(16→4, 1x1x1) + BN + ReLU
    │
    ▼
Conv3D(4→32, stride=2) + BN + ReLU → Conv3D(32→32) + BN + ReLU → Conv3D(32→8, 1x1x1) + BN + ReLU
    │
    ▼
Conv3D(8→64, stride=2) + BN + ReLU → Conv3D(64→64) + BN + ReLU → Conv3D(64→16, 1x1x1) + BN + ReLU
    │
    ▼
# Specialized Decoder
Reshape to (batch_size, 16, 1, 1, 1)
    │
    ▼
Upsample to (16, 8, 16, 16)
    │
    ▼
GroupedConv3D(16→16, groups=16) # Each channel gets its own filter
    │
    ▼
Conv3D(16→16, 1x1x1) # Mix channels
    │
    ▼
ReLU
    │
    ▼
Upsample(2x) → Conv3D(16→8) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(8→4) + BN + ReLU
    │
    ▼
Upsample(2x) → Conv3D(4→4) + BN + ReLU
    │
    ▼
Conv3D(4→1, kernel_size=1)
    │
    ▼
Output (1×64×128×128)
```
