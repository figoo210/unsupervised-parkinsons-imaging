# Autoencoder Model Architecture Summary

This document provides a summary of the different autoencoder model architectures used in the project.

## 1. Direct Autoencoder (`direct`)

A memory-optimized 3D Autoencoder for 64×128×128 medical volumes that connects encoder directly to decoder without a latent bottleneck.

### Encoder
- Initial: Conv3D(1→16)
- Down1: Conv3D(16→32, stride=2) → Conv3D(32→32)
- Down2: Conv3D(32→64, stride=2) → Conv3D(64→64)
- Down3: Conv3D(64→128, stride=2) → Conv3D(128→128)
- Down4: Conv3D(128→256, stride=2) → Conv3D(256→256)

### Decoder
- Up1: Upsample(2x) → Conv3D(256→128) → Conv3D(128→128)
- Up2: Upsample(2x) → Conv3D(128→64) → Conv3D(64→64)
- Up3: Upsample(2x) → Conv3D(64→32) → Conv3D(32→32)
- Up4: Upsample(2x) → Conv3D(32→16) → Conv3D(16→16)
- Final: Conv3D(16→1, kernel_size=1)

### Features
- Uses standard 3D convolutions with batch normalization and ReLU
- Progressive channel increase in encoder (16→32→64→128→256)
- Progressive channel decrease in decoder (256→128→64→32→16→1)
- No explicit bottleneck layer

## 2. Light Autoencoder (`light`)

A lightweight 3D Autoencoder with reduced channel counts for 64×128×128 medical volumes.

### Encoder
- Initial: Conv3D(1→8)
- Down1: Conv3D(8→16, stride=2) → Conv3D(16→16)
- Down2: Conv3D(16→32, stride=2) → Conv3D(32→32)
- Down3: Conv3D(32→64, stride=2) → Conv3D(64→64)
- Down4: Conv3D(64→128, stride=2) → Conv3D(128→128)

### Decoder
- Up1: Upsample(2x) → Conv3D(128→64) → Conv3D(64→64)
- Up2: Upsample(2x) → Conv3D(64→32) → Conv3D(32→32)
- Up3: Upsample(2x) → Conv3D(32→16) → Conv3D(16→16)
- Up4: Upsample(2x) → Conv3D(16→8) → Conv3D(8→8)
- Final: Conv3D(8→1, kernel_size=1)

### Features
- Similar to Direct Autoencoder but with fewer channels throughout
- Starts with 8 channels instead of 16
- Final decoder layer has 8 channels instead of 16

## 3. Grouped Convolution Autoencoder (`grouped`)

A memory-efficient 3D Autoencoder using grouped convolutions for 64×128×128 medical volumes.

### Encoder
- Initial: Regular Conv3D(1→16)
- Down1: GroupedConv3D(16→32, stride=2, groups=4) → GroupedConv3D(32→32, groups=4)
- Down2: GroupedConv3D(32→64, stride=2, groups=4) → GroupedConv3D(64→64, groups=4)
- Down3: GroupedConv3D(64→128, stride=2, groups=4) → GroupedConv3D(128→128, groups=4)
- Down4: GroupedConv3D(128→256, stride=2, groups=4) → GroupedConv3D(256→256, groups=4)

### Decoder
- Up1: Upsample(2x) → GroupedConv3D(256→128, groups=4) → GroupedConv3D(128→128, groups=4)
- Up2: Upsample(2x) → GroupedConv3D(128→64, groups=4) → GroupedConv3D(64→64, groups=4)
- Up3: Upsample(2x) → GroupedConv3D(64→32, groups=4) → GroupedConv3D(32→32, groups=4)
- Up4: Upsample(2x) → GroupedConv3D(32→16, groups=4) → GroupedConv3D(16→16, groups=4)
- Final: Regular Conv3D(16→1, kernel_size=1)

### Features
- Uses grouped convolutions (groups=4) for memory efficiency
- Regular convolution only for first and last layers
- Similar channel progression to Direct Autoencoder

## 4. Efficient Autoencoder (`efficient`)

An efficient 3D Autoencoder using depthwise separable convolutions for 64×128×128 medical volumes.

### Encoder
- Initial: Regular Conv3D(1→16)
- Down1: DepthwiseSeparableConv3D(16→32, stride=2) → DepthwiseSeparableConv3D(32→32)
- Down2: DepthwiseSeparableConv3D(32→64, stride=2) → DepthwiseSeparableConv3D(64→64)
- Down3: DepthwiseSeparableConv3D(64→128, stride=2) → DepthwiseSeparableConv3D(128→128)
- Down4: DepthwiseSeparableConv3D(128→256, stride=2) → DepthwiseSeparableConv3D(256→256)

### Decoder
- Up1: Upsample(2x) → DepthwiseSeparableConv3D(256→128) → DepthwiseSeparableConv3D(128→128)
- Up2: Upsample(2x) → DepthwiseSeparableConv3D(128→64) → DepthwiseSeparableConv3D(64→64)
- Up3: Upsample(2x) → DepthwiseSeparableConv3D(64→32) → DepthwiseSeparableConv3D(32→32)
- Up4: Upsample(2x) → DepthwiseSeparableConv3D(32→16) → DepthwiseSeparableConv3D(16→16)
- Final: Regular Conv3D(16→1, kernel_size=1)

### Features
- Uses depthwise separable convolutions (depthwise + pointwise) for memory efficiency
- Each depthwise separable block consists of:
  - Depthwise: Conv3D with groups=in_channels (one filter per input channel)
  - Pointwise: 1x1x1 convolution for channel mixing
- Similar channel progression to Direct Autoencoder

## 5. Optimized Autoencoder (`optimized`)

An optimized 3D Autoencoder implementing multiple memory and performance optimizations.

### Encoder
- Down1: OptimizedConvBlock(1→4, stride=2) → OptimizedConvBlock(4→32) → ChannelReduction(32→8)
- Down2: OptimizedConvBlock(8→16, stride=2) → OptimizedConvBlock(16→16) → ChannelReduction(16→4)
- Down3: OptimizedConvBlock(4→32, stride=2) → OptimizedConvBlock(32→32) → ChannelReduction(32→8)
- Down4: OptimizedConvBlock(8→64, stride=2) → OptimizedConvBlock(64→64) → ChannelReduction(64→16)

### Decoder (LatentSpaceDecoder)
- Up1: Upsample(2x) → OptimizedConvBlock(16→32) → OptimizedConvBlock(32→32) → ChannelReduction(32→8)
- Up2: Upsample(2x) → OptimizedConvBlock(8→16) → OptimizedConvBlock(16→16) → ChannelReduction(16→4)
- Up3: Upsample(2x) → OptimizedConvBlock(4→8) → OptimizedConvBlock(8→8) → ChannelReduction(8→2)
- Up4: Upsample(2x) → OptimizedConvBlock(2→4) → OptimizedConvBlock(4→4)
- Final: Conv3D(4→1, kernel_size=1)

### Features
- Starts directly with strided convolution and fewer filters
- Uses channel reduction blocks (1x1 convolutions) to reduce channels at the end of each block
- Aggressive channel reduction for memory efficiency
- Direct connection between encoder and decoder (no bottleneck)

## 6. Grouped Latent Autoencoder (`grouped_latent`)

An autoencoder with optimized encoder and a specific grouped convolution decoder.

### Encoder
- Same as Optimized Autoencoder encoder

### Decoder (GroupedLatentDecoder)
- Reshape latent vector to (batch_size, latent_dim, 1, 1, 1)
- Upsample to (8, 16, 16) shape
- Apply grouped convolution (groups=latent_dim, each channel gets its own filter)
- Mix channels with 1x1 convolution (latent_dim→16)
- Up1: Upsample(2x) → OptimizedConvBlock(16→8)
- Up2: Upsample(2x) → OptimizedConvBlock(8→4)
- Up3: Upsample(2x) → OptimizedConvBlock(4→4)
- Final: Conv3D(4→1, kernel_size=1)

### Features
- Uses the optimized encoder from the Optimized Autoencoder
- Specialized decoder with grouped convolutions in the latent space
- Implements the specific bottleneck architecture with grouped convolutions
- Aggressive channel reduction throughout the network
