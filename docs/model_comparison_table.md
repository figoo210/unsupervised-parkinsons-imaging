# Autoencoder Model Comparison Table

| Feature | Direct | Light | Grouped | Efficient | Optimized | Grouped Latent |
|---------|--------|-------|---------|-----------|-----------|----------------|
| **Initial Channels** | 16 | 8 | 16 | 16 | 4 | 4 |
| **Max Channels** | 256 | 128 | 256 | 256 | 64 | 64 |
| **Final Decoder Channels** | 16 | 8 | 16 | 16 | 4 | 4 |
| **Convolution Type** | Standard | Standard | Grouped | Depthwise Separable | Standard + 1x1 | Standard + Grouped |
| **Channel Reduction** | No | No | No | No | Yes (1x1 conv) | Yes (1x1 conv) |
| **Groups Parameter** | 1 | 1 | 4 | in_channels | 1 | 16 (latent space) |
| **Special Features** | Direct connection | Reduced channels | Memory efficiency | Parameter efficiency | Aggressive channel reduction | Specialized latent decoder |
| **Encoder-Decoder Connection** | Direct | Direct | Direct | Direct | Direct | Specialized bottleneck |

## Memory Efficiency Techniques

| Technique | Direct | Light | Grouped | Efficient | Optimized | Grouped Latent |
|-----------|--------|-------|---------|-----------|-----------|----------------|
| **Inplace ReLU** | Yes | Yes | Yes | Yes | Yes | Yes |
| **Reduced Channels** | No | Yes | No | No | Yes | Yes |
| **Grouped Convolutions** | No | No | Yes | Partial | No | Partial |
| **Depthwise Separable** | No | No | No | Yes | No | No |
| **Channel Reduction Blocks** | No | No | No | No | Yes | Yes |
| **Specialized Latent Space** | No | No | No | No | No | Yes |

## Approximate Parameter Count Comparison

| Model | Relative Parameter Count | Memory Efficiency |
|-------|--------------------------|-------------------|
| Direct | 100% (baseline) | Baseline |
| Light | ~25% | Higher than Direct |
| Grouped | ~25% | Higher than Direct |
| Efficient | ~10% | Higher than Grouped |
| Optimized | ~5% | Higher than Efficient |
| Grouped Latent | ~5% | Highest |
