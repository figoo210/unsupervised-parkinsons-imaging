# 3D Autoencoder: Selected Model Overview

## Goal of the module
- **Purpose** Compress and reconstruct 3D brain scans with several encoder/decoder flavors so we can test different latent bottlenecks.
- **Input shape** Expected tensors follow `(batch, channels=1, depth=64, height=128, width=128)`.
- **Output shape** All decoders rebuild tensors with the same spatial size and one channel.

## Helper utilities
- **`_compute_ladder_shapes()`** Plans the spatial sizes used while upsampling. It starts from a small tensor and grows it step by step so we land exactly on the final shape. This keeps every decoder stage aligned with the target dimensions.

## Shared building blocks
- **`BaseConvBlock`** A `Conv3d → BatchNorm3d → ReLU` trio. Kernel size is 3 with padding 1, so the spatial size stays unchanged. ReLU is in-place to save memory.
- **`BaseChannelReductionBlock`** A fast channel compressor using a `1×1×1` convolution. Default reduction factor is 4, so 32 channels shrink to 8, etc.

## Encoders
- **`BaseEncoder`** Four downsampling stages. Each stage halves the spatial size (via stride 2), boosts channels, then uses `BaseChannelReductionBlock` to keep the channel count manageable. Final latent map has shape `(batch, 128, 4, 8, 8)`.
- **`BottleneckEncoder`** Wraps `BaseEncoder` but finishes with adaptive average pooling to squeeze the latent map into `(batch, latent_dim, 1, 1, 1)`. Optional `1×1×1` projection matches any chosen `latent_dim`.

## Decoders
- **`BaseDecoder`** Mirror of `BaseEncoder`. It repeatedly upsamples by factor 2 using trilinear resize, then applies `BaseConvBlock`. A final `1×1×1` convolution produces the reconstruction.
- **`BottleneckDecoder`** Starts from a `(1,1,1)` latent cube. Steps:
  1. Resize to the small ladder shape.
  2. Run a depthwise grouped `Conv3d` that lets each latent channel paint its own basis patch.
  3. Mix channels with a `1×1×1` convolution and ReLU.
  4. Walk through the upsampling ladder computed by `_compute_ladder_shapes()`.
- **`BottleneckBasisDecoder`** Variation where the grouped convolution uses the full initial spatial kernel. Supports different group settings: full depthwise, single group, partial, or custom int.
- **`BottleneckDeconvDecoder`** Uses a single `ConvTranspose3d` to inflate the latent vector to the ladder entry shape before regular upsampling blocks.
- **`BottleneckHybridDecoder`** Combines components: trilinear resize, depthwise `3×3×3` conv, channel mix, small `ConvTranspose3d`, then the ladder.

## Parkinson imaging scenarios
- **`BaseModel` (spatial path)** Good first pass for DaT-SPECT volumes. Keeping a spatial latent map helps the network learn smooth intensity trends across basal ganglia without losing anatomy, which is useful for baseline reconstruction and denoising tasks.
- **`BottleneckEncoder` + `BottleneckDecoder`** The hard squeeze to `(1,1,1)` makes the latent vector easy to track over time. You can cluster those vectors to separate control vs prodromal vs diagnosed cases, or monitor therapy response using longitudinal scans.
- **`BottleneckBasisDecoder`** Depthwise filters per latent channel are handy when you want each latent to focus on a specific structure (e.g., substantia nigra, putamen, cerebellum). Switching group modes lets you test how much cross-structure mixing improves tremor severity prediction.
- **`BottleneckDeconvDecoder`** ConvTranspose3d gives the decoder more freedom to paint fine-grained uptake patterns. That is helpful when working with high-resolution neuromelanin-sensitive MRI where small signal changes matter.
- **`BottleneckHybridDecoder`** Mixes local texture modeling (depthwise conv) with small deconvs, so it copes better with mixed-modal datasets (MRI combined with synthetic dopaminergic maps). It can reduce ringing artifacts that often confuse volumetric classifiers.
- **Latent inspections** Whichever decoder you choose, the latent tensors can feed downstream Parkinson tasks: anomaly scoring for early diagnosis, region-wise attention maps to explain hypointensity zones, or as inputs to graph models.

## End-to-end wrappers
- **`BaseModel`** Switches between two autoencoder topologies:
  - Spatial path: `BaseEncoder` + `BaseDecoder` keeps a spatial latent map.
  - Bottleneck path: `BottleneckEncoder` + `BottleneckDecoder` collapses the latent into a vector before decoding.
- **`SpatialAE`** Generic wrapper that chains any encoder producing a spatial tensor with a matching decoder.
- **`BottleneckAE`** Wrapper for encoder/decoder pairs that pass around `(batch, latent_dim, bz, by, bx)` tensors (default `1×1×1`).

## Practical notes
- **Latent size control** Adjust `initial_filters` on encoders and `latent_dim` on bottleneck variants to balance compression and detail.
- **Group choices** Grouped convolutions let us experiment with how much interaction we want between latent channels before full decoding.
- **Interpolation vs deconvolution** We provide both resize-based and transposed-convolution paths so you can probe artifacts or sharpness differences in reconstructions.
