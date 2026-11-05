# Topological ResNet

A customized ResNet architecture engineered to maintain rigid spatial dimensionality across sequential blocks, enabling the empirical application of qualitative dynamical systems theory.

## The Engineering Challenge
Standard ResNet architectures utilize downsampling (Max Pooling, strided convolutions) which aggressively destroys spatial information and topological structure. To evaluate theoretical dynamical systems properties across continuous feature maps, spatial dimensionality must be strictly preserved.

## The Solution: Deterministic Space-to-Depth
This repository implements a custom `SpatialClipper` patch embedding layer (functioning as a deterministic Space-to-Depth operation). It replaces lossy downsampling by rigidly reshaping spatial dimensions into the channel dimension before subsequent convolutional passes, preserving absolute topological integrity for layer-wise feature evolution tracking.

*Note: The core ResNet backbone is adapted from the official PyTorch `torchvision.models` library, modified specifically to bypass internal downsampling modules and integrate the `SpatialClipper` routing.*
