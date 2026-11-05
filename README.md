# Topological ResNet

A customized ResNet architecture engineered to maintain rigid spatial dimensionality across sequential blocks, enabling the empirical application of qualitative dynamical systems theory.

## The Engineering Challenge
Standard ResNet architectures utilize downsampling (Max Pooling, strided convolutions) and global average pooling, which aggressively destroy spatial information and topological structure. To evaluate theoretical dynamical systems properties across continuous feature maps, spatial dimensionality must be strictly preserved throughout the entire network.

## The Solution & Architectural Modifications
This repository modifies standard ResNet-18 and ResNet-50 backbones to maintain uniform data dimensions. Key engineering modifications include:

* **Deterministic Patch Embedding:** Implemented a custom `SpatialClipper` layer (a deterministic Space-to-Depth operation) to handle channel expansion without lossy downsampling.
* **Isolated Group Convolutions:** Routed the patched outputs through grouped convolutions to compute features independently for each spatial patch, preventing premature topological mingling.
* **Stripped Downsampling & Pooling:** Retained the initial convolution and max-pooling stem, but completely eliminated all subsequent downsample functions, channel expansion functions, and the final global average pooling layer to minimize information loss.
* **Bottleneck Alignment:** Upgraded the ResNet-18 architecture to utilize bottleneck blocks, ensuring structural parity with ResNet-50 for strictly controlled comparative experiments.

*Note: The core ResNet backbone is adapted from the official PyTorch `torchvision.models` library, modified specifically to bypass internal downsampling modules and integrate the custom topological routing.*
