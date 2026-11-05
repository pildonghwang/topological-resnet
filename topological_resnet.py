import torch
import torch.nn as nn
from typing import Type, Union, List, Optional, Callable
from torch import Tensor

class SpatialClipper(nn.Module):
    """
    A deterministic Space-to-Depth patch embedding layer.
    
    Replaces lossy downsampling (like MaxPool or strided convolutions) by rigidly 
    reshaping spatial dimensions into the channel dimension. This preserves absolute 
    topological integrity for dynamical systems stability analysis in ResNets.
    """
    def __init__(self, h_split: int, w_split: int):
        super().__init__()
        self.h_split = h_split
        self.w_split = w_split

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        
        # Ensure dimensions are divisible by the split constraints
        assert h % self.h_split == 0, f"Height {h} must be divisible by {self.h_split}"
        assert w % self.w_split == 0, f"Width {w} must be divisible by {self.w_split}"
        
        h_size = h // self.h_split
        w_size = w // self.w_split
        
        # Vectorized space-to-depth operation (replaces nested loops for GPU efficiency)
        x = x.view(b, c, self.h_split, h_size, self.w_split, w_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(b, c * (self.h_split * self.w_split), h_size, w_size)
        
        return x


class TopologicalResNet(nn.Module):
    """
    A modified ResNet architecture engineered to maintain rigid spatial dimensionality 
    across sequential blocks, enabling the empirical application of qualitative 
    dynamical systems theory (g-expansive, g-shadowing, and Lipschitz continuity).
    """
    def __init__(
        self,
        block: Type[Union['BasicBlock', 'Bottleneck']], # Assuming PyTorch base blocks are imported
        layers: List[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial stem
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Custom deterministic patch embedding layers replacing standard downsampling
        self.clip1 = SpatialClipper(2, 2)
        self.clip2 = SpatialClipper(2, 1)
        self.clip3 = SpatialClipper(1, 2)

        # Residual layers (stripped of internal stride/downsampling)
        self.layer1 = self._make_layer(block, 64, layers[0], groups=4)
        self.layer2 = self._make_layer(block, 128, layers[1], groups=8)
        self.layer3 = self._make_layer(block, 256, layers[2], groups=16)
        self.layer4 = self._make_layer(block, 512, layers[3], groups=32)
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes: int, blocks: int, groups: int = None) -> nn.Sequential:
        norm_layer = self._norm_layer
        if groups is None:
            groups = self.groups

        layers = [block(self.inplanes, planes, groups=groups, base_width=self.base_width, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=groups, base_width=self.base_width, norm_layer=norm_layer
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Topological block routing
        x = self.clip1(x)
        x = self.layer1(x)
        
        x = self.clip2(x)
        x = self.layer2(x)
        
        x = self.clip3(x)
        x = self.layer3(x)
        
        x = self.clip2(x)
        x = self.layer4(x)

        # Classifier head
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
