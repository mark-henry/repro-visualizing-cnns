import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class LayerState:
    """Container for layer's intermediate state during forward pass"""
    output: torch.Tensor
    pre_pool: Optional[torch.Tensor] = None
    pool_indices: Optional[torch.Tensor] = None

class ConvLayer(nn.Module):
    """A single convolutional layer with pooling and normalization"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        
    def forward(self, x):
        """Forward pass through the layer
        
        Args:
            x: Input tensor
            
        Returns:
            LayerState containing output and intermediate states
        """
        # Convolution and activation
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        # Store pre-pool features
        pre_pool = x
        
        # Pooling
        x, indices = self.pool(x)
        
        return LayerState(output=x, pre_pool=pre_pool, pool_indices=indices)

class DeconvLayer(nn.Module):
    """A single deconvolutional layer with unpooling"""
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=pool_size, stride=2)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                       kernel_size=kernel_size, stride=1, padding=1)
        
    def forward(self, x, max_indices, pre_pool_size):
        # Ensure feature map dimensions match pooling indices
        if x.shape != max_indices.shape:
            x = F.interpolate(x, size=max_indices.shape[2:], mode='nearest')
            
        x = self.unpool(x, max_indices, output_size=pre_pool_size)
        x = F.relu(x)
        x = self.deconv(x)
        return x 