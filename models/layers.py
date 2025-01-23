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

class LocalContrastNorm(nn.Module):
    """Local contrast normalization as described in Jarrett et al."""
    def __init__(self, num_features, kernel_size=9):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Create gaussian kernel for local averaging
        kernel = self._create_gaussian_kernel(kernel_size)
        self.register_buffer('kernel', kernel.expand(num_features, 1, -1, -1))
    
    def _create_gaussian_kernel(self, size, sigma=1.0):
        coords = torch.arange(size).float() - (size - 1) / 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        return kernel / kernel.sum()
    
    def forward(self, x):
        # Subtractive normalization
        local_mean = F.conv2d(x, self.kernel, padding='same', groups=x.size(1))
        centered = x - local_mean
        
        # Calculate local standard deviation
        local_var = F.conv2d(centered.pow(2), self.kernel, padding='same', groups=x.size(1))
        local_std = torch.sqrt(local_var)
        
        # Set c to mean(σjk) for each sample as per paper
        c = local_std.mean(dim=(1, 2, 3), keepdim=True)
        
        # Divisive normalization using max(c, σjk)
        normalized = centered / torch.maximum(c, local_std)
        
        return normalized

class ConvLayer(nn.Module):
    """A single convolutional layer with pooling and normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=1, padding=1)
        self.norm = LocalContrastNorm(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=2, return_indices=True)
        
    def forward(self, x: torch.Tensor) -> LayerState:
        """Forward pass through the layer
        
        Args:
            x: Input tensor
            
        Returns:
            LayerState containing output and intermediate states
        """
        # Convolution and normalization
        x = F.relu(self.conv(x))
        x = self.norm(x)
        pre_pool = x
        
        # Pooling
        x, indices = self.pool(x)
        
        return LayerState(x, pre_pool, indices)

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