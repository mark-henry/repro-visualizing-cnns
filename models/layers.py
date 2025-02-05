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
    """A single convolutional layer with pooling and contrast normalization"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        # Calculate padding to maintain spatial dimensions
        padding = ((stride - 1) + kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.radius = 1e-1  # Fixed radius for filter renormalization (original value from paper)
        
    def renormalize_filters(self):
        """Renormalize filters whose RMS exceeds fixed radius back to that radius"""
        with torch.no_grad():
            # Calculate RMS of each filter
            weight = self.conv.weight
            rms = torch.sqrt(torch.mean(weight.pow(2), dim=[1,2,3]))
            
            # Find filters exceeding radius
            mask = rms > self.radius
            
            # Rescale those filters
            scale = self.radius / rms
            scale[~mask] = 1.0  # Don't change filters within radius
            
            # Apply scaling to each filter
            self.conv.weight.data *= scale.view(-1, 1, 1, 1)
        
    def contrast_normalize(self, x, size=5, k=1):
        """Local contrast normalization as per ZF2013
        Args:
            x: Input tensor [batch, channels, height, width]
            size: Size of local region for normalization (paper uses 5x5)
            k: Constant added to denominator for numerical stability
        """
        # Calculate local mean
        kernel = torch.ones(1, 1, size, size, device=x.device) / (size * size)
        padding = size // 2
        
        # For each channel separately
        normalized = []
        for c in range(x.shape[1]):
            # Get this channel
            xc = x[:, c:c+1]  # Keep dim
            
            # Calculate mean using average pooling
            mean = F.conv2d(xc, kernel, padding=padding)
            
            # Subtract mean (local mean subtraction)
            centered = xc - mean
            
            # Calculate local standard deviation
            var = F.conv2d(centered ** 2, kernel, padding=padding)
            std = torch.sqrt(var + k)
            
            # Divide by standard deviation
            normalized.append(centered / std)
        
        # Stack channels back together
        return torch.cat(normalized, dim=1)
        
    def forward(self, x):
        """Forward pass through the layer
        
        Args:
            x: Input tensor
            
        Returns:
            LayerState containing output and intermediate states
        """
        # Renormalize filters
        self.renormalize_filters()
        
        # Convolution
        x = self.conv(x)
        
        # Contrast normalization
        x = self.contrast_normalize(x)
        
        # ReLU
        x = F.relu(x)
        
        # Store pre-pool features
        pre_pool = x
        
        # Pooling
        x, indices = self.pool(x)
        
        return LayerState(output=x, pre_pool=pre_pool, pool_indices=indices)

class DeconvLayer(nn.Module):
    """A single deconvolutional layer with unpooling"""
    def __init__(self, conv_layer):
        super().__init__()
        self.conv_layer = conv_layer
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x, max_indices, pre_pool_size):
        """
        Args:
            x: Input tensor from the layer below
            max_indices: Indices from the max pooling operation in forward pass
            pre_pool_size: Size of the tensor before max pooling in forward pass
        """
        # Unpool to match the exact size from forward pass
        x = self.unpool(x, max_indices, output_size=pre_pool_size)
        
        # Apply ReLU after unpooling as per ZF2013
        x = F.relu(x)
        
        # Flip kernel dimensions for transposed convolution
        weight = self.conv_layer.conv.weight.flip([2, 3])
        
        # Use same stride and padding as original conv
        stride = self.conv_layer.conv.stride
        padding = self.conv_layer.conv.padding
        output_padding = 1 if stride[0] > 1 else 0

        x = F.conv_transpose2d(x, weight, stride=stride, padding=padding, output_padding=output_padding)
        return x 