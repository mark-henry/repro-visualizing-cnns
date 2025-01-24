import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from models.layers import LayerState, ConvLayer, DeconvLayer

def test_conv_layer():
    """Test a single convolutional layer"""
    batch_size, in_channels, out_channels = 1, 1, 1  # Use just 1 output channel for clarity
    input_size = 4  # Use a 4x4 input for clarity
    layer = ConvLayer(in_channels, out_channels, kernel_size=3)
    
    # Create a simple input with a known pattern
    x = torch.zeros(batch_size, in_channels, input_size, input_size)
    x[0, 0] = torch.tensor([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
        [13., 14., 15., 16.]
    ])
    
    # Test forward pass
    state = layer(x)
    
    print("\nInput:")
    print(x[0, 0])
    print("\nPre-pool feature map:")
    print(state.pre_pool[0, 0])
    print("\nPooling indices (showing which elements were maximum in each 2x2 region):")
    print(state.pool_indices[0, 0])
    print("\nOutput (the maximum values):")
    print(state.output[0, 0])
    
    # Test shape assertions
    assert state.output.shape == (batch_size, out_channels, input_size//2, input_size//2)
    assert state.pre_pool.shape == (batch_size, out_channels, input_size, input_size)
    assert state.pool_indices.shape == (batch_size, out_channels, input_size//2, input_size//2)

def test_deconv_layer():
    """Test a single deconvolutional layer"""
    batch_size, in_channels, out_channels = 1, 64, 32
    small_size = 7
    
    # Create conv layer first
    conv = ConvLayer(in_channels, out_channels, kernel_size=3)
    layer = DeconvLayer(conv)
    
    # Create inputs
    x = torch.randn(batch_size, in_channels, small_size, small_size)
    # For unpooling, the output size should be approximately double the input size
    large_size = (small_size - 1) * 2 + 2  # This follows PyTorch's unpooling size formula
    pool_indices = torch.randint(0, small_size*small_size, 
                               (batch_size, in_channels, small_size, small_size),
                               dtype=torch.int64)
    pre_pool_size = (batch_size, out_channels, large_size, large_size)
    
    # Test forward pass
    output = layer(x, pool_indices, pre_pool_size)
    assert output.shape == (batch_size, out_channels, large_size, large_size)

def test_end_to_end_conv_deconv():
    """Test that a conv layer followed by its corresponding deconv layer works"""
    batch_size, channels = 1, 32
    input_size = 28
    
    conv = ConvLayer(1, channels, kernel_size=3)
    deconv = DeconvLayer(conv)
    
    # Create input with known pattern - make it more pronounced
    x = torch.zeros(batch_size, 1, input_size, input_size)
    x[0, 0, 0:2, 0:2] = torch.tensor([[5., 0.], [0., 5.]])  # Stronger values
    
    # Forward through conv
    state = conv(x)
    
    # Forward through deconv using state components
    decoded = deconv(state.output, state.pool_indices, state.pre_pool.size())
    
    # First check shape
    assert decoded.shape == x.shape
    
    # Get the regions we want to compare
    top_left = decoded[0, 0, :2, :2]
    rest = decoded[0, 0, 2:, :]
    
    # Normalize each region separately to compare their internal structure
    if top_left.abs().max() > 0:
        top_left = top_left / top_left.abs().max()
    if rest.abs().max() > 0:
        rest = rest / rest.abs().max()
    
    # The reconstruction won't be exact, but the normalized activations
    # in the top-left should be stronger than the rest
    assert top_left.abs().mean() > rest.abs().mean(), \
        "Top-left region should have stronger relative activations" 