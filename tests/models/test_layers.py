import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from models.layers import LayerState, ConvLayer, DeconvLayer

def test_conv_layer():
    """Test a single convolutional layer"""
    batch_size, in_channels, out_channels = 1, 1, 1  # Use just 1 output channel for clarity
    input_size = 8  # Use an 8x8 input for clarity with stride 2
    layer = ConvLayer(in_channels, out_channels, kernel_size=3, stride=2)
    
    # Create a simple input with a known pattern
    x = torch.zeros(batch_size, in_channels, input_size, input_size)
    x[0, 0] = torch.tensor([
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [9., 10., 11., 12., 13., 14., 15., 16.],
        [17., 18., 19., 20., 21., 22., 23., 24.],
        [25., 26., 27., 28., 29., 30., 31., 32.],
        [33., 34., 35., 36., 37., 38., 39., 40.],
        [41., 42., 43., 44., 45., 46., 47., 48.],
        [49., 50., 51., 52., 53., 54., 55., 56.],
        [57., 58., 59., 60., 61., 62., 63., 64.]
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
    # After stride 2 conv: 4x4
    # After pool: 2x2
    assert state.output.shape == (batch_size, out_channels, 2, 2)
    assert state.pre_pool.shape == (batch_size, out_channels, 4, 4)
    assert state.pool_indices.shape == (batch_size, out_channels, 2, 2)

def test_pool_unpool_first_layer():
    """Test pooling/unpooling dimensions for first layer (14x14 -> 7x7 -> 14x14)"""
    # Setup
    batch_size = 1
    channels = 32
    input_size = 14
    
    # Create layer
    layer = ConvLayer(channels, channels, kernel_size=3, stride=1)
    deconv = DeconvLayer(layer)
    
    # Forward pass
    x = torch.randn(batch_size, channels, input_size, input_size)
    layer_state = layer(x)
    
    # Check pooled size
    assert layer_state.output.shape == (batch_size, channels, 7, 7)
    
    # Check unpooling restores original size
    output = deconv(layer_state.output, layer_state.pool_indices, layer_state.pre_pool.shape)
    assert output.shape == (batch_size, channels, 14, 14)

def test_pool_unpool_second_layer():
    """Test pooling/unpooling dimensions for second layer (7x7 -> 3x3 -> 7x7)"""
    # Setup
    batch_size = 1
    channels = 64
    input_size = 7
    
    # Create layer
    layer = ConvLayer(channels, channels, kernel_size=3, stride=1)
    deconv = DeconvLayer(layer)
    
    # Forward pass
    x = torch.randn(batch_size, channels, input_size, input_size)
    layer_state = layer(x)
    
    # Check pooled size
    assert layer_state.output.shape == (batch_size, channels, 3, 3)
    
    # Check unpooling restores original size
    output = deconv(layer_state.output, layer_state.pool_indices, layer_state.pre_pool.shape)
    assert output.shape == (batch_size, channels, 7, 7)

def test_conv_deconv_reconstruction():
    """Test that deconvolution approximately reconstructs the input pattern
    when going through the full ConvLayer -> DeconvLayer pipeline.
    """
    batch_size, channels = 1, 1  # Single channel for clarity
    input_size = 8

    # Create layers
    conv = ConvLayer(channels, channels, kernel_size=3, stride=1)
    deconv = DeconvLayer(conv)

    with torch.no_grad():
        # Set a simple edge detection filter (Sobel filter for vertical edges)
        conv.conv.weight.data[0,0] = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
        conv.conv.bias.data.zero_()

    # Create input with a vertical edge
    x = torch.zeros(batch_size, channels, input_size, input_size)
    x[:, :, :, input_size//2:] = 1.0  # Right half is white, left half is black

    # Forward through conv
    state = conv(x)

    # Check edge detection response at the edge location (middle columns)
    edge_response = state.pre_pool[0,0,:,input_size//2].mean()
    assert edge_response > 0, "Edge detector should respond positively to vertical edge"

    # Forward through deconv
    output = deconv(state.output, state.pool_indices, state.pre_pool.shape)

    # Verify dimensions
    assert output.shape == x.shape

    # Verify reconstruction quality
    # The reconstruction won't be perfect due to max pooling information loss
    # and the nature of the convolution operation
    edge_diff = (output - x).abs().mean()
    assert edge_diff < 0.6, "Reconstruction should approximately preserve the input pattern"

def test_conv_deconv_stride():
    """Test that deconvolution correctly handles different stride values"""
    batch_size, channels = 1, 1
    
    # Test with stride=2
    conv_stride2 = ConvLayer(channels, channels, kernel_size=3, stride=2)
    deconv_stride2 = DeconvLayer(conv_stride2)
    
    # Create a simple 8x8 input
    x = torch.zeros(batch_size, channels, 8, 8)
    x[0, 0, 3:5, 3:5] = 1.0  # 2x2 square in the middle
    
    print("\nStride 2 Test:")
    print("Input:")
    print(x[0, 0])
    
    # Forward through conv (8x8 -> 4x4 -> 2x2)
    state = conv_stride2(x)
    print("\nAfter stride-2 conv (pre-pool):")
    print(state.pre_pool[0, 0])
    print("\nAfter pooling:")
    print(state.output[0, 0])
    
    # Forward through deconv
    output = deconv_stride2(state.output, state.pool_indices, state.pre_pool.shape)
    print("\nAfter deconv:")
    print(output[0, 0])
    
    # The output should be same size as input
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"
    
    # Test with stride=1
    conv_stride1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
    deconv_stride1 = DeconvLayer(conv_stride1)
    
    print("\nStride 1 Test:")
    # Forward through conv (8x8 -> 8x8 -> 4x4)
    state = conv_stride1(x)
    print("\nAfter stride-1 conv (pre-pool):")
    print(state.pre_pool[0, 0])
    print("\nAfter pooling:")
    print(state.output[0, 0])
    
    # Forward through deconv
    output = deconv_stride1(state.output, state.pool_indices, state.pre_pool.shape)
    print("\nAfter deconv:")
    print(output[0, 0])
    
    # The output should be same size as input
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}" 