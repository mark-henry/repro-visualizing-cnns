import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from v import LocalContrastNorm, SimpleCNN, ConvLayer, DeconvLayer
from types import SimpleNamespace

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
    
    # Test forward pass with pooling indices
    output, pool_indices = layer(x, store_indices=True)
    
    print("\nInput:")
    print(x[0, 0])
    print("\nPre-pool feature map:")
    print(pool_indices['pre_pool'][0, 0])
    print("\nPooling indices (showing which elements were maximum in each 2x2 region):")
    print(pool_indices['pool'][0, 0])
    print("\nPooled output (the maximum values):")
    print(output[0, 0])
    
    # Assertions
    assert output.shape == (batch_size, out_channels, input_size//2, input_size//2)
    assert 'pre_pool' in pool_indices
    assert 'pool' in pool_indices
    assert pool_indices['pre_pool'].shape == (batch_size, out_channels, input_size, input_size)
    assert pool_indices['pool'].shape == (batch_size, out_channels, input_size//2, input_size//2)

def test_deconv_layer():
    """Test a single deconvolutional layer"""
    batch_size, in_channels, out_channels = 1, 64, 32
    small_size = 7
    layer = DeconvLayer(in_channels, out_channels, kernel_size=3)
    
    # Create inputs
    x = torch.randn(batch_size, in_channels, small_size, small_size)
    # For unpooling, the output size should be approximately double the input size
    large_size = (small_size - 1) * 2 + 2  # This follows PyTorch's unpooling size formula
    max_indices = torch.randint(0, small_size*small_size, 
                           (batch_size, in_channels, small_size, small_size),
                           dtype=torch.int64)
    pre_pool_size = (batch_size, out_channels, large_size, large_size)
    
    # Test forward pass
    output = layer(x, max_indices, pre_pool_size)
    assert output.shape == (batch_size, out_channels, large_size, large_size)

def test_end_to_end_conv_deconv():
    """Test that a conv layer followed by its corresponding deconv layer works"""
    batch_size, channels = 1, 32
    input_size = 28
    
    conv = ConvLayer(1, channels, kernel_size=3)
    deconv = DeconvLayer(channels, 1, kernel_size=3)
    
    # Create input with known pattern - make it more pronounced
    x = torch.zeros(batch_size, 1, input_size, input_size)
    x[0, 0, 0:2, 0:2] = torch.tensor([[5., 0.], [0., 5.]])  # Stronger values
    
    # Forward through conv
    encoded, pool_indices = conv(x, store_indices=True)
    
    # Forward through deconv
    decoded = deconv(encoded, pool_indices['pool'], pool_indices['pre_pool'].size())
    
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

def test_deconv_visualization():
    """Test that deconv_visualization works correctly for both layers"""
    # Setup
    config = SimpleNamespace(
        conv1_channels=32,
        conv2_channels=64,
        kernel_size=3,
        pool_size=2,
        fc_units=10
    )
    model = SimpleCNN(config)
    batch_size = 1
    
    # Create mock feature maps and pooling indices
    # Adjust sizes to match PyTorch's unpooling requirements
    pool_indices = {
        'pool1': torch.randint(0, 196, (batch_size, 32, 14, 14)),
        'pool2': torch.randint(0, 49, (batch_size, 64, 7, 7)),
        'pre_pool1': torch.randn(batch_size, 32, 28, 28),
        'pre_pool2': torch.randn(batch_size, 64, 14, 14),
    }
    
    # Test Layer 1
    feature_maps_1 = torch.randn(batch_size, 32, 14, 14)  # Small feature map
    output_1 = model.deconv_visualization(feature_maps_1, pool_indices, layer=1)
    assert output_1.shape == (batch_size, 1, 28, 28), \
        "Layer 1 visualization should output original image size"
    
    # Test Layer 2
    feature_maps_2 = torch.randn(batch_size, 64, 7, 7)  # Smaller feature map
    output_2 = model.deconv_visualization(feature_maps_2, pool_indices, layer=2)
    assert output_2.shape == (batch_size, 1, 28, 28), \
        "Layer 2 visualization should output original image size"
    
    # Test invalid layer
    with pytest.raises(ValueError):
        model.deconv_visualization(feature_maps_1, pool_indices, layer=3)

def test_local_contrast_norm():
    """Test local contrast normalization behavior and parameters"""
    batch_size, channels = 1, 1
    size = 16
    
    # Test different kernel sizes
    for kernel_size in [3, 5, 7, 9]:
        # Create layer
        norm = LocalContrastNorm(num_features=channels, kernel_size=kernel_size)
        
        # Create input with a step edge (high contrast) and a uniform region (low contrast)
        x = torch.zeros(batch_size, channels, size, size)
        x[0, 0, :, :size//2] = 0.0  # Left half black
        x[0, 0, :, size//2:] = 1.0  # Right half white
        
        # Apply normalization
        output = norm(x)
        
        # Test 1: Edge response
        # The response should be strongest at the edge (center columns)
        edge_response = output[0, 0, :, size//2-1:size//2+1].abs().mean()
        uniform_response = output[0, 0, :, :size//4].abs().mean()  # Sample from uniform region
        assert edge_response > uniform_response, \
            f"Edge response ({edge_response:.4f}) should be stronger than uniform region ({uniform_response:.4f}) with kernel_size={kernel_size}"
        
        # Test 2: Uniform region response
        # Response in uniform regions should be close to zero
        assert uniform_response < 0.1, \
            f"Response in uniform region ({uniform_response:.4f}) should be close to zero with kernel_size={kernel_size}"
        
        # Test 3: Zero mean
        # The local mean should be subtracted, but due to edge effects and the step function,
        # the mean might not be exactly zero
        assert abs(output.mean()) < 0.5, \
            f"Output should have approximately zero mean, got {output.mean():.4f} with kernel_size={kernel_size}"
        
        # Test 4: No numerical issues
        assert not torch.isnan(output).any(), f"NaN values found with kernel_size={kernel_size}"
        assert not torch.isinf(output).any(), f"Inf values found with kernel_size={kernel_size}" 