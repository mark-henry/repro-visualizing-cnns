import torch
import pytest
from models.layers import LayerState
from models.cnn import ModelState, SimpleCNN
from dataclasses import dataclass
import matplotlib.pyplot as plt

def test_deconv_visualization(normal_config):
    """Test that deconv_visualization works correctly for both layers"""
    model = SimpleCNN(normal_config)
    batch_size = 1
    
    # Create mock layer states
    # After first conv (stride 2): 14x14
    # After first pool: 7x7
    # After second conv (stride 1): 7x7
    # After second pool: 3x3

    # Layer 1: 7x7 -> 14x14 (unpool) -> 28x28 (deconv)
    # For 7x7 input and 14x14 output with 2x2 pooling:
    # Each index should be within range [0, 14*14 - 1]
    layer1_state = LayerState(
        output=torch.randn(batch_size, 32, 7, 7),  # After pool
        pre_pool=torch.randn(batch_size, 32, 14, 14),  # After conv
        pool_indices=torch.randint(0, 14*14, (batch_size, 32, 7, 7))
    )
    
    # Layer 2: 3x3 -> 7x7 (unpool) -> 14x14 (deconv)
    # For 3x3 input and 7x7 output with 2x2 pooling:
    # Each index should be within range [0, 7*7 - 1]
    layer2_state = LayerState(
        output=torch.randn(batch_size, 64, 3, 3),  # After pool
        pre_pool=torch.randn(batch_size, 64, 7, 7),  # After conv
        pool_indices=torch.randint(0, 7*7, (batch_size, 64, 3, 3))
    )
    
    # Create mock model state
    model_state = ModelState(
        logits=torch.randn(batch_size, 10),
        layer_states=[layer1_state, layer2_state],
        features=layer2_state.output
    )
    
    # Test Layer 1
    feature_maps_1 = torch.randn(batch_size, 32, 7, 7)  # Small feature map
    output_1 = model.deconv_visualization(feature_maps_1, model_state, layer=1)
    assert output_1.shape == (batch_size, 1, 28, 28), \
        "Layer 1 visualization should output original image size"
    
    # Test Layer 2
    feature_maps_2 = torch.randn(batch_size, 64, 3, 3)  # Smaller feature map
    output_2 = model.deconv_visualization(feature_maps_2, model_state, layer=2)
    assert output_2.shape == (batch_size, 1, 28, 28), \
        "Layer 2 visualization should output original image size"
    
    # Test invalid layer
    with pytest.raises(ValueError):
        model.deconv_visualization(feature_maps_1, model_state, layer=3)

def test_filter_normalization(small_config):
    """Test that filter normalization correctly handles filters exceeding the radius"""
    model = SimpleCNN(small_config)
    
    # Set known weights that will exceed the radius
    with torch.no_grad():
        # First layer: one filter above radius, one below
        model.conv_layers[0].conv.weight.data[0].fill_(0.2)  # RMS = 0.2, exceeds radius
        model.conv_layers[0].conv.weight.data[1].fill_(0.05)  # RMS = 0.05, within radius
        
        # Second layer: both filters below radius
        model.conv_layers[1].conv.weight.data.fill_(0.05)  # All RMS = 0.05
    
    # Force normalization and get info
    norm_info = model.normalize_filters(force=True)
    
    # Test first layer
    assert 0 in norm_info, "First layer should have been normalized"
    assert norm_info[0]['exceeded'][0].item(), "First filter should have exceeded radius"
    assert not norm_info[0]['exceeded'][1].item(), "Second filter should not have exceeded radius"
    
    # Calculate expected scale for exceeded filter
    expected_scale = model.filter_radius / 0.2
    assert torch.allclose(norm_info[0]['scale'][0], torch.tensor(expected_scale)), \
        "Scale factor should match expected value"
    assert torch.allclose(norm_info[0]['scale'][1], torch.tensor(1.0)), \
        "Non-exceeded filter should have scale 1.0"
    
    # Test second layer
    assert 1 not in norm_info, "Second layer should not have been normalized"
    
    # Test that weights were actually scaled
    assert torch.allclose(model.conv_layers[0].conv.weight.data[0].mean(), 
                         torch.tensor(0.2 * expected_scale)), \
        "Weights should be scaled by the computed factor"
    assert torch.allclose(model.conv_layers[0].conv.weight.data[1].mean(), 
                         torch.tensor(0.05)), \
        "Non-exceeded filter weights should remain unchanged"
    
    # Test periodic normalization timing
    model.steps_until_next_norm = model.norm_frequency  # Reset clock
    assert not model.is_filter_norm_due(), "Should not normalize when clock just reset"
    
    # Tick clock until just before it's due
    for _ in range(model.norm_frequency - 1):
        assert not model.tick_filter_norm_clock(), "Should not normalize before clock reaches zero"
        assert not model.is_filter_norm_due(), "Should not be due before clock reaches zero"
    
    # Final tick should trigger normalization
    assert model.tick_filter_norm_clock(), "Should normalize when clock reaches zero"
    assert model.steps_until_next_norm == model.norm_frequency, "Clock should reset after triggering"

@dataclass
class TestConfig:
    conv1_channels: int = 32
    conv2_channels: int = 64
    kernel_size: int = 7
    fc_units: int = 10

def test_single_forward_pass_with_visualization():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    config = TestConfig()
    model = SimpleCNN(config)
    model.eval()  # Set to eval mode
    print("\n=== Model Architecture ===")
    print(model)
    
    # Create a diagonal line image (1 channel, 28x28)
    test_image = torch.zeros(1, 1, 28, 28)
    for i in range(28):
        # Create diagonal line from bottom left (0,27) to top right (27,0)
        col = i        # x goes from 0 to 27
        row = 27 - i   # y goes from 27 to 0
        
        # Make the line 3 pixels thick for better visibility
        if col > 0:  # Add pixel to the left
            test_image[0, 0, row, col-1] = 1.0
        test_image[0, 0, row, col] = 1.0  # Main diagonal
        if col < 27:  # Add pixel to the right
            test_image[0, 0, row, col+1] = 1.0
            
    # Print some sample positions to verify the line
    print(f"\n=== Input Image Shape: {test_image.shape} ===")
    print("\nInput Image Values (sample diagonal positions):")
    for i in range(0, 28, 4):  # Sample every 4th position
        col = i
        row = 27 - i
        print(f"Position ({row}, {col}): {test_image[0, 0, row, col]:.1f}")  # Should be 1.0 for all
    
    # Forward pass
    print("\n=== Forward Pass ===")
    with torch.no_grad():
        model_state = model(test_image)
    
    # Print shapes and sample values after each layer
    print("\nLayer 1:")
    print(f"After conv1: {model_state.layer_states[0].pre_pool.shape}")
    print("Sample conv1 values (first feature map):")
    print(model_state.layer_states[0].pre_pool[0, 0])
    print(f"After pool1: {model_state.layer_states[0].output.shape}")
    print("Sample pool1 values (first feature map):")
    print(model_state.layer_states[0].output[0, 0])
    print(f"Pool1 indices shape: {model_state.layer_states[0].pool_indices.shape}")
    print("Sample pool1 indices (first feature map):")
    print(model_state.layer_states[0].pool_indices[0, 0])
    
    print("\nLayer 2:")
    print(f"After conv2: {model_state.layer_states[1].pre_pool.shape}")
    print("Sample conv2 values (first feature map):")
    print(model_state.layer_states[1].pre_pool[0, 0])
    print(f"After pool2: {model_state.layer_states[1].output.shape}")
    print("Sample pool2 values (first feature map):")
    print(model_state.layer_states[1].output[0, 0])
    print(f"Pool2 indices shape: {model_state.layer_states[1].pool_indices.shape}")
    print("Sample pool2 indices (first feature map):")
    print(model_state.layer_states[1].pool_indices[0, 0])
    
    print(f"\nFinal features shape: {model_state.final_features.shape}")
    print("Sample final features (first feature map):")
    print(model_state.final_features[0, 0])
    print(f"Logits shape: {model_state.logits.shape}")
    print("Logits values:")
    print(model_state.logits[0])
    
    # Test visualization of both layers
    print("\n=== Visualization ===")
    for layer in [1, 2]:
        print(f"\nVisualizing Layer {layer}")
        # Get feature maps for this layer
        feature_maps = (model_state.final_features if layer == len(model.conv_layers) 
                       else model_state.layer_states[layer-1].output)
        print(f"Feature maps shape: {feature_maps.shape}")
        print(f"First feature map values:")
        print(feature_maps[0, 0])
        
        # Create a copy with only first feature map active
        zeroed_maps = torch.zeros_like(feature_maps)
        zeroed_maps[0, 0] = feature_maps[0, 0]
        print(f"Zeroed maps shape: {zeroed_maps.shape}")
        print("Verification of zeroing (should show only first channel has values):")
        print(f"Sum of values in first channel: {zeroed_maps[0, 0].sum():.3f}")
        print(f"Sum of values in all other channels: {zeroed_maps[0, 1:].sum():.3f}")
        
        # Get reconstruction
        reconstruction = model.deconv_visualization(zeroed_maps, model_state, layer)
        print(f"Reconstruction shape: {reconstruction.shape}")
        print(f"Reconstruction min/max values: {reconstruction.min():.3f}/{reconstruction.max():.3f}")
        print("Reconstruction values (showing non-zero elements):")
        nonzero = (reconstruction.detach()[0, 0].abs() > 0.1).nonzero()
        for idx in range(min(10, len(nonzero))):  # Show first 10 non-zero elements
            pos = nonzero[idx]
            print(f"Position ({pos[0]}, {pos[1]}): {reconstruction[0, 0, pos[0], pos[1]].item():.3f}")
        
        # Visualize the reconstruction
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(test_image[0, 0].numpy(), cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title(f"Layer {layer} Feature Map")
        plt.imshow(feature_maps[0, 0].detach().numpy(), cmap='gray')
        
        plt.subplot(1, 3, 3)
        plt.title(f"Layer {layer} Reconstruction")
        plt.imshow(reconstruction[0, 0].detach().numpy(), cmap='RdBu_r')
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics about the feature maps
        print(f"\nFeature map statistics for layer {layer}:")
        print(f"Mean activation: {feature_maps.mean():.3f}")
        print(f"Std deviation: {feature_maps.std():.3f}")
        print(f"Max activation: {feature_maps.max():.3f}")
        print(f"Min activation: {feature_maps.min():.3f}")

if __name__ == "__main__":
    test_single_forward_pass_with_visualization() 