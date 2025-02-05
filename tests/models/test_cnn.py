import torch
import pytest
from models.layers import LayerState
from models.cnn import ModelState, SimpleCNN
from dataclasses import dataclass
import matplotlib.pyplot as plt

def test_deconv_visualization(normal_config):
    """Test that deconv_visualization works correctly for all layers"""
    model = SimpleCNN(normal_config)
    batch_size = 1
    
    # Create mock layer states for 224x224 input
    # After first conv (stride 2): 112x112
    # After first pool: 56x56
    # After second conv (stride 1): 56x56
    # After second pool: 28x28
    # After third conv (stride 1): 28x28
    # After third pool: 14x14
    # After fourth conv (stride 1): 14x14
    # After fourth pool: 7x7

    # Layer 1: 56x56 -> 112x112 (unpool) -> 224x224 (deconv)
    layer1_state = LayerState(
        output=torch.randn(batch_size, 96, 56, 56),  # After pool
        pre_pool=torch.randn(batch_size, 96, 112, 112),  # After conv
        pool_indices=torch.randint(0, 112*112, (batch_size, 96, 56, 56))
    )
    
    # Layer 2: 28x28 -> 56x56 (unpool) -> 112x112 (deconv)
    layer2_state = LayerState(
        output=torch.randn(batch_size, 256, 28, 28),  # After pool
        pre_pool=torch.randn(batch_size, 256, 56, 56),  # After conv
        pool_indices=torch.randint(0, 56*56, (batch_size, 256, 28, 28))
    )
    
    # Layer 3: 14x14 -> 28x28 (unpool) -> 56x56 (deconv)
    layer3_state = LayerState(
        output=torch.randn(batch_size, 384, 14, 14),  # After pool
        pre_pool=torch.randn(batch_size, 384, 28, 28),  # After conv
        pool_indices=torch.randint(0, 28*28, (batch_size, 384, 14, 14))
    )
    
    # Layer 4: 7x7 -> 14x14 (unpool) -> 28x28 (deconv)
    layer4_state = LayerState(
        output=torch.randn(batch_size, 384, 7, 7),  # After pool
        pre_pool=torch.randn(batch_size, 384, 14, 14),  # After conv
        pool_indices=torch.randint(0, 14*14, (batch_size, 384, 7, 7))
    )
    
    # Create mock model state
    model_state = ModelState(
        logits=torch.randn(batch_size, 1000),  # 1000 ImageNet classes
        layer_states=[layer1_state, layer2_state, layer3_state, layer4_state],
        features=layer4_state.output
    )
    
    # Test all layers
    feature_maps_sizes = [(96, 56, 56), (256, 28, 28), (384, 14, 14), (384, 7, 7)]
    for layer, size in enumerate(feature_maps_sizes, 1):
        feature_maps = torch.randn(batch_size, *size)
        output = model.deconv_visualization(feature_maps, model_state, layer)
        assert output.shape == (batch_size, 3, 224, 224), \
            f"Layer {layer} visualization should output original image size"
    
    # Test invalid layer
    with pytest.raises(ValueError):
        model.deconv_visualization(feature_maps, model_state, layer=5)

def test_filter_normalization(small_config):
    """Test that filter normalization correctly handles filters exceeding the radius"""
    model = SimpleCNN(small_config)
    
    # Set known weights that will exceed the radius
    with torch.no_grad():
        # First layer: one filter above radius, one below
        model.conv_layers[0].conv.weight.data[0].fill_(0.2)  # RMS = 0.2, exceeds radius
        model.conv_layers[0].conv.weight.data[1].fill_(0.05)  # RMS = 0.05, within radius
        
        # Other layers: all filters below radius
        for layer in model.conv_layers[1:]:
            layer.conv.weight.data.fill_(0.05)  # All RMS = 0.05
    
    # Run normalization and get info
    norm_info = model.normalize_filters()
    
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
    
    # Test other layers
    for i in range(1, 4):
        assert i not in norm_info, f"Layer {i} should not have been normalized"
    
    # Test that weights were actually scaled
    assert torch.allclose(model.conv_layers[0].conv.weight.data[0].mean(), 
                         torch.tensor(0.2 * expected_scale)), \
        "Weights should be scaled by the computed factor"
    assert torch.allclose(model.conv_layers[0].conv.weight.data[1].mean(), 
                         torch.tensor(0.05)), \
        "Non-exceeded filter weights should remain unchanged"

@dataclass
class TestConfig:
    conv1_channels: int = 96
    conv2_channels: int = 256
    conv3_channels: int = 384
    conv4_channels: int = 384
    kernel_size: int = 11
    fc_units: int = 1000

def test_single_forward_pass_with_visualization():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    config = TestConfig()
    model = SimpleCNN(config)
    model.eval()  # Set to eval mode
    print("\n=== Model Architecture ===")
    print(model)
    
    # Create a test image (3 channels, 224x224)
    test_image = torch.zeros(1, 3, 224, 224)
    # Create a simple pattern - diagonal lines in different colors
    for i in range(224):
        # Red diagonal
        test_image[0, 0, i, i] = 1.0
        # Green diagonal (offset)
        if i < 223:
            test_image[0, 1, i, i+1] = 1.0
        # Blue diagonal (offset other way)
        if i > 0:
            test_image[0, 2, i, i-1] = 1.0
            
    # Forward pass
    print("\n=== Forward Pass ===")
    with torch.no_grad():
        model_state = model(test_image)
    
    # Test visualization of all layers
    print("\n=== Visualization ===")
    for layer in range(1, 5):
        print(f"\nVisualizing Layer {layer}")
        # Get feature maps for this layer
        feature_maps = (model_state.final_features if layer == len(model.conv_layers) 
                       else model_state.layer_states[layer-1].output)
        print(f"Feature maps shape: {feature_maps.shape}")
        
        # Create a copy with only first feature map active
        zeroed_maps = torch.zeros_like(feature_maps)
        zeroed_maps[0, 0] = feature_maps[0, 0]
        
        # Get reconstruction
        reconstruction = model.deconv_visualization(zeroed_maps, model_state, layer)
        print(f"Reconstruction shape: {reconstruction.shape}")
        
        # Visualize the reconstruction
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.title("Original Image (RGB)")
        plt.imshow(test_image[0].permute(1, 2, 0).numpy())
        
        plt.subplot(1, 4, 2)
        plt.title(f"Layer {layer} Feature Map")
        plt.imshow(feature_maps[0, 0].detach().numpy(), cmap='gray')
        
        plt.subplot(1, 4, 3)
        plt.title(f"Reconstruction (R channel)")
        plt.imshow(reconstruction[0, 0].detach().numpy(), cmap='RdBu_r')
        
        plt.subplot(1, 4, 4)
        plt.title(f"Reconstruction (RGB)")
        plt.imshow(reconstruction[0].permute(1, 2, 0).detach().numpy())
        
        plt.tight_layout()
        plt.show()
        
        # Print feature map statistics
        print(f"\nFeature map statistics for layer {layer}:")
        print(f"Mean activation: {feature_maps.mean():.3f}")
        print(f"Std deviation: {feature_maps.std():.3f}")
        print(f"Max activation: {feature_maps.max():.3f}")
        print(f"Min activation: {feature_maps.min():.3f}")

    print("\nFeature map statistics for layer 1:")
    print(f"Mean activation: {model_state.layer_states[0].pre_pool.mean():.3f}")
    print(f"Std deviation: {model_state.layer_states[0].pre_pool.std():.3f}")
    print(f"Max activation: {model_state.layer_states[0].pre_pool.max():.3f}")
    print(f"Min activation: {model_state.layer_states[0].pre_pool.min():.3f}")

    print("\nVisualizing Layer 2")
    print(f"Feature maps shape: {model_state.layer_states[1].output.shape}")
    print("First feature map values:")
    print(model_state.layer_states[1].output[0, 0])

    # Add detailed statistics for each feature map in layer 2
    print("\nLayer 2 feature map statistics:")
    for i in range(model_state.layer_states[1].output.shape[1]):
        fmap = model_state.layer_states[1].output[0, i]  # Get i-th feature map
        if fmap.max() > 0.1:  # Only show maps with significant activation
            print(f"\nFeature map {i}:")
            print(f"Mean: {fmap.mean():.3f}")
            print(f"Max: {fmap.max():.3f}")
            print(f"Non-zero elements: {(fmap > 0).sum().item()}/{fmap.numel()}")

    print("\nOverall Layer 2 Statistics:")
    print(f"Mean activation across all maps: {model_state.layer_states[1].output.mean():.3f}")
    print(f"Std deviation across all maps: {model_state.layer_states[1].output.std():.3f}")
    print(f"Max activation across all maps: {model_state.layer_states[1].output.max():.3f}")
    print(f"Number of maps with max > 0.1: {(model_state.layer_states[1].output.max(dim=2)[0].max(dim=2)[0] > 0.1).sum().item()}")
    print(f"Percentage of active neurons: {(model_state.layer_states[1].output > 0).float().mean().item()*100:.1f}%")

    # Continue with zeroing and reconstruction
    print("\nZeroed maps shape:", zeroed_maps.shape)

if __name__ == "__main__":
    test_single_forward_pass_with_visualization() 