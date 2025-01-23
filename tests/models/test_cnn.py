import torch
import pytest
from models.layers import LayerState
from models.cnn import ModelState, SimpleCNN

def test_deconv_visualization(normal_config):
    """Test that deconv_visualization works correctly for both layers"""
    model = SimpleCNN(normal_config)
    batch_size = 1
    
    # Create mock layer states
    layer1_state = LayerState(
        output=torch.randn(batch_size, 32, 14, 14),
        pre_pool=torch.randn(batch_size, 32, 28, 28),
        pool_indices=torch.randint(0, 196, (batch_size, 32, 14, 14))
    )
    
    layer2_state = LayerState(
        output=torch.randn(batch_size, 64, 7, 7),
        pre_pool=torch.randn(batch_size, 64, 14, 14),
        pool_indices=torch.randint(0, 49, (batch_size, 64, 7, 7))
    )
    
    # Create mock model state
    model_state = ModelState(
        logits=torch.randn(batch_size, 10),
        layer_states=[layer1_state, layer2_state],
        features=layer2_state.output
    )
    
    # Test Layer 1
    feature_maps_1 = torch.randn(batch_size, 32, 14, 14)  # Small feature map
    output_1 = model.deconv_visualization(feature_maps_1, model_state, layer=1)
    assert output_1.shape == (batch_size, 1, 28, 28), \
        "Layer 1 visualization should output original image size"
    
    # Test Layer 2
    feature_maps_2 = torch.randn(batch_size, 64, 7, 7)  # Smaller feature map
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