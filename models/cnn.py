import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional
from .layers import ConvLayer, DeconvLayer, LayerState

@dataclass
class ModelState:
    """Container for model's full state during forward pass"""
    logits: torch.Tensor
    layer_states: List[LayerState]
    features: Optional[torch.Tensor] = None  # Final features before classification

    @property
    def final_features(self) -> torch.Tensor:
        """Get features from the final layer"""
        return self.features if self.features is not None else self.layer_states[-1].output

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Create convolutional layers
        self.conv_layers = nn.ModuleList([
            ConvLayer(1, config.conv1_channels, config.kernel_size),
            ConvLayer(config.conv1_channels, config.conv2_channels, config.kernel_size)
        ])
        
        # Create corresponding deconvolutional layers with references to conv layers
        self.deconv_layers = nn.ModuleList([
            DeconvLayer(self.conv_layers[1]),  # Second conv layer
            DeconvLayer(self.conv_layers[0])   # First conv layer
        ])
        
        # Final classification layer
        self.fc = nn.Linear(config.conv2_channels * 7 * 7, config.fc_units)
        
        # Filter normalization parameters
        self.filter_radius = 1e-1  # As per ZF2013
        self.steps_until_next_norm = self.norm_frequency = 50  # Normalize every 50 steps
        
        # Device handling
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def tick_filter_norm_clock(self):
        """Advance the filter normalization clock and return whether it's time to normalize.
        
        Returns:
            bool: True if it's time to normalize filters (clock has reached frequency)
        """
        self.steps_until_next_norm -= 1
        if self.steps_until_next_norm <= 0:
            self.steps_until_next_norm = self.norm_frequency
            return True
        return False
    
    def is_filter_norm_due(self):
        """Check if it's time to normalize filters based on the clock without advancing it"""
        return self.steps_until_next_norm <= 0
    
    def normalize_filters(self, force=False):
        """Normalize filters whose RMS exceeds a fixed radius to that fixed radius (ZF2013 Sec 3)
        
        Args:
            force: If True, normalize regardless of clock. Useful for testing.
            
        Returns:
            dict: Information about normalization for each layer:
                {layer_idx: {'exceeded': tensor of which filters exceeded,
                           'scale': tensor of scaling factors applied}}
        """
        if not (force or self.tick_filter_norm_clock()):
            return {}
            
        normalization_info = {}
        with torch.no_grad():
            for i, layer in enumerate(self.conv_layers):
                # Calculate RMS for each filter
                weight = layer.conv.weight.data
                rms = torch.sqrt(torch.mean(weight.pow(2), dim=(1,2,3)))
                
                # Find filters exceeding the radius
                exceeded = rms > self.filter_radius
                
                # Only normalize if any filters exceed the radius
                if exceeded.any():
                    scale = torch.ones_like(rms)
                    scale[exceeded] = self.filter_radius / rms[exceeded]
                    layer.conv.weight.data *= scale.view(-1, 1, 1, 1)
                    normalization_info[i] = {
                        'exceeded': exceeded,
                        'scale': scale
                    }
        
        return normalization_info

    def _collect_layer_state(self, layer_idx: int, state: LayerState) -> LayerState:
        """Collect intermediate state from a layer's output"""
        if layer_idx == len(self.conv_layers):
            # Store features for final layer
            return LayerState(
                output=state.output.clone(),
                pre_pool=state.pre_pool,
                pool_indices=state.pool_indices
            )
        return state

    def forward(self, x: torch.Tensor) -> ModelState:
        """Forward pass through the network
        
        Args:
            x: Input tensor
            
        Returns:
            ModelState containing logits and intermediate states
        """
        layer_states = []
        
        # Forward through convolutional layers
        for i, layer in enumerate(self.conv_layers, 1):
            layer_output = layer(x)
            layer_state = self._collect_layer_state(i, layer_output)
            layer_states.append(layer_state)
            x = layer_output.output
        
        # Classification layer
        logits = self.fc(x.view(x.size(0), -1))
        
        return ModelState(logits=logits, layer_states=layer_states, features=x)

    def deconv_visualization(self, feature_maps: torch.Tensor, model_state: ModelState, layer: int) -> torch.Tensor:
        """Project feature maps back to input space using deconvnet approach (ZF2013 Sec 2.1)
        Args:
            feature_maps: The feature maps to visualize
            model_state: Model state containing pooling indices and pre-pool sizes
            layer: Which layer's features to visualize (1-based indexing)
        """
        if not 1 <= layer <= len(self.conv_layers):
            raise ValueError(f"Layer {layer} not supported for visualization")
        
        x = feature_maps
        
        # Process through deconv layers in reverse order
        start_idx = len(self.deconv_layers) - layer
        for i, deconv_layer in enumerate(self.deconv_layers[start_idx:], start_idx + 1):
            layer_num = len(self.conv_layers) - i + 1
            layer_state = model_state.layer_states[layer_num - 1]  # Convert to 0-based indexing
            x = deconv_layer(x, layer_state.pool_indices, layer_state.pre_pool.size())
        
        return x 

    def load_state_dict(self, state_dict):
        """Custom state dict loading to handle architectural changes"""
        # Create new state dict with remapped keys
        new_state_dict = {}
        
        # Copy over conv layer and fc weights directly
        for key in state_dict:
            if key.startswith('conv_layers') or key.startswith('fc'):
                new_state_dict[key] = state_dict[key]
        
        # Load the state dict with strict=False to ignore missing deconv keys
        super().load_state_dict(new_state_dict, strict=False) 