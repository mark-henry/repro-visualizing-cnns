import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm
from types import SimpleNamespace
import argparse

# Set random seeds for reproducibility
for seed in [random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:
    seed(42)
torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        
    def forward(self, x, store_indices=False):
        pool_indices = {}
        
        x = F.relu(self.conv(x))
        x = self.norm(x)
        if store_indices:
            pool_indices['pre_pool'] = x
        x, indices = self.pool(x)
        if store_indices:
            pool_indices['pool'] = indices
            
        return x, pool_indices

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

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Create convolutional layers
        self.conv_layers = nn.ModuleList([
            ConvLayer(1, config.conv1_channels, config.kernel_size),
            ConvLayer(config.conv1_channels, config.conv2_channels, config.kernel_size)
        ])
        
        # Create corresponding deconvolutional layers
        self.deconv_layers = nn.ModuleList([
            DeconvLayer(config.conv2_channels, config.conv1_channels, config.kernel_size),
            DeconvLayer(config.conv1_channels, 1, config.kernel_size)
        ])
        
        # Final classification layer
        self.fc = nn.Linear(config.conv2_channels * 7 * 7, config.fc_units)
        
        # Filter normalization parameters
        self.filter_radius = 1e-1  # As per ZF2013
        self.steps_until_next_norm = self.norm_frequency = 50  # Normalize every 50 steps

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

    def forward(self, x, store_indices=False):
        pool_indices = {}
        
        # Forward through convolutional layers
        for i, layer in enumerate(self.conv_layers, 1):
            x, layer_indices = layer(x, store_indices)
            if store_indices:
                pool_indices[f'pre_pool{i}'] = layer_indices['pre_pool']
                pool_indices[f'pool{i}'] = layer_indices['pool']
                if i == len(self.conv_layers):
                    pool_indices['features'] = x.clone()
        
        # Classification layer
        x = self.fc(x.view(x.size(0), -1))
        
        return (x, pool_indices) if store_indices else x

    def deconv_visualization(self, feature_maps, pool_indices, layer):
        """Project feature maps back to input space using deconvnet approach (ZF2013 Sec 2.1)
        Args:
            feature_maps: The feature maps to visualize
            pool_indices: Dictionary containing pooling indices and pre-pool sizes
            layer: Which layer's features to visualize (1-based indexing)
        """
        if not 1 <= layer <= len(self.conv_layers):
            raise ValueError(f"Layer {layer} not supported for visualization")
            
        x = feature_maps
        
        # Process through deconv layers in reverse order
        start_idx = len(self.deconv_layers) - layer
        for i, deconv_layer in enumerate(self.deconv_layers[start_idx:], start_idx + 1):
            layer_num = len(self.conv_layers) - i + 1
            x = deconv_layer(x, 
                           pool_indices[f'pool{layer_num}'],
                           pool_indices[f'pre_pool{layer_num}'].size())
            
        return x

def get_data(config):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)
    )

def train(model, train_loader, config):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    wandb.watch(model, criterion, log="all", log_freq=50)
    
    # Move model to device once
    model = model.to(device)
    
    for epoch in tqdm(range(config.epochs)):
        model.train()
        epoch_loss = epoch_accuracy = num_batches = 0
        
        for images, labels in train_loader:
            # Move batch to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass and loss
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Periodic filter normalization
            model.normalize_filters()
            
            # Track metrics
            with torch.no_grad():  # Don't track metrics computation
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == labels).sum().item() / labels.size(0)
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                num_batches += 1
            
            # Log batch metrics
            wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "accuracy": accuracy,
                "examples": num_batches * config.batch_size,
            })
        
        # Log epoch metrics
        wandb.log({
            "epoch_loss": epoch_loss / num_batches,
            "epoch_accuracy": epoch_accuracy / num_batches,
        })

def find_strongest_activations(model, data_loader, num_samples=1000):
    """Find strongest activations for each feature map"""
    model.eval()
    strongest = {layer: {'activations': None, 'images': None} for layer in [1, 2]}
    
    print(f"Finding strongest activations across {num_samples} samples...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(data_loader)):
            if batch_idx * images.size(0) >= num_samples:
                break
                
            # Get feature maps
            images = images.to(device)
            _, pool_indices = model(images, store_indices=True)
            
            # Process each layer
            for layer, key in [(1, 'pre_pool1'), (2, 'features')]:
                acts = pool_indices[key]
                
                # Initialize storage if needed
                if strongest[layer]['activations'] is None:
                    strongest[layer]['activations'] = torch.zeros((acts.size(1),), device=device)
                    strongest[layer]['images'] = torch.zeros((acts.size(1), 1, 28, 28), device=device)
                
                # Find max activations for each feature map
                for f in range(acts.size(1)):
                    feature_acts = acts[:, f]
                    # Use total activation across spatial dimensions instead of max
                    batch_total_acts = feature_acts.sum(dim=(1, 2))
                    max_batch = batch_total_acts.argmax()
                    total_act = batch_total_acts[max_batch]
                    
                    if total_act > strongest[layer]['activations'][f]:
                        strongest[layer]['activations'][f] = total_act
                        strongest[layer]['images'][f] = images[max_batch]
    
    return strongest

def visualize_features(model, strongest_activations, layer):
    """Visualize patterns that cause strongest activations using deconvnet approach (ZF2013)"""
    model.eval()
    with torch.no_grad():
        # Setup visualization
        fig, axes = plt.subplots(5, 8, figsize=(15, 10))
        plt.suptitle(f'Layer {layer} Features - Strongest Activations', y=1.02)
        
        # Process each feature
        num_features = strongest_activations[layer]['activations'].size(0)
        for idx in range(min(32, num_features)):
            # Get original image that caused strongest activation
            image = strongest_activations[layer]['images'][idx:idx+1]
            _, pool_indices = model(image.to(device), store_indices=True)
            
            # Get the feature maps for this layer
            feature_maps = pool_indices['features'] if layer == 2 else pool_indices['pre_pool1']
            
            # Create a copy with all feature maps zeroed except the target one
            zeroed_maps = torch.zeros_like(feature_maps)
            zeroed_maps[0, idx] = feature_maps[0, idx]  # Keep entire feature map
            
            # Get reconstruction through deconvnet
            reconstruction = model.deconv_visualization(zeroed_maps, pool_indices, layer)
            
            # Plot
            ax = axes[idx // 8, idx % 8]
            img = reconstruction.squeeze().cpu()
            vmax = torch.abs(img).max().item()
            ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.axis('off')
            
            # Add input image inset
            ax_ins = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
            ax_ins.imshow(image.squeeze().cpu(), cmap='gray')
            ax_ins.axis('off')
        
        plt.tight_layout()
        wandb.log({f"layer_{layer}_strongest_features": wandb.Image(plt)})
        plt.close()

if __name__ == "__main__":
    # Configuration
    config = {
        "architecture": "SimpleCNN",
        "dataset": "MNIST",
        "epochs": 2,
        "batch_size": 64,
        "learning_rate": 0.001,
        "conv1_channels": 32,
        "conv2_channels": 64,
        "kernel_size": 3,
        "pool_size": 2,
        "fc_units": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train or visualize a CNN model')
    parser.add_argument('--mode', type=str, choices=['train', 'visualize'], required=True,
                      help='Whether to train a new model or visualize an existing one')
    parser.add_argument('--model_path', type=str, default='model.pth',
                      help='Path to save/load the model (default: model.pth)')
    parser.add_argument('--num_images', type=int, default=1000,
                      help='Number of images to search for strongest activations (default: 1000)')
    parser.add_argument('--layers', type=str, default='1,2',
                      help='Comma-separated list of layers to visualize (default: 1,2)')
    args = parser.parse_args()
    
    # Main execution
    if args.mode == 'train':
        print("Training new model...")
        with wandb.init(project="cnn-feature-visualization", config=config):
            config = wandb.config
            train_loader, test_loader = get_data(config)
            model = SimpleCNN(config).to(device)
            train(model, train_loader, config)
            
            print(f"Saving model to {args.model_path}...")
            torch.save(model.state_dict(), args.model_path, _use_new_zipfile_serialization=False)
            wandb.save(args.model_path)
            print("Training complete!")
            
    else:  # visualize mode
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found at {args.model_path}")
            
        print(f"Loading model from {args.model_path}...")
        model = SimpleCNN(SimpleNamespace(**config)).to(device)
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        
        _, test_loader = get_data(SimpleNamespace(**config))
        layers = [int(layer) for layer in args.layers.split(',')]
        
        with wandb.init(project="cnn-feature-visualization", config=config, job_type="visualization"):
            strongest = find_strongest_activations(model, test_loader, args.num_images)
            for layer in layers:
                if layer not in [1, 2]:
                    print(f"Warning: Layer {layer} not supported. Skipping...")
                    continue
                print(f"Visualizing strongest activations for layer {layer}...")
                visualize_features(model, strongest, layer)
            print("Visualization complete!")
    