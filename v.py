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
    """Local contrast normalization as described in ZF2013 Section 2"""
    def __init__(self, num_features, k=2, n=5, alpha=1e-4, beta=0.75):
        super().__init__()
        self.n = n  # Size of local region
        self.alpha = alpha  # Scale parameter
        self.beta = beta  # Exponent
        
        # Create gaussian kernel for local averaging
        kernel = torch.ones(num_features, 1, n, n) / (n * n)
        self.register_buffer('kernel', kernel)
    
    def forward(self, x):
        # All kernels are already on the correct device thanks to register_buffer
        # Calculate local average using convolution
        local_mean = F.conv2d(x, self.kernel, padding='same', groups=x.size(1))
        
        # Subtract mean and normalize
        centered = x - local_mean
        
        # Calculate local standard deviation
        local_var = F.conv2d(centered.pow(2), self.kernel, padding='same', groups=x.size(1))
        local_std = (local_var + self.alpha).pow(self.beta)
        
        # Normalize by local standard deviation
        normalized = centered / local_std
        
        return normalized

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Encoder (CNN) layers
        self.conv1 = nn.Conv2d(1, config.conv1_channels, kernel_size=config.kernel_size, stride=1, padding=1)
        self.norm1 = LocalContrastNorm(config.conv1_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=config.pool_size, stride=2, return_indices=True)
        
        self.conv2 = nn.Conv2d(config.conv1_channels, config.conv2_channels, kernel_size=config.kernel_size, stride=1, padding=1)
        self.norm2 = LocalContrastNorm(config.conv2_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=config.pool_size, stride=2, return_indices=True)
        
        self.fc = nn.Linear(config.conv2_channels * 7 * 7, config.fc_units)
        
        # Decoder (Deconv) layers for visualization
        self.unpool1 = nn.MaxUnpool2d(kernel_size=config.pool_size, stride=2)
        self.deconv1 = nn.ConvTranspose2d(config.conv2_channels, config.conv1_channels, kernel_size=config.kernel_size, stride=1, padding=1)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=config.pool_size, stride=2)
        self.deconv2 = nn.ConvTranspose2d(config.conv1_channels, 1, kernel_size=config.kernel_size, stride=1, padding=1)
        
        # Store conv layers for normalization
        self.conv_layers = [self.conv1, self.conv2]
        self.filter_radius = 1e-1  # As per ZF2013
        self.steps_since_norm = 0
        self.norm_frequency = 50  # Normalize every 50 steps

    def normalize_filters(self):
        """Normalize filters whose RMS exceeds a fixed radius to that fixed radius (ZF2013 Sec 3)"""
        self.steps_since_norm += 1
        if self.steps_since_norm >= self.norm_frequency:
            with torch.no_grad():
                for conv in self.conv_layers:
                    # Calculate RMS for each filter
                    weight = conv.weight.data
                    rms = torch.sqrt(torch.mean(weight.pow(2), dim=(1,2,3)))
                    
                    # Find filters exceeding the radius
                    exceeded = rms > self.filter_radius
                    
                    # Only normalize if any filters exceed the radius
                    if exceeded.any():
                        scale = torch.ones_like(rms)
                        scale[exceeded] = self.filter_radius / rms[exceeded]
                        conv.weight.data *= scale.view(-1, 1, 1, 1)
            
            self.steps_since_norm = 0

    def forward(self, x, store_switches=False):
        # Forward pass through encoder
        switches = {}
        
        # Layer 1
        x = F.relu(self.conv1(x))
        x = self.norm1(x)  # Apply contrast normalization
        if store_switches:
            switches['pre_pool1'] = x
        x, switch1 = self.pool1(x)
        if store_switches:
            switches['pool1'] = switch1
            
        # Layer 2
        x = F.relu(self.conv2(x))
        x = self.norm2(x)  # Apply contrast normalization
        if store_switches:
            switches['pre_pool2'] = x
        x, switch2 = self.pool2(x)
        if store_switches:
            switches['pool2'] = switch2
            switches['features'] = x.clone()
            
        # Classification layer
        x = self.fc(x.view(x.size(0), -1))
        
        return (x, switches) if store_switches else x

    def deconv_visualization(self, feature_maps, switches, layer):
        """Project feature maps back to input space using deconvnet approach (ZF2013 Sec 2.1)
        Steps for each layer:
        1. Unpool - Use stored switch locations
        2. Rectify - Apply ReLU
        3. Filter - Use transposed convolution filters
        """
        x = feature_maps
        
        if layer == 2:
            # Unpool layer 2
            x = self.unpool1(x, switches['pool2'])
            # Rectify
            x = F.relu(x)
            # Filter with transposed weights
            x = self.deconv1(x)
            
            # Unpool layer 1
            x = self.unpool2(x, switches['pool1'])
            # Rectify
            x = F.relu(x)
            # Filter with transposed weights
            x = self.deconv2(x)
        else:  # layer 1
            # Unpool layer 1
            x = self.unpool2(x, switches['pool1'])
            # Rectify
            x = F.relu(x)
            # Filter with transposed weights
            x = self.deconv2(x)
            
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
            _, switches = model(images, store_switches=True)
            
            # Process each layer
            for layer, key in [(1, 'pre_pool1'), (2, 'features')]:
                acts = switches[key]
                
                # Initialize storage if needed
                if strongest[layer]['activations'] is None:
                    strongest[layer]['activations'] = torch.zeros((acts.size(1),), device=device)
                    strongest[layer]['images'] = torch.zeros((acts.size(1), 1, 28, 28), device=device)
                
                # Find max activations for each feature map
                for f in range(acts.size(1)):
                    feature_acts = acts[:, f]
                    batch_max_acts, _ = feature_acts.reshape(feature_acts.size(0), -1).max(1)
                    max_batch = batch_max_acts.argmax()
                    max_act = batch_max_acts[max_batch]
                    
                    if max_act > strongest[layer]['activations'][f]:
                        strongest[layer]['activations'][f] = max_act
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
            _, switches = model(image.to(device), store_switches=True)
            
            # Get the feature maps for this layer
            feature_maps = switches['features'] if layer == 2 else switches['pre_pool1']
            
            # Create a copy with all activations set to zero except the target feature
            zeroed_maps = torch.zeros_like(feature_maps)
            
            # For the target feature map, keep only its strongest activation
            target_map = feature_maps[0, idx].clone()
            max_val, max_idx = target_map.view(-1).max(0)
            target_map_zeroed = torch.zeros_like(target_map)
            target_map_zeroed.view(-1)[max_idx] = max_val
            zeroed_maps[0, idx] = target_map_zeroed
            
            # Get reconstruction through deconvnet
            reconstruction = model.deconv_visualization(zeroed_maps, switches, layer)
            
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
    