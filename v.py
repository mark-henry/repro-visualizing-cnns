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

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        # CNN architecture with max pooling
        self.conv1 = nn.Conv2d(1, config.conv1_channels, kernel_size=config.kernel_size, 
                              stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=config.pool_size, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(config.conv1_channels, config.conv2_channels, 
                              kernel_size=config.kernel_size, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=config.pool_size, stride=2, return_indices=True)
        
        # Classification layer
        self.fc = nn.Linear(config.conv2_channels * 7 * 7, config.fc_units)
        
        # Deconv layers for visualization
        self.unpool1 = nn.MaxUnpool2d(kernel_size=config.pool_size, stride=2)
        self.deconv1 = nn.ConvTranspose2d(config.conv2_channels, config.conv1_channels, 
                                         kernel_size=config.kernel_size, stride=1, padding=1)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=config.pool_size, stride=2)
        self.deconv2 = nn.ConvTranspose2d(config.conv1_channels, 1, 
                                         kernel_size=config.kernel_size, stride=1, padding=1)

    def forward(self, x, store_switches=False):
        switches = {}
        
        # Layer 1
        x = F.relu(self.conv1(x))
        x1 = x  # Store pre-pooling activations
        x, switch1 = self.pool1(x)
        if store_switches:
            switches['pool1'] = switch1
            switches['pre_pool1'] = x1
            
        # Layer 2
        x = F.relu(self.conv2(x))
        x2 = x  # Store pre-pooling activations
        x, switch2 = self.pool2(x)
        if store_switches:
            switches['pool2'] = switch2
            switches['pre_pool2'] = x2
            switches['features'] = x.clone()
            
        x = x.view(x.size(0), -1)
        x = self.fc(x)
            
        if store_switches:
            return x, switches
        return x

    def deconv_visualization(self, feature_maps, switches, layer):
        """
        Reconstruct input that would activate given feature maps
        layer: which layer's feature maps to visualize (1 or 2)
        """
        if layer == 2:
            # For layer 2, start from the second layer features
            x = feature_maps  # [1, 64, 7, 7]
            x = self.unpool1(x, switches['pool2'])  # [1, 64, 14, 14]
            x = F.relu(self.deconv1(x))  # [1, 32, 14, 14]
            x = self.unpool2(x, switches['pool1'])  # [1, 32, 28, 28]
            x = self.deconv2(x)  # [1, 1, 28, 28]
        else:
            # For layer 1, we need to handle the first layer features
            x = feature_maps  # [1, 32, 14, 14]
            # Create a properly sized output tensor for unpooling
            output_size = (feature_maps.size(0), feature_maps.size(1), 28, 28)
            x = F.interpolate(x, size=(28, 28), mode='nearest')  # Simple upsampling instead of unpooling
            x = self.deconv2(x)  # [1, 1, 28, 28]
        
        return x

def get_data(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=config.batch_size,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=config.batch_size,
                                            shuffle=False)
    return train_loader, test_loader

def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss

def train(model, train_loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    example_ct = 0
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct += len(images)
            batch_ct += 1
            
            if ((batch_ct + 1) % 25) == 0:
                # Log metrics
                wandb.log({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "examples": example_ct,
                })

def visualize_features(model, test_image, layer):
    model.eval()
    with torch.no_grad():
        _, switches = model(test_image.to(device), store_switches=True)
        
        # Get the appropriate feature maps based on layer
        if layer == 2:
            feature_maps = switches['features']  # [1, 64, 7, 7]
            num_features = feature_maps.size(1)
        else:
            feature_maps = switches['pre_pool1']  # [1, 32, 14, 14]
            num_features = feature_maps.size(1)
        
        fig, axes = plt.subplots(4, 8, figsize=(15, 8))
        for feature_idx in range(min(32, num_features)):
            zeroed_maps = torch.zeros_like(feature_maps)
            zeroed_maps[0, feature_idx] = feature_maps[0, feature_idx]
            
            reconstruction = model.deconv_visualization(zeroed_maps, switches, layer)
            
            ax = axes[feature_idx//8, feature_idx%8]
            ax.imshow(reconstruction.squeeze().cpu(), cmap='gray')
            ax.axis('off')
                
        plt.tight_layout()
        wandb.log({f"layer_{layer}_features": wandb.Image(plt)})
        plt.close()

def model_pipeline(config):
    with wandb.init(project="cnn-feature-visualization", config=config):
        config = wandb.config
        
        # Get data, model, and training components
        train_loader, test_loader = get_data(config)
        model = SimpleCNN(config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Train the model
        train(model, train_loader, criterion, optimizer, config)
        
        # Visualize features
        test_image = next(iter(test_loader))[0][0:1]
        visualize_features(model, test_image, layer=1)
        visualize_features(model, test_image, layer=2)
        
        # Save model
        torch.onnx.export(model, test_image, "model.onnx")
        wandb.save("model.onnx")
        
        return model

if __name__ == "__main__":
    config = {
        "architecture": "SimpleCNN",
        "dataset": "MNIST",
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
        "conv1_channels": 32,
        "conv2_channels": 64,
        "kernel_size": 3,
        "pool_size": 2,
        "fc_units": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    model = model_pipeline(config)
    