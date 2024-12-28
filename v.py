import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import AutoModelForImageClassification
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Simple CNN architecture with max pooling
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Corresponding deconv layers
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, store_switches=False):
        # Forward pass with option to store pooling indices
        switches = {}
        
        # Layer 1
        x = F.relu(self.conv1(x))
        x, switch1 = self.pool1(x)
        if store_switches:
            switches['pool1'] = switch1
            
        # Layer 2
        x = F.relu(self.conv2(x))
        x, switch2 = self.pool2(x)
        if store_switches:
            switches['pool2'] = switch2
            
        if store_switches:
            return x, switches
        return x

    def deconv_visualization(self, feature_maps, switches, layer):
        """
        Reconstruct input that would activate given feature maps
        layer: which layer's feature maps to visualize
        """
        x = feature_maps
        
        if layer == 2:
            # Unpool and deconv from layer 2
            x = self.unpool1(x, switches['pool2'])
            x = F.relu(self.deconv1(x))
            
        # Unpool and deconv from layer 1
        x = self.unpool2(x, switches['pool1'])
        x = self.deconv2(x)
        
        return x

def visualize_layer_features(model, image, layer=1):
    """
    Visualize what features a given layer has learned
    """
    # Forward pass storing pooling switches
    model.eval()
    with torch.no_grad():
        feature_maps, switches = model(image, store_switches=True)
        
        # Zero out all activations except one at a time
        visualizations = []
        for feature_idx in range(feature_maps.size(1)):
            zeroed_maps = torch.zeros_like(feature_maps)
            zeroed_maps[0, feature_idx] = feature_maps[0, feature_idx]
            
            # Reconstruct input using deconv network
            reconstruction = model.deconv_visualization(zeroed_maps, switches, layer)
            visualizations.append(reconstruction.squeeze().numpy())
            
        # Plot results
        fig, axes = plt.subplots(4, 8, figsize=(15, 8))
        for idx, vis in enumerate(visualizations[:32]):  # Show first 32 features
            ax = axes[idx//8, idx%8]
            ax.imshow(vis, cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

# Load pre-trained model or train new one
def get_model():
    # Option 1: Load from HuggingFace
    # return AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    
    # Option 2: Train simple CNN
    model = SimpleCNN()
    # Add training code here if needed
    return model

# Example usage
def main():
    # Get MNIST test image
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    mnist = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_image = mnist[0][0].unsqueeze(0)  # Add batch dimension
    
    # Load model and visualize
    model = get_model()
    visualize_layer_features(model, test_image, layer=1)
    visualize_layer_features(model, test_image, layer=2)

if __name__ == "__main__":
    main()