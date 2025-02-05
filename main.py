import os
import torch
import wandb
import argparse
from types import SimpleNamespace

from models.cnn import SimpleCNN
from utils.training import get_data, train
from visualization.feature_vis import find_strongest_activations, visualize_features

if __name__ == "__main__":
    # Configuration
    config = {
        "architecture": "SimpleCNN",
        "dataset": "MNIST",
        "epochs": 1,
        "batch_size": 64,
        "learning_rate": 0.001,
        "conv1_channels": 8,
        "conv2_channels": 16,
        "kernel_size": 5,
        "pool_size": 2,
        "fc_units": 10,
    }
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train or visualize a CNN model')
    parser.add_argument('--mode', type=str, choices=['train', 'visualize'], required=True,
                      help='Whether to train a new model or visualize an existing one')
    parser.add_argument('--model_path', type=str, default='model.pth',
                      help='Path to save/load the model (default: model.pth)')
    parser.add_argument('--num_images', type=int, default=10000,
                      help='Number of images to search for strongest activations (default: 10000)')
    parser.add_argument('--layers', type=str, default='1,2',
                      help='Comma-separated list of layers to visualize (default: 1,2)')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of epochs to train for (default: 1)')
    args = parser.parse_args()
    
    # Update config with command line arguments
    config["epochs"] = args.epochs
    
    # Main execution
    if args.mode == 'train':
        print("Training new model...")
        with wandb.init(project="cnn-feature-visualization", config=config):
            config = wandb.config
            train_loader, test_loader = get_data(config)
            model = SimpleCNN(config)
            train(model, train_loader, test_loader, config)
            
            print(f"Saving model to {args.model_path}...")
            torch.save(model.state_dict(), args.model_path, _use_new_zipfile_serialization=False)
            wandb.save(args.model_path)
            print("Training complete!")
            
    else:  # visualize mode
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found at {args.model_path}")
            
        print(f"Loading model from {args.model_path}...")
        model = SimpleCNN(SimpleNamespace(**config))
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