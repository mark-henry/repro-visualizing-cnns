import torch
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm

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
            images = images.to(model.device)
            model_state = model(images)
            
            # Process each layer
            for layer in [1, 2]:
                # Get feature maps for this layer
                acts = (model_state.final_features if layer == len(model.conv_layers)
                       else model_state.layer_states[layer-1].pre_pool)
                
                # Initialize storage if needed
                if strongest[layer]['activations'] is None:
                    strongest[layer]['activations'] = torch.zeros((acts.size(1),), device=model.device)
                    strongest[layer]['images'] = torch.zeros((acts.size(1), 1, 28, 28), device=model.device)
                
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
            model_state = model(image.to(model.device))
            
            # Get the feature maps for this layer
            feature_maps = (model_state.final_features if layer == len(model.conv_layers) 
                          else model_state.layer_states[layer-1].pre_pool)
            
            # Create a copy with all feature maps zeroed except the target one
            zeroed_maps = torch.zeros_like(feature_maps)
            zeroed_maps[0, idx] = feature_maps[0, idx]  # Keep entire feature map
            
            # Get reconstruction through deconvnet
            reconstruction = model.deconv_visualization(zeroed_maps, model_state, layer)
            
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