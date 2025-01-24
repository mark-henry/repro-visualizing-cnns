import torch
import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm
import heapq

def find_strongest_activations(model, data_loader, num_samples=1000, top_k=9):
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
                       else model_state.layer_states[layer-1].output)
                
                # Initialize storage if needed
                if strongest[layer]['activations'] is None:
                    strongest[layer]['activations'] = [[] for _ in range(acts.size(1))]
                    strongest[layer]['images'] = [[] for _ in range(acts.size(1))]
                
                # Find max activations for each feature map
                for f in range(acts.size(1)):
                    feature_acts = acts[:, f]
                    # Use total activation across spatial dimensions
                    batch_total_acts = feature_acts.sum(dim=(1, 2))
                    
                    # Process each image in batch
                    for img_idx in range(images.size(0)):
                        total_act = batch_total_acts[img_idx].item()
                        
                        # Store activation and image
                        if len(strongest[layer]['activations'][f]) < top_k:
                            heapq.heappush(strongest[layer]['activations'][f], total_act)
                            strongest[layer]['images'][f].append(images[img_idx].clone())
                        elif total_act > strongest[layer]['activations'][f][0]:
                            # Remove smallest activation and its corresponding image
                            heapq.heapreplace(strongest[layer]['activations'][f], total_act)
                            strongest[layer]['images'][f][0] = images[img_idx].clone()
                            # Ensure images stay aligned with activations
                            idx = strongest[layer]['activations'][f].index(total_act)
                            strongest[layer]['images'][f][0], strongest[layer]['images'][f][idx] = \
                                strongest[layer]['images'][f][idx], strongest[layer]['images'][f][0]
    
    return strongest

def visualize_features(model, strongest_activations, layer):
    """Visualize patterns that cause strongest activations using deconvnet approach (ZF2013)"""
    model.eval()
    with torch.no_grad():
        num_features = len(strongest_activations[layer]['activations'])
        num_features = min(32, num_features)  # Limit to 32 features max
        
        # Calculate grid layout for feature blocks
        grid_cols = 4  # We'll do 4 features per row
        grid_rows = (num_features + grid_cols - 1) // grid_cols
        
        # Create figure
        # Each 3x3 block is treated as one unit
        fig = plt.figure(figsize=(5 * grid_cols, 5 * grid_rows))
        plt.suptitle(f'Layer {layer} Features - Top 9 Activations Each', y=0.95, fontsize=16)
        
        # Create grid for feature blocks
        gs = plt.GridSpec(grid_rows, grid_cols, figure=fig)
        
        # Process each feature
        for feature_idx in range(num_features):
            # Calculate this feature's position in the grid
            grid_row = feature_idx // grid_cols
            grid_col = feature_idx % grid_cols
            
            # Create subgrid for this feature's 3x3 activations
            subgs = gs[grid_row, grid_col].subgridspec(3, 3, wspace=0, hspace=0)
            
            # Sort activations by strength
            activations = strongest_activations[layer]['activations'][feature_idx]
            sorted_acts = sorted(activations, reverse=True)
            images = strongest_activations[layer]['images'][feature_idx]
            
            # Add feature number in top left corner
            ax_title = fig.add_subplot(gs[grid_row, grid_col])
            ax_title.text(0.02, 0.98, f'F{feature_idx}', 
                        transform=ax_title.transAxes,
                        fontsize=10, fontweight='bold',
                        verticalalignment='top')
            ax_title.axis('off')
            
            # Process top 9 activations for this feature
            for i, act in enumerate(sorted_acts[:9]):
                row = i // 3
                col = i % 3
                
                # Create subplot for this activation
                ax = fig.add_subplot(subgs[row, col])
                
                # Get original image that caused strong activation
                img_idx = activations.index(act)
                image = images[img_idx].unsqueeze(0)
                model_state = model(image.to(model.device))
                
                # Get the feature maps for this layer
                feature_maps = (model_state.final_features if layer == len(model.conv_layers) 
                              else model_state.layer_states[layer-1].output)
                
                # Create a copy with all feature maps zeroed except the target one
                zeroed_maps = torch.zeros_like(feature_maps)
                zeroed_maps[0, feature_idx] = feature_maps[0, feature_idx]
                
                # Get reconstruction through deconvnet
                reconstruction = model.deconv_visualization(zeroed_maps, model_state, layer)
                
                # Plot reconstruction
                img = reconstruction.squeeze().cpu()
                vmax = torch.abs(img).max().item()
                ax.imshow(img, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                ax.axis('off')
                
                # Add input image inset
                ax_ins = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
                ax_ins.imshow(image.squeeze().cpu(), cmap='gray')
                ax_ins.axis('off')
                
                # Add activation value in top-left corner
                ax.text(0.02, 0.98, f'{act:.1f}', 
                       transform=ax.transAxes,
                       fontsize=8,
                       verticalalignment='top')
        
        plt.tight_layout()
        wandb.log({f"layer_{layer}_all_features": wandb.Image(plt)})
        plt.close() 