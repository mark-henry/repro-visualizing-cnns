import torch
import torch.nn as nn
from torchvision import datasets, transforms
import wandb
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def get_data(config):
    """Load CIFAR-100 dataset"""
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to match ImageNet dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],  # CIFAR-100 mean
                           std=[0.2675, 0.2565, 0.2761])     # CIFAR-100 std
    ])
    
    train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100('./data', train=False, transform=transform)
    
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4),
        torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)
    )

def evaluate(model, test_loader):
    """Evaluate model on test set
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test set
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = total_correct = total_samples = 0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            model_state = model(images)
            loss = criterion(model_state.logits, labels)
            
            _, predicted = torch.max(model_state.logits.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
    
    return total_loss / total_samples, total_correct / total_samples

def train(model, train_loader, test_loader, config):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Cosine annealing scheduler with warm restarts
    # T_0: number of iterations for the first restart
    # T_mult: factor to increase T_i after a restart
    # We'll do 2 restarts per epoch, and double the cycle length after each restart
    steps_per_epoch = len(train_loader)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=steps_per_epoch // 2,  # First restart after half epoch
        T_mult=2,  # Double the period after each restart
        eta_min=config.learning_rate * 0.01  # Min learning rate is 1% of initial
    )
    
    wandb.watch(model, criterion, log="all", log_freq=50)
    
    # Move model to device once
    device = next(model.parameters()).device
    
    # Track best model
    best_test_acc = 0
    eval_frequency = 500  # Evaluate every 500 batches
    norm_frequency = 50   # Normalize filters every 50 batches
    global_step = 0
    
    for epoch in tqdm(range(config.epochs)):
        model.train()
        epoch_loss = epoch_accuracy = num_batches = 0
        
        for images, labels in train_loader:
            # Move batch to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass and loss
            optimizer.zero_grad()
            model_state = model(images)
            loss = criterion(model_state.logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Periodic filter normalization
            if global_step % norm_frequency == 0:
                normalization_info = model.normalize_filters()
                if normalization_info:  # Log if any filters were normalized
                    wandb.log({
                        "filters_normalized": len(normalization_info),
                        "global_step": global_step
                    })
            
            # Track metrics
            with torch.no_grad():  # Don't track metrics computation
                _, predicted = torch.max(model_state.logits.data, 1)
                accuracy = (predicted == labels).sum().item() / labels.size(0)
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                num_batches += 1
            
            # Log batch metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
                "learning_rate": current_lr,
                "examples": num_batches * config.batch_size,
            })
            
            # Evaluate on test set periodically
            global_step += 1
            if global_step % eval_frequency == 0:
                test_loss, test_accuracy = evaluate(model, test_loader)
                wandb.log({
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                    "global_step": global_step
                })
                
                # Save best model
                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy
                    wandb.run.summary["best_test_accuracy"] = best_test_acc
                
                # Switch back to train mode
                model.train()
        
        # Log epoch metrics
        wandb.log({
            "epoch_train_loss": epoch_loss / num_batches,
            "epoch_train_accuracy": epoch_accuracy / num_batches,
        }) 