import torch
import torch.nn as nn
from torchvision import datasets, transforms
import wandb
from tqdm.auto import tqdm

def get_data(config):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    return (
        torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)
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