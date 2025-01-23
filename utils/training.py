import torch
import torch.nn as nn
from torchvision import datasets, transforms
import wandb
from tqdm.auto import tqdm

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
    device = next(model.parameters()).device
    
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
            model.normalize_filters()
            
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
                "loss": loss.item(),
                "accuracy": accuracy,
                "examples": num_batches * config.batch_size,
            })
        
        # Log epoch metrics
        wandb.log({
            "epoch_loss": epoch_loss / num_batches,
            "epoch_accuracy": epoch_accuracy / num_batches,
        }) 