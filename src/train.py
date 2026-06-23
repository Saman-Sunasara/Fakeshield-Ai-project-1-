import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import logging
from .evaluate import evaluate

logger = logging.getLogger("FakeShield")

def train_model(model, train_loader, epochs, lr, device, val_loader=None):
    """
    Standard training loop for the PyTorch model.
    """
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar if logging level allows
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f}")
        
        if val_loader:
            val_loss, val_metrics = evaluate(model, val_loader, device)
            logger.info(f"Validation - Loss: {val_loss:.4f} - Acc: {val_metrics['accuracy']:.4f}")
            model.train() # Switch back to train mode

def get_parameters(model):
    """Returns the parameters of a PyTorch model as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    """Sets the parameters of a PyTorch model from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
