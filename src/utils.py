import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fakeshield.log', mode='a')
    ]
)
logger = logging.getLogger("FakeShield")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
    plt.close()

def plot_training_metrics(history, save_path=None):
    """Plot accuracy and loss from federated learning history."""
    if not history:
        return
        
    rounds = [m[0] for m in history['accuracy']]
    accs = [m[1] for m in history['accuracy']]
    losses = [m[1] for m in history['loss']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(rounds, accs, marker='o', color='b')
    ax1.set_title('Global Model Accuracy vs Rounds')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True)
    
    ax2.plot(rounds, losses, marker='o', color='r')
    ax2.set_title('Global Model Loss vs Rounds')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path)
    plt.close()

def save_metrics(metrics, filepath):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
