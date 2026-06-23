import flwr as fl
import torch
import logging
from src.model import FakeNewsModel
from src.train import train_model, get_parameters, set_parameters
from src.evaluate import evaluate

logger = logging.getLogger("FakeShield")

class FakeNewsClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader, device, model_name='distilbert-base-uncased', epochs=5, lr=2e-5):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        
        self.model = FakeNewsModel(model_name=model_name).to(self.device)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        
        # We can receive hyperparameters from the server config
        epochs = config.get("epochs", self.epochs)
        lr = config.get("lr", self.lr)
        
        logger.info(f"Client starting training for {epochs} epochs")
        
        # Train locally
        train_model(self.model, self.train_loader, epochs=epochs, lr=lr, device=self.device)
        
        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        
        loss, metrics = evaluate(self.model, self.test_loader, self.device)
        
        return float(loss), len(self.test_loader.dataset), metrics
