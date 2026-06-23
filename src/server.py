import flwr as fl
from typing import Dict, List, Optional, Tuple
from flwr.common import Metrics
import logging
import json
import os
from src.utils import ensure_dir

logger = logging.getLogger("FakeShield")

# Global variables to store metrics for the dashboard
history_metrics = {
    "accuracy": [],
    "loss": [],
    "f1": [],
    "communication_cost": [] # Simulated
}

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "server_round": server_round,
        # Dynamically adjust epochs based on round if needed
        "epochs": 1, 
    }
    return config

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics over all clients."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    f1_scores = [num_examples * m["f1"] for num_examples, m in metrics]
    
    # Aggregate and return
    agg_acc = sum(accuracies) / sum(examples)
    agg_f1 = sum(f1_scores) / sum(examples)
    
    return {"accuracy": agg_acc, "f1": agg_f1}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_acc = 0.0

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics using FedAvg."""
        if not results:
            return None, {}
            
        # Call the default FedAvg aggregation
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        # Save history for dashboard
        if metrics_aggregated:
            acc = metrics_aggregated.get("accuracy", 0.0)
            f1 = metrics_aggregated.get("f1", 0.0)
            
            history_metrics["accuracy"].append((server_round, acc))
            history_metrics["loss"].append((server_round, float(loss_aggregated)))
            history_metrics["f1"].append((server_round, f1))
            
            # Simulate communication cost (e.g., 200MB per round per client)
            # 200MB * 5 clients * 2 (up/down) = 2000MB
            history_metrics["communication_cost"].append((server_round, 2000 * server_round))
            
            logger.info(f"Round {server_round} aggregated metrics: Loss={loss_aggregated:.4f}, Acc={acc:.4f}, F1={f1:.4f}")
            
            # Save history to file so dashboard can read it
            ensure_dir('reports')
            with open('reports/fl_history.json', 'w') as f:
                json.dump(history_metrics, f)
                
        return loss_aggregated, metrics_aggregated

def get_strategy(num_rounds=10):
    """Returns the custom FedAvg strategy."""
    strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=2, # Minimum number of clients to be sampled for next round
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
    )
    return strategy
