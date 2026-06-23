import flwr as fl
import torch
import logging
import argparse
from src.dataset import load_data, partition_data, create_dataloaders
from src.client import FakeNewsClient
from src.server import get_strategy
from src.utils import ensure_dir
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FakeShield")

def start_simulation(num_clients=5, num_rounds=10, epochs=1, iid=True, model_name='distilbert-base-uncased'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    logger.info("Loading and preprocessing data...")
    texts, labels = load_data('dataset')
    
    # 2. Partition Data
    logger.info(f"Partitioning data among {num_clients} clients (IID={iid})...")
    partitions = partition_data(texts, labels, num_clients=num_clients, iid=iid)
    
    # 3. Create Clients
    def client_fn(cid: str) -> fl.client.Client:
        logger.info(f"Creating client {cid}")
        cid_idx = int(cid)
        c_texts, c_labels = partitions[cid_idx]
        train_loader, test_loader = create_dataloaders(c_texts, c_labels, batch_size=16)
        
        return FakeNewsClient(
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            model_name=model_name,
            epochs=epochs
        ).to_client()

    # 4. Configure Strategy
    strategy = get_strategy(num_rounds=num_rounds)
    
    # 5. Start Simulation
    logger.info(f"Starting Federated Learning Simulation for {num_rounds} rounds...")
    
    # Clear previous history
    ensure_dir('reports')
    if os.path.exists('reports/fl_history.json'):
        os.remove('reports/fl_history.json')
        
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1 if torch.cuda.is_available() else 0},
    )
    
    logger.info("Simulation completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeShield FL Simulation")
    parser.add_argument("--clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--non-iid", action="store_true", help="Use Non-IID data distribution")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Model name")
    
    args = parser.parse_args()
    
    start_simulation(
        num_clients=args.clients, 
        num_rounds=args.rounds, 
        epochs=args.epochs,
        iid=not args.non_iid,
        model_name=args.model
    )
