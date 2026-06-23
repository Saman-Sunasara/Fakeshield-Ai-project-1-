from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import subprocess
import os
import json
import torch
from transformers import AutoTokenizer
from src.model import FakeNewsModel
from src.preprocess import clean_text

app = FastAPI(title="FakeShield AI API", description="Privacy-Preserving Fake News Detection")

# Global variables for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = FakeNewsModel(model_name=model_name).to(device)
model.eval() # Set to evaluation mode

# We would ideally load the best global model weights here.
# For demonstration, we just initialize the model.

class PredictRequest(BaseModel):
    text: str

class TrainRequest(BaseModel):
    rounds: int = 5
    clients: int = 5
    epochs: int = 1

def run_fl_simulation(rounds, clients, epochs):
    """Runs the FL simulation as a background process."""
    cmd = f"python -m src.federated --rounds {rounds} --clients {clients} --epochs {epochs}"
    subprocess.Popen(cmd, shell=True)

@app.post("/predict")
async def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    cleaned_text = clean_text(req.text)
    
    inputs = tokenizer(
        cleaned_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
    label = "Real" if pred_class == 1 else "Fake"
    
    return {
        "text": req.text,
        "prediction": label,
        "confidence": float(confidence)
    }

@app.post("/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Triggers a background Federated Learning simulation."""
    background_tasks.add_task(run_fl_simulation, req.rounds, req.clients, req.epochs)
    return {"message": f"Federated Learning started with {req.clients} clients for {req.rounds} rounds."}

@app.get("/metrics")
async def get_metrics():
    """Returns the latest training metrics."""
    history_file = "reports/fl_history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    return {"message": "No training metrics available yet."}

@app.get("/status")
async def get_status():
    return {"status": "Active", "model": model_name, "device": str(device)}
