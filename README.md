# 🛡️ FakeShield AI: Privacy-Preserving Fake News Detection

![FakeShield AI Architecture](architecture/system_architecture.png)

## 📌 Project Overview
FakeShield AI is a complete, production-ready B.Tech Artificial Intelligence final year project. It demonstrates the use of **Federated Learning** to train a highly accurate **Transformer (DistilBERT)** model for Fake News Detection without compromising data privacy. 

Traditional ML models require data centralization, which raises security and ownership concerns. FakeShield AI solves this by deploying models to clients (nodes), training them locally, and only sharing model parameters with a central server via the **FedAvg** algorithm.

## ✨ Features
- **Privacy-Preserving**: Raw news data never leaves the local client.
- **Deep Learning NLP**: Uses HuggingFace DistilBERT for state-of-the-art text classification.
- **Federated Learning Orchestration**: Powered by the Flower (FLWR) framework.
- **Interactive Dashboard**: A beautiful Streamlit frontend to interact with the model and view training analytics.
- **FastAPI Backend**: Robust asynchronous API for model inference and triggering FL training loops.
- **Colab Ready**: Includes a Jupyter Notebook for easy execution on Google Colab free GPUs.

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/FakeShieldAI.git
cd FakeShieldAI
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Dataset Preparation
Place your dataset (`Fake.csv` and `True.csv`) inside the `dataset/` folder. If they are missing, the system will auto-generate dummy data for testing purposes.

---

## 💻 Usage

### 1. Start the FastAPI Backend
Open a terminal and run:
```bash
uvicorn src.api:app --reload
```
The API will be available at `http://localhost:8000`. You can view the Swagger UI at `http://localhost:8000/docs`.

### 2. Start the Streamlit Dashboard
Open a new terminal and run:
```bash
streamlit run src/dashboard.py
```
The interactive UI will open in your browser at `http://localhost:8501`.

### 3. Run Federated Learning Simulation via CLI (Optional)
If you prefer not to use the dashboard, you can trigger the FL simulation directly:
```bash
python -m src.federated --clients 5 --rounds 10 --epochs 1
```

---

## 📂 Project Structure
```text
FakeShieldAI/
├── dataset/                # Place Fake.csv and True.csv here
├── src/                    # Source code directory
│   ├── api.py              # FastAPI backend endpoints
│   ├── client.py           # Flower FL client implementation
│   ├── dashboard.py        # Streamlit frontend application
│   ├── dataset.py          # Data loading and FL partitioning logic
│   ├── evaluate.py         # Evaluation metrics calculation
│   ├── federated.py        # Main FL simulation script
│   ├── model.py            # DistilBERT PyTorch model definition
│   ├── preprocess.py       # Text cleaning functions
│   └── utils.py            # Logging and plotting utilities
├── reports/                # Project reports, PPT content, Viva QA
├── models/                 # Saved model weights
├── architecture/           # Architecture diagrams
├── FakeShield_Colab.ipynb  # Google Colab deployment notebook
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 📊 Results & Architecture
### System Architecture
The system consists of N edge clients and 1 central server. Clients pull the global model, train on local news data, and push gradients back.
### Evaluation Metrics
The model is evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Federated learning typically achieves near-centralized accuracy (e.g., >95% on the Kaggle Fake News dataset) while maintaining 100% data privacy.

## 🔮 Future Scope
- **Differential Privacy**: Adding noise to gradients before sending them to the server to prevent membership inference attacks.
- **Secure Multi-Party Computation (SMPC)**: Encrypting weights during aggregation.
- **Handling Non-IID Data**: Implementing advanced FL strategies like FedProx to handle severe data imbalance across clients.

## 👨‍💻 Contributors
- Saman Sunasara - Lead AI Developer

## 📝 License
This project is licensed under the MIT License.
