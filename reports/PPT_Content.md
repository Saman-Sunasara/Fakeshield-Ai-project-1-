# Slide 1: Title Slide
**FakeShield AI**
*Privacy-Preserving Fake News Detection using Federated Learning and BERT*
[Your Name/Team Name]
[Date]

# Slide 2: Introduction
- Fake news is a critical global issue affecting politics, finance, and society.
- AI (specifically NLP) is highly effective at detecting deceptive text.
- **The Problem:** Training powerful AI models requires massive centralized datasets.
- Centralization compromises user privacy and creates security vulnerabilities.

# Slide 3: Problem Statement
- Traditional systems force users to upload data to a central server.
- This leads to:
  - Privacy Violations
  - Data Ownership Issues
  - Security Risks (Single point of failure)
- **Goal:** Build an accurate fake news detector without centralizing data.

# Slide 4: Proposed Solution
- **Federated Learning (FL):** Bring the model to the data, not the data to the model.
- **Deep Learning (BERT):** Use state-of-the-art Transformers for NLP.
- **FakeShield AI:** A distributed system where local edge devices train the model collaboratively.

# Slide 5: System Objectives
1. Detect fake news accurately.
2. Strictly preserve user data privacy.
3. Implement Federated Learning architecture.
4. Compare centralized vs. federated performance.
5. Provide a real-time interactive dashboard.

# Slide 6: Technology Stack
- **Language:** Python 3.11
- **Machine Learning:** PyTorch, HuggingFace Transformers
- **Federated Learning:** Flower (FLWR) framework
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Data:** Pandas, Scikit-learn

# Slide 7: The Dataset
- **Source:** Kaggle Fake News Dataset (Fake.csv, True.csv)
- **Size:** 40,000+ articles
- **Labels:** 0 (Fake), 1 (Real)
- **Partitioning:** Data is split among simulated clients (IID or Non-IID distribution) to mimic real-world scenarios.

# Slide 8: Data Preprocessing
- Lowercasing text
- Removing URLs, HTML tags, and special characters
- **Tokenization:** WordPiece tokenization handled by HuggingFace AutoTokenizer.
- Converting text into numerical Tensors for PyTorch.

# Slide 9: Model Architecture: DistilBERT
- **Why DistilBERT?**
  - 40% smaller than BERT base.
  - 60% faster inference and training.
  - Retains 97% of BERT's language understanding.
- Crucial for Federated Learning where client devices (like phones) have limited compute.

# Slide 10: Federated Learning Architecture
- **Central Server:** Initializes model, aggregates weights.
- **Clients (1 to N):** Hold local data, perform local training.
- **Process:**
  1. Server sends weights to clients.
  2. Clients train on local data.
  3. Clients send updated weights back.
  4. Server aggregates weights.

# Slide 11: The FedAvg Algorithm
- Federated Averaging (FedAvg) is used to combine client updates.
- The server takes a weighted average of the received model parameters based on the number of training samples on each client.
- Formula: Global Weight = Sum(Client Weight * Client Data Size) / Total Data Size.

# Slide 12: Backend API (FastAPI)
- High-performance asynchronous API.
- Endpoints:
  - `POST /predict`: Takes text, returns Real/Fake prediction.
  - `POST /train`: Triggers the FL simulation process.
  - `GET /metrics`: Retrieves the latest training analytics.

# Slide 13: Frontend Dashboard (Streamlit)
- User-friendly web interface.
- **Features:**
  - Real-time prediction tool.
  - Control panel to start FL rounds.
  - Live charts showing Accuracy, Loss, and F1-Score over federated rounds.

# Slide 14: Experimental Results (Expected)
- **Centralized Model Accuracy:** ~98%
- **Federated Model Accuracy:** ~96.5% (after 10 rounds)
- The slight accuracy trade-off is justified by the 100% guarantee of data privacy.

# Slide 15: Evaluation Metrics
- Evaluated using standard ML metrics:
  - Accuracy
  - Precision, Recall, F1-Score
  - ROC-AUC Curve
  - Confusion Matrix
- Tracked Communication Cost (MB transferred per round).

# Slide 16: Security & Privacy Analysis
- **What is protected?** Raw text, reading habits, user identity.
- **What is shared?** Only floating-point numbers (model weights/gradients).
- Reverse engineering text from gradients is extremely difficult and mathematically complex.

# Slide 17: Advantages of FakeShield AI
1. Privacy by design.
2. Highly scalable to millions of devices.
3. Utilizes cutting-edge NLP technology.
4. Modular and production-ready architecture.

# Slide 18: Limitations
- **Communication Overhead:** Transferring model weights (260MB) repeatedly requires high bandwidth.
- **Client Heterogeneity:** Different devices have varying compute power, causing bottlenecks.
- **Non-IID Data:** Highly skewed data on clients can slow down model convergence.

# Slide 19: Future Scope
- **Differential Privacy:** Adding noise to gradients to prevent membership inference attacks.
- **Secure Aggregation:** Encrypting weights in transit.
- **Quantization:** Reducing the model size from 32-bit float to 8-bit integer to save bandwidth.

# Slide 20: Conclusion
- FakeShield AI proves that we do not need to sacrifice privacy for performance.
- By combining Transformers and Federated Learning, we can create secure, robust systems to combat misinformation globally.
- **Thank You! Questions?**
