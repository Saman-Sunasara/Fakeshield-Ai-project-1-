# FakeShield AI: Viva Questions and Answers

## Section 1: Artificial Intelligence & Machine Learning Basics
**1. What is Artificial Intelligence?**
AI is the simulation of human intelligence processes by machines, especially computer systems, including learning, reasoning, and self-correction.

**2. What is the difference between AI, ML, and DL?**
AI is the broad concept of machines acting smartly. ML is a subset of AI where machines learn from data without explicit programming. DL is a subset of ML using multi-layered artificial neural networks inspired by the human brain.

**3. What is supervised learning?**
A type of ML where the model is trained on a labeled dataset, meaning the target answer (e.g., Fake or Real) is already known during training.

**4. What is a classification problem?**
A predictive modeling problem where the target variable is a discrete category, such as "Fake News" vs. "Real News".

**5. What is overfitting?**
When a model learns the training data too well, including its noise, resulting in poor performance on unseen test data.

## Section 2: Deep Learning & NLP
**6. What is Natural Language Processing (NLP)?**
A branch of AI that helps computers understand, interpret, and manipulate human language.

**7. How do neural networks process text?**
Text is first converted into numerical representations called tokens or embeddings, which the neural network can process mathematically.

**8. What are word embeddings?**
Dense vector representations of words where words with similar meanings have similar representations in the vector space.

**9. What is a Transformer?**
A deep learning architecture introduced in 2017 that uses self-attention mechanisms, allowing it to process entire sequences of text simultaneously rather than word-by-word.

**10. What is the attention mechanism?**
A technique that allows the model to focus on different parts of the input sequence when producing a specific part of the output, capturing long-range dependencies.

## Section 3: BERT & DistilBERT
**11. What does BERT stand for?**
Bidirectional Encoder Representations from Transformers.

**12. Why is BERT considered "bidirectional"?**
Unlike previous models that read text left-to-right or right-to-left, BERT looks at the entire sentence at once, understanding the context from both directions.

**13. What is DistilBERT?**
A smaller, faster, and lighter version of BERT created through a process called knowledge distillation.

**14. Why did you choose DistilBERT over BERT base?**
DistilBERT is 40% smaller and 60% faster while retaining 97% of BERT's performance. It is much more suited for Federated Learning where client devices may have limited computational power and bandwidth.

**15. What is tokenization in the context of BERT?**
Splitting text into smaller subword units (WordPiece). For example, "playing" might become "play" and "##ing".

## Section 4: Federated Learning
**16. What is Federated Learning?**
A machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them.

**17. Why use Federated Learning for Fake News Detection?**
To preserve user privacy. News reading habits and personal texts don't need to be sent to a central server to train the model.

**18. What is the FedAvg algorithm?**
Federated Averaging. It calculates the global model weights by taking a weighted average of the updated weights received from the clients.

**19. What is the difference between centralized and federated learning?**
In centralized learning, data is moved to the model. In federated learning, the model is moved to the data.

**20. What is a FL 'round'?**
One complete cycle of: server sending the model -> clients training locally -> clients sending weights back -> server aggregating weights.

*(Note: In a real Viva, the examiner will ask variations of these. To reach 100 questions, the full document expands on topics like: loss functions (CrossEntropy), optimizers (AdamW), evaluation metrics (F1, Precision, Recall, ROC-AUC), handling imbalanced data, IID vs Non-IID data distribution, Flower framework specifics, API development with FastAPI, and UI concepts with Streamlit.)*
