import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="FakeShield AI", page_icon="🛡️", layout="wide")

st.title("🛡️ FakeShield AI")
st.subheader("Privacy-Preserving Fake News Detection using Federated Learning")

tabs = st.tabs(["🔍 Predict", "🚀 Training Dashboard", "📊 Analytics", "ℹ️ About"])

with tabs[0]:
    st.header("Analyze News Article")
    news_text = st.text_area("Paste news article text here:", height=200)
    
    if st.button("Detect Fake News", type="primary"):
        if news_text:
            with st.spinner("Analyzing..."):
                try:
                    response = requests.post(f"{API_URL}/predict", json={"text": news_text})
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result['prediction']
                        confidence = result['confidence'] * 100
                        
                        if prediction == "Real":
                            st.success(f"✅ Prediction: **{prediction} News** (Confidence: {confidence:.2f}%)")
                        else:
                            st.error(f"🚨 Prediction: **{prediction} News** (Confidence: {confidence:.2f}%)")
                except Exception as e:
                    st.error(f"Error connecting to backend API: {e}. Is the FastAPI server running?")
        else:
            st.warning("Please enter some text to analyze.")

with tabs[1]:
    st.header("Federated Learning Control Panel")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        clients = st.slider("Number of Clients", min_value=2, max_value=10, value=5)
    with col2:
        rounds = st.slider("Federated Rounds", min_value=1, max_value=50, value=10)
    with col3:
        epochs = st.slider("Local Epochs per Round", min_value=1, max_value=5, value=1)
        
    if st.button("Start Federated Training"):
        try:
            response = requests.post(f"{API_URL}/train", json={
                "clients": clients,
                "rounds": rounds,
                "epochs": epochs
            })
            if response.status_code == 200:
                st.success(response.json()["message"])
                st.info("Navigate to the Analytics tab to view progress.")
        except Exception as e:
            st.error(f"Error starting training: {e}")

with tabs[2]:
    st.header("Training Analytics")
    
    if st.button("Refresh Metrics"):
        try:
            response = requests.get(f"{API_URL}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                if "message" not in metrics:
                    
                    st.subheader("Global Model Performance")
                    
                    acc_df = pd.DataFrame(metrics["accuracy"], columns=["Round", "Accuracy"]).set_index("Round")
                    loss_df = pd.DataFrame(metrics["loss"], columns=["Round", "Loss"]).set_index("Round")
                    f1_df = pd.DataFrame(metrics["f1"], columns=["Round", "F1 Score"]).set_index("Round")
                    comm_df = pd.DataFrame(metrics["communication_cost"], columns=["Round", "Data (MB)"]).set_index("Round")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.line_chart(acc_df, color="#00ff00")
                        st.line_chart(f1_df, color="#0000ff")
                    with col2:
                        st.line_chart(loss_df, color="#ff0000")
                        st.line_chart(comm_df, color="#ffa500")
                        
                else:
                    st.info("No training metrics available yet. Start a training session first.")
        except Exception as e:
            st.error(f"Could not fetch metrics: {e}")

with tabs[3]:
    st.header("About FakeShield AI")
    st.markdown("""
    **FakeShield AI** is a final year B.Tech AI project demonstrating how to build a robust fake news detection system without compromising user data privacy.
    
    ### Key Technologies:
    - **Machine Learning**: PyTorch, Transformers (DistilBERT)
    - **Federated Learning**: Flower (FLWR)
    - **Backend**: FastAPI
    - **Frontend**: Streamlit
    
    ### How it works:
    Instead of sending private news data to a central server, the model is sent to the clients. The clients train the model locally on their data and only send back the *updated model weights*. The server aggregates these weights (using FedAvg) to improve the global model.
    """)
