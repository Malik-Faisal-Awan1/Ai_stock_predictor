import streamlit as st
from src.data import get_stock_data
from src.features import make_features_and_labels
from src.models.knn_model import train_knn, predict_knn
from src.models.mlp_model import train_mlp, predict_mlp
from src.evaluate import evaluate_classification
from src.db import log_run, get_run_history

st.title("Stock Movement Prediction (AI-Lab Term Project)")
st.write("Project by: Malik-Faisal-Awan1")

# Sidebar for user options
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
years = st.sidebar.slider("Years of Data", 2, 10, 2)
model_type = st.sidebar.selectbox("Model", ["KNN", "Neural Network (MLP)"])

# Fetch and display data
df = get_stock_data(ticker, years)

if df is not None:
    st.write("Raw Data", df.tail(5))
    X, y, feature_names = make_features_and_labels(df)
    st.write(f"Features: {list(feature_names)} | {len(X)} samples")
    
    train_size = st.sidebar.slider("Training Size (%)", 50, 90, 80)
    n_train = int(len(X) * train_size / 100)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    if st.sidebar.button("Train"):
        if model_type == "KNN":
            model = train_knn(X_train, y_train)
            preds = predict_knn(model, X_test)
        else:
            model = train_mlp(X_train, y_train)
            preds = predict_mlp(model, X_test)
        
        acc, cm_fig = evaluate_classification(y_test, preds)
        st.success(f"Test Accuracy: {acc:.2%}")
        st.pyplot(cm_fig)
        log_run(ticker, model_type, acc)
        
        st.write("Run History", get_run_history())
else:
    st.warning("No data - check ticker.")

st.sidebar.markdown("---")
st.sidebar.write("© UCP AI-Lab Project")
