import streamlit as st
import numpy as np
import pickle  # Import pickle to load the tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load pre-trained GRU model
model_gru = load_model('gru_model.keras', compile=False)

# Load the saved tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set max sequence length (ensure this matches the value used during training)
max_sequence_length = 100

# Define a function for preprocessing the input URL
def preprocess_url(url):
    sequences = tokenizer.texts_to_sequences([url])
    return pad_sequences(sequences, maxlen=max_sequence_length)

# Define a function for making predictions
def predict_url(url):
    processed_url = preprocess_url(url)
    prediction = model_gru.predict(processed_url)
    confidence = prediction[0][0]
    return confidence, "Phishing" if confidence > 0.5 else "Not Phishing"

# Streamlit App Interface
st.title("Real-Time Phishing URL Detection")
st.write("Enter a URL below to check if it is phishing or not using the GRU model.")

# Input URL
test_url = st.text_input("Enter a URL:", "")

if st.button("Check URL"):
    if not test_url:
        st.warning("Please enter a URL to check.")
    else:
        result = predict_url(test_url)
        confidence, prediction_result = result
        st.write(f"### Prediction: {prediction_result}")
        st.write(f"Confidence: {confidence:.2f}")
