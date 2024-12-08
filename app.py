import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load pre-trained GRU model
model_gru = load_model('gru_model.h5')

# Define the tokenizer and max sequence length (adjust these values to match your trained model)
# Replace with the actual tokenizer and sequence length used during training
tokenizer = None  # Load your tokenizer here (e.g., via pickle or joblib)
max_sequence_length = 100  # Replace with the correct value

# Define a function for preprocessing the input URL
def preprocess_url(url):
    if tokenizer is None:
        st.error("Tokenizer is not loaded. Please load the tokenizer correctly.")
        return None
    sequences = tokenizer.texts_to_sequences([url])
    return pad_sequences(sequences, maxlen=max_sequence_length)

# Define a function for making predictions
def predict_url(url):
    processed_url = preprocess_url(url)
    if processed_url is None:
        return None
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
        try:
            confidence, result = predict_url(test_url)
            st.write(f"### Prediction: {result}")
            st.write(f"Confidence: {confidence:.2f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.write("\n**Note:** Ensure the tokenizer and the model are correctly loaded for this app to work.")
