import streamlit as st
import numpy as np
import pickle  # Import pickle to load the tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load pre-trained GRU model
try:
    model_gru = load_model('gru_model.keras')
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_gru = None

# Load the saved tokenizer
try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        st.success("Tokenizer loaded successfully.")
except Exception as e:
    st.error(f"Failed to load tokenizer: {e}")
    tokenizer = None

# Set max sequence length (ensure this matches the value used during training)
max_sequence_length = 100


# Define a function for preprocessing the input URL
def preprocess_url(url):
    if tokenizer is None:
        st.error("Tokenizer isn't loaded. Ensure the tokenizer is available.")
        return None
    sequences = tokenizer.texts_to_sequences([url])
    return pad_sequences(sequences, maxlen=max_sequence_length)


# Define a function for making predictions
def predict_url(url):
    if model_gru is None:
        st.error("Model isn't loaded. Prediction cannot proceed.")
        return None

    if tokenizer is None:
        st.error("Tokenizer isn't loaded. Prediction cannot proceed.")
        return None

    processed_url = preprocess_url(url)
    if processed_url is None:
        return None
    try:
        prediction = model_gru.predict(processed_url)
        confidence = prediction[0][0]
        return confidence, "Phishing" if confidence > 0.5 else "Not Phishing"
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None


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
            result = predict_url(test_url)
            if result is None:
                st.error("Could not make a prediction. Ensure all components are loaded properly.")
            else:
                confidence, prediction_result = result
                st.write(f"### Prediction: {prediction_result}")
                st.write(f"Confidence: {confidence:.2f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
