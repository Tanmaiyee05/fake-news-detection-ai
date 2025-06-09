# app.py

import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
@st.cache_resource
def load_resources():
    with open("models/tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    model = load_model("models/lstm_model.h5")
    return tokenizer, model

tokenizer, model = load_resources()

# App UI
st.title("📰 Fake News Detection (LSTM Model)")

news = st.text_area("Enter the news article content below:")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([news])
        padded = pad_sequences(sequence, maxlen=300)  # Adjust to model's input size

        # Predict
        prediction = model.predict(padded)[0][0]

        if prediction >= 0.5:
            st.success("✅ Real News")
        else:
            st.error("🚨 Fake News")
