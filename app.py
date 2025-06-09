import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("models/lstm_model.h5")
with open("models/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Constants
max_length = 200  # set same as used during training

# Title
st.title("📰 Fake News Detector")
st.markdown("Enter a news headline or article to check if it's **Real or Fake**.")

# Input
text_input = st.text_area("Paste News Text Here:")

# Predict button
if st.button("Check News"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded)[0][0]
        label = "FAKE 🛑" if prediction < 0.5 else "REAL ✅"
        
        st.subheader("Prediction:")
        st.write(f"**{label}**")
        st.caption(f"(Confidence: {prediction:.2f})")
