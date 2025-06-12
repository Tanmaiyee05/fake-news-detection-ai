import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_data()

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean and preprocess text same as training data"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = load_model('lstm_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Streamlit App
def main():
    # Title and description
    st.title("📰 Fake News Detection System")
    st.markdown("### Powered by LSTM Deep Learning Model")
    st.write("Enter a news article below to check if it's real or fake!")
    
    # Model disclaimer
    st.info("⚠️ **Model Accuracy Note:** This is an educational project model and may not be 100% accurate. Always verify news from multiple reliable sources!")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("❌ Could not load the model. Please make sure model files are available.")
        return
    
    # Input section
    st.markdown("---")
    st.subheader("📝 Enter News Article")
    
    # Text input options
    input_method = st.radio(
        "Choose input method:",
        ["Type/Paste Text", "Upload Text File"]
    )
    
    user_input = ""
    
    if input_method == "Type/Paste Text":
        user_input = st.text_area(
            "Paste your news article here:",
            height=200,
            placeholder="Enter the news article text here..."
        )
    else:
        uploaded_file = st.file_uploader("Choose a text file", type="txt")
        if uploaded_file is not None:
            user_input = str(uploaded_file.read(), "utf-8")
            st.text_area("Uploaded content:", user_input, height=150)
    
    # Prediction section
    if st.button("🔍 Analyze News", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing the article..."):
                try:
                    # Preprocess the input
                    cleaned_input = clean_text(user_input)
                    
                    # Convert to sequence
                    sequence = tokenizer.texts_to_sequences([cleaned_input])
                    padded_sequence = pad_sequences(sequence, maxlen=300)
                    
                    # Make prediction
                    prediction_prob = float(model.predict(padded_sequence)[0][0])
                    
                    # Determine result with adjusted thresholds for better balance
                    if prediction_prob > 0.6:  # Higher threshold for REAL
                        result = "REAL"
                        confidence = prediction_prob
                        color = "green"
                        icon = "✅"
                    elif prediction_prob < 0.3:  # Lower threshold for FAKE
                        result = "FAKE"
                        confidence = 1 - prediction_prob
                        color = "red"  
                        icon = "❌"
                    else:  # Uncertain range
                        result = "UNCERTAIN"
                        confidence = 0.5 + abs(prediction_prob - 0.5)
                        color = "orange"
                        icon = "❓"
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"### {icon} Prediction")
                        st.markdown(f"<h2 style='color: {color};'>{result} NEWS</h2>", 
                                   unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 📊 Confidence")
                        st.markdown(f"<h2>{confidence:.1%}</h2>", unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.progress(confidence)
                    
                    # Additional info
                    st.markdown("---")
                    st.markdown("### 📋 Analysis Details")
                    
                    if result == "FAKE":
                        st.warning("⚠️ This article shows characteristics of fake news. Be cautious about sharing or believing this information.")
                        st.info("🔍 Consider checking multiple reliable sources before accepting this information.")
                    elif result == "REAL":
                        st.success("✅ This article appears to be legitimate news.")
                        st.info("📰 Remember to always verify information from multiple trusted sources.")
                    else:  # UNCERTAIN
                        st.warning("❓ The model is uncertain about this article's authenticity.")
                        st.info("🤔 This article falls in the uncertain range. Please verify from multiple reliable sources before making any conclusions.")
                    
                    # Technical details (expandable)
                    with st.expander("🔧 Technical Details"):
                        st.write(f"**Model Type:** LSTM Neural Network")
                        st.write(f"**Raw Prediction Score:** {prediction_prob:.6f}")
                        st.write(f"**Processed Text Length:** {len(cleaned_input.split())} words")
                        st.write(f"**Original Text Length:** {len(user_input.split())} words")
                        st.write(f"**Cleaned Text Preview:** {cleaned_input[:200]}...")
                        st.write(f"**Sequence Length:** {len(sequence[0]) if sequence[0] else 0}")
                        st.write(f"**Padded Sequence Shape:** {padded_sequence.shape}")
                        
                        st.write(f"**First 10 tokens in sequence:**")
                        st.write(sequence[0][:10] if sequence[0] else "Empty sequence")
                        
                        # Prediction interpretation
                        st.markdown("**Prediction Interpretation:**")
                        if prediction_prob > 0.6:
                            st.success(f"Strong confidence this is REAL news (score: {prediction_prob:.3f})")
                        elif prediction_prob > 0.4:
                            st.warning(f"Uncertain prediction (score: {prediction_prob:.3f}) - could be either")
                        else:
                            st.error(f"Strong confidence this is FAKE news (score: {prediction_prob:.3f})")
                            
                        st.info("💡 **Tip:** Scores closer to 1.0 = Real News, Scores closer to 0.0 = Fake News")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    
    # Debug section (expandable)
    with st.expander("🔍 Debug Mode - Test Different Inputs"):
        st.write("Try these test cases to see if the model is working correctly:")
        
        test_cases = [
            "The stock market closed higher today after positive economic news.",
            "Scientists discover aliens living among us in secret underground bases.",
            "The government announced new healthcare policies effective next month.",
            "Breaking: World ends tomorrow according to ancient prophecy."
        ]
        
        for i, test_text in enumerate(test_cases):
            if st.button(f"Test Case {i+1}: {test_text[:50]}..."):
                cleaned_test = clean_text(test_text)
                test_sequence = tokenizer.texts_to_sequences([cleaned_test])
                test_padded = pad_sequences(test_sequence, maxlen=300)
                test_prob = float(model.predict(test_padded)[0][0])
                
                st.write(f"**Original:** {test_text}")
                st.write(f"**Cleaned:** {cleaned_test}")
                st.write(f"**Prediction:** {test_prob:.6f}")
                st.write(f"**Result:** {'REAL' if test_prob > 0.5 else 'FAKE'}")
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
