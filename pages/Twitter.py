import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib 

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+|http?://\S+', '', text)
    # Remove special characters, numbers, and punctuations
    text = re.sub(r'\W', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word not in stop_words])
    return text

# Load the pre-trained model and vectorizer
model_path = 'Models/sv_model.pkl'  # Change this path to your model file path
vectorizer_path = 'Models/tfidf_vectorizer.pkl'  # Change this path to your vectorizer file path
svm_model = SVC()
tfidf_vectorizer = TfidfVectorizer()

# Load the saved model and vectorizer
svm_model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)


# Streamlit app
def main():
    st.title("Twitter Sentiment Analysis")

    # Input text from user
    user_input = st.text_area("Enter a tweet:", "")

    if st.button("Predict"):
        if user_input:
            # Preprocess user input
            user_input_processed = preprocess_text(user_input)
            # Vectorize the processed text
            user_input_vectorized = tfidf_vectorizer.transform([user_input_processed])
            # Make prediction
            prediction = svm_model.predict(user_input_vectorized)

            st.write("Prediction:")
            if prediction[0] == 1:
                st.write("This is a disaster tweet.")
            else:
                st.write("This is a normal tweet.")
        else:
            st.warning("Please enter a tweet for prediction.")

if __name__ == "__main__":
    main()
