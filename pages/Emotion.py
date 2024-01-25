import streamlit as st
import numpy as np 
import pandas as pd 
import pickle
import nltk
from nltk.stem import PorterStemmer
import re

## load the model
model=pickle.load(open('Models/logistic_regresion.pkl', 'rb'))
lb=pickle.load(open('Models/label_encoder.pkl', 'rb'))
tfidf_vectorizer=pickle.load(open('Models_1/tfidf_vectorizer.pkl', 'rb'))


# Custom function

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = model.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label =  np.max(model.predict(input_vectorized))

    return predicted_emotion,label

stopwords = set(nltk.corpus.stopwords.words('english'))
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)



# App
st.title("Emotion Detection Application")
st.write("Joy, Fear, Love, Anger, Sadness, Surprise")
input_text=st.text_input("Please write something")

if st.button("predict"):
    predicted_emotion,label=predict_emotion(input_text)
    st.write("Predicted Emotion: ",predicted_emotion)
    st.write("Predicted Level: ",label)


