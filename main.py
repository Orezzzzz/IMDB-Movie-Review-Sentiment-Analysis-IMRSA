import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

model = load_model('LSTM_imdb.keras')

#func to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,"?") for i in encoded_review])

#func to pre-process user input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    encoded_review = [word_index.get(word,2) +3 for word in words]
    padded_review = pad_sequences([encoded_review],maxlen = 500)
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    score = prediction[0][0]

    if score > 0.6:
        sentiment = 'Positive'

    elif score < 0.4:
        sentiment = 'Negative'

    else:
        sentiment = 'Neutral'
    return sentiment,score
    return sentiment,score


import streamlit as st
##streamlit app
st.title('IMDB Movie Review Sentiment Analysis(IMRSA).')
st.write('Enter a movie review for Interstellar to classify it as Positive, Neutral or Negative.')
user_input = st.text_area('Movie Review')
if st.button('Classify'):

    if user_input:

        preprocessed_input = preprocess_text(user_input)

        # prediction
        prediction = model.predict(preprocessed_input)

        score = prediction[0][0]

        if score > 0.6:
            sentiment = 'Positive'

        elif score < 0.4:
            sentiment = 'Negative'

        else:
            sentiment = 'Neutral'

        st.write(f'Review : {user_input}')
        st.write(f'Sentiment : {sentiment}')
        st.write(f'Prediction score : {score}')

    else:
        st.write('Please enter a movie review')