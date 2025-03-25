# -*- coding: utf-8 -*-
"""Untitled5.ipynb
Original file is located at
    https://colab.research.google.com/drive/167KHckx3q-9qklfGz62Sxtk6eCnEpDJW
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
from imblearn.over_sampling import RandomOverSampler

# Load and Preprocess Data
df = pd.read_csv("/content/emotion_sentimen_dataset.csv")

# Rename column if necessary
if 'content' in df.columns:
    df.rename(columns={'content': 'text'}, inplace=True)

# Verify if 'sentiment' exists before dropping NaN values
df.dropna(subset=[col for col in ['text', 'Emotion'] if col in df.columns], inplace=True)

# Clean text
df['clean_text'] = df['text'].str.lower().str.replace(r'\W+', ' ', regex=True)

#Convert Text to TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['clean_text'])

#Handle Class Imbalance with Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_tfidf, df['Emotion'])

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

#Evaluate Model
y_pred = model.predict(X_test)
print("Model Performance:\n", classification_report(y_test, y_pred, zero_division=1))

#Save Model
joblib.dump((model, vectorizer), "emotion_model.pkl")

#Load Model & Predict on Sample Text
model, vectorizer = joblib.load("emotion_model.pkl")
sample_text = "I am feeling really happy today!"
sample_tfidf = vectorizer.transform([sample_text])
predicted_emotion = model.predict(sample_tfidf)[0]

print(f"Predicted Emotion: {predicted_emotion}")

# Predict emotion from a sample text 2
sample_text = "i love mangos "
sample_tfidf = vectorizer.transform([sample_text])
predicted_emotion = model.predict(sample_tfidf)[0]

print(f"Predicted Emotion: {predicted_emotion}")

import whisper

# Load Whisper model
whisper_model = whisper.load_model("base")

# Transcribe audio file
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Example usage
audio_text = transcribe_audio("/content/Record.mp3")
print("Transcribed Text:", audio_text)

# Predict emotion from transcribed text
sample_tfidf = vectorizer.transform([audio_text])
predicted_emotion = model.predict(sample_tfidf)[0]

print(f"Predicted Emotion: {predicted_emotion}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Example test dataset
X_test = ["I am happy today", "I feel very sad", "This is amazing", "I am angry"]

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(["happy day", "very sad", "amazing time", "angry words"])  # Dummy train data
X_test_tfidf = vectorizer.transform(X_test) 

# Dummy emotion labels for testing
y_test = ["happy", "sad", "happy", "angry"]  # Actual labels
y_pred = ["happy", "sad", "happy", "sad"] 

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["happy", "sad", "angry"])

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["happy", "sad", "angry"], yticklabels=["happy", "sad", "angry"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Emotion Detection")
plt.show()

model, vectorizer = joblib.load("emotion_model.pkl")

import ipywidgets as widgets
from IPython.display import display
import joblib

# Load the trained model and vectorizer
model, vectorizer = joblib.load("emotion_model.pkl")

# Input widget
text_input = widgets.Text(placeholder="Enter text")
output_label = widgets.Label(value="Predicted Emotion: ")

# Function to process text
def predict_emotion(change):
    text = text_input.value
    if text.strip():  
        sample_tfidf = vectorizer.transform([text]) 
        predicted_emotion = model.predict(sample_tfidf)[0] 
        output_label.value = f"Predicted Emotion: {predicted_emotion}"
    else:
        output_label.value = "Predicted Emotion: "

# Observe changes in text input
text_input.observe(predict_emotion, names="value")

# Display widgets
display(text_input, output_label)

