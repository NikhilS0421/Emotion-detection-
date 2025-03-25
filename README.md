# Emotion Detection using NLP and Machine Learning

## Project Overview
This project focuses on detecting emotions from text using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The model is trained on a labeled dataset and can predict emotions from text input, including transcriptions from audio files.

## Features
- **Preprocessing & Data Handling**: Cleans and transforms text data using TF-IDF vectorization.
- **Machine Learning Model**: Implements a Multinomial Naive Bayes classifier to predict emotions.
- **Class Imbalance Handling**: Uses oversampling to balance the dataset.
- **Audio Transcription Integration**: Utilizes OpenAI's Whisper model to transcribe audio and predict emotions.
- **Interactive Interface**: Provides an interactive text input widget and a Gradio-based web interface.

## Dependencies
The project requires the following libraries:
```bash
pip install pandas numpy scikit-learn imbalanced-learn joblib whisper seaborn matplotlib gradio
```

## Dataset
The model is trained on `emotion_sentimen_dataset.csv`. Ensure that this dataset is available in the specified location.

## Model Training
1. Load and preprocess the dataset.
2. Convert text data into TF-IDF features.
3. Handle class imbalance using RandomOverSampler.
4. Split the data into training and testing sets.
5. Train a Multinomial Naive Bayes classifier.
6. Evaluate the model using classification metrics.
7. Save the trained model and vectorizer for later use.

## Running the Model
To predict emotion from a text input:
```python
import joblib

# Load model
model, vectorizer = joblib.load("emotion_model.pkl")

# Predict emotion
sample_text = "I am feeling really happy today!"
sample_tfidf = vectorizer.transform([sample_text])
predicted_emotion = model.predict(sample_tfidf)[0]
print(f"Predicted Emotion: {predicted_emotion}")
```

## Audio Transcription & Emotion Detection
The project includes Whisper for speech-to-text conversion. To transcribe an audio file and predict its emotion:
```python
import whisper

# Load Whisper model
whisper_model = whisper.load_model("base")

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Predict emotion from transcribed text
audio_text = transcribe_audio("/path/to/audio.mp3")
sample_tfidf = vectorizer.transform([audio_text])
predicted_emotion = model.predict(sample_tfidf)[0]
print(f"Predicted Emotion: {predicted_emotion}")
```

## Interactive Interface
Run the Gradio interface to interact with the model:
```python
import gradio as gr

model, vectorizer = joblib.load("emotion_model.pkl")

def predict_emotion(text):
    if text.strip():
        sample_tfidf = vectorizer.transform([text])
        return f"Predicted Emotion: {model.predict(sample_tfidf)[0]}"
    return "Please enter valid text."

demo = gr.Interface(fn=predict_emotion, inputs="text", outputs="text", title="Emotion Detection", description="Enter a sentence to predict its emotion.")
demo.launch(share=True, debug=True)
```

## Visualization
A confusion matrix is generated to evaluate model performance:
```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["happy", "sad", "angry"])

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["happy", "sad", "angry"], yticklabels=["happy", "sad", "angry"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Emotion Detection")
plt.show()
```

## Author
This project was developed using Google Colab by Nikhil


