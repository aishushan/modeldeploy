import streamlit as st
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load the pre-trained model
model = load_model('samplemodel.h5')

# Define a Streamlit SessionState to persist the scaler
class SessionState:
    def __init__(self):
        self.scaler = None

# Initialize the session state
state = SessionState()

# Function to define and load the scaler
def load_scaler():
    scaler = StandardScaler()
    scaler.mean_ = np.load('scaler_mean.npy')
    scaler.scale_ = np.load('scaler_scale.npy')
    return scaler

# Function to extract MFCC features from an audio file
def extract_mfcc(wav_file_name, scaler):
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    
    # Normalize the features using the loaded scaler
    audio_features = np.hstack((mfccs, chroma, mel))
    audio_features = scaler.transform(audio_features.reshape(1, -1))
    
    return audio_features

# Streamlit UI
st.title('Emotion Identification from Audio')

# Upload an audio file
audio_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

if audio_file is not None:
    try:
        # Load the scaler if not already loaded
        if state.scaler is None:
            state.scaler = load_scaler()

        # Extract MFCC features
        audio_features = extract_mfcc(audio_file, state.scaler)

        # Make predictions using the trained model
        predicted_class = np.argmax(model.predict(audio_features), axis=-1)

        # Map the predicted class back to an emotion label
        emotion_labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
        predicted_emotion = emotion_labels[predicted_class[0]]

        st.success(f"Predicted emotion: {predicted_emotion}")
    except Exception as e:
        st.error(f"Error processing audio: {e}")
