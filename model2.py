import streamlit as st
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load the pre-trained model
model = load_model('samplemodel.h5')

# Function to extract MFCC features from an audio file
def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    return np.hstack((mfccs, chroma, mel))

# Streamlit UI
st.title('Emotion Identification from Audio')

# Upload an audio file
audio_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])

if audio_file is not None:
    # Load the scaler that was fitted on your training data
    scaler = StandardScaler()
    scaler.mean_ = np.load('scaler_mean.npy')
    scaler.scale_ = np.load('scaler_scale.npy')

    # Perform emotion prediction when an audio file is uploaded
    try:
        # Extract MFCC features
        audio_features = extract_mfcc(audio_file)

        # Normalize the features using the same scaler fitted on training data
        audio_features = scaler.transform(audio_features.reshape(1, -1))

        # Make predictions using the trained model
        predicted_class = np.argmax(model.predict(audio_features), axis=-1)

        # Map the predicted class back to an emotion label
        emotion_labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
        predicted_emotion = emotion_labels[predicted_class[0]]

        st.success(f"Predicted emotion: {predicted_emotion}")
    except Exception as e:
        st.error(f"Error processing audio: {e}")
