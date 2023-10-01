import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
import os

# Check if the model file exists
model_file_path = 'samplemodel.h5'

if not os.path.exists(model_file_path):
    st.error(f"Model file '{model_file_path}' not found.")
else:
    # Load the trained model
    try:
        loaded_model = load_model(model_file_path)

        # Function to predict emotion
        def predict_emotion(wav_file_name):
            # Extract MFCC features from the audio file
            y, sr = librosa.load(wav_file_name)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
            mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
            features = np.hstack((mfccs, chroma, mel))

            # Make predictions using the trained model
            predicted_class = np.argmax(loaded_model.predict(features.reshape(1, -1)), axis=-1)

            # Map the predicted class back to an emotion label
            emotion_labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
            predicted_emotion = emotion_labels[predicted_class[1]]

            return predicted_emotion

        # Streamlit UI
        st.title('Emotion Recognition Web App')
        st.write('Upload an audio file to recognize the emotion.')

        # File Upload
        audio_file = st.file_uploader("Upload Audio File", type=["wav"])

        if audio_file is not None:
            st.audio(audio_file)
            predicted_emotion = predict_emotion(audio_file)
            st.write(f"Predicted emotion: {predicted_emotion}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
