import librosa
import numpy as np
import torch
import tensorflow as tf
from transformers import Wav2Vec2Processor, Wav2Vec2Model, AutoFeatureExtractor, WavLMForXVector
import os

#padding the smaller duration audios as done in training
def pad_audio_to_minimum_length(audio, target_length=16000):
    if len(audio) < target_length:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
    return audio


# Loading models
wave2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
wavlm_model = WavLMForXVector.from_pretrained("microsoft/wavlm-large")
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
siamese_model = tf.keras.models.load_model("C://Users//Abdul Kavi Chaudhary//feature_extractor_model.h5", compile=False)
ensemble_classifier_model = tf.keras.models.load_model('C://Users//Abdul Kavi Chaudhary//ensemble_classifier.keras')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move models to GPU
wave2vec_model.to(device)
wavlm_model.to(device)

#extract mel spectrogram
def mel_spectrogram_gen(audio):
    signal, sample_rate = librosa.load(audio, sr=22050, duration=2)
    hop_length = 512
    n_mels = 128
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return S_dB

#pad mel spectrograms
def pad_mel_spectrograms(mel_spectrogram, max_pad_len=87):
    pad_width = max_pad_len - mel_spectrogram.shape[1]
    if pad_width > 0:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_pad_len]
    return mel_spectrogram

# Function to extract Wav2Vec2 features
def extract_wave2vec_features(audio_file):
    audio_input, _ = librosa.load(audio_file, sr=16000)
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to device
    with torch.no_grad():
        outputs = wave2vec_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Function to extract WavLM features
def extract_xvector_features(audio_file):
    audio_input, _ = librosa.load(audio_file, sr=16000)
    inputs = feature_extractor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to device
    with torch.no_grad():
        outputs = wavlm_model(**inputs).embeddings
    return outputs.mean(dim=1).squeeze().cpu().numpy()

# Function to normalize data
def normalize_data(data):
    return (data - data.min()) / (data.max() - data.min())

def get_audio_duration(audio_file):
    signal, sr = librosa.load(audio_file, sr=None)
    duration = len(signal) / sr
    return duration

# Main inference function
def predict_audio(audio_file):
    duration = get_audio_duration(audio_file)
    # Mel-spectrogram extraction and padding
    mel_spectrogram = mel_spectrogram_gen(audio_file)
    max_pad_len = 87  # Ensure this matches the value used in training
    mel_spectrogram_padded = pad_mel_spectrograms(mel_spectrogram, max_pad_len)
    mel_spectrogram_padded = normalize_data(mel_spectrogram_padded)
    
    # Adjust shape to match model input
    mel_spectrogram_padded = np.expand_dims(mel_spectrogram_padded, axis=-1)  # Shape: (128, 87, 1)
    mel_spectrogram_padded = np.expand_dims(mel_spectrogram_padded, axis=0)   # Shape: (1, 128, 87, 1)

    # Feature extraction
    wave2vec_features = extract_wave2vec_features(audio_file)
    xvector_features = extract_xvector_features(audio_file)
    siamese_features = siamese_model.predict(mel_spectrogram_padded)

    # Reshape features
    wave2vec_features = wave2vec_features.reshape((1, -1))
    siamese_features = siamese_features.reshape((1, -1))
    xvector_features = xvector_features.reshape((1, -1))

    # Concatenate features for ensemble classifier
    combined_features = np.concatenate([wave2vec_features, siamese_features, xvector_features], axis=1)

    # Predict using ensemble classifier
    prediction = ensemble_classifier_model.predict(combined_features)

    # Convert prediction to percentage
    percentage = prediction[0][0] * 100  # Get the first value as a percentage
    print(f"Prediction probability: {percentage:.2f}% fake")

    # Output result
    if prediction[0][0] >= 0.5:  # Using the probability threshold
        return ("Fake Audio",duration)
    else:
        return ("Real Audio",duration)
    


# Example usage
audio_file = "C:\\Users\\Abdul Kavi Chaudhary\\Desktop\\Test\\fake\\11636.wav"
result = predict_audio(audio_file)
print(result)
