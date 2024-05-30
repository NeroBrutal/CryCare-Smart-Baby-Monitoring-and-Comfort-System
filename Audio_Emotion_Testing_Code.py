import numpy as np
import librosa
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Function to extract features from a single audio file
def extract_features_single(file_path, mfcc=True, chroma=True, mel=True):
    try:
        X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

    features = []
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
        features.append(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        features.append(mel)
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        features.append(mfccs)
    return np.concatenate(features) if len(features) > 0 else None

# Load the saved model
saved_model_path = 'baby_cry_model_new_one.h5'
loaded_model = load_model(saved_model_path)

# Function to predict the class of a single audio file
def predict_single_audio(file_path, model, label_encoder):
    feature = extract_features_single(file_path)
    if feature is None:
        print("Error extracting features from the audio file.")
        return None, None

    # Reshape feature for prediction
    feature = np.expand_dims(feature, axis=0)

    # Predict class probabilities
    predicted_probabilities = model.predict(feature)

    # Get the predicted class index
    predicted_class_index = np.argmax(predicted_probabilities)
    
    # Decode the predicted class
    predicted_class = label_encoder.classes_[predicted_class_index]
    
    return predicted_class, predicted_probabilities[0]

# Provide the path to the audio file you want to predict
audio_file_path = './donateacry_corpus_cleaned_and_updated_data/hungry/02c3b725-26e4-4a2c-9336-04ddc58836d9-1430726196216-1.7-m-04-hu.wav'

# Make predictions for the single audio file
predicted_class, predicted_probabilities = predict_single_audio(audio_file_path, loaded_model, label_encoder)

if predicted_class is not None:
    print("Predicted Class:", predicted_class)

    # Calculate accuracy, precision, and F1-score
    accuracy = accuracy_score([np.argmax(y_test)], [np.argmax(predicted_probabilities)])
    precision = precision_score([np.argmax(y_test)], [np.argmax(predicted_probabilities)], average='weighted')
    f1 = f1_score([np.argmax(y_test)], [np.argmax(predicted_probabilities)], average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("F1-score:", f1)
else:
    print("Prediction failed.")

