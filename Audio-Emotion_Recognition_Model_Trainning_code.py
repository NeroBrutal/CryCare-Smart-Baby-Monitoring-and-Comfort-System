import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Function to extract features from audio files
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
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

def load_data(data_dir):
    X, y = [], []
    counts = {}  # Dictionary to store counts for each emotion folder
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue  # Skip if not a directory
        counts[folder] = 0  # Initialize count for this folder
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if not os.path.isfile(file_path) or not file.endswith('.wav'):
                continue  # Skip if not a .wav file
            feature = extract_features(file_path)
            if feature is not None:
                X.append(feature)
                y.append(folder)  # Assuming folder name is the label
                counts[folder] += 1  # Increment count for this folder
    for folder, count in counts.items():
        print(f"Number of audio files in '{folder}' folder: {count}")
    return np.array(X), np.array(y)

# Define the model architecture
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# Load data
data_dir = './donateacry_corpus_cleaned_and_updated_data'
X, y = load_data(data_dir)

# Encode labelslabel_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)


# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=num_classes)
print(f"Number of audio files used: {len(X)}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
input_shape = X_train.shape[1:]
model = create_model(input_shape, num_classes)
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate classification report
classification_rep = classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)

# Print the classification report
print("\nClassification Report:")
print(classification_rep)

# Save the model
model.save('baby_cry_model_new_one.h5')
