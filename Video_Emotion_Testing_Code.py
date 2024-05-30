import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

# Load the trained model
model = load_model('emotion_detection_model.h5')

# Function to preprocess the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Provide the path to the input image
image_path = './archive/test/angry/PrivateTest_10131363.jpg'

# Preprocess the input image
input_image = preprocess_image(image_path)

# Make predictions
predictions = model.predict(input_image)

# Get the predicted class
predicted_class = np.argmax(predictions)

# Define the true label (ground truth) of the image
# Replace this with the actual ground truth label if available
true_label = 0

# Calculate metrics
accuracy = accuracy_score([true_label], [predicted_class])
precision = precision_score([true_label], [predicted_class], average='weighted')
f1 = f1_score([true_label], [predicted_class], average='weighted')

# Generate classification report
classification_rep = classification_report([true_label], [predicted_class], labels=[0, 1, 2, 3, 4, 5, 6])

# Print the results
print("Predicted Class:", predicted_class)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1-score:", f1)

# Print the classification report
print("\nClassification Report:")
print(classification_rep)
