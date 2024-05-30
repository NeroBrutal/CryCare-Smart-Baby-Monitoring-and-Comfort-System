import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import RPi.GPIO as GPIO # type: ignore
import time
import BlynkLib
import pygame
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Initialize Blynk
BLYNK_AUTH_TOKEN = "TAbOBe5FOmmLmIhThdY2xSyASlNhygIj"
blynk = BlynkLib.Blynk(BLYNK_AUTH_TOKEN)

# Function to send emotion to Blynk virtual pin 4
def send_emotion_to_blynk(emotion):
    blynk.virtual_write(4, f"Your baby is {emotion}")

# Initialize Pygame for audio playback
pygame.init()

# Define the pin for controlling cradle servo movement and LED
CRADLE_SERVO_CONTROL_PIN = 0
LED_CRADLE_PIN = 13
# Define the pin for controlling toy servo movement and LED
TOY_SERVO_CONTROL_PIN = 2
LED_TOY_PIN = 15
# Define the pin for controlling music playback and LED
MUSIC_CONTROL_PIN = 1
LED_MUSIC_PIN = 40

# Define the audio file path
AUDIO_FILE_PATH = './Audio.mp3'

# Set GPIO mode and pin for cradle servo
GPIO.setmode(GPIO.BOARD)
cradle_servo_pin = 11
GPIO.setup(cradle_servo_pin, GPIO.OUT)
cradle_pwm = GPIO.PWM(cradle_servo_pin, 50)  # PWM with frequency 50Hz
cradle_pwm.start(0)

# Set GPIO mode and pin for toy servo
toy_servo_pin = 12
GPIO.setup(toy_servo_pin, GPIO.OUT)
toy_pwm = GPIO.PWM(toy_servo_pin, 50)  # PWM with frequency 50Hz
toy_pwm.start(0)

# Set GPIO mode and pin for LED connected to cradle servo
GPIO.setup(LED_CRADLE_PIN, GPIO.OUT)

# Set GPIO mode and pin for LED connected to toy servo
GPIO.setup(LED_TOY_PIN, GPIO.OUT)

# Set GPIO mode and pin for LED connected to music control
GPIO.setup(LED_MUSIC_PIN, GPIO.OUT)

# Flag variable to indicate cradle servo movement interruption
cradle_servo_interrupt_flag = True

# Function to set cradle servo angle
def set_cradle_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(cradle_servo_pin, True)
    cradle_pwm.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(cradle_servo_pin, False)
    cradle_pwm.ChangeDutyCycle(0)

# Flag variable to indicate toy servo movement interruption
toy_servo_interrupt_flag = True

# Function to set toy servo angle
def set_toy_angle(angle):
    duty = angle / 18 + 2
    GPIO.output(toy_servo_pin, True)
    toy_pwm.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(toy_servo_pin, False)
    toy_pwm.ChangeDutyCycle(0)

# Function to control the LED connected to the cradle servo
def control_led_cradle(state):
    GPIO.output(LED_CRADLE_PIN, state)

# Function to control the LED connected to the toy servo
def control_led_toy(state):
    GPIO.output(LED_TOY_PIN, state)

# Function to control the LED connected to the music control
def control_led_music(state):
    GPIO.output(LED_MUSIC_PIN, state)

# Function to play the audio
def play_audio():
    pygame.mixer.music.load(AUDIO_FILE_PATH)
    pygame.mixer.music.play()

# Function to stop the audio
def stop_audio():
    pygame.mixer.music.stop()

# Function to run cradle servo movement for 30 seconds or until interrupted
def run_cradle_servo():
    global cradle_servo_interrupt_flag
    duration = 30  # seconds
    start_time = time.time()
    cradle_servo_interrupt_flag = False  # Allow cradle servo movement
    while time.time() - start_time < duration and not cradle_servo_interrupt_flag:
        # Move cradle servo to one extreme position
        set_cradle_angle(0)
        time.sleep(0.1)  # Adjust delay as needed
        # Move cradle servo to the other extreme position
        set_cradle_angle(90)
        time.sleep(0.1)  # Adjust delay as needed
    cradle_servo_interrupt_flag = True  # Reset the interrupt flag

# Function to run toy servo movement for 30 seconds or until interrupted
def run_toy_servo():
    global toy_servo_interrupt_flag
    duration = 30  # seconds
    start_time = time.time()
    toy_servo_interrupt_flag = False  # Allow toy servo movement
    while time.time() - start_time < duration and not toy_servo_interrupt_flag:
        # Move toy servo to one extreme position
        set_toy_angle(0)
        time.sleep(0.1)  # Adjust delay as needed
        # Move toy servo to the other extreme position
        set_toy_angle(90)
        time.sleep(0.1)  # Adjust delay as needed
    toy_servo_interrupt_flag = True  # Reset the interrupt flag

# Blynk handler for cradle servo control
def cradle_servo_control(value):
    global cradle_servo_interrupt_flag
    button_state = int(value[0])
    if button_state == 1:  # If button is pressed
        if cradle_servo_interrupt_flag:
            # Start a new thread to run the cradle servo movement
            threading.Thread(target=run_cradle_servo).start()
            # Turn on LED connected to cradle servo
            control_led_cradle(GPIO.HIGH)
    else:  # If button is released
        cradle_servo_interrupt_flag = True  # Stop cradle servo movement
        # Turn off LED connected to cradle servo
        control_led_cradle(GPIO.LOW)

# Blynk handler for toy servo control
def toy_servo_control(value):
    global toy_servo_interrupt_flag
    button_state = int(value[0])
    if button_state == 1:  # If button is pressed
        if toy_servo_interrupt_flag:
            # Start a new thread to run the toy servo movement
            threading.Thread(target=run_toy_servo).start()
            # Turn on LED connected to toy servo
            control_led_toy(GPIO.HIGH)
    else:  # If button is released
        toy_servo_interrupt_flag = True  # Stop toy servo movement
        # Turn off LED connected to toy servo
        control_led_toy(GPIO.LOW)

# Blynk handler for music control
def music_control(value):
    global music_playing
    button_state = int(value[0])
    if button_state == 1:  # If button is pressed
        play_audio()
        control_led_music(GPIO.HIGH)  # Turn on LED
    else:  # If button is released
        stop_audio()
        control_led_music(GPIO.LOW)  # Turn off LED

# Register virtual write handlers
blynk.on("V{}".format(CRADLE_SERVO_CONTROL_PIN), cradle_servo_control)
blynk.on("V{}".format(TOY_SERVO_CONTROL_PIN), toy_servo_control)
blynk.on("V{}".format(MUSIC_CONTROL_PIN), music_control)

# Load the pre-trained emotion detection model
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'emotion_detection_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Function to extract features from audio files
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate).T, axis=0)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    return np.concatenate((mfccs, chroma, mel))

# Load the pre-trained model
model = load_model('./baby_cry_model_updated.h5')

# Function to predict emotion from audio file
def predict_emotion(audio_file):
    # Extract features from audio file
    features = extract_features(audio_file)
    # Reshape features to match model input shape
    features = np.expand_dims(features, axis=0)
    # Predict emotion label
    prediction = model.predict(features)
    # Get the predicted label index
    predicted_index = np.argmax(prediction)
    # Map index to emotion label
    emotion_labels = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
    predicted_emotion = emotion_labels[predicted_index]
    return predicted_emotion

# MJPEG server class
class MJPEGServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Unable to capture frame.")
                        break

                    # Detect emotions
                    labels = []
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_classifier.detectMultiScale(gray)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                        # Convert the grayscale image to RGB format
                        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)

                        if np.sum([roi_rgb]) != 0:
                            roi = roi_rgb.astype('float') / 255.0
                            roi = img_to_array(roi)
                            roi = np.expand_dims(roi, axis=0)

                            prediction = classifier.predict(roi)[0]
                            label = emotion_labels[prediction.argmax()]
                            label_position = (x, y)
                            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                            # Send emotion to Blynk
                            send_emotion_to_blynk(label)

                            # Check for continuous sad emotion
                            check_sad_emotion(label)

                    # Encode the frame as JPEG
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    frame_bytes = jpeg.tobytes()

                    # Send the frame as part of the MJPEG stream
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')

                    # Add a small delay to control the frame rate
                    time.sleep(0.05)
            except Exception as e:
                print("Error:", str(e))

# Start the MJPEG server in a separate thread
mjpeg_thread = threading.Thread(target=HTTPServer(('', 8000), MJPEGServer).serve_forever)
mjpeg_thread.start()
print("Tha Audio Model is recording")

# Initialize global variables for sad emotion detection
sad_count = 0
sad_start_time = None
emotion_triggered = False

# Function to check if the detected emotion is sad continuously for a specified duration
def check_sad_emotion(emotion):
    global sad_count, sad_start_time, emotion_triggered

    if emotion == 'Sad':
        if sad_count == 0:
            sad_start_time = time.time()
        sad_count += 1
        if sad_count >= 10:
            # Trigger music, cradle servo, and toy servo
            if not emotion_triggered:
                play_audio()
                threading.Thread(target=run_cradle_servo).start()
                threading.Thread(target=run_toy_servo).start()
                # Turn on LEDs
                control_led_cradle(GPIO.HIGH)
                control_led_toy(GPIO.HIGH)
                control_led_music(GPIO.HIGH)
                emotion_triggered = True
                # Update Blynk button status
                blynk.virtual_write(MUSIC_CONTROL_PIN, 1)  # Update music button state
                blynk.virtual_write(CRADLE_SERVO_CONTROL_PIN, 1)  # Update cradle servo button state
                blynk.virtual_write(TOY_SERVO_CONTROL_PIN, 1)  # Update toy servo button state
    else:
        sad_count = 0
        if emotion_triggered:
            stop_audio()
            # Turn off LEDs
            control_led_cradle(GPIO.LOW)
            control_led_toy(GPIO.LOW)
            control_led_music(GPIO.LOW)
            emotion_triggered = False
            # Update Blynk button status
            blynk.virtual_write(MUSIC_CONTROL_PIN, 0)  # Update music button state
            blynk.virtual_write(CRADLE_SERVO_CONTROL_PIN, 0)  # Update cradle servo button state
            blynk.virtual_write(TOY_SERVO_CONTROL_PIN, 0)  # Update toy servo button state


# Keep Blynk running
while True:
    blynk.run()
    

