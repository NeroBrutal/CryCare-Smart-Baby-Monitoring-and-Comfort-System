# CryCare System

## Overview
CryCare System is a project aimed at creating a smart monitoring system using Raspberry Pi and Blynk platform. It allows users to remotely monitor and control devices such as servos and LEDs through a mobile application.

## Setup Instructions

### Blynk Account Setup
1. Create a Blynk account at [Blynk website](https://blynk.io/).
2. Create a new project and design the desired template by dragging and dropping buttons in the Blynk app.

### Raspberry Pi Setup
1. You'll need a Raspberry Pi 4B or newer version with more than 4GB of RAM.
2. Plug the servos into the GPIO pins as follows:
    - Cradle servo: GPIO 11
    - Toy servo: GPIO 13
    - LED 1: GPIO 15
    - LED 2: GPIO 22
    - LED 3: GPIO 37
3. Power the Raspberry Pi using a 12V supply.
4. Install the necessary libraries like Blynk, TensorFlow, librosa, and OpenCV on the Raspberry Pi.

### Running the System
1. Clone or download the CryCare System code repository.
2. Navigate to the directory containing the code.
3. Run the following command to start the system:
    ```
    python crycare-system-file.py
    ```
4. Once the system is running, you can control it from the Blynk app and view real-time live feed.

## Contributors
- Mohamed Razith

## License
This project is licensed under the [MIT License](LICENSE).

## Working Prototype Video
You can watch the working video of the CryCare prototype [here](https://drive.google.com/file/d/1ZKLnExg1X9YR5_bckzWbVcVprzM05TOc/view?usp=share_link).
