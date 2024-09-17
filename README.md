# Real Time Iris Tracking
Real-time iris detection refers to the process of identifying and locating irises within live or recorded video streams in real-time.
This technology involves detecting the presence of eyes, extracting iris regions, and potentially recognizing individuals based on their iris patterns, all in a continuous and immediate manner.
## Background
Iris detection involves identifying and localizing the iris region within an eye image or video frame. It utilizes computer vision techniques to detect distinct features and patterns unique to the iris.Typically, iris detection involves preprocessing steps like image segmentation and feature extraction, followed by classification or recognition algorithms.It finds applications in biometric identification systems, authentication, and medical diagnostics, leveraging the iris's uniqueness for accurate identification and analysis.
## MediaPipe
MediaPipe is a comprehensive framework developed by Google for building machine learning pipelines to process perceptual data such as audio and video. It provides pre-trained machine learning models for various tasks, allowing developers to perform complex tasks like facial landmark detection without training their own models from scratch. 

## Methodology
1. Initialization: Import required libraries and modules such as OpenCV, NumPy, Mediapipe, and Math. Define landmark indices for the eyes and iris regions.
2. Facial Landmark Detection: Utilize the Mediapipe face mesh model to detect facial landmarks in real-time from the webcam feed.
3. Iris Position Calculation: Calculate the position of the iris (left, center, or right) based on the detected landmarks and their relative distances.
4. Accuracy Evaluation: Evaluate the accuracy of iris detection by comparing the predicted iris positions with predefined ground truth values, calculating Mean Squared Error (MSE) and Root 
   Mean Squared Error (RMSE).
5. Visualization: Visualize the detected facial landmarks, iris positions, and evaluation metrics on the webcam feed in real-time using OpenCV

## Results
   ![image](https://github.com/user-attachments/assets/073cc0d6-b580-42dd-ab6c-ce0bc61ef008)


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/itskutush/Iris-Tracking-Model.git
2. Requirements:
```bash
pip install -r requirements.txt

