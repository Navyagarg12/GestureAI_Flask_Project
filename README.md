# Gesture AI - Complete Flask Project

## Project
Context-Aware Gesture-to-Speech Assistive System with:
- Hand gesture recognition
- MediaPipe hand landmarks
- Machine-learning gesture classifier
- Facial emotion recognition
- Time-aware/context-aware responses
- Text-to-speech
- Flask web frontend
- Live browser camera stream

## Project structure

GestureAI_Flask_Project_Complete/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── gesture_model.pkl
│   └── emotion_model.hdf5   <-- ADD YOUR EMOTION MODEL HERE
│
├── templates/
│   ├── index.html
│   ├── live.html
│   ├── gestures.html
│   ├── how_it_works.html
│   ├── emotions.html
│   └── about.html
│
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── live.js

## Important

The supplied gesture model contains some accidental labels from data collection.
`app.py` normalizes the intended labels and maps unsupported labels to `UNKNOWN`.
For best accuracy, retraining the gesture model with clean labels is recommended.

## Setup

1. Open this project folder in PowerShell.
2. Create a virtual environment:

   python -m venv venv

3. Activate it:

   .\venv\Scripts\activate

4. Install dependencies:

   pip install -r requirements.txt

5. Put your emotion model here:

   models\emotion_model.hdf5

6. Run:

   python app.py

7. Open:

   http://127.0.0.1:5000

## Flask integration

The original desktop OpenCV loop has been converted to a Flask streaming architecture.
There is no `cv2.imshow()` window and no `cv2.waitKey()` loop.

The browser receives:
GET /video_feed

The frontend reads live recognition data from:
GET /api/status

TTS still runs on the computer where Flask is running.
