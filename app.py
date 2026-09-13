import atexit
import pickle
import threading
import time
import os
import queue
import subprocess
import base64
import re
from collections import Counter, deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from flask import Flask, Response, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ============================================================
# MODEL LOADING
# ============================================================
GESTURE_MODEL_PATH = "models/gesture_model.pkl"
EMOTION_MODEL_PATH = "models/emotion_model.hdf5"

gesture_model = pickle.load(open(GESTURE_MODEL_PATH, "rb"))
#emotion_model = load_model(EMOTION_MODEL_PATH)
emotion_model = load_model(
    EMOTION_MODEL_PATH,
    compile=False
)

emotion_labels = [
    "Angry", "Disgust", "Fear", "Happy",
    "Sad", "Surprise", "Neutral"
]

# Only these are presented as supported gestures.
SUPPORTED_GESTURES = {
    "HELLO",
    "HELP",
    "NO",
    "THANK YOU",
    "YES",
    "UNKNOWN",
}

# ============================================================
# FLASK / APPLICATION STATE
# ============================================================
state_lock = threading.Lock()

state = {
    "gesture": "No gesture",
    "confidence": 0.0,
    "emotion": "Neutral",
    "response": "Waiting for gesture...",
    "speaking": False,
    "time": "--:--:--",
}

prediction_history = deque(maxlen=8)
last_spoken = ""
last_response_time = 0.0

# Prevent several TTS threads from using pyttsx3 simultaneously.
tts_lock = threading.Lock()
#engine = pyttsx3.init()
#engine.setProperty("rate", 145)
#engine.setProperty("volume", 1.0)

# Prefer a clear English voice when Windows exposes one.
try:
    voices = engine.getProperty("voices")
    preferred = None
    for voice in voices:
        info = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
        if any(x in info for x in ["zira", "david", "english", "en-us", "en_us"]):
            preferred = voice.id
            break
    if preferred:
        engine.setProperty("voice", preferred)
except Exception as e:
    print("TTS voice selection warning:", repr(e))

# =========================
# POLISHED WINDOWS TTS
# =========================

tts_queue = queue.Queue(maxsize=1)
tts_lock = threading.Lock()

TTS_VOICE = "Microsoft David Desktop - English (United States)"
TTS_RATE = -1
TTS_VOLUME = 100


def clean_speech(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"!{2,}", "!", text)
    return text


def speak_worker():
    while True:
        text = tts_queue.get()

        if text is None:
            tts_queue.task_done()
            break

        try:
            text = clean_speech(text)

            if not text:
                tts_queue.task_done()
                continue

            with state_lock:
                state["speaking"] = True

            # Encode speech text safely.
            # This prevents apostrophes/special characters
            # from breaking the PowerShell command.
            encoded_text = base64.b64encode(
                text.encode("utf-8")
            ).decode("ascii")

            ps_script = f'''
Add-Type -AssemblyName System.Speech

$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer

$speaker.Rate = {TTS_RATE}
$speaker.Volume = {TTS_VOLUME}

$textBytes = [Convert]::FromBase64String("{encoded_text}")
$text = [System.Text.Encoding]::UTF8.GetString($textBytes)

try {{
    $speaker.SelectVoice("{TTS_VOICE}")
}} catch {{
    # Use default Windows voice
}}

$speaker.SetOutputToDefaultAudioDevice()
$speaker.Speak($text)
$speaker.Dispose()
'''

            encoded_script = base64.b64encode(
                ps_script.encode("utf-16le")
            ).decode("ascii")

            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-EncodedCommand",
                    encoded_script
                ],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            if result.returncode != 0:
                print("[TTS ERROR]", result.stderr.strip())
            else:
                print("[TTS] Spoken:", text)

        except Exception as e:
            print("[TTS ERROR]", repr(e))

        finally:
            with state_lock:
                state["speaking"] = False

            tts_queue.task_done()


# Start ONE permanent TTS worker
tts_thread = threading.Thread(
    target=speak_worker,
    daemon=True
)

tts_thread.start()


def speak(text):
    text = clean_speech(text)

    if not text:
        return

    # Remove an old waiting message.
    try:
        while True:
            tts_queue.get_nowait()
            tts_queue.task_done()
    except queue.Empty:
        pass

    # Add latest speech
    try:
        tts_queue.put_nowait(text)
    except queue.Full:
        pass
# ============================================================
# GESTURE LABEL CLEANING
# ============================================================
def normalize_gesture(raw):
    """Convert model labels into clean application labels."""
    raw = str(raw).strip()

    if raw in {"THANK YOU", "THANKYOU"}:
        return "THANK YOU"

    if raw in {"HELLO", "HELP", "NO", "YES", "UNKNOWN"}:
        return raw

    # The supplied model contains accidental/non-gesture labels.
    return "UNKNOWN"


# ============================================================
# MEDIAPIPE HAND SETUP
# ============================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ============================================================
# FACE / EMOTION SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

cascade_candidates = [
    os.path.join(cv2.data.haarcascades, CASCADE_FILENAME),
    os.path.join(BASE_DIR, "models", CASCADE_FILENAME),
    os.path.join(BASE_DIR, "assets", CASCADE_FILENAME),
]

face_cascade = None
cascade_path = None

for candidate in cascade_candidates:
    if os.path.isfile(candidate):
        detector = cv2.CascadeClassifier(candidate)
        if not detector.empty():
            face_cascade = detector
            cascade_path = candidate
            break

if face_cascade is None:
    print("WARNING: Haar face cascade XML was not found.")
    print("Emotion detection will return Neutral until the XML is installed.")
    for candidate in cascade_candidates:
        print("Checked:", candidate)
else:
    print("SUCCESS: Face detector loaded.")
    print("Cascade path:", cascade_path)


def detect_emotion(frame):

    # If the face detector failed to load,
    # don't crash the complete gesture recognition system.
    if face_cascade is None or face_cascade.empty():
        return "Neutral"

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            roi = gray[y:y + h, x:x + w]

            if roi.size == 0:
                continue

            roi = cv2.resize(roi, (64, 64))

            roi = roi.astype("float32") / 255.0

            roi = np.reshape(
                roi,
                (1, 64, 64, 1)
            )

            try:
                preds = emotion_model.predict(
                    roi,
                    verbose=0
                )

                emotion_index = int(np.argmax(preds))

                return emotion_labels[emotion_index]

            except Exception as e:
                print("Emotion model error:", repr(e))
                return "Neutral"

    except Exception as e:
        print("Face detection error:", repr(e))
        return "Neutral"

    return "Neutral"

# ============================================================
# CONTEXT-AWARE RESPONSE
# ============================================================
def generate_response(gesture, emotion):
    hour = datetime.now().hour

    if gesture == "HELLO":
        if hour < 12:
            base = "Good Morning"
        elif hour < 18:
            base = "Good Afternoon"
        else:
            base = "Good Evening"

    elif gesture == "HELP":
        base = "Please help me"

    elif gesture == "YES":
        base = "Yes"

    elif gesture == "NO":
        base = "No"

    elif gesture == "THANK YOU":
        base = "Thank you"

    else:
        base = "Unknown gesture"

    # Emotion-aware variation.
    if emotion == "Happy":
        return base + ". You look happy!"
    elif emotion == "Sad":
        return base + ". I hope you feel better."
    elif emotion == "Angry":
        return base + ". Please stay calm."
    elif emotion == "Surprise":
        return base + ". You seem surprised!"
    elif emotion == "Fear":
        return base + ". You seem worried."

    return base


# ============================================================
# DRAW INFORMATION ON CAMERA FRAME
# ============================================================
def draw_overlay(frame, gesture, confidence, emotion, response):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 150), (20, 20, 35), -1)

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (15, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Emotion: {emotion}",
        (15, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 200, 0),
        2,
    )

    cv2.putText(
        frame,
        response[:70],
        (15, 133),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    


# ============================================================
# PROCESS ONE FRAME
# ============================================================
def process_frame(frame):
    global last_spoken, last_response_time

    emotion = detect_emotion(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No gesture"
    confidence = 0.0
    response = "Show a supported hand gesture"

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )

        # Same landmark normalization approach as the original program.
        base_point = landmarks[0]
        landmarks = landmarks - base_point
        row = landmarks.flatten().tolist()

        try:
            probabilities = gesture_model.predict_proba([row])[0]
            max_probability = float(np.max(probabilities))
        except Exception:
            max_probability = 0.0

        confidence = max_probability

        if max_probability >= 0.75:
            try:
                raw_prediction = gesture_model.predict([row])[0]
                prediction = normalize_gesture(raw_prediction)
            except Exception:
                prediction = "UNKNOWN"

            # Ignore accidental model labels by mapping them to UNKNOWN.
            prediction_history.append(prediction)

            if len(prediction_history) == prediction_history.maxlen:
                final_prediction = Counter(prediction_history).most_common(1)[0][0]

                gesture = final_prediction
                response = generate_response(final_prediction, emotion)

                now = time.time()

                # Do not speak the same response repeatedly.
                # A short cooldown also protects against rapid repeated predictions.
                if (
                    response != last_spoken
                    and now - last_response_time > 1.5
                    and final_prediction != "UNKNOWN"
                ):
                    last_spoken = response
                    last_response_time = now
                    speak(response)
        else:
            prediction_history.clear()
    else:
        prediction_history.clear()

    # Draw MediaPipe landmarks after processing.
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

    draw_overlay(
        frame,
        gesture,
        confidence,
        emotion,
        response,
    )

    with state_lock:
        state["gesture"] = gesture
        state["confidence"] = round(confidence, 4)
        state["emotion"] = emotion
        state["response"] = response
        state["time"] = datetime.now().strftime("%H:%M:%S")

    return frame


# ============================================================
# CAMERA
# ============================================================

camera_lock = threading.Lock()

def open_camera():
    """
    Open the Windows webcam using DirectShow.
    """
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Camera could not be opened.")
        return None

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("SUCCESS: Camera opened successfully.")

    return camera


cap = open_camera()


def generate_frames():
    global cap

    if cap is None:
        print("ERROR: No camera available.")

        # Send a simple error image instead of crashing Flask
        error_frame = np.zeros((480, 800, 3), dtype=np.uint8)

        cv2.putText(
            error_frame,
            "Camera could not be opened",
            (80, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

        cv2.putText(
            error_frame,
            "Check camera permissions or another app using the camera",
            (30, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        success, buffer = cv2.imencode(".jpg", error_frame)

        if success:
            frame_bytes = buffer.tobytes()

            while True:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
                time.sleep(1)

        return

    while True:

        try:
            with camera_lock:
                success, frame = cap.read()

            if not success:
                print("WARNING: Could not read camera frame.")
                time.sleep(0.1)
                continue

            # Process gesture + emotion
            try:
                frame = process_frame(frame)
            except Exception as e:
                print("PROCESS FRAME ERROR:", repr(e))

                # Keep displaying camera even if AI processing fails
                cv2.putText(
                    frame,
                    "AI processing error",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

            # Convert frame to JPEG
            success, buffer = cv2.imencode(".jpg", frame)

            if not success:
                print("WARNING: JPEG encoding failed.")
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        except GeneratorExit:
            print("Video stream closed.")
            break

        except Exception as e:
            print("CAMERA STREAM ERROR:", repr(e))
            time.sleep(0.2)

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/gestures")
def gestures():
    return render_template("gestures.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/emotions")
def emotions():
    return render_template("emotions.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(dict(state))


@app.route("/api/gestures")
def api_gestures():
    return jsonify(sorted(SUPPORTED_GESTURES))


# ============================================================
# CLEANUP
# ============================================================
def cleanup():
    global cap

    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass

    try:
        hands.close()
    except Exception:
        pass

atexit.register(cleanup)


# ============================================================
# START FLASK
# ============================================================
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
