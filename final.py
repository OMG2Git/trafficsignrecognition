from roboflow import Roboflow
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
import queue
import time

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="YOUR_API_KEY")
detection_model = rf.workspace("YOUR_WORKSPACE_NAME").project("tabela_v1.2").version(12).model

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()
speaking = threading.Event()

def text_to_speech_worker():
    while True:
        text = tts_queue.get()
        if text is None:  # Exit condition
            break
        speaking.set()  # Mark speaking as active
        tts_engine.say(text)
        tts_engine.runAndWait()
        speaking.clear()  # Mark speaking as done
        tts_queue.task_done()

# Start the text-to-speech thread
tts_queue = queue.Queue()
tts_thread = threading.Thread(target=text_to_speech_worker, daemon=True)
tts_thread.start()

# Function to preprocess the frame (resize and normalize)
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized / 255.0
    return frame_normalized

# Load the classification model
classification_model = tf.keras.models.load_model("finetuned_train_v2.keras")
class_labels = {
    0: "30",
    1: "50",
    2: "60",
    3: "70",
    4: "80",
    5: "right"
}

# Connect to DroidCam via USB
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not connect to DroidCam.")
    exit()

# Set resolution
frame_width = 526
frame_height = 526
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

latest_prediction = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Resize frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Perform detection
    predictions = detection_model.predict(frame, confidence=50, overlap=30).json()
    predictions['predictions'] = [p for p in predictions['predictions'] if p['confidence'] > 0.80]

    for pred in predictions['predictions']:
        class_name = pred['class']
        confidence = pred['confidence']
        x1, y1, width_pred, height_pred = pred['x'], pred['y'], pred['width'], pred['height']

        if x1 <= 1 and width_pred <= 1:
            x1, width_pred = x1 * frame_width, width_pred * frame_width
            y1, height_pred = y1 * frame_height, height_pred * frame_height

        cx, cy, w, h = x1, y1, width_pred, height_pred
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)

        cropped_img = frame[y1:y2, x1:x2]

        if cropped_img.size == 0:
            print(f"Error: Empty cropped image for bounding box ({x1}, {y1}), ({x2}, {y2})")
            continue

        # Preprocess the cropped image for classification
        cropped_img_resized = cv2.resize(cropped_img, (64, 64))  # Resize to match input size (64x64)
        cropped_img_resized = cv2.cvtColor(cropped_img_resized, cv2.COLOR_BGR2RGB)  # Convert to RGB
        cropped_img_resized = np.expand_dims(cropped_img_resized, axis=0)  # Add batch dimension
        cropped_img_resized = cropped_img_resized / 255.0  # Normalize the image

        predictions = classification_model.predict(cropped_img_resized)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_label = class_labels[predicted_class]

        # Update latest prediction only if TTS is not speaking
        if not speaking.is_set():
            latest_prediction = f"Predicted traffic sign is {predicted_label}. Confidence is {confidence * 100:.2f} percent."
            tts_queue.put(latest_prediction)

    # Display video feed
    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Stop the TTS thread
tts_queue.put(None)  # Signal the thread to exit
tts_thread.join()
