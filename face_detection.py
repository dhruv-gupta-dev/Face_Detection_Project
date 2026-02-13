"""
Real-Time Face Detection using OpenCV DNN (Caffe SSD ResNet-10)

Features:
- Webcam face detection
- Confidence threshold filtering (>0.6)
- Bounding box + confidence display
- Real-time face count
- FPS calculation
- Automatic saving of detected faces
- Robust error handling
- Production-ready code structure
"""

import cv2
import os
import time
from datetime import datetime


# ==============================
# Configuration
# ==============================
CONFIDENCE_THRESHOLD = 0.6
MODEL_FOLDER = "models"
SAVED_FACES_FOLDER = "saved_faces"


# ==============================
# Utility Functions
# ==============================

def get_absolute_path(*paths):
    """
    Returns absolute path constructed using os.path.join
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, *paths)


def load_model():
    """
    Loads Caffe face detection model.
    Returns:
        net (cv2.dnn_Net): Loaded DNN model
    Raises:
        FileNotFoundError: If model files are missing
    """
    prototxt_path = get_absolute_path(MODEL_FOLDER, "deploy.prototxt")
    model_path = get_absolute_path(MODEL_FOLDER, "res10_300x300_ssd_iter_140000.caffemodel")

    if not os.path.exists(prototxt_path):
        raise FileNotFoundError(f"[ERROR] Prototxt file not found: {prototxt_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Caffe model file not found: {model_path}")

    print("[INFO] Loading face detection model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("[INFO] Model loaded successfully.")

    return net


def initialize_camera():
    """
    Initializes webcam.
    Returns:
        cap (cv2.VideoCapture): Video capture object
    Raises:
        RuntimeError: If camera cannot be accessed
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("[ERROR] Cannot access webcam.")

    print("[INFO] Webcam initialized.")
    return cap


def save_detected_face(frame, box, face_id):
    """
    Saves detected face image to saved_faces folder.
    """
    (startX, startY, endX, endY) = box
    face = frame[startY:endY, startX:endX]

    if face.size == 0:
        return

    save_dir = get_absolute_path(SAVED_FACES_FOLDER)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"face_{face_id}_{timestamp}.jpg"
    save_path = os.path.join(save_dir, filename)

    cv2.imwrite(save_path, face)


def detect_faces(net, frame):
    """
    Performs face detection on a frame.
    Returns:
        detections, frame dimensions
    """
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    return detections, w, h


# ==============================
# Main Application
# ==============================

def main():
    try:
        net = load_model()
        cap = initialize_camera()

        prev_time = 0
        face_counter = 0

        print("[INFO] Starting video stream... Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame.")
                break

            detections, w, h = detect_faces(net, frame)
            current_face_count = 0

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > CONFIDENCE_THRESHOLD:
                    current_face_count += 1
                    face_counter += 1

                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    (startX, startY, endX, endY) = box.astype("int")

                    # Ensure bounding box is within frame
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w - 1, endX), min(h - 1, endY)

                    # Draw bounding box
                    label = f"{confidence * 100:.2f}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)

                    y = startY - 10 if startY - 10 > 10 else startY + 20
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                    # Save face
                    save_detected_face(frame, (startX, startY, endX, endY), face_counter)

            # FPS calculation
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time

            # Display face count and FPS
            cv2.putText(frame, f"Faces: {current_face_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Real-Time Face Detection", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Application closed successfully.")

    except FileNotFoundError as e:
        print(str(e))

    except RuntimeError as e:
        print(str(e))

    except Exception as e:
        print(f"[CRITICAL ERROR] {str(e)}")


# ==============================
# Entry Point
# ==============================

if __name__ == "__main__":
    main()
