# Real-Time Face Detection using OpenCV DNN 📸

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

A lightweight, real-time face detection application built with Python and OpenCV. It utilizes a pre-trained **Caffe Deep Neural Network (SSD ResNet-10)** for high-accuracy detection, significantly outperforming standard Haar Cascades.

## 🚀 Features
* **Real-Time Detection:** Processes webcam feed instantly.
* **High Accuracy:** Uses a DNN (Deep Neural Network) model rather than older cascade classifiers.
* **Confidence Filtering:** Only displays detections with >60% confidence to reduce false positives.
* **Auto-Save:** Automatically crops and saves detected faces to a local folder for dataset creation.
* **Live Stats:** Displays real-time FPS (Frames Per Second) and face count on screen.

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/dhruv-gupta-dev/Face_Detection_Project.git
    cd Face_Detection_Project
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Models**
    Ensure you have the following Caffe model files inside a `models/` folder in the root directory:
    * `deploy.prototxt`
    * `res10_300x300_ssd_iter_140000.caffemodel`

    *(Note: These files are required for the DNN module to work.)*

## 💻 Usage

Run the main script:
```bash
python face_detection.py
```

* `Press q to quit the application.`

* `Saved Faces: Check the saved_faces/ directory to see crops of detected faces.`

```bash
📂 Project Structure
Plaintext
├── models/                  # Caffe model files (prototxt & caffemodel)
├── saved_faces/             # Auto-generated folder for face crops
├── face_detection.py        # Main application script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```
```bash
🧠 How it Works
The application uses OpenCV's DNN module to load a pre-trained Single Shot Detector (SSD) model with a ResNet-10 architecture. 
It resizes video frames to 300x300 blobs, passes them through the network, and filters out weak predictions based on the confidence threshold.

Built by Dhruv Gupta
```