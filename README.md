Real-Time Object & Text Detection App (notebook + streamlit.py file)
Overview
This project demonstrates a real-time application that detects objects and reads visible text from a webcam feed. The goal was to create a simple yet effective tool for live computer vision tasks, combining state-of-the-art deep learning for object detection with fast OCR for reading text from physical environments.

Features
Detects multiple objects from live video using YOLOv8

Reads printed or handwritten text using EasyOCR from the camera feed

Live annotation overlays for both object and text detection

Dynamic activity log that records detections frame by frame

Clean and user-friendly web interface powered by Streamlit

Technology Stack
Python 3.10

YOLOv8 (Ultralytics) for object detection

EasyOCR for text recognition

OpenCV for webcam and image processing

Streamlit for the web interface

How It Works
Launch the app with streamlit run app.py.

The web UI guides the user—click "Start Camera" to begin real-time detection.

The app continuously captures frames from the webcam.

Each frame is:

Passed through YOLOv8 for object detection and bounding box annotation.

Processed with EasyOCR to detect and overlay any visible text.

The activity log updates after each frame, showing detected objects and text transcriptions.

Click "Stop Camera" to pause the video and release the webcam.
