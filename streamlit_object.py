import streamlit as st
import cv2
from ultralytics import YOLO
import easyocr
import time

st.set_page_config(page_title="Real-Time Object & Text Detection", layout="wide")
st.markdown("""
    <style>
    .stButton>button {height: 3em; width: 100%; font-size: 1.2em;}
    .activity-log {background-color: #f9f9fb; border-radius:7px; padding:12px; font-size:1.08em; min-height:180px;}
    </style>
    """, unsafe_allow_html=True
)
# Sidebar info
with st.sidebar:
    st.header("📷 Real-Time Detection")
    st.info(
        "Detects objects (YOLOv8) and reads visible text (OCR) in real time from your webcam.\n\n"
        "- Click 'Start Camera' to begin.\n"
        "- Click 'Stop Camera' to finish.\n"
        "- Activity log shows what was detected each frame."
    )
    st.caption("Built with Streamlit, OpenCV, YOLOv8, EasyOCR.")

st.title("🔎 Object & Text Detector")

# Session setup
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "last_detected" not in st.session_state:
    st.session_state.last_detected = []
if "last_texts" not in st.session_state:
    st.session_state.last_texts = []

camera_placeholder = st.empty()
log_placeholder = st.container()

col1, col2 = st.columns([1, 1])
with col1:
    start = st.button("▶️ Start Camera", key="start_camera_btn")
with col2:
    stop = st.button("⏹️ Stop Camera", key="stop_camera_btn")

if start:
    st.session_state.run_camera = True
if stop:
    st.session_state.run_camera = False

if st.session_state.run_camera:
    cap = cv2.VideoCapture(0)
    model = YOLO('yolov8n.pt')
    reader = easyocr.Reader(['en'])  # Use 'en' for English

    with log_placeholder:
        st.markdown("<b> balck Activity Log</b>", unsafe_allow_html=True)
        log_box = st.empty()

    while st.session_state.run_camera:
        ret, frame = cap.read()
        if not ret:
            st.warning("Can't read from webcam!")
            break

        # YOLO object detection
        results = model(frame)
        classes = results[0].names
        detected = [classes[int(c)] for c in results[0].boxes.cls]
        st.session_state.last_detected = detected

        # EasyOCR text detection
        ocr_results = reader.readtext(frame)
        detected_texts = [res[1] for res in ocr_results]
        st.session_state.last_texts = detected_texts

        # Annotate frame with OCR text
        annotated_frame = results[0].plot()
        for bbox, text, conf in ocr_results:
            topleft = tuple(map(int, bbox[0]))
            cv2.putText(annotated_frame, text, topleft, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        camera_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

        # Log
        activity = f"🟢 Objects: <b>{', '.join(detected) or 'None'}</b>"
        if detected_texts:
            activity += f" | 🔤 Text: <b>{', '.join(detected_texts)}</b>"
        if not st.session_state.activity_log or activity != st.session_state.activity_log[-1]:
            st.session_state.activity_log.append(activity)
            logs = "<div class='activity-log'>" + "<br>".join(st.session_state.activity_log[-8:]) + "</div>"
            log_box.markdown(logs, unsafe_allow_html=True)

        # Remove cv2.waitKey(), use sleep for timing
        time.sleep(0.07)  # Adjust for FPS vs. system load

        if not st.session_state.run_camera or stop:
            break

    cap.release()
    camera_placeholder.empty()
else:
    camera_placeholder.info("Click ▶️ **Start Camera** to begin object & text detection.")
