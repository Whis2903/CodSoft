import cv2
import streamlit as st
import numpy as np
from PIL import Image
import tempfile

# Load the face detection model
face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_default.xml")

# Title and description
st.title("Face Detection and Image Recognition App")
st.write("This app detects faces and recognizes objects in real-time, uploaded videos, or images.")

# Select the input mode
mode = st.sidebar.selectbox("Choose input mode", ["Webcam", "Image", "Video"])

if mode == "Webcam":
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height

    while run:
        success, img = cap.read()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
        num_faces = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        st.write(f"Number of people detected: {num_faces}")

    cap.release()

elif mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        img = np.array(Image.open(uploaded_file))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
        num_faces = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        st.image(img, channels="RGB")
        st.write(f"Number of people detected: {num_faces}")

elif mode == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        FRAME_WINDOW = st.image([])

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img_gray, 1.1, 4)
            num_faces = len(faces)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            st.write(f"Number of people detected: {num_faces}")

        cap.release()
