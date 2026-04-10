import streamlit as st
import cv2 as cv
import numpy as np
import os
from sklearn.svm import LinearSVC
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------- MODEL ----------------
class Model:
    def __init__(self):
        self.model = LinearSVC()
        self.is_trained = False
        self.label_map = {}

    def train_model(self):
        img_list = []
        class_list = []
        label_map = {}
        current_label = 0

        # Loop through all object folders
        if not os.path.exists("dataset"):
            return False

        for folder in os.listdir("dataset"):
            folder_path = os.path.join("dataset", folder)

            if not os.path.isdir(folder_path):
                continue

            current_label += 1
            label_map[current_label] = folder

            for file in os.listdir(folder_path):
                path = os.path.join(folder_path, file)
                img = cv.imread(path, cv.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                img = cv.resize(img, (150, 150))
                img_list.append(img.flatten())
                class_list.append(current_label)

        if len(img_list) == 0:
            return False

        X = np.array(img_list)
        y = np.array(class_list)

        self.model.fit(X, y)
        self.is_trained = True
        self.label_map = label_map
        return True

    def predict(self, frame):
        if not self.is_trained:
            return None

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img = cv.resize(gray, (150, 150)).flatten().reshape(1, -1)

        pred = self.model.predict(img)[0]
        return self.label_map.get(pred, "Unknown")


# ---------------- VIDEO CLASS ----------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img


# ---------------- INIT ----------------
st.title("🤖 Smart Object Classifier (Unlimited Classes)")

model = Model()

os.makedirs("dataset", exist_ok=True)

# ---------------- OBJECT INPUT ----------------
object_name = st.text_input("Enter Object Name (e.g. phone, bottle, pen)")

# ---------------- LIVE CAMERA ----------------
st.subheader("📹 Live Camera")

ctx = webrtc_streamer(
    key="camera",
    video_transformer_factory=VideoTransformer
)

# ---------------- ACTIONS ----------------
if ctx.video_transformer:
    frame = ctx.video_transformer.frame

    if frame is not None:

        col1, col2, col3 = st.columns(3)

        # SAVE IMAGE
        if col1.button("📸 Capture Image"):
            if object_name.strip() == "":
                st.warning("Enter object name first!")
            else:
                folder = f"dataset/{object_name}"
                os.makedirs(folder, exist_ok=True)

                count = len(os.listdir(folder)) + 1
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                img = cv.resize(gray, (150, 150))

                cv.imwrite(f"{folder}/img{count}.jpg", img)
                st.success(f"Saved {object_name} image #{count}")

        # TRAIN
        if col2.button("🧠 Train Model"):
            if model.train_model():
                st.success("Model trained successfully!")
            else:
                st.error("No data found!")

        # PREDICT
        if col3.button("🔍 Predict"):
            result = model.predict(frame)

            if result:
                st.success(f"Prediction: {result}")
            else:
                st.warning("Model not trained")


# ---------------- AUTO PREDICTION ----------------
st.divider()

auto = st.checkbox("Enable Auto Prediction")

if auto and ctx.video_transformer:
    frame = ctx.video_transformer.frame

    if frame is not None:
        result = model.predict(frame)
        if result:
            st.info(f"Live Prediction: {result}")


# ---------------- DATASET VIEW ----------------
st.divider()
st.subheader("📁 Dataset Overview")

if os.path.exists("dataset"):
    for folder in os.listdir("dataset"):
        path = os.path.join("dataset", folder)
        if os.path.isdir(path):
            count = len(os.listdir(path))
            st.write(f"{folder}: {count} images")


# ---------------- RESET ----------------
if st.button("🗑 Reset Dataset"):
    for folder in os.listdir("dataset"):
        folder_path = os.path.join("dataset", folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
    model.is_trained = False
    st.warning("Dataset cleared!")
    ###.....###