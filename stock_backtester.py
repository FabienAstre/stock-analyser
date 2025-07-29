import os
import mss  # type: ignore
import cv2
import numpy as np
import time
import glob
import streamlit as st
from ultralytics import YOLO
from openpyxl import Workbook
from PIL import Image

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📈 YOLO Stock Pattern Detection - Live Monitor")

run_detection = st.toggle("Start Detection")

# Define dynamic paths
home_dir = os.path.expanduser("~")
save_path = os.path.join(home_dir, "yolo_detection")
screenshots_path = os.path.join(save_path, "screenshots")
detect_path = os.path.join(save_path, "runs", "detect")

# Ensure necessary directories exist
os.makedirs(screenshots_path, exist_ok=True)
os.makedirs(detect_path, exist_ok=True)

# Load YOLOv8 model
model_path = "model.pt"
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()
model = YOLO(model_path)

# Define pattern classes
classes = ['Head and shoulders bottom', 'Head and shoulders top', 'M_Head', 'StockLine', 'Triangle', 'W_Bottom']

# Create an Excel file
excel_file = os.path.join(save_path, "classification_results.xlsx")
wb = Workbook()
ws = wb.active
ws.append(["Timestamp", "Predicted Image Path", "Label"])

# Screen capture region (adjust based on your screen)
monitor = {"top": 0, "left": 683, "width": 683, "height": 768}

# Layout
col1, col2 = st.columns([1, 2])
status_placeholder = col1.empty()
image_placeholder = col2.empty()

# Main loop
if run_detection:
    with mss.mss() as sct:
        last_capture_time = time.time()
        frame_count = 0
        while run_detection:
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            current_time = time.time()
            if current_time - last_capture_time >= 60:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                image_name = f"predicted_images_{timestamp}_{frame_count}.png"
                image_path = os.path.join(screenshots_path, image_name)
                cv2.imwrite(image_path, img)

                results = model(image_path, save=True)
                predict_path = results[0].save_dir if results else None
                final_image_path = image_path

                if predict_path and os.path.exists(predict_path):
                    annotated_images = sorted(glob.glob(os.path.join(predict_path, "*.jpg")), key=os.path.getmtime, reverse=True)
                    final_image_path = annotated_images[0] if annotated_images else image_path

                if results and results[0].boxes:
                    class_indices = results[0].boxes.cls.tolist()
                    predicted_label = classes[int(class_indices[0])]
                else:
                    predicted_label = "No pattern detected"

                ws.append([timestamp, final_image_path, predicted_label])
                wb.save(excel_file)

                annotated_img = cv2.imread(final_image_path)
                if annotated_img is not None:
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    image_placeholder.image(annotated_img, channels="RGB", use_column_width=True)

                status_placeholder.info(f"Frame {frame_count}: {predicted_label}")
                frame_count += 1
                last_capture_time = current_time

            # Display current screen for debugging
            preview_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            preview_img = Image.fromarray(preview_img)
            image_placeholder.image(preview_img, caption="Live Feed", use_column_width=True)

            if not run_detection:
                break
