import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
from PIL import Image

st.title("📈 YOLO Stock Pattern Detection - Upload Model & Image")

# Step 1: Upload YOLOv8 model
uploaded_model = st.file_uploader("Upload your YOLO model (.pt)", type=["pt"])

if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
        tmp_model.write(uploaded_model.read())
        model_path = tmp_model.name

    # Load the model
    try:
        model = YOLO(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    # Step 2: Upload image to detect on
    uploaded_image = st.file_uploader("Upload an image (screenshot, chart, etc.)", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        img_array = np.array(image)

        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            image.save(tmp_img.name)
            img_path = tmp_img.name

        # Run detection
        results = model(img_path, save=True)  # saves annotated image to results[0].save_dir

        # Get annotated image path
        save_dir = results[0].save_dir
        annotated_images = list(save_dir.glob("*.jpg")) + list(save_dir.glob("*.png"))
        if annotated_images:
            annotated_img_path = str(annotated_images[0])
            annotated_img = Image.open(annotated_img_path)
        else:
            annotated_img = image

        st.image(annotated_img, caption="Annotated Image", use_column_width=True)

        # Display detected classes
        if results and results[0].boxes:
            class_ids = results[0].boxes.cls.tolist()
            detected_labels = [model.names[int(c)] for c in class_ids]
            st.write("📝 Detected labels:", detected_labels)
        else:
            st.write("No objects detected.")
else:
    st.info("Upload your YOLO model (.pt) to get started.")
