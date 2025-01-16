import streamlit as st
import os
import requests
import numpy as np
from PIL import Image, ImageDraw
import io
import random
from ultralytics import YOLO
import time

###############################################################################
# 1. Chaotic Logistic Map Encryption Functions
###############################################################################


def logistic_map(r, x):
    return r * x * (1 - x)


def generate_key(seed, n):
    """
    Generate a chaotic key (array of size n) using a logistic map and the given seed.
    """
    key = []
    x = seed
    for _ in range(n):
        x = logistic_map(3.9, x)
        key.append(int(x * 255) % 256)  # map float to 0-255
    return np.array(key, dtype=np.uint8)


def shuffle_pixels(img_array, seed):
    """
    Shuffle the pixels in img_array based on a random sequence seeded by 'seed'.
    """
    h, w, c = img_array.shape
    num_pixels = h * w
    flattened = img_array.reshape(-1, c)
    indices = np.arange(num_pixels)
    random.seed(seed)
    random.shuffle(indices)
    shuffled = flattened[indices]
    return shuffled.reshape(h, w, c), indices


def encrypt_image(img_array, seed):
    """
    Encrypt the given image array using a two-layer XOR + pixel shuffling approach.
    """
    h, w, c = img_array.shape
    flat_image = img_array.flatten()
    # First chaotic key
    chaotic_key_1 = generate_key(seed, len(flat_image))
    # XOR-based encryption (first layer)
    encrypted_flat_1 = [p ^ chaotic_key_1[i] for i, p in enumerate(flat_image)]
    encrypted_array_1 = np.array(
        encrypted_flat_1, dtype=np.uint8).reshape(h, w, c)
    # Shuffle
    shuffled_array, _ = shuffle_pixels(encrypted_array_1, seed)
    # Second chaotic key
    chaotic_key_2 = generate_key(seed * 1.1, len(flat_image))
    shuffled_flat = shuffled_array.flatten()
    encrypted_flat_2 = [p ^ chaotic_key_2[i]
                        for i, p in enumerate(shuffled_flat)]
    doubly_encrypted_array = np.array(
        encrypted_flat_2, dtype=np.uint8).reshape(h, w, c)
    return doubly_encrypted_array

###############################################################################
# 2. YOLOv8 License Plate Detection
###############################################################################


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    """
    Loads the YOLOv8 model from local .pt weights.
    """
    model = YOLO(weights_path)
    return model


def detect_license_plates(model, pil_image):
    """
    Runs YOLOv8 detection on the PIL image.
    Returns:
      - image_with_boxes: PIL image with bounding boxes drawn
      - bboxes: list of (x1, y1, x2, y2) for detected license plates
    """
    np_image = np.array(pil_image)
    results = model.predict(np_image)
    if not results or len(results) == 0:
        return pil_image, []
    result = results[0]
    if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
        return pil_image, []
    bboxes = []
    draw = ImageDraw.Draw(pil_image)
    for box in result.boxes:
        coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        cls_id = int(box.cls[0].item())
        cls_name = model.names.get(cls_id, "Unknown")
        if cls_name.lower() == "licenseplate" or cls_id == 0:
            x1, y1, x2, y2 = map(int, coords)
            bboxes.append((x1, y1, x2, y2))
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return pil_image, bboxes

###############################################################################
# 3. Streamlit App
###############################################################################


def main():
    st.title("YOLOv8 + Chaotic Encryption for License Plates")
    st.write(
        """
        **Instructions**:
        1. Provide an image (URL or file upload).
        2. If a license plate is detected, only that region will be **encrypted** using Chaotic Logistic Map.
        3. Download the final result.
        """
    )

    default_model_path = "best.pt"
    model_path = st.sidebar.text_input(
        "YOLOv8 Weights (.pt)", value=default_model_path)
    if not os.path.isfile(model_path):
        st.warning(f"Model file '{
                   model_path}' not found. Please upload or provide a correct path.")
        st.stop()

    with st.spinner("Loading YOLOv8 model..."):
        model = load_model(model_path)
    st.success("Model loaded successfully!")

    image_url = st.text_input("Image URL (optional)")
    uploaded_file = st.file_uploader(
        "Or upload an image file", type=["jpg", "jpeg", "png"])

    key_seed = st.slider("Encryption Key Seed (0 < seed < 1)",
                         0.001, 0.999, 0.5, step=0.001)

    if st.button("Detect & Encrypt"):
        if image_url and not uploaded_file:
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image_bytes = io.BytesIO(response.content)
                pil_image = Image.open(image_bytes).convert("RGB")
            except Exception as e:
                st.error(f"Failed to load image from URL. Error: {str(e)}")
                return
        elif uploaded_file:
            try:
                pil_image = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"Failed to open uploaded image. Error: {str(e)}")
                return
        else:
            st.warning("Please either paste a valid URL or upload an image.")
            return

        st.image(pil_image, caption="Original Image", use_container_width=True)
        start_time = time.time()

        with st.spinner("Detecting license plates..."):
            image_with_boxes, bboxes = detect_license_plates(
                model, pil_image.copy())

        st.image(image_with_boxes, caption="Detected Plate(s)",
                 use_container_width=True)

        if not bboxes:
            st.warning("No license plates detected.")
            return

        with st.spinner("Encrypting license plates..."):
            np_img = np.array(pil_image)
            encrypted_np = np_img.copy()
            for (x1, y1, x2, y2) in bboxes:
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, encrypted_np.shape[1])
                y2 = min(y2, encrypted_np.shape[0])
                plate_region = encrypted_np[y1:y2, x1:x2]
                if plate_region.size == 0:
                    st.warning(f"Detected plate region ({x1}, {y1}, {
                               x2}, {y2}) is invalid or empty.")
                    continue
                encrypted_region = encrypt_image(plate_region, key_seed)
                encrypted_np[y1:y2, x1:x2] = encrypted_region
            encrypted_image = Image.fromarray(encrypted_np)

        elapsed_time = time.time() - start_time
        st.write(
            f"Total time taken for detection and encryption: **{elapsed_time:.2f} seconds**")
        st.image(encrypted_image, caption="Encrypted Image",
                 use_container_width=True)

        buf = io.BytesIO()
        encrypted_image.save(buf, format="PNG")
        buf.seek(0)
        st.download_button(
            label="Download Encrypted Image",
            data=buf,
            file_name="encrypted_plate.png",
            mime="image/png"
        )


if __name__ == "__main__":
    main()
