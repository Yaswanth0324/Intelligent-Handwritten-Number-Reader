import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load CNN model without compiling (avoids compatibility issues)
model = load_model("model/digit_model.h5", compile=False)

st.set_page_config(page_title="Intelligent Handwritten Number Reader", page_icon="🔢", layout="centered")

st.title("🔢 Intelligent Handwritten Number Reader")
st.write("Upload an image OR draw multi-digit numbers (e.g., 245, 1002, 987).")

# --------------------------
# FILE UPLOAD SECTION
# --------------------------
uploaded_file = st.file_uploader("📤 Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

st.write("### OR draw your digits below 👇")

# --------------------------
# CANVAS DRAWING SECTION
# --------------------------
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    height=300,
    width=600,
    drawing_mode="freedraw",
    key="canvas",
)


# =============================
#   DIGIT SEGMENTATION LOGIC
# =============================
def segment_digits(img: Image.Image):
    img = img.convert("L")
    img = np.array(img)

    # MNIST uses white digit on black background → invert
    img = cv2.bitwise_not(img)

    # Binarize
    _, thresh = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)

    # Extract contours (each contour = a digit)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_images = []
    boxes = []

    # Filter noise
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:     # Avoid small specks
            boxes.append((x, y, w, h))

    # Sort contours left -> right
    boxes = sorted(boxes, key=lambda b: b[0])

    # Prepare each digit
    for (x, y, w, h) in boxes:
        digit = thresh[y:y+h, x:x+w]

        # Resize to 20x20
        digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

        # Put inside a 28x28 MNIST-style frame
        padded = np.zeros((28, 28), dtype=np.uint8)
        xo = (28 - 20) // 2
        yo = (28 - 20) // 2
        padded[yo:yo+20, xo:xo+20] = digit

        padded = padded.astype("float32") / 255.0
        padded = padded.reshape(1, 28, 28, 1)

        digit_images.append(padded)

    return digit_images


# =============================
#   PREDICTION BUTTON
# =============================
if st.button("Predict Number"):
    image = None

    # Use uploaded file if exists
    if uploaded_file:
        image = Image.open(uploaded_file)

    # Else use the canvas
    elif canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype("uint8"))

    else:
        st.warning("Please upload an image OR draw a number.")
        st.stop()

    digit_list = segment_digits(image)

    if len(digit_list) == 0:
        st.error("❌ No digits detected. Try drawing thicker or uploading a clearer image.")
        st.stop()

    final_number = ""

    for d in digit_list:
        prediction = model.predict(d, verbose=0)
        digit = str(np.argmax(prediction))
        final_number += digit

    st.success(f"### 🎉 Predicted Number: **{final_number}**")
    st.image(image, width=300, caption="Your Input")
