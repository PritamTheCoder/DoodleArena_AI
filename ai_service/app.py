# Save this as app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import base64
from PIL import Image
import io
import numpy as np

# -----------------------
# Config
# -----------------------
API_URL = "http://127.0.0.1:8000/recognize"  # FastAPI endpoint
CANVAS_SIZE = 448  # Canvas size for drawing

CLASSES_30 = [
    "cat", "dog", "bird", "fish", "cow",
    "apple", "banana", "pizza", "cake", "ice cream",
    "car", "bicycle", "airplane", "bus", "train",
    "house", "tree", "flower", "sun", "cloud",
    "star", "moon", "hand", "face", "clock",
    "book", "chair", "shoe", "key", "umbrella"
]
# -----------------------
# Streamlit UI
# -----------------------
st.title(" DoodleNet Recognition")
st.write("Draw a doodle, select the target prompt, and see what the model predicts!")

# Prompt selection
prompt = st.selectbox("Select target prompt:", CLASSES_30)

# Drawing canvas
canvas = st_canvas(
    fill_color="white",
    stroke_width=16,
    stroke_color="black",
    background_color="white",
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    drawing_mode="freedraw",
    key="canvas"
)

# -----------------------
# Helper functions
# -----------------------
def encode_image_to_base64(pil_image):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def recognize_doodle(base64_image, prompt):
    """Send doodle to API and get predictions."""
    payload = {"prompt": prompt, "image_base64": base64_image}
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text}

# -----------------------
# Run recognition
# -----------------------
if st.button("Recognize"):
    if canvas.image_data is not None:
        # Convert numpy canvas to PIL
        img = Image.fromarray(np.uint8(canvas.image_data[:, :, :3]))
        
        # Encode to base64
        img_base64 = encode_image_to_base64(img)
        
        # Send to FastAPI
        result = recognize_doodle(img_base64, prompt)
        
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"Target class: {result['prompt']}")
            st.write(f"Confidence: {result['confidence']:.4f}")
            st.subheader("Top predictions")
            for pred in result["top_predictions"]:
                st.write(f"{pred['class']}: {pred['confidence']:.4f}")
    else:
        st.warning("Please draw something on the canvas before recognizing.")