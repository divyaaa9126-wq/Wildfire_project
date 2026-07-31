import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# --- Dynamic Path Resolution ---
# Sets the directory relative to this app.py file location (Wild_fire/)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model' / 'best_model_mobilenet.keras'
LABELS_PATH = BASE_DIR / 'labels.json'

# --- Page Configuration ---
st.set_page_config(
    page_title="Wildfire Detection AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        text-align: center;
        color: #A0A0A0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model & Labels (Cached for speed) ---
@st.cache_resource
def load_wildfire_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_data
def load_labels():
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
    return labels

model = load_wildfire_model()
class_names = load_labels()

# --- Grad-CAM Helper Functions ---
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
        elif hasattr(layer, 'layers'):
            sub_name = find_last_conv_layer(layer)
            if sub_name:
                return sub_name
    return None

def generate_gradcam(img_array, model):
    last_conv_name = find_last_conv_layer(model)
    if not last_conv_name:
        return None
    
    _ = model(img_array)
    base_model = model.layers[0] if len(model.layers) > 1 and hasattr(model.layers[0], 'get_layer') else model
    target_layer = base_model.get_layer(last_conv_name)
    
    grad_model = tf.keras.models.Model(
        inputs=base_model.inputs,
        outputs=[target_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(img_array)
        x = base_outputs
        for layer in model.layers[1:]:
            x = layer(x)
        preds = x
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(pil_img, heatmap, alpha=0.4):
    img_np = np.array(pil_img.convert('RGB').resize((224, 224)))
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = cv2.resize(jet_heatmap, (img_np.shape[1], img_np.shape[0]))
    superimposed_img = jet_heatmap * alpha * 255 + img_np * (1 - alpha)
    return np.uint8(np.clip(superimposed_img, 0, 255))

# --- UI Header ---
st.markdown("<div class='main-title'>🔥 Wildfire Detection Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Satellite & Aerial Image Analysis using MobileNetV2</div>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.info("""
**Model Specs:**
* **Architecture:** MobileNetV2
* **Accuracy:** 96.92%
* **Size:** ~12.3 MB
* **Input Resolution:** 224x224
""")

show_gradcam = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)
confidence_threshold = st.sidebar.slider("Confidence Alert Threshold", 50, 100, 80)

# --- Main App Body ---
uploaded_file = st.file_uploader("Upload an aerial image for analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file)
    
    # Preprocess
    img_resized = image.convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    
    # Run Prediction
    with st.spinner("Analyzing image features..."):
        preds = model.predict(img_array, verbose=0)
        prob_wildfire = float(preds[0][1]) if len(preds[0]) > 1 else float(preds[0][0])
        prob_nowildfire = 1.0 - prob_wildfire
        
        predicted_idx = 1 if prob_wildfire >= 0.5 else 0
        predicted_label = class_names.get(str(predicted_idx), "Unknown")
        confidence = prob_wildfire if predicted_idx == 1 else prob_nowildfire

    # Display Results in Columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("📊 Diagnostic Summary")
        
        # Status Card
        if predicted_label.lower() == 'wildfire':
            st.error(f"⚠️ **WILDFIRE DETECTED**\nConfidence: {confidence * 100:.2f}%")
        else:
            st.success(f"✅ **NO WILDFIRE DETECTED**\nConfidence: {confidence * 100:.2f}%")
            
        # Probability Bars
        st.write("**Class Probability Distribution:**")
        st.progress(prob_wildfire, text=f"Wildfire: {prob_wildfire * 100:.1f}%")
        st.progress(prob_nowildfire, text=f"No Wildfire: {prob_nowildfire * 100:.1f}%")

    # Grad-CAM Section
    if show_gradcam:
        st.markdown("---")
        st.subheader("🔍 Explainability (Grad-CAM)")
        st.caption("Highlights regions of the image that contributed most to the prediction.")
        
        with st.spinner("Generating attention heatmap..."):
            heatmap = generate_gradcam(img_array, model)
            if heatmap is not None:
                gradcam_img = overlay_heatmap(image, heatmap)
                
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.image(img_resized, caption="Resized Input (224x224)", use_container_width=True)
                with g_col2:
                    st.image(gradcam_img, caption="Grad-CAM Focus Area", use_container_width=True)
            else:
                st.warning("Grad-CAM visualization could not be computed for this layer structure.")
else:
    st.info("👆 Please upload a JPEG or PNG image to start detection.")
