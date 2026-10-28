import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page config
st.set_page_config(
    page_title="🐱 vs 🐶 Cats vs Dogs: Method Comparison",
    page_icon="🐾",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class=main-header>🐾 Cats vs. Dogs: Deep Learning Method Comparison</div>", unsafe_allow_html=True)
st.markdown("<div class=sub-header>Comparative study of Custom CNN (trained from scratch) vs. Pretrained MobileNetV2 on small data.</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Choose Prediction Mode",
    ["Compare Both Models", "MobileNetV2 (Transfer Learning)", "Custom CNN (From Scratch)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📊 Benchmark Summary
- **MobileNetV2**: 96.67% Test Acc (F1: 0.97)
- **Custom CNN**: 54.67% Test Acc (F1: 0.44)
- **Dataset**: ~1000 images (Cat-and-Dog subset)
""" )

tab1, tab2 = st.tabs(["🔮 Real-Time Classifier", "📈 Comparative Report & Metrics"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png", "webp"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("🎯 Classification Results")

        if uploaded_file is not None:
            models_dir = PROJECT_ROOT / "models"
            mobilenet_path = models_dir / "mobilenetv2_best.keras"
            custom_path = models_dir / "custom_cnn_best.keras"

            # Check if models exist
            has_mobilenet = mobilenet_path.exists()
            has_custom = custom_path.exists()

            def run_inference(m_path, m_type):
                from src.inference.predictor import CatDogPredictor
                predictor = CatDogPredictor(m_path, model_type=m_type)
                return predictor.predict(image)

            if model_choice in ("Compare Both Models", "MobileNetV2 (Transfer Learning)"):
                st.markdown("#### 🚀 MobileNetV2 (Transfer Learning)")
                if has_mobilenet:
                    res_m = run_inference(mobilenet_path, "mobilenetv2")
                    st.write(f"**Prediction:** {res_m[label]} ({res_m[confidence_percent]:.1f}% confidence)")
                    st.progress(res_m[dog_confidence])
                    st.caption(f"Cat: {res_m[cat_confidence]*100:.1f}% | Dog: {res_m[dog_confidence]*100:.1f}%")
                else:
                    st.info("ℹ️ MobileNetV2 checkpoint can be trained with `python scripts/train.py --model mobilenetv2`")

            if model_choice in ("Compare Both Models", "Custom CNN (From Scratch)"):
                st.markdown("#### 🧠 Custom CNN (From Scratch)")
                if has_custom:
                    res_c = run_inference(custom_path, "custom_cnn")
                    st.write(f"**Prediction:** {res_c[label]} ({res_c[confidence_percent]:.1f}% confidence)")
                    st.progress(res_c[dog_confidence])
                    st.caption(f"Cat: {res_c[cat_confidence]*100:.1f}% | Dog: {res_c[dog_confidence]*100:.1f}%")
                else:
                    st.info("ℹ️ Custom CNN checkpoint can be trained with `python scripts/train.py --model custom_cnn`")
        else:
            st.info("👆 Upload an image to evaluate predictions.")

with tab2:
    st.subheader("📑 Empirical Benchmark Results")
    st.markdown("""
    | Metric | Custom CNN (From Scratch) | MobileNetV2 (Transfer Learning) |
    | :--- | :--- | :--- |
    | **Convergence Epoch** | Epoch 59 | Epoch 4 (Early stopped at 6) |
    | **Validation Accuracy** | ~53.33% | ~99.33% |
    | **Test Accuracy** | **54.67%** | **96.67%** |
    | **Test Loss** | 0.7211 | 0.0993 |
    | **Macro F1-Score** | 0.44 | 0.97 |
    """ )

    st.markdown("---")
    st.subheader("🖼️ Confusion Matrices & Learning Curves")
    out_dir = PROJECT_ROOT / "outputs"
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Custom CNN (From Scratch)**")
        if (out_dir / "custom_cnn_confusion_matrix.png").exists():
            st.image(str(out_dir / "custom_cnn_confusion_matrix.png"), caption="Custom CNN Confusion Matrix")
        if (out_dir / "custom_cnn_history.png").exists():
            st.image(str(out_dir / "custom_cnn_history.png"), caption="Custom CNN Training History")

    with c2:
        st.markdown("**MobileNetV2 (Transfer Learning)**")
        if (out_dir / "mobilenetv2_confusion_matrix.png").exists():
            st.image(str(out_dir / "mobilenetv2_confusion_matrix.png"), caption="MobileNetV2 Confusion Matrix")
        if (out_dir / "mobilenetv2_history.png").exists():
            st.image(str(out_dir / "mobilenetv2_history.png"), caption="MobileNetV2 Training History")
