import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

from car_damage_detector import CarDamageDetector

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Car Damage Assessment AI",
    page_icon="🚗",
    layout="wide"
)

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    return CarDamageDetector(model_path="vehicle_damage_vijithaa.h5")

model = load_model()

# =========================
# Simple decision logic
# =========================
def decide_action(detections):
    has_severe = any(d["severity"] == "Severe" for d in detections)
    has_moderate = any(d["severity"] == "Moderate" for d in detections)
    has_multiple = len(detections) > 1

    if has_severe:
        return "REJECT", "Severe damage detected. Claim cannot be auto-approved."

    if has_multiple or has_moderate:
        return "HUMAN_REVIEW", "Moderate or multiple damages detected. Manual review required."

    return "AUTO_APPROVE", "Only minor damage detected."

# =========================
# Main app
# =========================
def main():

    st.title("🚗 Car Damage Assessment AI")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.1, 1.0, 0.5, 0.05
    )

    uploaded_files = st.file_uploader(
        "Upload vehicle images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Upload one or more images to start analysis.")
        return

    all_detections = []
    processed_images = []

    if st.button("Analyze Damage", type="primary"):

        with st.spinner("Analyzing images..."):

            for uploaded in uploaded_files:
                image = Image.open(uploaded).convert("RGB")
                processed, detections = model.predict(np.array(image))

                detections = [
                    d for d in detections
                    if d["confidence"] >= confidence_threshold
                ]

                for d in detections:
                    d["image"] = uploaded.name

                all_detections.extend(detections)
                processed_images.append((uploaded.name, processed))

    if not all_detections:
        st.warning("No significant damage detected.")
        return

    # =========================
    # Decision
    # =========================
    decision, reason = decide_action(all_detections)

    st.subheader(f"Decision: {decision}")
    st.caption(reason)

    # =========================
    # Images
    # =========================
    st.markdown("## Processed Images")
    for name, img in processed_images:
        st.image(img, caption=name, use_container_width=True)

    # =========================
    # Summary
    # =========================
    st.markdown("## Assessment Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Images", len(uploaded_files))
    c2.metric("Damages", len(all_detections))
    c3.metric(
        "Avg Confidence",
        f"{np.mean([d['confidence'] for d in all_detections]):.2f}"
    )
    c4.metric(
        "Estimated Cost",
        f"₹{sum(d['estimated_cost'] for d in all_detections)}"
    )

    # =========================
    # Report
    # =========================
    df = pd.DataFrame(all_detections)[[
        "image", "type", "severity",
        "confidence", "area_percentage", "estimated_cost"
    ]]

    st.markdown("## Damage Report")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Report",
        csv,
        f"damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

# =========================
if __name__ == "__main__":
    main()
