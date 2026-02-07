from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

from agentic.decision_agent import DecisionAgent
from agentic.adapters import detection_to_damage_signal
from agentic.strategies import build_damage_story, build_repair_strategies

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
# Load model (REAL)
# =========================
@st.cache_resource
def load_model():
    return CarDamageDetector(model_path="vehicle_damage_vijithaa.h5")

model = load_model()

# =========================
# Severity Scoring
# =========================
def severity_score(d):
    score = d["confidence"] * (d["area_percentage"] / 10)

    if d["severity"] == "Mild":
        score *= 0.8
    elif d["severity"] == "Moderate":
        score *= 1.2
    elif d["severity"] == "Severe":
        score *= 1.6

    return score

# =========================
# Main App
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
        st.info("Upload one or more vehicle images to begin.")
        return

    all_detections = []
    processed_images = []

    if st.button("Analyze Damage", type="primary"):

        with st.spinner("Running AI damage assessment..."):

            for uploaded in uploaded_files:
                image = Image.open(uploaded).convert("RGB")
                processed, detections = model.predict(np.array(image))

                detections = [
                    d for d in detections
                    if d["confidence"] >= confidence_threshold
                ]

                for d in detections:
                    d["image"] = uploaded.name
                    d["severity_score"] = severity_score(d)

                all_detections.extend(detections)
                processed_images.append((uploaded.name, processed))

    if not all_detections:
        st.warning("No significant damage detected.")
        return

    # =========================
    # AGENTIC DECISION
    # =========================
    agent = DecisionAgent(policies_dir="policies")

    signals = [detection_to_damage_signal(d) for d in all_detections]
    decision = agent.decide(signals)

    # =========================
    # OUTPUT
    # =========================
    st.subheader(f"Decision: {decision.action}")
    st.caption(decision.reason)

    # =========================
    # Images
    # =========================
    st.markdown("## Processed Images")
    for name, img in processed_images:
        st.image(img, caption=name, use_container_width=True)

    # =========================
    # Summary Metrics
    # =========================
    st.markdown("## Assessment Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Images", len(uploaded_files))
    c2.metric("Damages", len(all_detections))
    c3.metric("Avg Confidence",
              f"{np.mean([d['confidence'] for d in all_detections]):.2f}")
    c4.metric("Total Cost",
              f"₹{sum(d['estimated_cost'] for d in all_detections)}")

    # =========================
    # Damage Table
    # =========================
    df = pd.DataFrame(all_detections)[[
        "image", "type", "severity",
        "confidence", "area_percentage",
        "estimated_cost", "severity_score"
    ]]

    st.markdown("## Damage Report")
    st.dataframe(df, use_container_width=True)

    # =========================
    # Repair Strategies
    # =========================
    st.markdown("## Repair Recommendations")

    for d in all_detections:
        strategies = build_repair_strategies(d)
        with st.expander(f"{d['image']} – {d['type']} ({d['severity']})"):
            for s in strategies:
                st.write(f"• {s.summary}")

    # =========================
    # Export
    # =========================
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Assessment Report",
        csv,
        f"damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

# =========================
if __name__ == "__main__":
    main()
