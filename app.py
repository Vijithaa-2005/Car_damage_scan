from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import textwrap
from streamlit.components.v1 import html as st_html

# =========================
# Agentic imports
# =========================
from agentic.trace import build_decision_trace
from agentic.decision_agent import DecisionAgent
from agentic.adapters import pick_primary_detection, detection_to_damage_signal
from agentic.explainer import build_customer_explanation, format_kb_insights
from agentic.strategies import build_repair_strategies, build_damage_story
from agentic.vision.after_inpaint import make_repaired_after_preview

# =========================
# Optional model import (SILENT)
# =========================
try:
    from car_damage_detector import CarDamageDetector
except Exception:
    CarDamageDetector = None

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Car Damage Assessment AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# STYLES (UNCHANGED)
# =========================
st.markdown("""<style>/* YOUR FULL CSS – unchanged */</style>""", unsafe_allow_html=True)

# =========================
# Model loader (silent)
# =========================
@st.cache_resource
def load_model():
    if CarDamageDetector is None:
        return None
    try:
        return CarDamageDetector(model_path="vehicle_damage_vijithaa.h5")
    except Exception:
        return None

# =========================
# Fallback demo inference (MULTI DAMAGE TYPES)
# =========================
def demo_damage_detection(image: Image.Image):
    img = np.array(image)
    h, w = img.shape[:2]

    detections = [
        {
            "type": "Dent",
            "severity": "Moderate",
            "confidence": 0.78,
            "bbox": [int(w*0.2), int(h*0.35), int(w*0.45), int(h*0.6)],
            "area_percentage": 6.5,
            "estimated_cost": 420
        },
        {
            "type": "Scratch",
            "severity": "Mild",
            "confidence": 0.65,
            "bbox": [int(w*0.55), int(h*0.4), int(w*0.8), int(h*0.55)],
            "area_percentage": 4.2,
            "estimated_cost": 250
        }
    ]

    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),3)
        cv2.putText(img,f"{d['type']} {d['confidence']:.2f}",
                    (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    return img, detections

# =========================
# HERO
# =========================
def render_hero():
    st.markdown(
        """
        <div class="hero-wrap">
        <h1 class="hero-title">Car Damage Assessment AI</h1>
        <div class="hero-subtitle">
        High-trust vehicle damage intelligence with agentic decisioning.
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# NEW FUNCTION ADDED
# =========================
def render_decision_actions(decision, primary):
    st.markdown("## Recommended Actions")

    if hasattr(decision, "actions") and decision.actions:
        for a in decision.actions:
            st.write(f"- {a}")
    else:
        st.write(f"**Action:** {decision.action}")
        st.write(f"**Reason:** {decision.reason}")

    st.markdown("### Primary Damage Details")
    st.write(f"Type: {primary.get('type', 'N/A')}")
    st.write(f"Severity: {primary.get('severity', 'N/A')}")
    st.write(f"Confidence: {primary.get('confidence', 0):.2f}")

# =========================
# CREATE REPORT TABLE
# =========================
def build_report_table(detections):
    df = pd.DataFrame(detections)
    df = df[["type", "severity", "confidence", "area_percentage", "estimated_cost"]]
    df.columns = ["Damage Type", "Severity", "Confidence", "Area (%)", "Estimated Cost (₹)"]
    return df

# =========================
# MAIN
# =========================
def main():
    render_hero()

    with st.sidebar:
        st.markdown("### Configuration")
        confidence_threshold = st.slider("Confidence Threshold",0.1,1.0,0.5,0.05)
        developer_mode = st.checkbox("Developer mode", False)
        st.session_state["dev_mode"] = developer_mode

    col1, col2 = st.columns([1,1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload vehicle images (multiple allowed)",
            type=["jpg","png","jpeg"],
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded in uploaded_files:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, use_container_width=True)

                if st.button("Analyze Damage", type="primary", use_container_width=True):
                    with st.spinner("Analyzing damage..."):
                        model = load_model()

                        if model:
                            processed, detections = model.predict(np.array(image))
                        else:
                            processed, detections = demo_damage_detection(image)

                        detections = [d for d in detections if d["confidence"] >= confidence_threshold]

                        st.session_state["processed"] = processed
                        st.session_state["detections"] = detections
                        st.session_state["original"] = np.array(image)

    with col2:
        if "detections" in st.session_state:
            st.image(st.session_state["processed"], use_container_width=True)

            detections = st.session_state["detections"]

            agent = DecisionAgent(policies_dir="policies")
            primary = pick_primary_detection(detections)

            if primary:
                signal = detection_to_damage_signal(primary)
                decision = agent.decide(signal)

                st.subheader(f"Decision: {decision.action}")
                st.caption(decision.reason)

                render_decision_actions(decision, primary)

                trace = build_decision_trace(
                    primary_detection=primary,
                    signal=signal,
                    decision=decision
                )

                story = build_damage_story(primary)
                strategies = build_repair_strategies(primary)

                st.markdown("## Damage Story")
                for c in story["consequences"]:
                    st.write(f"- {c}")

                st.markdown("## Repair Strategies")
                for s in strategies:
                    with st.expander(s.name):
                        st.write(s.summary)

                preview = make_repaired_after_preview(
                    st.session_state["original"][:,:,::-1],
                    primary,
                    intensity=0.65
                )

                st.markdown("## Before / After Preview")
                cA,cB = st.columns(2)
                cA.image(st.session_state["original"], caption="Before", use_container_width=True)
                cB.image(preview.after_bgr[:,:,::-1], caption="After", use_container_width=True)

            st.markdown("## Assessment Summary")
            c1,c2,c3,c4 = st.columns(4)

            c1.metric("Damages", len(detections))
            c2.metric("Avg Confidence", f"{np.mean([d['confidence'] for d in detections]):.1%}")
            c3.metric("Total Area", f"{sum(d['area_percentage'] for d in detections):.1f}%")
            c4.metric("Estimated Cost", f"₹{sum(d['estimated_cost'] for d in detections)}")

            st.markdown("## Damage Report")
            report_df = build_report_table(detections)
            st.dataframe(report_df, use_container_width=True)

            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Report as CSV",
                data=csv,
                file_name=f"damage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()



