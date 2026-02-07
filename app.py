# ===============================
# ENV
# ===============================
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import textwrap
from streamlit.components.v1 import html as st_html

# ===============================
# Agentic imports (UNCHANGED)
# ===============================
from agentic.trace import build_decision_trace
from agentic.decision_agent import DecisionAgent
from agentic.adapters import pick_primary_detection, detection_to_damage_signal
from agentic.explainer import build_customer_explanation, format_kb_insights
from agentic.strategies import build_repair_strategies, build_damage_story
from agentic.vision.after_inpaint import make_repaired_after_preview

# ===============================
# Optional model import (SAFE)
# ===============================
try:
    from car_damage_detector import CarDamageDetector
    from utils import enhance_image, calculate_damage_stats
except ImportError:
    CarDamageDetector = None


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Car Damage Assessment AI",
    page_icon="🚗",
    layout="wide",
)

# ===============================
# WHITE THEME (CHANGE #2)
# ===============================
st.markdown("""
<style>
.stApp {
    background: #ffffff;
    color: #111111;
}
html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}
h1,h2,h3,h4 {
    color: #111111;
}
.stMarkdown p {
    color: #333333;
}
section[data-testid="stSidebar"] {
    background: #f8f9fa;
    border-right: 1px solid #e5e5e5;
}
.stButton button {
    background: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #cccccc !important;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HERO
# ===============================
def render_hero():
    st.title("🚗 Car Damage Assessment AI")
    st.caption("AI-powered vehicle damage analysis with clean, reviewer-friendly UI")

# ===============================
# DEMO DAMAGE DETECTION (UNCHANGED)
# ===============================
def demo_damage_detection(image: Image.Image):
    img = np.array(image)
    h, w = img.shape[:2]

    detections = [
        {
            "type": "Scratch",
            "severity": "Light",
            "confidence": 0.88,
            "bbox": [int(w*0.2), int(h*0.3), int(w*0.4), int(h*0.5)],
            "area_percentage": 2.4,
            "estimated_cost": 150,
        }
    ]

    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

    return img, detections

# ===============================
# MAIN
# ===============================
def main():
    render_hero()

    uploaded = st.file_uploader(
        "Upload car image",
        type=["jpg","png","jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

        if st.button("Analyze Damage"):
            with st.spinner("Analyzing image..."):
                processed, detections = demo_damage_detection(image)

            st.image(processed, caption="Detected Damage", use_container_width=True)

            primary = pick_primary_detection(detections)
            agent = DecisionAgent(policies_dir="policies")

            if primary:
                signal = detection_to_damage_signal(primary)
                decision = agent.decide(signal)

                expl = build_customer_explanation(
                    decision_action=decision.action,
                    decision_reason=decision.reason,
                    policy_refs=list(decision.policy_refs or []),
                    next_steps=list(decision.next_steps or []),
                    sop_text=None,
                    signal=signal,
                )

                st.subheader("Assessment Result")
                st.success(expl["summary"])

                if expl["why_bullets"]:
                    st.write("**Why:**")
                    for w in expl["why_bullets"]:
                        st.write(f"- {w}")

                if expl["next_steps"]:
                    st.write("**Next Steps:**")
                    for n in expl["next_steps"]:
                        st.write(f"- {n}")

    else:
        st.info("Upload an image to begin analysis.")

if __name__ == "__main__":
    main()
