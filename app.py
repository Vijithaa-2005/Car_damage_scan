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

from agentic.trace import build_decision_trace
from agentic.decision_agent import DecisionAgent
from agentic.adapters import pick_primary_detection, detection_to_damage_signal
from agentic.explainer import (
    build_customer_explanation,
    format_kb_insights,
    build_expert_insight,
)
from agentic.strategies import build_repair_strategies, build_damage_story
from agentic.vision.after_inpaint import make_repaired_after_preview

try:
    from car_damage_detector import CarDamageDetector
    from utils import enhance_image, calculate_damage_stats
except ImportError:
    CarDamageDetector = None


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Car Damage Assessment AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# WHITE BACKGROUND THEME
# =========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #ffffff;
    color: #000000;
}

h1, h2, h3, h4 {
    color: #000000;
    letter-spacing: -0.04em;
}

.stMarkdown p {
    color: #333333;
}

section[data-testid="stSidebar"] {
    background-color: #f7f7f7;
    border-right: 1px solid #e0e0e0;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 12px;
}

div[data-testid="stExpander"] {
    border: 1px solid #e5e5e5;
    border-radius: 12px;
    background-color: #ffffff;
}

.stButton button {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-radius: 10px;
    font-weight: 700;
}

[data-testid="stDownloadButton"] button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cccccc !important;
    border-radius: 10px;
    font-weight: 600;
}

.js-plotly-plot {
    background-color: #ffffff !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HERO
# =========================
def render_hero():
    st.markdown(
        """
        <div style="padding:20px; border:1px solid #e5e5e5; border-radius:16px;">
            <h1>Car Damage Assessment AI</h1>
            <p>
                AI-assisted vehicle damage analysis using computer vision and decision intelligence.
                Upload an image to receive damage insights, severity, and repair guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# DEMO DAMAGE DETECTION
# =========================
def demo_damage_detection(image: Image.Image):
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    detections = [
        {
            "type": "Dent",
            "severity": "Moderate",
            "confidence": 0.78,
            "bbox": [int(w*0.25), int(h*0.35), int(w*0.45), int(h*0.55)],
            "area_percentage": 6.4,
            "estimated_cost": 420,
        }
    ]

    annotated = img_array.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)

    return annotated, detections


# =========================
# CHARTS
# =========================
def create_damage_distribution_chart(detections):
    counts = {}
    for d in detections:
        counts[d["type"]] = counts.get(d["type"], 0) + 1

    fig = px.pie(
        values=list(counts.values()),
        names=list(counts.keys()),
        title="Damage Type Distribution",
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


def create_severity_chart(detections):
    sev = {}
    for d in detections:
        sev[d["severity"]] = sev.get(d["severity"], 0) + 1

    fig = go.Figure([go.Bar(x=list(sev.keys()), y=list(sev.values()))])
    fig.update_layout(
        title="Severity Distribution",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
    )
    return fig


# =========================
# MAIN
# =========================
def main():
    render_hero()

    with st.sidebar:
        st.markdown("## Configuration")
        st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        st.checkbox("Enable Image Enhancement", value=True)
        st.checkbox("Generate Assessment Report", value=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Image")
        uploaded = st.file_uploader("Upload vehicle image", type=["jpg", "png", "jpeg"])

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, use_container_width=True)

            if st.button("Analyze Damage", use_container_width=True):
                processed, detections = demo_damage_detection(image)
                st.session_state.processed = processed
                st.session_state.detections = detections
                st.session_state.original = np.array(image)
                st.session_state.image_info = image.size

    with col2:
        st.subheader("Results")

        if "detections" in st.session_state:
            st.image(st.session_state.processed, use_container_width=True)

            detections = st.session_state.detections
            agent = DecisionAgent(policies_dir="policies")
            primary = pick_primary_detection(detections)

            if primary:
                signal = detection_to_damage_signal(primary)
                decision = agent.decide(signal)

                st.markdown(f"### Decision: {decision.action}")
                st.write(decision.reason)

                trace = build_decision_trace(primary, signal, decision)
                with st.expander("Decision Trace"):
                    st.json(trace)

            colA, colB = st.columns(2)
            with colA:
                st.plotly_chart(create_damage_distribution_chart(detections), use_container_width=True)
            with colB:
                st.plotly_chart(create_severity_chart(detections), use_container_width=True)

        else:
            st.info("Upload and analyze an image to view results.")


if __name__ == "__main__":
    main()
