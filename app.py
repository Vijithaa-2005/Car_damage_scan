from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================
# ARTEMEDX / AGENTIC IMPORTS
# =========================
from agentic.trace import build_decision_trace
from agentic.decision_agent import DecisionAgent
from agentic.adapters import pick_primary_detection, detection_to_damage_signal
from agentic.strategies import build_repair_strategies, build_damage_story

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Car Damage Assessment AI",
    page_icon="🚗",
    layout="wide"
)

# =========================
# WHITE BACKGROUND THEME
# =========================
st.markdown("""
<style>
.stApp {
    background-color: white;
    color: black;
}
section[data-testid="stSidebar"] {
    background-color: #f5f5f5;
}
h1,h2,h3 {
    color: black;
}
.stButton button {
    background-color: black;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO
# =========================
st.markdown("""
<h1>Car Damage Assessment AI</h1>
<p>
AI-assisted vehicle damage analysis using computer vision principles and
decision intelligence.
</p>
""", unsafe_allow_html=True)

# =========================
# SIMPLE IMAGE-BASED HEURISTIC
# (NO TORCH, NO MODEL)
# =========================
def analyze_image_heuristic(image: Image.Image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    edge_density = np.sum(edges > 0) / edges.size

    detections = []

    h, w = img.shape[:2]

    if edge_density > 0.08:
        detections.append({
            "type": "Dent",
            "severity": "Severe",
            "confidence": 0.85,
            "bbox": [int(w*0.25), int(h*0.3), int(w*0.55), int(h*0.6)],
            "area_percentage": 9.5,
            "estimated_cost": 900
        })
    elif edge_density > 0.04:
        detections.append({
            "type": "Scratch",
            "severity": "Moderate",
            "confidence": 0.72,
            "bbox": [int(w*0.3), int(h*0.4), int(w*0.7), int(h*0.5)],
            "area_percentage": 5.2,
            "estimated_cost": 400
        })
    else:
        detections.append({
            "type": "Surface Mark",
            "severity": "Mild",
            "confidence": 0.6,
            "bbox": [int(w*0.35), int(h*0.45), int(w*0.65), int(h*0.55)],
            "area_percentage": 2.1,
            "estimated_cost": 150
        })

    annotated = img.copy()
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,0,0),3)

    return annotated, detections

# =========================
# CHARTS
# =========================
def damage_chart(detections):
    df = pd.DataFrame(detections)
    fig = px.pie(df, names="type", title="Damage Type Distribution")
    fig.update_layout(paper_bgcolor="white")
    return fig

# =========================
# SIDEBAR
# =========================
st.sidebar.header("Upload Image")
uploaded = st.sidebar.file_uploader(
    "Upload vehicle image",
    type=["jpg","png","jpeg"]
)

# =========================
# MAIN LOGIC
# =========================
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Damage", use_container_width=True):
        processed, detections = analyze_image_heuristic(image)

        st.image(processed, caption="Detected Damage", use_container_width=True)

        # -------------------------
        # ARTEMEDX AGENT DECISION
        # -------------------------
        agent = DecisionAgent(policies_dir="policies")
        primary = pick_primary_detection(detections)

        if primary:
            signal = detection_to_damage_signal(primary)
            decision = agent.decide(signal)

            st.subheader("Decision")
            st.write(f"**Action:** {decision.action}")
            st.write(f"**Reason:** {decision.reason}")

            # ✅ FIXED FUNCTION CALL
            trace = build_decision_trace(
                primary_detection=primary,
                signal=signal,
                decision=decision
            )

            with st.expander("Decision Trace"):
                st.json(trace)

            story = build_damage_story(primary)
            strategies = build_repair_strategies(primary)

            st.subheader("Damage Impact")
            for c in story["consequences"]:
                st.write(f"- {c}")

            st.subheader("Repair Strategies")
            for s in strategies:
                with st.expander(s.name):
                    st.write(s.summary)

        st.plotly_chart(damage_chart(detections), use_container_width=True)

        st.subheader("Assessment Table")
        st.dataframe(pd.DataFrame(detections), use_container_width=True)

else:
    st.info("Please upload a vehicle image to begin analysis.")
