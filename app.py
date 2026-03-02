import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Car Damage Assessment AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- WHITE / LIGHT UI ---
st.markdown(
    """
    <style>
    .stApp {background-color: #ffffff; color: #000000;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Demo Damage Detection
# --------------------------
def demo_damage_detection(image: Image.Image):
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    detections = [
        {"type": "Scratch", "severity": "Light", "confidence": 0.89, "bbox":[int(w*0.2), int(h*0.3), int(w*0.4), int(h*0.5)], "area_percentage":2.5, "estimated_cost":150},
        {"type": "Dent", "severity": "Moderate", "confidence": 0.76, "bbox":[int(w*0.6), int(h*0.2), int(w*0.8), int(h*0.4)], "area_percentage":8.3, "estimated_cost":450},
        {"type": "Paint Damage", "severity": "Light", "confidence": 0.82, "bbox":[int(w*0.1), int(h*0.6), int(w*0.25), int(h*0.8)], "area_percentage":3.2, "estimated_cost":200},
    ]

    img_annot = img_array.copy()
    colors = {"Scratch": (0,255,0), "Dent": (255,165,0), "Paint Damage": (255,0,255)}
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        cv2.rectangle(img_annot, (x1,y1), (x2,y2), colors.get(d["type"], (255,0,0)), 3)
        label = f"{d['type']} ({d['confidence']:.2f})"
        cv2.putText(img_annot, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    return img_annot, detections

# --------------------------
# Charts
# --------------------------
def create_damage_distribution_chart(detections):
    damage_counts = {}
    for d in detections: damage_counts[d["type"]] = damage_counts.get(d["type"],0)+1
    fig = px.pie(values=list(damage_counts.values()), names=list(damage_counts.keys()), title="Damage Type Distribution")
    fig.update_traces(textinfo='percent+label')
    return fig

def create_severity_chart(detections):
    severity_counts = {}
    for d in detections: severity_counts[d["severity"]] = severity_counts.get(d["severity"],0)+1
    fig = go.Figure([go.Bar(x=list(severity_counts.keys()), y=list(severity_counts.values()), marker_color=["#cfcfcf","#9a9a9a","#f2f2f2"])])
    fig.update_layout(title="Damage Severity Distribution", xaxis_title="Severity", yaxis_title="Count")
    return fig

# --------------------------
# Assessment Report
# --------------------------
def generate_report(detections, image: Image.Image):
    total_cost = sum([d["estimated_cost"] for d in detections])
    total_area = sum([d["area_percentage"] for d in detections])
    avg_conf = np.mean([d["confidence"] for d in detections])
    highest_sev = max(detections, key=lambda x: {"Light":1,"Moderate":2,"Severe":3}.get(x["severity"],0))["severity"]

    report = {
        "Timestamp": datetime.now(),
        "Image Width": image.width,
        "Image Height": image.height,
        "Total Damages": len(detections),
        "Total Area %": total_area,
        "Estimated Cost ($)": total_cost,
        "Average Confidence": avg_conf,
        "Highest Severity": highest_sev
    }
    return report

# --------------------------
# Human Review / Decisions
# --------------------------
def render_decision_ui():
    st.markdown("---")
    st.subheader("Decision Actions")
    decision = st.radio("Select Decision Type:", ["Auto-Approve", "Human Review", "Escalate"])
    if decision == "Auto-Approve":
        st.success("✅ Auto-Approve: Ready to create repair ticket.")
        if st.button("Create Repair Ticket"): st.info("Repair ticket created (demo).")
    elif decision == "Human Review":
        st.warning("⚠️ Human Review: Complete checklist.")
        c1 = st.checkbox("Verify Vehicle ID / VIN")
        c2 = st.checkbox("Confirm severity & area")
        c3 = st.checkbox("Request additional images")
        c4 = st.checkbox("Check repair vs replacement")
        ready = all([c1,c2,c3,c4])
        notes = st.text_area("Operator Notes", placeholder="Write notes for reviewer...")
        if st.button("Submit for Review", disabled=not ready): st.info("Submitted for review (demo).")
    else:
        st.error("🚨 Escalate to specialist assessor.")
        if st.button("Assign Senior Assessor"): st.info("Assigned to senior assessor (demo).")

# --------------------------
# Main UI
# --------------------------
st.title("Car Damage Assessment AI")
st.markdown("Upload a vehicle image for analysis.")

uploaded_file = st.file_uploader("Select image", type=["png","jpg","jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Analyze Damage"):
        processed_img, detections = demo_damage_detection(image)
        st.image(processed_img, caption="Damage Annotated", use_column_width=True)
        st.success(f"Detected {len(detections)} damage areas.")

        # Charts
        st.plotly_chart(create_damage_distribution_chart(detections), use_container_width=True)
        st.plotly_chart(create_severity_chart(detections), use_container_width=True)

        # Report
        report = generate_report(detections, image)
        st.table(pd.DataFrame([report]))

        # CSV download
        df_dmg = pd.DataFrame(detections)
        csv = df_dmg.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Report", csv, "damage_report.csv", "text/csv")

        # Human Review / Decision UI
        render_decision_ui()
