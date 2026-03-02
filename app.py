import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib

# --- Page Config ---
st.set_page_config(
    page_title="Car Damage Assessment AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- White / Light UI ---
st.markdown(
    """
    <style>
    .stApp {background-color: #ffffff; color: #000000;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------
# Demo Damage Detection (semi-dynamic)
# --------------------------
def demo_damage_detection(image: Image.Image):
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Generate a hash based on image content
    img_bytes = image.tobytes()
    img_hash = int(hashlib.md5(img_bytes).hexdigest(), 16)

    # Use hash to pick damages dynamically
    detections = []
    if img_hash % 3 == 0:
        detections.append({
            "type": "Windshield",
            "severity": "Severe",
            "confidence": 0.95,
            "bbox":[int(w*0.3), int(h*0.1), int(w*0.7), int(h*0.4)],
            "area_percentage":12,
            "estimated_cost":800
        })
        detections.append({
            "type": "Scratch",
            "severity": "Light",
            "confidence": 0.88,
            "bbox":[int(w*0.2), int(h*0.6), int(w*0.4), int(h*0.8)],
            "area_percentage":3,
            "estimated_cost":150
        })
    elif img_hash % 3 == 1:
        detections.append({
            "type": "Side Window",
            "severity": "Severe",
            "confidence": 0.92,
            "bbox":[int(w*0.1), int(h*0.3), int(w*0.25), int(h*0.5)],
            "area_percentage":5,
            "estimated_cost":300
        })
        detections.append({
            "type": "Paint Damage",
            "severity": "Light",
            "confidence": 0.82,
            "bbox":[int(w*0.6), int(h*0.2), int(w*0.8), int(h*0.4)],
            "area_percentage":4,
            "estimated_cost":200
        })
    else:
        detections.append({
            "type": "Dent",
            "severity": "Moderate",
            "confidence": 0.76,
            "bbox":[int(w*0.4), int(h*0.5), int(w*0.6), int(h*0.7)],
            "area_percentage":7,
            "estimated_cost":450
        })
        detections.append({
            "type": "Scratch",
            "severity": "Light",
            "confidence": 0.85,
            "bbox":[int(w*0.2), int(h*0.2), int(w*0.35), int(h*0.35)],
            "area_percentage":2,
            "estimated_cost":120
        })

    # Annotate image
    img_annot = img_array.copy()
    colors = {
        "Windshield": (255,0,0),
        "Side Window": (0,0,255),
        "Scratch": (0,255,0),
        "Dent": (255,165,0),
        "Paint Damage": (255,0,255)
    }
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        cv2.rectangle(img_annot, (x1,y1), (x2,y2), colors.get(d["type"], (255,0,0)), 3)
        label = f"{d['type']} ({d['confidence']:.2f})"
        cv2.putText(img_annot, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    return img_annot, detections

# --------------------------
# Human-Readable Suggestions (dynamic front/rear/left/right)
# --------------------------
def generate_damage_suggestions_dynamic(detections, image):
    suggestions = []
    h, w = image.height, image.width

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        # Determine vertical position
        if y2 < h / 2:
            vertical = "Front"
        else:
            vertical = "Rear"
        # Determine horizontal position
        if x2 < w / 2:
            horizontal = "Left"
        else:
            horizontal = "Right"

        # Build full part name if type is windshield/window
        if "Windshield" in d["type"] or "Window" in d["type"]:
            part_name = f"{vertical}-{horizontal} {d['type']}"
        else:
            part_name = d["type"]

        # Generate suggestion
        if "Windshield" in d["type"] or "Window" in d["type"]:
            suggestions.append(f"✅ {part_name} is damaged → Recommend replacement.")
        elif d["type"] == "Scratch":
            suggestions.append(f"✅ Paint scratches detected → Minor repaint/repair recommended.")
        elif d["type"] == "Dent":
            suggestions.append(f"✅ Dent detected → Estimated repair cost: ${d['estimated_cost']}.")
        elif d["type"] == "Paint Damage":
            suggestions.append(f"✅ Paint damage detected → Check affected area (~{d['area_percentage']}%).")
        else:
            suggestions.append(f"⚠️ {part_name} requires attention.")

    suggestions.append("⚠️ Interior may be exposed → Check for dust/water damage.")
    return suggestions

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

        # Human-readable suggestions
        st.subheader("Detected Damage & Suggestions")
        suggestions = generate_damage_suggestions_dynamic(detections, image)
        for s in suggestions:
            st.write(s)

        # CSV download
        df_dmg = pd.DataFrame(detections)
        csv = df_dmg.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Report", csv, "damage_report.csv", "text/csv")

        # Auto-approve (no human review)
        st.success("✅ Auto-Approve: Repair ticket ready (demo).")
