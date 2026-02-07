import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from fpdf import FPDF
import tempfile
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Damage Assessment AI",
    page_icon="🚗",
    layout="wide"
)

# ---------------- WHITE THEME ----------------
st.markdown("""
<style>
.stApp {
    background-color: white;
    color: black;
}
h1, h2, h3 {
    color: black;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🚗 Car Damage Assessment AI")
st.write("Upload **multiple car images**, analyze damage, and download **repair suggestions as PDF**.")

# ---------------- SIMPLE DAMAGE LOGIC ----------------
def analyze_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    if edge_density < 0.02:
        severity = "Light"
        damage_type = random.choice(["Minor Scratch", "Paint Fade"])
        cost = random.randint(100, 300)
    elif edge_density < 0.05:
        severity = "Moderate"
        damage_type = random.choice(["Dent", "Deep Scratch"])
        cost = random.randint(400, 800)
    else:
        severity = "Severe"
        damage_type = random.choice(["Body Damage", "Broken Panel"])
        cost = random.randint(1000, 2000)

    suggestion_map = {
        "Minor Scratch": "Polishing and paint touch-up recommended.",
        "Paint Fade": "Repainting of affected area suggested.",
        "Dent": "Dent removal and repainting required.",
        "Deep Scratch": "Sanding and repainting required.",
        "Body Damage": "Panel replacement may be required.",
        "Broken Panel": "Immediate replacement advised."
    }

    return {
        "damage_type": damage_type,
        "severity": severity,
        "estimated_cost": cost,
        "suggestion": suggestion_map[damage_type]
    }

# ---------------- PDF GENERATOR ----------------
def generate_pdf(report_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Car Damage Assessment Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now()}", ln=True)
    pdf.ln(5)

    for idx, item in enumerate(report_data, 1):
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Image {idx}", ln=True)

        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, 
            f"Damage Type: {item['damage_type']}\n"
            f"Severity: {item['severity']}\n"
            f"Estimated Repair Cost: ${item['estimated_cost']}\n"
            f"Repair Suggestion: {item['suggestion']}"
        )
        pdf.ln(4)

    file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf.output(file_path)
    return file_path

# ---------------- IMAGE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload car images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

results = []

if uploaded_files:
    st.subheader("🔍 Analysis Results")

    for i, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")
        result = analyze_image(image)
        results.append(result)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption=f"Image {i+1}", use_container_width=True)

        with col2:
            st.markdown(f"""
            **Damage Type:** {result['damage_type']}  
            **Severity:** {result['severity']}  
            **Estimated Cost:** ${result['estimated_cost']}  
            **Suggestion:** {result['suggestion']}
            """)

    # ---------------- PDF DOWNLOAD ----------------
    st.markdown("---")
    st.subheader("📄 Download Repair Report")

    pdf_path = generate_pdf(results)

    with open(pdf_path, "rb") as f:
        st.download_button(
            label="⬇️ Download PDF Report",
            data=f,
            file_name="car_damage_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Upload one or more car images to start analysis.")
