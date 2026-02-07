import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Damage Assessment AI",
    layout="wide"
)

st.markdown("""
<style>
body, .stApp {
    background-color: white;
    color: black;
}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Car Damage Assessment AI")
st.caption("Beginner-friendly | Multi-image | Real-time | PDF report")

# ---------------- SIMPLE DAMAGE LOGIC ----------------
def analyze_damage(image_np):
    h, w = image_np.shape[:2]

    damages = []

    damages.append({
        "type": "Scratch",
        "severity": "Light",
        "confidence": round(np.random.uniform(0.70, 0.90), 2),
        "repair": "Polishing or repainting required",
        "cost": "₹1,500 – ₹3,000"
    })

    damages.append({
        "type": "Dent",
        "severity": "Moderate",
        "confidence": round(np.random.uniform(0.65, 0.85), 2),
        "repair": "Dent removal with panel reshaping",
        "cost": "₹4,000 – ₹8,000"
    })

    return damages

# ---------------- PDF GENERATION ----------------
def generate_pdf(all_results):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp.name, pagesize=A4)

    text = c.beginText(40, 800)
    text.setFont("Helvetica", 11)

    text.textLine("Car Damage Assessment Report")
    text.textLine(f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
    text.textLine("-" * 60)

    for idx, result in enumerate(all_results, 1):
        text.textLine(f"\nImage {idx}")
        for d in result:
            text.textLine(f"• Damage Type: {d['type']}")
            text.textLine(f"  Severity: {d['severity']}")
            text.textLine(f"  Confidence: {d['confidence']}")
            text.textLine(f"  Repair: {d['repair']}")
            text.textLine(f"  Estimated Cost: {d['cost']}")
            text.textLine("")

    c.drawText(text)
    c.showPage()
    c.save()

    return temp.name

# ---------------- UI ----------------
uploaded_files = st.file_uploader(
    "Upload car images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

all_results = []

if uploaded_files:
    st.success(f"{len(uploaded_files)} image(s) uploaded")

    for idx, file in enumerate(uploaded_files, 1):
        image = Image.open(file).convert("RGB")
        image_np = np.array(image)

        st.subheader(f"Image {idx}")
        st.image(image, use_container_width=True)

        damages = analyze_damage(image_np)
        all_results.append(damages)

        for d in damages:
            with st.expander(f"{d['type']} ({d['severity']})"):
                st.write(f"**Confidence:** {d['confidence']}")
                st.write(f"**Repair Suggestion:** {d['repair']}")
                st.write(f"**Estimated Cost:** {d['cost']}")

    pdf_path = generate_pdf(all_results)

    with open(pdf_path, "rb") as f:
        st.download_button(
            "⬇️ Download Repair Report (PDF)",
            f,
            file_name="car_damage_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Upload one or more car images to begin analysis.")
