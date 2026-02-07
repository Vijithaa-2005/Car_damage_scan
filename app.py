import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Artemedx-style imports
from artemedx.models.damage_classifier import DamageClassifier
from artemedx.policy.engine import PolicyEngine
from artemedx.policy.loader import load_policies

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Damage Scan",
    layout="centered"
)

# White background
st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL & POLICY ----------------
@st.cache_resource
def load_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DamageClassifier()
    model.load_pretrained()
    model.to(device)
    model.eval()

    policies = load_policies("policies/")
    policy_engine = PolicyEngine(policies)

    return model, policy_engine, device

model, policy_engine, device = load_system()

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------- ANALYSIS FUNCTION ----------------
def analyze_image(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)

    severity = float(probs.max().item())
    damage_class = int(torch.argmax(probs).item())

    damage_map = {
        0: "No Damage",
        1: "Scratch",
        2: "Dent",
        3: "Broken Part"
    }

    prediction = {
        "damage_type": damage_map.get(damage_class, "Unknown"),
        "severity": round(severity, 2)
    }

    decision = policy_engine.evaluate(prediction)

    return prediction, decision

# ---------------- UI ----------------
st.title("🚗 Car Damage Detection System")
st.write("Upload a car image to detect damage using AI and policy rules.")

uploaded = st.file_uploader(
    "Upload car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        prediction, decision = analyze_image(image)

    st.subheader("🔍 Damage Analysis")
    st.write(f"**Damage Type:** {prediction['damage_type']}")
    st.write(f"**Severity Score:** {prediction['severity']}")

    st.subheader("📋 Policy Decision")
    st.write(f"**Risk Level:** {decision['risk_level']}")
    st.write(f"**Recommendation:** {decision['recommendation']}")
