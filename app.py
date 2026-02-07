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

# Agentic imports
from agentic.trace import build_decision_trace
from agentic.decision_agent import DecisionAgent
from agentic.adapters import pick_primary_detection, detection_to_damage_signal
from agentic.explainer import build_customer_explanation, format_kb_insights
from agentic.strategies import build_repair_strategies, build_damage_story
from agentic.vision.after_inpaint import make_repaired_after_preview

# Custom modules (demo fallback)
try:
    from car_damage_detector import CarDamageDetector
    from utils import enhance_image, calculate_damage_stats
except ImportError:
    st.warning("Custom modules not found. Running in demo mode.")
    CarDamageDetector = None

# ---------------------- Page setup ----------------------
st.set_page_config(
    page_title="Car Damage Assessment AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- CSS Styling ----------------------
st.markdown(
    r"""
<style>
/* ... same CSS as before ... */
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------- Hero SVG ----------------------
def hero_svg_car_damage() -> str:
    svg = r"""<div style="border-radius:14px; border:1px solid rgba(255,255,255,0.10); ... </svg></div>"""
    return textwrap.dedent(svg).strip()

# ---------------------- Demo Damage Detection ----------------------
def demo_damage_detection(image: Image.Image):
    img_array = np.array(image)
    height, width = img_array.shape[:2]

    detections = [
        {"type": "Scratch", "severity": "Light", "confidence": 0.89,
         "bbox": [int(width*0.2), int(height*0.3), int(width*0.4), int(height*0.5)],
         "area_percentage": 2.5, "estimated_cost": 150},
        {"type": "Dent", "severity": "Moderate", "confidence": 0.76,
         "bbox": [int(width*0.6), int(height*0.2), int(width*0.8), int(height*0.4)],
         "area_percentage": 8.3, "estimated_cost": 450},
        {"type": "Paint Damage", "severity": "Light", "confidence": 0.82,
         "bbox": [int(width*0.1), int(height*0.6), int(width*0.25), int(height*0.8)],
         "area_percentage": 3.2, "estimated_cost": 200},
    ]

    img_with_annotations = img_array.copy()
    colors = {"Scratch": (0,255,0), "Dent": (255,165,0), "Paint Damage": (255,0,255), "Broken Part": (255,0,0)}

    return img_with_annotations, detections, colors

# ---------------------- Charts ----------------------
def create_damage_distribution_chart(detections):
    damage_counts = {}
    for d in detections:
        damage_counts[d["type"]] = damage_counts.get(d["type"],0)+1
    fig = px.pie(values=list(damage_counts.values()), names=list(damage_counts.keys()), title="Damage Type Distribution")
    fig.update_traces(textposition="inside", textinfo="percent+label", insidetextorientation='radial', sort=False)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="rgba(255,255,255,0.86)"), title_font=dict(size=18))
    return fig

def create_severity_chart(detections):
    severity_counts = {}
    for d in detections:
        severity_counts[d["severity"]] = severity_counts.get(d["severity"],0)+1
    colors = {"Light":"#cfcfcf","Moderate":"#9a9a9a","Severe":"#f2f2f2"}
    fig = go.Figure(data=[go.Bar(x=list(severity_counts.keys()), y=list(severity_counts.values()),
                                 marker_color=[colors.get(k,"#8a8a8a") for k in severity_counts.keys()])])
    fig.update_layout(title="Damage Severity Distribution", xaxis_title="Severity Level", yaxis_title="Number of Damages",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="rgba(255,255,255,0.86)"), title_font=dict(size=18))
    return fig

# ---------------------- Assessment Report ----------------------
def generate_assessment_report(detections, image_info):
    total_cost = sum([d.get("estimated_cost",0) for d in detections])
    total_area = sum([d["area_percentage"] for d in detections])
    avg_confidence = np.mean([d["confidence"] for d in detections])
    severity_priority = {"Severe":3, "Moderate":2, "Light":1}
    highest_severity = max([severity_priority.get(d["severity"],0) for d in detections])
    severity_names = {3:"Severe",2:"Moderate",1:"Light"}
    report = {
        "timestamp": datetime.now(),
        "image_dimensions": image_info,
        "total_damages": len(detections),
        "total_affected_area": total_area,
        "estimated_repair_cost": total_cost,
        "average_confidence": avg_confidence,
        "highest_severity": severity_names.get(highest_severity,"None"),
        "damage_breakdown": detections
    }
    return report

# ---------------------- Repaired Preview ----------------------
def make_repaired_preview(before_rgb: np.ndarray, primary: dict, intensity: float = 0.65):
    if before_rgb is None or primary is None:
        return {"after": before_rgb, "diff": None, "mask": None}
    before_bgr = before_rgb[:, :, ::-1].copy()
    try:
        res = make_repaired_after_preview(before_bgr, primary, intensity=float(intensity), method="cv")
        after_bgr = res.after_bgr
        diff_bgr = getattr(res, "diff_bgr", None)
        mask = getattr(res, "mask", None) or getattr(res, "mask_bgr", None) or getattr(res, "mask_gray", None)
        after_rgb = after_bgr[:, :, ::-1] if after_bgr is not None else None
        diff_rgb = diff_bgr[:, :, ::-1] if diff_bgr is not None else None
        if mask is None:
            mask_vis = None
        else:
            mask_vis = mask[:, :, ::-1] if mask.ndim==3 else mask
        return {"after": after_rgb, "diff": diff_rgb, "mask": mask_vis}
    except Exception as e:
        st.warning(f"Preview generation failed: {e}")
        return {"after": before_rgb, "diff": None, "mask": None}

# ---------------------- Hero UI ----------------------
def render_hero():
    left, right = st.columns([1.35,1])
    with left:
        st.markdown(
            f"""<div class="hero-wrap">
            <h1 class="hero-title">Car Damage Assessment AI</h1>
            <div class="hero-subtitle">High-trust vehicle damage intelligence. Computer vision detection and agentic decisioning.</div>
            <div class="tag-row"><span class="tag">Real-time inference</span><span class="tag">Policy workflow</span><span class="tag">Analytics</span></div></div>""",
            unsafe_allow_html=True
        )
    with right:
        st.markdown('<div class="hero-right">', unsafe_allow_html=True)
        st_html(hero_svg_car_damage(), height=255)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Main App ----------------------
def main():
    render_hero()

    # Sidebar
    with st.sidebar:
        st.markdown("## Configuration")
        st.markdown("---")
        developer_mode = st.checkbox("Developer mode (show debug)", value=False)
        st.session_state["dev_mode"] = developer_mode
        confidence_threshold = st.slider("Confidence Threshold", 0.1,1.0,0.5,0.05)
        damage_types = st.multiselect("Damage Types", ["Scratch","Dent","Broken Part","Paint Damage"],
                                      default=["Scratch","Dent","Paint Damage"])
        enhance_image_option = st.checkbox("Enable Image Enhancement", value=True)
        show_confidence = st.checkbox("Display Confidence Scores", value=True)
        generate_report = st.checkbox("Generate Assessment Report", value=True)
        st.markdown("---")

    col1, col2 = st.columns([1,1])

    # Left: Upload
    with col1:
        st.markdown("## Image Upload")
        uploaded_file = st.file_uploader("Select vehicle image", type=["png","jpg","jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_container_width=True)
            st.session_state.original_image = np.array(image)
            st.session_state.image_info = image.size

            if st.button("Analyze Damage"):
                import time
                with st.spinner("Processing..."):
                    time.sleep(1)
                    img_with_annotations, detections, colors = demo_damage_detection(image)

                    # ---- APPLY DAMAGE TYPE FILTER ----
                    filtered_detections = [d for d in detections if d["type"] in damage_types]

                    # ---- DRAW ANNOTATIONS ----
                    img_annotated = np.array(image).copy()
                    for d in filtered_detections:
                        x1, y1, x2, y2 = d["bbox"]
                        color = colors.get(d["type"], (255,255,0))
                        cv2.rectangle(img_annotated, (x1,y1), (x2,y2), color, 3)
                        if show_confidence:
                            label = f"{d['type']} ({d['confidence']:.2f})"
                        else:
                            label = f"{d['type']}"
                        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(img_annotated, (x1, y1-size[1]-10), (x1+size[0], y1), color, -1)
                        cv2.putText(img_annotated, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                    st.session_state.processed_image = img_annotated
                    st.session_state.detections = filtered_detections

                st.success(f"Found {len(filtered_detections)} damage areas.")

    # Right: Results
    with col2:
        st.markdown("## Analysis Results")
        if "detections" in st.session_state and st.session_state.detections:
            st.image(st.session_state.processed_image, caption="Detected Damage Areas", use_container_width=True)
            detections = st.session_state.detections

            # Charts
            st.plotly_chart(create_damage_distribution_chart(detections), use_container_width=True)
            st.plotly_chart(create_severity_chart(detections), use_container_width=True)

            # Report CSV & Repair Cost
            report = generate_assessment_report(detections, st.session_state.image_info)
            st.download_button("📥 Download Report CSV", data=pd.DataFrame([report["damage_breakdown"]]).to_csv(index=False),
                               file_name="damage_report.csv", mime="text/csv")
            st.info(f"💰 Estimated Total Repair Cost: ${report['estimated_repair_cost']}")

            # Continue with Agentic KB, decision trace, repair story etc...
            agent = DecisionAgent(policies_dir="policies")
            primary = pick_primary_detection(detections)

            if primary:
                signal = detection_to_damage_signal(primary)
                decision = agent.decide(signal)
                trace = build_decision_trace(primary_detection=primary, signal=signal, decision=decision)
                with st.expander("🧾 Decision Trace", expanded=True):
                    for i, step in enumerate(trace["steps"],1):
                        st.markdown(f"**{i}. {step['title']}**")
                        for d in step["details"]:
                            st.write(f"- {d}")

                expl = build_customer_explanation(
                    decision_action=decision.action,
                    decision_reason=decision.reason,
                    policy_refs=list(decision.policy_refs or []),
                    next_steps=list(decision.next_steps or []),
                    sop_text=getattr(decision,"evidence",None),
                    signal=signal
                )
                badge_emoji = {"AUTO_APPROVE":"✅","HUMAN_REVIEW":"⚠️","ESCALATE":"🔺"}.get(decision.action,"ℹ️")
                st.markdown(f"#### {badge_emoji} {decision.action}")
                st.caption(expl["summary"])
                if expl["why_bullets"]:
                    st.write("**Why:**")
                    for b in expl["why_bullets"]: st.write(f"- {b}")
                if expl["next_steps"]:
                    st.write("**Next steps:**")
                    for s in expl["next_steps"]: st.write(f"- {s}")

                # Before/After preview
                st.markdown("---")
                st.subheader("✨ Before / After Vision (Preview)")
                preview_intensity = st.slider("Preview intensity", 0.0,1.0,0.65,0.05)
                before_rgb = st.session_state.original_image
                preview = make_repaired_preview(before_rgb, primary, intensity=preview_intensity)
                cA, cB = st.columns(2)
                with cA: st.image(before_rgb, caption="Before", use_container_width=True)
                with cB: st.image(preview["after"], caption="After", use_container_width=True)
                if preview.get("diff") is not None: st.image(preview["diff"], caption="Diff map", use_container_width=True)
                if preview.get("mask") is not None:
                    with st.expander("Mask applied for inpainting"): st.image(preview["mask"], use_container_width=True)

                # Knowledge Base Insights
                q = f"{signal.get('damage_type','')} {signal.get('severity','')} repair guidance checklist risks"
                chunks = agent.retriever.retrieve(q, top_k=3)
                insights = format_kb_insights(chunks, max_items=4)
                if insights:
                    st.markdown("---")
                    st.subheader("📚 Knowledge Base Insights")
                    for item in insights: st.write(f"- {item}")

# ---------------------- Run App ----------------------
if __name__ == "__main__":
    main()
