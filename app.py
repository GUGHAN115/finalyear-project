import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Fix Anaconda OpenMP conflict on Windows

"""
app.py — Streamlit interactive dashboard for Robust Deepfake Detection.

Features:
  • Glassmorphic dark-mode UI with animated gradient accents
  • Static image verification: upload -> face detection -> verdict + bbox visualisation
  • Video sequence analysis: upload -> equidistant sampling -> temporal charts
  • Diagnostic panel: forensic feature heatmap + stage-level confidence bars
  • Model info sidebar with architecture summary
"""

import io
import os
import sys
import tempfile
import numpy as np
import cv2
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.inference import load_model, infer_image, infer_video, get_device
from src.preprocessor import preprocess_image

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DeepGuard — Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Glassmorphic dark theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #0a0c14;
    color: #e2e8f0;
}

/* ── Animated gradient title ─────────────────────────────────────────── */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 40%, #c084fc 75%, #f472b6 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradShift 4s ease infinite;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
@keyframes gradShift {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}
.hero-sub {
    font-size: 1.05rem;
    color: #64748b;
    font-weight: 400;
    letter-spacing: 0.02em;
    margin-bottom: 2.5rem;
}

/* ── Verdict badge ───────────────────────────────────────────────────── */
.verdict-real {
    display: inline-block;
    background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(16,185,129,0.06));
    border: 1px solid rgba(16,185,129,0.45);
    border-radius: 16px;
    padding: 18px 36px;
    font-size: 2.2rem;
    font-weight: 800;
    color: #34d399;
    letter-spacing: 0.08em;
    box-shadow: 0 0 30px rgba(16,185,129,0.12);
    animation: pulseGreen 2.5s ease-in-out infinite;
}
.verdict-fake {
    display: inline-block;
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(239,68,68,0.06));
    border: 1px solid rgba(239,68,68,0.45);
    border-radius: 16px;
    padding: 18px 36px;
    font-size: 2.2rem;
    font-weight: 800;
    color: #f87171;
    letter-spacing: 0.08em;
    box-shadow: 0 0 30px rgba(239,68,68,0.12);
    animation: pulseRed 2.5s ease-in-out infinite;
}
.verdict-unknown {
    display: inline-block;
    background: rgba(100,116,139,0.12);
    border: 1px solid rgba(100,116,139,0.35);
    border-radius: 16px;
    padding: 18px 36px;
    font-size: 2.2rem;
    font-weight: 800;
    color: #94a3b8;
    letter-spacing: 0.08em;
}
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 18px rgba(16,185,129,0.10); }
    50%       { box-shadow: 0 0 42px rgba(16,185,129,0.28); }
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 18px rgba(239,68,68,0.10); }
    50%       { box-shadow: 0 0 42px rgba(239,68,68,0.28); }
}

/* ── Glass cards ─────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(129,140,248,0.28);
    box-shadow: 0 8px 40px rgba(129,140,248,0.08);
}

/* ── Stat chips ──────────────────────────────────────────────────────── */
.stat-chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 12px 18px;
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    min-width: 110px;
}
.stat-val  { font-size: 1.5rem; font-weight: 700; color: #e2e8f0; }
.stat-lbl  { font-size: 0.72rem; color: #64748b; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Section headers ─────────────────────────────────────────────────── */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

/* ── Confidence bar ──────────────────────────────────────────────────── */
.conf-bar-wrap { margin-bottom: 10px; }
.conf-label {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
}
.conf-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

/* ── Feature vector heatmap label ────────────────────────────────────── */
.feat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #64748b;
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(10,12,20,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
.sidebar-logo {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.arch-item {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #818cf8;
    border-radius: 0 10px 10px 0;
    padding: 8px 12px;
    margin-bottom: 8px;
    font-size: 0.82rem;
    color: #94a3b8;
}
.arch-item b { color: #c4b5fd; font-weight: 600; }

/* ── Upload zone ─────────────────────────────────────────────────────── */
.stFileUploader > div { background: rgba(255,255,255,0.03) !important; border: 2px dashed rgba(129,140,248,0.35) !important; border-radius: 16px !important; }
.stFileUploader > div:hover { border-color: rgba(129,140,248,0.70) !important; background: rgba(129,140,248,0.04) !important; }

/* ── Streamlit overrides ─────────────────────────────────────────────── */
.stButton>button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.6rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(79,70,229,0.30) !important;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(79,70,229,0.45) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom-color: #818cf8 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CACHE
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading DeepfakeDetector model…")
def get_model():
    checkpoint = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
    device = get_device()
    model, device = load_model(
        checkpoint_path=checkpoint if os.path.isfile(checkpoint) else None,
        pretrained=True,
        device=device,
    )
    return model, device


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛡️ DeepGuard</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.8rem;margin-bottom:1.5rem;">Robust Deepfake Detection System</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧠 Architecture")

    arch_items = [
        ("Backbone", "ResNet-50 (ImageNet pretrained)"),
        ("Stage 1 tap", "layer1 — 256ch, 56×56<br>Low-level edges & blending"),
        ("Stage 2 tap", "layer2 — 512ch, 28×28<br>Facial geometry & landmarks"),
        ("Stage 4 tap", "layer4 — 2048ch, 7×7<br>Semantics & illumination"),
        ("Fusion", "GAP → [256,256,512] → concat → 1024-d"),
        ("Head", "BN → FC(512) → ReLU → Dropout(0.4) → FC(2)"),
    ]
    for name, desc in arch_items:
        st.markdown(
            f'<div class="arch-item"><b>{name}</b><br>{desc}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    margin = st.slider(
        "Face bbox margin", 0.05, 0.40, 0.15, 0.01,
        help="Fractional expansion of detected face bounding box"
    )
    n_frames = st.slider(
        "Video frames to sample", 8, 64, 32, 4,
        help="Number of equidistant frames extracted for video analysis"
    )

    st.markdown("---")
    device_label = "CUDA GPU" if torch.cuda.is_available() else \
                   ("Apple MPS" if torch.backends.mps.is_available() else "CPU")
    st.markdown(f"**Device:** `{device_label}`")
    st.markdown(f"**PyTorch:** `{torch.__version__}`")
    st.markdown(f"**OpenCV:** `{cv2.__version__}`")


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🛡️ DeepGuard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Multi-level forensic deepfake detection · ResNet-50 backbone · Real-time inference</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
try:
    model, device = get_model()
    model_loaded = True
    model_status = "✅ Model loaded"
except Exception as e:
    model_loaded = False
    model_status = f"⚠️ Model error: {e}"

st.caption(model_status)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def bgr_to_rgb_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def make_confidence_bar(label: str, value: float, color: str) -> str:
    pct = value * 100
    return f"""
    <div class="conf-bar-wrap">
      <div class="conf-label">
        <span>{label}</span><span>{pct:.1f}%</span>
      </div>
      <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
      </div>
    </div>
    """


def render_verdict(label: str, confidence: float):
    cls = {"REAL": "verdict-real", "FAKE": "verdict-fake"}.get(label, "verdict-unknown")
    icon = {"REAL": "✅", "FAKE": "🚨"}.get(label, "❓")
    st.markdown(
        f'<div class="{cls}">{icon} {label} &nbsp; <span style="font-size:1.2rem;opacity:0.8;">{confidence*100:.1f}%</span></div>',
        unsafe_allow_html=True,
    )


def forensic_heatmap(vector: np.ndarray) -> go.Figure:
    """Reshape 1024-d vector into a 2D heatmap for diagnostic visualisation."""
    v = vector.reshape(32, 32)
    fig = go.Figure(go.Heatmap(
        z=v,
        colorscale=[
            [0.0, "#0f172a"], [0.3, "#1e40af"],
            [0.6, "#7c3aed"], [0.8, "#db2777"],
            [1.0, "#f97316"],
        ],
        showscale=True,
        colorbar=dict(
            title="Activation",
            titlefont=dict(color="#64748b", size=11),
            tickfont=dict(color="#64748b"),
            thickness=12,
        ),
    ))
    fig.update_layout(
        title=dict(text="Forensic Feature Vector (1024-d → 32×32)", font=dict(color="#94a3b8", size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=280,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    return fig


def timeline_chart(frame_results: list) -> go.Figure:
    """Temporal fake-probability timeline for video analysis."""
    frames     = [r["frame_index"] for r in frame_results if r["face_detected"]]
    fake_probs = [r["prob_fake"]   for r in frame_results if r["face_detected"]]

    fig = go.Figure()
    # Fill area
    fig.add_trace(go.Scatter(
        x=frames, y=fake_probs,
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.10)",
        line=dict(color="#f87171", width=2.5),
        mode="lines+markers",
        marker=dict(
            size=7,
            color=fake_probs,
            colorscale=[[0,"#34d399"],[0.5,"#fbbf24"],[1,"#f87171"]],
            cmin=0, cmax=1,
            showscale=False,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
        ),
        name="P(FAKE)",
        hovertemplate="Frame %{x}<br>P(FAKE): %{y:.3f}<extra></extra>",
    ))
    # Decision threshold line
    fig.add_hline(
        y=0.5, line_dash="dash",
        line_color="rgba(251,191,36,0.6)",
        annotation_text="Decision threshold (0.50)",
        annotation_font_color="#fbbf24",
        annotation_font_size=11,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Temporal Fake-Probability Timeline", font=dict(color="#94a3b8", size=13)),
        xaxis=dict(title="Sampled Frame Index", color="#64748b", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="P(FAKE)", range=[0, 1], color="#64748b", gridcolor="rgba(255,255,255,0.05)"),
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(font=dict(color="#94a3b8")),
        showlegend=True,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_img, tab_vid, tab_about = st.tabs([
    "📷 Image Analysis",
    "🎬 Video Analysis",
    "ℹ️ About & Architecture",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — IMAGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_img:
    st.markdown('<div class="section-header">Upload an Image for Deepfake Verification</div>', unsafe_allow_html=True)

    uploaded_img = st.file_uploader(
        "Drop an image (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key="img_uploader",
        label_visibility="collapsed",
    )

    if uploaded_img is not None:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("❌ Could not decode the uploaded image.")
        elif not model_loaded:
            st.error("❌ Model is not loaded. Check the sidebar for details.")
        else:
            with st.spinner("🔍 Detecting face and running inference…"):
                result = infer_image(img_bgr, model, device, margin=margin)

            # ── Layout: 3 columns ─────────────────────────────────────────
            col_orig, col_ann, col_results = st.columns([1, 1, 1.2], gap="large")

            with col_orig:
                st.markdown('<div class="section-header">Original</div>', unsafe_allow_html=True)
                st.image(bgr_to_rgb_pil(img_bgr), use_container_width=True)

            with col_ann:
                st.markdown('<div class="section-header">Detection</div>', unsafe_allow_html=True)
                if result["annotated_bgr"] is not None:
                    st.image(bgr_to_rgb_pil(result["annotated_bgr"]), use_container_width=True)
                    if result["face_crop_rgb"] is not None:
                        with st.expander("Face crop (224×224)"):
                            st.image(result["face_crop_rgb"], width=224)
                else:
                    st.warning("⚠️ No face detected in the image.")
                    st.image(bgr_to_rgb_pil(img_bgr), use_container_width=True)

            with col_results:
                st.markdown('<div class="section-header">Verdict</div>', unsafe_allow_html=True)
                render_verdict(result["label"], result["confidence"])

                st.markdown("<br>", unsafe_allow_html=True)

                # Confidence bars
                real_color = "#34d399"
                fake_color = "#f87171"
                st.markdown(
                    make_confidence_bar("P(REAL)", result["prob_real"], real_color) +
                    make_confidence_bar("P(FAKE)", result["prob_fake"], fake_color),
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Bounding box info
                if result["bbox"]:
                    x1, y1, x2, y2 = result["bbox"]
                    st.markdown(
                        f"""<div class="glass-card" style="font-size:0.82rem;color:#94a3b8;">
                        <b style="color:#c4b5fd;">Face Region</b><br>
                        Top-left: ({x1}, {y1})<br>
                        Bottom-right: ({x2}, {y2})<br>
                        Size: {x2-x1} × {y2-y1} px
                        </div>""",
                        unsafe_allow_html=True,
                    )

            # ── Forensic diagnostic ───────────────────────────────────────
            st.markdown("---")
            diag_col1, diag_col2 = st.columns([1.5, 1], gap="large")

            with diag_col1:
                st.plotly_chart(
                    forensic_heatmap(result["forensic_vector"]),
                    use_container_width=True,
                )

            with diag_col2:
                st.markdown('<div class="section-header">Stage Contributions</div>', unsafe_allow_html=True)
                vec = result["forensic_vector"]
                # Split into stage windows: low[0:256], mid[256:512], high[512:1024]
                low_energy  = float(np.linalg.norm(vec[:256]))
                mid_energy  = float(np.linalg.norm(vec[256:512]))
                high_energy = float(np.linalg.norm(vec[512:]))
                total = low_energy + mid_energy + high_energy + 1e-9

                stages = [
                    ("Stage 1 — Edges / Blending",     low_energy  / total, "#38bdf8"),
                    ("Stage 2 — Geometry / Landmarks",  mid_energy  / total, "#818cf8"),
                    ("Stage 4 — Semantics / Illumin.",  high_energy / total, "#c084fc"),
                ]
                html_bars = ""
                for name, val, color in stages:
                    html_bars += make_confidence_bar(name, val, color)
                st.markdown(html_bars, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f"""<div class="glass-card" style="font-size:0.82rem;">
                    <b style="color:#c4b5fd;">Forensic Vector Stats</b><br>
                    <span style="color:#64748b;">
                    Norm: {np.linalg.norm(vec):.2f}<br>
                    Mean: {vec.mean():.4f}<br>
                    Std: {vec.std():.4f}<br>
                    Max: {vec.max():.4f}
                    </span>
                    </div>""",
                    unsafe_allow_html=True,
                )

    else:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#334155;">
            <div style="font-size:4rem;margin-bottom:1rem;">🔍</div>
            <div style="font-size:1.2rem;font-weight:600;">Upload an image to begin analysis</div>
            <div style="font-size:0.9rem;margin-top:0.5rem;">Supports JPG, PNG, WEBP</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — VIDEO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_vid:
    st.markdown('<div class="section-header">Upload a Video for Temporal Deepfake Analysis</div>', unsafe_allow_html=True)

    uploaded_vid = st.file_uploader(
        "Drop a video (MP4 / AVI / MOV / WEBM)",
        type=["mp4", "avi", "mov", "webm", "mkv"],
        key="vid_uploader",
        label_visibility="collapsed",
    )

    if uploaded_vid is not None:
        if not model_loaded:
            st.error("❌ Model is not loaded.")
        else:
            # Save to temp file
            suffix = os.path.splitext(uploaded_vid.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_vid.read())
                tmp_path = tmp.name

            st.video(uploaded_vid)

            with st.spinner(f"🎞️ Sampling {n_frames} frames and running temporal inference…"):
                try:
                    vid_result = infer_video(
                        tmp_path, model, device,
                        n_frames=n_frames, margin=margin,
                    )
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Video processing failed: {e}")
                    os.unlink(tmp_path)
                    st.stop()

            # ── Top-level verdict row ─────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            vc1, vc2, vc3, vc4 = st.columns(4, gap="medium")

            with vc1:
                render_verdict(vid_result["verdict"], vid_result["mean_fake_prob"])

            def stat_chip(val: str, lbl: str) -> str:
                return f'<div class="stat-chip"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>'

            with vc2:
                st.markdown(stat_chip(
                    f"{vid_result['mean_fake_prob']*100:.1f}%", "Mean P(FAKE)"
                ), unsafe_allow_html=True)

            with vc3:
                st.markdown(stat_chip(
                    f"{vid_result['peak_fake_prob']*100:.1f}%", "Peak Anomaly"
                ), unsafe_allow_html=True)

            with vc4:
                st.markdown(stat_chip(
                    f"{vid_result['faces_detected']}/{vid_result['total_sampled']}",
                    "Faces Detected"
                ), unsafe_allow_html=True)

            # ── Timeline chart ────────────────────────────────────────────
            st.markdown("---")
            st.plotly_chart(
                timeline_chart(vid_result["frame_results"]),
                use_container_width=True,
            )

            # ── Per-frame breakdown ───────────────────────────────────────
            with st.expander("🔎 Per-frame Breakdown"):
                import pandas as pd
                df_frames = pd.DataFrame([
                    {
                        "Frame #":       r["frame_index"],
                        "Label":         r["label"],
                        "Confidence":    f"{r['confidence']*100:.1f}%",
                        "P(FAKE)":       f"{r['prob_fake']*100:.2f}%",
                        "P(REAL)":       f"{r['prob_real']*100:.2f}%",
                        "Face Detected": "✅" if r["face_detected"] else "❌",
                    }
                    for r in vid_result["frame_results"]
                ])
                st.dataframe(df_frames, use_container_width=True, hide_index=True)

            # ── Video meta ────────────────────────────────────────────────
            st.markdown(
                f"""<div class="glass-card" style="font-size:0.82rem;color:#64748b;margin-top:12px;">
                <b style="color:#c4b5fd;">Video Metadata</b><br>
                Total frames: <b style="color:#94a3b8;">{vid_result['video_total_frames']}</b> &nbsp;|&nbsp;
                FPS: <b style="color:#94a3b8;">{vid_result['video_fps']:.1f}</b> &nbsp;|&nbsp;
                Frames sampled: <b style="color:#94a3b8;">{vid_result['total_sampled']}</b> &nbsp;|&nbsp;
                Peak anomaly @ frame: <b style="color:#f87171;">{vid_result['peak_frame_index']}</b>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#334155;">
            <div style="font-size:4rem;margin-bottom:1rem;">🎬</div>
            <div style="font-size:1.2rem;font-weight:600;">Upload a video to begin temporal analysis</div>
            <div style="font-size:0.9rem;margin-top:0.5rem;">Supports MP4, AVI, MOV, WEBM</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT & ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    col_a, col_b = st.columns([1.2, 1], gap="large")

    with col_a:
        st.markdown("## 🧠 System Architecture")
        st.markdown("""
        **DeepGuard** is a hierarchical multi-level deepfake detection system built on a
        ResNet-50 backbone pretrained on ImageNet.

        ### Feature Extraction Pipeline
        | Stage | Layer | Resolution | Channels | Captures |
        |-------|-------|-----------|----------|---------|
        | 1 | `layer1` | 56 × 56 | 256 | Edge artifacts, blending boundaries |
        | 2 | `layer2` | 28 × 28 | 512 | Facial geometry, landmark patterns |
        | 4 | `layer4` | 7 × 7 | 2048 | Semantics, illumination coherence |

        ### Feature Fusion
        1. **GAP** (Global Average Pooling) applied to each stage output
        2. **Projection** through dedicated linear layers → `[256, 256, 512]`
        3. **Concatenation** → `1024-d` fused forensic vector
        4. **Classifier head**: `BN → FC(512) → ReLU → Dropout(0.4) → FC(2)`

        ### Preprocessing
        - Face detection via **OpenCV Haar Cascade** (frontalface_default)
        - Bounding box expanded by **15% margin** on all sides
        - Crop resized to **224 × 224** and normalised with ImageNet statistics

        ### Video Inference
        - **Equidistant frame sampling** (configurable, default: 32 frames)
        - **Temporal aggregation**: mean P(FAKE) as primary verdict signal
        - **Peak anomaly tracking**: identifies highest-risk individual frame
        """)

    with col_b:
        st.markdown("## 📐 Data Flow")
        st.code("""
Input Image (H×W×3)
    │
    ▼
OpenCV Haar Cascade
    │ face ROI + 15% margin
    ▼
Resize → 224×224
    │ ImageNet normalisation
    ▼
┌────────────────────────────┐
│   ResNet-50 Backbone       │
│  ┌──────────────────────┐  │
│  │ Stem (Conv+BN+Pool)  │  │
│  └─────────┬────────────┘  │
│            │               │
│  ┌─────────▼────────────┐  │
│  │ layer1 (256ch,56×56) │◄─┼── Tap 1: Low-level
│  └─────────┬────────────┘  │
│            │               │
│  ┌─────────▼────────────┐  │
│  │ layer2 (512ch,28×28) │◄─┼── Tap 2: Mid-level
│  └─────────┬────────────┘  │
│            │               │
│  ┌─────────▼────────────┐  │
│  │ layer3  (pass-thru)  │  │
│  └─────────┬────────────┘  │
│            │               │
│  ┌─────────▼─────────────┐ │
│  │layer4 (2048ch, 7×7)   │◄┼── Tap 3: High-level
│  └───────────────────────┘ │
└────────────────────────────┘
    │
    ▼
GAP × 3  →  [256] [512] [2048]
    │
    ▼
Projections → [256] [256] [512]
    │
    ▼
Concat → [1024] forensic vector
    │
    ▼
BN → FC(512) → ReLU → Dropout
    │
    ▼
FC(2) → Softmax
    │
    ▼
REAL / FAKE + Confidence
        """, language="text")

        st.markdown("### 🗂️ Project Structure")
        st.code("""
finalyear project/
├── app.py              ← Streamlit dashboard (this file)
├── train.py            ← Training script
├── requirements.txt    ← Python dependencies
├── src/
│   ├── __init__.py
│   ├── model.py        ← DeepfakeDetector architecture
│   ├── preprocessor.py ← OpenCV face detection pipeline
│   └── inference.py    ← Image + video inference engine
└── checkpoints/
    └── best_model.pth  ← Trained weights (place here)
        """, language="text")
