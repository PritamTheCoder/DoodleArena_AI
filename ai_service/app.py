"""
DoodleNet Recognition — Glassmorphism Streamlit Dashboard.
A UI for real-time doodle recognition powered by MobileNetV3.
"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import base64
from PIL import Image
import io
import numpy as np
import time

# === Config ==========================================================
API_URL = "http://127.0.0.1:8000"
RECOGNIZE_URL = f"{API_URL}/recognize"
HEALTH_URL = f"{API_URL}/health"
CLASSES_URL = f"{API_URL}/classes"
CANVAS_SIZE = 400

CLASSES_30 = [
    "cat", "dog", "bird", "fish", "cow",
    "apple", "banana", "pizza", "cake", "ice cream",
    "car", "bicycle", "airplane", "bus", "train",
    "house", "tree", "flower", "sun", "cloud",
    "star", "moon", "hand", "face", "clock",
    "book", "chair", "shoe", "key", "umbrella"
]

# === Page Config =========================================================
st.set_page_config(
    page_title="DoodleNet AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# == Glassmorphism CSS ===================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Reset & Base ─────────────────────────────────── */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1040 25%, #302b63 50%, #24243e 75%, #0f0c29 100%) !important;
        background-attachment: fixed !important;
    }

    /* Animated gradient orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 20% 50%, rgba(120, 80, 255, 0.12) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(255, 100, 200, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 60% 80%, rgba(50, 200, 255, 0.06) 0%, transparent 50%);
        animation: orbFloat 25s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes orbFloat {
        0% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(-30px, 20px) rotate(5deg); }
        100% { transform: translate(10px, -15px) rotate(-3deg); }
    }

    /* ── Glass Cards ──────────────────────────────────── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        gap: 1.5rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(120, 80, 255, 0.2) !important;
        box-shadow: 0 12px 48px rgba(120, 80, 255, 0.1),
                    inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    }

    /* ── Sidebar ──────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0d4ff !important;
    }

    /* ── Typography ───────────────────────────────────── */
    h1, h2, h3 {
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }

    .hero-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #a78bfa 0%, #c084fc 30%, #f472b6 60%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.45);
        font-weight: 400;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        color: rgba(167, 139, 250, 0.7) !important;
        font-weight: 600 !important;
        margin-bottom: 0.8rem !important;
    }

    /* ── Metrics & Stat Cards ─────────────────────────── */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(120, 80, 255, 0.3);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    /* ── Confidence Bar ───────────────────────────────── */
    .confidence-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin: 0.6rem 0;
    }

    .confidence-bar-bg {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        height: 14px;
        overflow: hidden;
        margin-top: 0.5rem;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .fill-high {
        background: linear-gradient(90deg, #34d399, #10b981);
        box-shadow: 0 0 15px rgba(52, 211, 153, 0.4);
    }
    .fill-medium {
        background: linear-gradient(90deg, #fbbf24, #f59e0b);
        box-shadow: 0 0 15px rgba(251, 191, 36, 0.4);
    }
    .fill-low {
        background: linear-gradient(90deg, #f87171, #ef4444);
        box-shadow: 0 0 15px rgba(248, 113, 113, 0.4);
    }

    /* ── Prediction Rows ──────────────────────────────── */
    .pred-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.7rem 1rem;
        border-radius: 12px;
        margin: 0.4rem 0;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        transition: all 0.2s ease;
    }

    .pred-row:hover {
        background: rgba(120, 80, 255, 0.06);
        border-color: rgba(120, 80, 255, 0.15);
    }

    .pred-rank {
        font-size: 0.7rem;
        font-weight: 700;
        color: rgba(167, 139, 250, 0.6);
        width: 20px;
    }

    .pred-name {
        flex: 1;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.9rem;
        text-transform: capitalize;
    }

    .pred-score {
        font-weight: 600;
        font-size: 0.85rem;
        color: #a78bfa;
        font-variant-numeric: tabular-nums;
    }

    .pred-bar-bg {
        width: 80px;
        height: 6px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 3px;
        overflow: hidden;
    }

    .pred-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #a78bfa, #c084fc);
    }

    /* ── Streamlit Overrides ──────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 50%, #5b21b6 100%) !important;
        color: white !important;
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 14px !important;
        padding: 0.8rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.25) !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.4) !important;
        border-color: rgba(167, 139, 250, 0.5) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    .stSelectbox label, .stSlider label {
        color: rgba(255, 255, 255, 0.6) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 1rem !important;
    }

    div[data-testid="stMetric"] label {
        color: rgba(255, 255, 255, 0.5) !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #a78bfa !important;
    }

    /* Canvas styling */
    canvas {
        border-radius: 16px !important;
        border: 2px solid rgba(167, 139, 250, 0.2) !important;
    }

    /* Status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .status-online {
        background: rgba(52, 211, 153, 0.1);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.2);
    }

    .status-offline {
        background: rgba(248, 113, 113, 0.1);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.2);
    }

    /* Divider */
    .glass-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.2), transparent);
        margin: 1.5rem 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(167, 139, 250, 0.2); border-radius: 3px; }

    /* Hide default Streamlit branding but keep sidebar toggle */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
    }

    /* Style the sidebar toggle button */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarNavCollapseButton"],
    [data-testid="collapsedControl"] {
        color: rgba(167, 139, 250, 0.7) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }

    button[data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="collapsedControl"]:hover {
        background: rgba(120, 80, 255, 0.12) !important;
        border-color: rgba(167, 139, 250, 0.3) !important;
        color: #c084fc !important;
    }

    /* Hide default Material Icon text (keyboard_double_arrow_right) */
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        color: transparent !important; /* Hides the text ligature */
    }

    button[data-testid="stSidebarCollapseButton"] * ,
    [data-testid="collapsedControl"] * {
        display: none !important; /* Hides nested spans/svgs */
    }

    /* Inject chevrons */
    button[data-testid="stSidebarCollapseButton"]::after {
        content: '«';
        font-family: 'Inter', sans-serif !important;
        font-size: 1.3rem;
        font-weight: 700;
        color: rgba(167, 139, 250, 0.8) !important;
        display: block;
        line-height: 1;
    }

    [data-testid="collapsedControl"]::after {
        content: '»';
        font-family: 'Inter', sans-serif !important;
        font-size: 1.3rem;
        font-weight: 700;
        color: rgba(167, 139, 250, 0.8) !important;
        display: block;
        line-height: 1;
        cursor: pointer;
    }

    /* Fix text colors */
    .stMarkdown, .stMarkdown p, .stText {
        color: rgba(255, 255, 255, 0.7) !important;
    }

</style>
""", unsafe_allow_html=True)


# === Helper Functions ===========================================================

def encode_image_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def recognize_doodle(base64_image: str, prompt: str) -> dict:
    """Send doodle to FastAPI backend and get predictions."""
    payload = {"prompt": prompt, "image_base64": base64_image}
    try:
        response = requests.post(RECOGNIZE_URL, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API error {response.status_code}: {response.text}"}
    except requests.ConnectionError:
        return {"error": "Cannot connect to AI service. Is the FastAPI server running?"}
    except Exception as e:
        return {"error": str(e)}


def check_api_health() -> dict:
    """Check if the FastAPI backend is healthy."""
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_confidence_class(conf: float) -> str:
    """Return CSS class name based on confidence level."""
    if conf >= 0.6:
        return "fill-high"
    elif conf >= 0.3:
        return "fill-medium"
    return "fill-low"


def render_confidence_bar(label: str, value: float, css_class: str) -> str:
    """Render a styled confidence bar."""
    pct = value * 100
    return f"""
    <div class="confidence-container">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <span style="color:rgba(255,255,255,0.6); font-size:0.8rem; font-weight:500;">{label}</span>
            <span style="color:#a78bfa; font-size:1.4rem; font-weight:700;">{pct:.1f}%</span>
        </div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill {css_class}" style="width:{pct}%;"></div>
        </div>
    </div>
    """


def render_prediction_row(rank: int, name: str, score: float) -> str:
    """Render a single prediction row."""
    pct = score * 100
    return f"""
    <div class="pred-row">
        <span class="pred-rank">#{rank}</span>
        <span class="pred-name">{name}</span>
        <div class="pred-bar-bg"><div class="pred-bar-fill" style="width:{pct}%;"></div></div>
        <span class="pred-score">{pct:.1f}%</span>
    </div>
    """


# === Sidebar ===========================================================================================
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem !important;">🎨 DoodleNet</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle" style="font-size:0.85rem;">AI-Powered Doodle Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)

    # API Health
    health_data = check_api_health()
    if health_data:
        st.markdown('<span class="status-badge status-online">● API Online</span>', unsafe_allow_html=True)
        if health_data.get("model_info"):
            info = health_data["model_info"]
            st.markdown(f"""
            <div class="metric-card" style="margin-top:1rem; text-align:left;">
                <div style="color:rgba(255,255,255,0.5); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">Model Info</div>
                <div style="color:rgba(255,255,255,0.8); font-size:0.85rem;">
                    <strong>Arch:</strong> {info.get('architecture', 'N/A')}<br/>
                    <strong>Input:</strong> {str(info.get('input_shape', 'N/A'))}<br/>
                    <strong>Params:</strong> {info.get('total_parameters', 0):,}<br/>
                    <strong>Device:</strong> {health_data.get('device', 'cpu')}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-offline">● API Offline</span>', unsafe_allow_html=True)
        st.caption("Start the FastAPI server first.")

    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)

    # Settings
    st.markdown('<div class="section-title">⚙ Settings</div>', unsafe_allow_html=True)
    prompt = st.selectbox("Target Prompt", CLASSES_30, index=0)
    stroke_width = st.slider("Brush Size", 4, 30, 14)
    stroke_color = st.color_picker("Brush Color", "#000000")

    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="color:rgba(255,255,255,0.3); font-size:0.7rem; text-align:center; margin-top:1rem;">
        DoodleNet v2.0 · MobileNetV3-Small<br/>
        30 classes · 96×96 input
    </div>
    """, unsafe_allow_html=True)


# == Main Layout ================================================================
st.markdown('<div class="hero-title">DoodleNet Recognition</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-subtitle">Draw a <strong style="color:#c084fc;">{prompt}</strong> on the canvas and hit Recognize</div>', unsafe_allow_html=True)

col_canvas, col_results = st.columns([1.1, 1], gap="large")

# === Canvas Column ==============================================================
with col_canvas:
    st.markdown('<div class="section-title">✏️ Drawing Canvas</div>', unsafe_allow_html=True)

    canvas = st_canvas(
        fill_color="white",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="white",
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode="freedraw",
        key="canvas"
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        recognize_clicked = st.button("🔍 Recognize", use_container_width=True)

# === Results Column ===========================================================
with col_results:
    st.markdown('<div class="section-title">📊 Recognition Results</div>', unsafe_allow_html=True)

    if recognize_clicked:
        if canvas.image_data is not None:
            # Check for blank canvas
            img_array = canvas.image_data[:, :, :3]
            if np.mean(img_array) > 252:
                st.markdown("""
                <div class="confidence-container" style="text-align:center;">
                    <span style="color:rgba(255,255,255,0.4); font-size:0.9rem;">✏️ Canvas appears empty — draw something first!</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                img = Image.fromarray(np.uint8(img_array))
                img_base64 = encode_image_to_base64(img)

                with st.spinner(""):
                    result = recognize_doodle(img_base64, prompt)

                if "error" in result:
                    st.error(result["error"])
                else:
                    confidence = result.get("confidence", 0)
                    css_class = get_confidence_class(confidence)

                    # Main confidence display
                    st.markdown(
                        render_confidence_bar(f'Match: "{prompt}"', confidence, css_class),
                        unsafe_allow_html=True
                    )

                    # Top predictions
                    top_preds = result.get("top_predictions", [])
                    if top_preds:
                        st.markdown('<div class="section-title" style="margin-top:1.5rem;">🏆 Top Predictions</div>', unsafe_allow_html=True)
                        preds_html = ""
                        for i, pred in enumerate(top_preds):
                            preds_html += render_prediction_row(
                                i + 1,
                                pred.get("class", "?"),
                                pred.get("confidence", 0)
                            )
                        st.markdown(preds_html, unsafe_allow_html=True)

                    # Quick stats
                    st.markdown('<div class="glass-divider"></div>', unsafe_allow_html=True)
                    stat1, stat2, stat3 = st.columns(3)
                    with stat1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{confidence*100:.0f}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with stat2:
                        top_cls = top_preds[0]["class"] if top_preds else "—"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="font-size:1.2rem;">{top_cls}</div>
                            <div class="metric-label">Best Guess</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with stat3:
                        match_emoji = "✅" if confidence > 0.5 else "⚠️" if confidence > 0.2 else "❌"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{match_emoji}</div>
                            <div class="metric-label">Match</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="confidence-container" style="text-align:center;">
                <span style="color:rgba(255,255,255,0.4); font-size:0.9rem;">Draw something on the canvas to begin</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Default state
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.02);
            border: 1px dashed rgba(167, 139, 250, 0.15);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            margin-top: 1rem;
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🖌️</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 1rem; font-weight: 500;">
                Draw a doodle, then press Recognize
            </div>
            <div style="color: rgba(255,255,255,0.25); font-size: 0.8rem; margin-top: 0.5rem;">
                The AI will analyze your drawing in real-time
            </div>
        </div>
        """, unsafe_allow_html=True)