"""
What's Your Vibe? - Streamlit face-scan + vibe generator

Fix vs. the original script:
- The old version pushed each animation frame to the browser one-by-one via
  st.image() inside a Python loop. Every push is a full server<->browser
  round trip, so the animation stuttered (Streamlit is not built for
  frame-by-frame video).
- This version renders every frame in memory (no UI updates in the hot
  loop), stitches them into a single animated GIF, and displays that GIF
  ONCE. The browser's native GIF decoder handles playback, so it's smooth
  regardless of network/render latency.
- Status text + progress bar are drawn directly onto each frame, so they
  are always perfectly in sync with the animation (no separate widgets to
  desync from GIF playback).
- Fixed a bug: `mp.Image(...)` was used without `import mediapipe as mp`.
- The face-landmarker model is cached with st.cache_resource so it isn't
  reloaded from disk on every click.
"""

import base64
import io
import re
import textwrap
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "face_landmarker.task"
MAX_WIDTH = 420

TOTAL_FRAMES = 40          # frames in the scanning GIF
FRAME_DURATION_MS = 120    # ms each frame is shown -> ~4.8s animation

NEON = (255, 160, 255)     # RGB accent color used everywhere

SELECTED_POINTS = [
    # Face outline
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    # Eyes
    33, 133, 159, 145, 362, 263, 386, 374,
    # Nose
    1, 2, 98, 327, 4,
    # Mouth
    61, 291, 13, 14, 78, 308, 95, 324,
]

FACIAL_LINES = [
    # Left eye
    (33, 133), (133, 159), (159, 145), (145, 33),
    # Right eye
    (362, 263), (263, 386), (386, 374), (374, 362),
    # Nose
    (1, 2), (2, 98), (98, 327), (327, 4),
    # Mouth
    (61, 291), (61, 13), (13, 14), (14, 291),
]


# =========================================================
# PAGE
# =========================================================

st.set_page_config(page_title="What's Your Vibe?", page_icon="✨", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% -10%, rgba(150,80,220,0.25), transparent 45%),
            radial-gradient(circle at 100% 10%, rgba(255,140,200,0.12), transparent 40%),
            #0a0810;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .block-container { padding-top: 2.5rem; max-width: 640px; }

    /* ---------- Header ---------- */
    .app-header { text-align: center; margin-bottom: 1.6rem; }
    .app-title {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.15;
        background: linear-gradient(90deg, #ffd36e, #ff8fd6, #b78bff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        filter: drop-shadow(0 0 18px rgba(199,130,255,0.35));
        margin-bottom: 0.4rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.65);
    }

    /* ---------- Tabs (Upload / Camera) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.04);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: rgba(255,255,255,0.6);
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(199,130,255,0.25), rgba(255,140,200,0.2));
        color: #fff !important;
        box-shadow: inset 0 0 0 1px rgba(214,158,255,0.4);
    }

    /* ---------- Upload / camera boxes ---------- */
    [data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] video {
        border-radius: 16px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1.5px dashed rgba(214,158,255,0.35) !important;
    }
    [data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(214,158,255,0.25);
        border-radius: 12px;
        color: #fff;
    }

    /* ---------- Generate button ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #c58bff, #ff8fd6);
        color: #17101f;
        font-weight: 700;
        border: none;
        border-radius: 999px;
        padding: 0.7rem 1rem;
        box-shadow: 0 8px 30px rgba(199,130,255,0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 36px rgba(199,130,255,0.5);
        color: #17101f;
    }

    .vibe-frame {
        position: relative;
        border-radius: 22px;
        overflow: hidden;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.06),
                    0 20px 60px rgba(0,0,0,0.55);
    }
    .vibe-frame img {
        display: block;
        width: 100%;
        height: auto;
    }

    .vibe-card {
        position: absolute;
        left: 16px;
        right: 16px;
        bottom: 16px;
        padding: 20px 22px;
        border-radius: 18px;
        background: linear-gradient(160deg, rgba(30,14,45,0.82), rgba(12,8,20,0.88));
        border: 1px solid rgba(214,158,255,0.35);
        box-shadow: 0 0 0 1px rgba(255,255,255,0.04),
                    0 0 40px rgba(190,110,255,0.25),
                    inset 0 1px 0 rgba(255,255,255,0.06);
        backdrop-filter: blur(6px);
        animation: vibeRise 0.6s cubic-bezier(0.22, 1, 0.36, 1) both;
    }

    .vibe-greeting {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }
    .vibe-label {
        font-size: 0.7rem;
        letter-spacing: 0.18em;
        font-weight: 700;
        text-transform: uppercase;
        color: #e0b6ff;
        margin-bottom: 6px;
    }
    .vibe-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.15;
        margin-bottom: 6px;
        text-shadow: 0 0 24px rgba(214,158,255,0.45);
    }
    .vibe-desc {
        font-size: 0.92rem;
        color: rgba(255,255,255,0.82);
        margin-bottom: 12px;
        max-width: 46ch;
    }
    .vibe-tags { display: flex; flex-wrap: wrap; gap: 8px; }
    .vibe-tag {
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 999px;
        color: #f1defe;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(214,158,255,0.4);
    }

    /* ---------- Why This Vibe / Vibe Match ---------- */
    .why-text {
        font-size: 0.9rem;
        line-height: 1.5;
        color: rgba(255,255,255,0.82);
    }
    .match-percent {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(214,158,255,0.4);
        margin-bottom: 8px;
    }
    .match-caption {
        font-size: 0.76rem;
        color: rgba(255,255,255,0.5);
        margin-top: 8px;
        line-height: 1.4;
    }

    /* ---------- Profile sections: palette / visual energy ---------- */
    .profile-section {
        margin-top: 20px;
        padding: 18px 20px;
        border-radius: 16px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .profile-heading {
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 700;
        color: #e0b6ff;
        margin-bottom: 12px;
    }

    .palette-row { display: flex; gap: 18px; flex-wrap: wrap; }
    .palette-swatch { text-align: center; }
    .palette-dot {
        width: 42px; height: 42px; border-radius: 50%;
        border: 2px solid rgba(255,255,255,0.15);
        box-shadow: 0 0 14px rgba(0,0,0,0.5);
        margin: 0 auto 6px auto;
    }
    .palette-hex {
        font-size: 0.68rem;
        color: rgba(255,255,255,0.55);
        font-family: monospace;
    }
    .palette-label {
        font-size: 0.62rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        margin-bottom: 2px;
    }

    .energy-tag {
        font-size: 0.7rem;
        font-weight: 500;
        color: rgba(255,255,255,0.5);
    }

    .energy-row { margin-bottom: 12px; }
    .energy-row:last-child { margin-bottom: 0; }
    .energy-label {
        display: flex; justify-content: space-between;
        font-size: 0.82rem; color: rgba(255,255,255,0.82);
        margin-bottom: 5px;
    }
    .energy-track {
        width: 100%; height: 8px; border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
    }
    .energy-fill {
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, #c58bff, #ff8fd6);
    }

    .stDownloadButton > button {
        background: rgba(255,255,255,0.06);
        color: #fff;
        font-weight: 700;
        border: 1px solid rgba(214,158,255,0.4);
        border-radius: 999px;
    }
    .stDownloadButton > button:hover {
        border-color: rgba(214,158,255,0.8);
        color: #fff;
    }

    @keyframes vibeRise {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">✨ What's Your Vibe?</div>
        <div class="app-subtitle">Upload your photo and let AI discover your aesthetic vibe.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

@st.cache_resource
def load_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_rgb.shape[1] > MAX_WIDTH:
        scale = MAX_WIDTH / image_rgb.shape[1]
        new_w = int(image_rgb.shape[1] * scale)
        new_h = int(image_rgb.shape[0] * scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image_rgb


def pixel_points(landmarks, indices, width, height):
    pts = []
    for i in indices:
        if i >= len(landmarks):
            continue
        lm = landmarks[i]
        x, y = int(lm.x * width), int(lm.y * height)
        if 0 <= x < width and 0 <= y < height:
            pts.append((x, y))
    return pts


def status_text(raw_progress):
    if raw_progress < 0.15:
        return "Preparing your vibe..."
    elif raw_progress < 0.30:
        return "Detecting your face..."
    elif raw_progress < 0.50:
        return "Mapping your features..."
    elif raw_progress < 0.70:
        return "Building your facial map..."
    elif raw_progress < 0.85:
        return "Analyzing your visual patterns..."
    return "Finalizing your vibe..."


def draw_bottom_bar(frame, raw_progress, text):
    """Burn status text + a thin progress bar into the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 34

    # Solid dark strip so text/bar are always readable
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (20, 20, 25), -1)

    # Progress bar track + fill
    track_y = h - 10
    cv2.line(frame, (10, track_y), (w - 10, track_y), (70, 70, 80), 2)
    fill_x = int(10 + (w - 20) * raw_progress)
    cv2.line(frame, (10, track_y), (fill_x, track_y), NEON, 2)

    # Status text
    cv2.putText(
        frame, text, (10, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 235), 1, cv2.LINE_AA,
    )
    return frame


def build_frame(base_image, landmarks, points, bounds, smooth, raw_progress, sparkle_points):
    width_full, height_full = base_image.shape[1], base_image.shape[0]
    face_left, face_right, face_top, face_bottom = bounds

    frame = base_image.copy()
    overlay = np.zeros_like(frame)

    # Progressive landmark dots
    visible_count = int(len(points) * smooth)
    for x, y in points[:visible_count]:
        cv2.circle(overlay, (x, y), 2, (255, 150, 255), -1)

    # Facial connection lines
    if smooth > 0.35:
        line_progress = min(1.0, (smooth - 0.35) / 0.45)
        for line_index, (start, end) in enumerate(FACIAL_LINES):
            if start >= len(landmarks) or end >= len(landmarks):
                continue
            if (line_index / len(FACIAL_LINES)) > line_progress:
                continue
            x1, y1 = int(landmarks[start].x * width_full), int(landmarks[start].y * height_full)
            x2, y2 = int(landmarks[end].x * width_full), int(landmarks[end].y * height_full)
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 120, 255), 1)

    glow = cv2.GaussianBlur(overlay, (0, 0), 6)
    frame = cv2.addWeighted(frame, 1.0, glow, 0.45, 0)
    frame = cv2.addWeighted(frame, 1.0, overlay, 0.75, 0)

    # Sparkles
    if smooth > 0.15:
        sparkle_alpha = min(1.0, (smooth - 0.15) / 0.45)
        sparkle_overlay = np.zeros_like(frame)
        for sx, sy in sparkle_points:
            cv2.circle(sparkle_overlay, (sx, sy), 2, (255, 180, 255), -1)
        sparkle_glow = cv2.GaussianBlur(sparkle_overlay, (0, 0), 8)
        frame = cv2.addWeighted(frame, 1.0, sparkle_glow, 0.4 * sparkle_alpha, 0)
        frame = cv2.addWeighted(frame, 1.0, sparkle_overlay, 0.35 * sparkle_alpha, 0)

    # Corner frame near the end
    if smooth > 0.72:
        frame_progress = min(1.0, (smooth - 0.72) / 0.28)
        corner = int(45 * frame_progress)
        frame_overlay = np.zeros_like(frame)

        corners = [
            ((face_left, face_top), (1, 1)),
            ((face_right, face_top), (-1, 1)),
            ((face_left, face_bottom), (1, -1)),
            ((face_right, face_bottom), (-1, -1)),
        ]
        for (cx, cy), (dx, dy) in corners:
            cv2.line(frame_overlay, (cx, cy), (cx + dx * corner, cy), NEON, 2)
            cv2.line(frame_overlay, (cx, cy), (cx, cy + dy * corner), NEON, 2)

        frame_glow = cv2.GaussianBlur(frame_overlay, (0, 0), 8)
        frame = cv2.addWeighted(frame, 1.0, frame_glow, 0.35 * frame_progress, 0)
        frame = cv2.addWeighted(frame, 1.0, frame_overlay, 0.8 * frame_progress, 0)

    frame = draw_bottom_bar(frame, raw_progress, status_text(raw_progress))
    return frame


def frame_to_data_uri(frame_rgb):
    buf = io.BytesIO()
    Image.fromarray(frame_rgb).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def compact_html(html):
    """Strip per-line indentation so Streamlit's markdown parser never
    mistakes an indented (4+ space) line for a code block."""
    return "\n".join(line.strip() for line in html.strip().splitlines())


def render_vibe_overlay(final_frame, vibe, description, tags, name=""):
    tags_html = "".join(f'<span class="vibe-tag">{t}</span>' for t in tags)
    img_uri = frame_to_data_uri(final_frame)
    greeting_html = f'<div class="vibe-greeting">Hi, {name}! ✨</div>' if name.strip() else ""

    html = f"""
    <div class="vibe-frame">
        <img src="{img_uri}" />
        <div class="vibe-card">
            {greeting_html}
            <div class="vibe-label">✨ Your Vibe</div>
            <div class="vibe-title">{vibe}</div>
            <div class="vibe-desc">{description}</div>
            <div class="vibe-tags">{tags_html}</div>
        </div>
    </div>
    """
    return compact_html(html)


def make_gif(frames, duration_ms):
    pil_frames = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    buf.seek(0)
    return buf


def extract_palette(image_rgb, k=4):
    """Dominant colors via k-means, sorted lightest -> darkest for display.
    RNG is seeded so the same photo always yields the same palette.

    Near-duplicate clusters (common on low color-variance photos, where
    k-means splits one visual color into several near-identical centers)
    are merged first, so the same color is never labeled as two different
    things. Semantic labels (Dominant/Secondary/Accent/Deep Tone) are
    based on how much of the photo each merged color actually covers."""
    cv2.setRNGSeed(42)
    small = cv2.resize(image_rgb, (80, 80), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(centers, 0, 255).astype(int)
    counts = np.bincount(labels.flatten(), minlength=k)

    merged = []  # each item: [color (np.array int), pixel_count]
    for idx in np.argsort(-counts):
        color, count = centers[idx], counts[idx]
        twin = next((m for m in merged if np.linalg.norm(m[0] - color) < 18), None)
        if twin is not None:
            twin[1] += count
        else:
            merged.append([color, int(count)])

    merged.sort(key=lambda m: -m[1])  # index 0 = most photo coverage = "Dominant"

    semantic_names = ["Dominant", "Secondary", "Accent", "Deep Tone"]
    display_order = sorted(range(len(merged)), key=lambda i: -merged[i][0].mean())

    palette = []
    for i in display_order:
        r, g, b = (int(v) for v in merged[i][0])
        label = semantic_names[i] if i < len(semantic_names) else f"Tone {i + 1}"
        palette.append({"rgb": (r, g, b), "hex": f"#{r:02X}{g:02X}{b:02X}", "label": label})
    return palette


def compute_energy(brightness, saturation, contrast, warmth):
    """Turn the raw color stats into 4 fun 0-100 'visual energy' scores."""
    b_norm = float(np.clip(brightness / 255.0, 0, 1))
    s_norm = float(np.clip(saturation / 255.0, 0, 1))
    c_norm = float(np.clip(contrast / 128.0, 0, 1))
    warmth_norm = float(np.clip((warmth + 1) / 2, 0, 1))

    softness = np.clip(0.6 * (1 - c_norm) + 0.4 * b_norm, 0, 1)
    vividness = np.clip(0.7 * s_norm + 0.3 * c_norm, 0, 1)

    return {
        "Softness": int(round(softness * 100)),
        "Warmth": int(round(warmth_norm * 100)),
        "Contrast": int(round(c_norm * 100)),
        "Vividness": int(round(vividness * 100)),
    }


# Same low/mid/high thresholds used everywhere else in the app (see the
# _describe_* helpers above) - kept consistent on purpose.
_ENERGY_WORDS = {
    "Softness": ("Firm", "Balanced", "Soft"),
    "Warmth": ("Cool", "Neutral", "Warm"),
    "Contrast": ("Gentle", "Balanced", "Dramatic"),
    "Vividness": ("Muted", "Balanced", "Vivid"),
}


def _energy_label(name, value):
    low, mid, high = _ENERGY_WORDS.get(name, ("Low", "Balanced", "High"))
    if value < 35:
        return low
    if value <= 65:
        return mid
    return high


def render_why_html(why_text):
    html = f"""
    <div class="profile-section">
        <div class="profile-heading">Why This Vibe?</div>
        <div class="why-text">{why_text}</div>
    </div>
    """
    return compact_html(html)


def render_match_html(match_pct):
    pct = int(round(match_pct))
    html = f"""
    <div class="profile-section">
        <div class="profile-heading">Vibe Match</div>
        <div class="match-percent">{pct}%</div>
        <div class="energy-track"><div class="energy-fill" style="width:{pct}%;"></div></div>
        <div class="match-caption">
            How closely your photo's visual tone matches this aesthetic profile —
            not a measure of personality, accuracy, or attractiveness.
        </div>
    </div>
    """
    return compact_html(html)


def render_secondary_html(secondary):
    if not secondary:
        return ""
    rows = "".join(
        f"""
        <div class="energy-row">
            <div class="energy-label">
                <span>{v['profile']['vibe']}</span><span>{int(round(v['match_pct']))}%</span>
            </div>
            <div class="energy-track">
                <div class="energy-fill" style="width:{int(round(v['match_pct']))}%;"></div>
            </div>
        </div>
        """
        for v in secondary
    )
    html = f"""
    <div class="profile-section">
        <div class="profile-heading">Your Other Vibes</div>
        {rows}
    </div>
    """
    return compact_html(html)


def render_palette_html(palette):
    swatches = "".join(
        f"""
        <div class="palette-swatch">
            <div class="palette-dot" style="background:{p['hex']};"></div>
            <div class="palette-label">{p['label']}</div>
            <div class="palette-hex">{p['hex']}</div>
        </div>
        """
        for p in palette
    )
    html = f"""
    <div class="profile-section">
        <div class="profile-heading">Your Palette</div>
        <div class="palette-row">{swatches}</div>
    </div>
    """
    return compact_html(html)


def render_energy_html(energy):
    rows = "".join(
        f"""
        <div class="energy-row">
            <div class="energy-label">
                <span>{label} <span class="energy-tag">{_energy_label(label, value)}</span></span>
                <span>{value}%</span>
            </div>
            <div class="energy-track"><div class="energy-fill" style="width:{value}%;"></div></div>
        </div>
        """
        for label, value in energy.items()
    )
    html = f"""
    <div class="profile-section">
        <div class="profile-heading">Your Visual Energy</div>
        {rows}
    </div>
    """
    return compact_html(html)


def sanitize_text(text):
    """Strip emoji/non-ASCII (PIL's fallback font can't render them) and
    normalize dashes so downloaded card text never breaks or shows boxes."""
    text = text.replace("—", "-").replace("–", "-")
    return re.sub(r"[^\x00-\x7F]+", "", text).strip()


def get_font(size, bold=False):
    names = (
        ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
        if bold else
        ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    )
    prefixes = [
        "", "/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/truetype/liberation/",
        "/System/Library/Fonts/Supplemental/", "C:\\Windows\\Fonts\\",
    ]
    for prefix in prefixes:
        for name in names:
            try:
                return ImageFont.truetype(prefix + name, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def compose_share_card(
    image_rgb, vibe, description, tags, why_text, match_pct, secondary,
    palette, energy, name="",
):
    """Render one downloadable PNG that mirrors the on-screen design and
    hierarchy: photo+vibe card -> Why This Vibe? -> Vibe Match ->
    Your Other Vibes -> Your Palette -> Your Visual Energy."""
    card_w = 720

    greeting_font = get_font(19, bold=True)
    label_font = get_font(15, bold=True)
    title_font = get_font(30, bold=True)
    match_font = get_font(30, bold=True)
    desc_font = get_font(17)
    tag_font = get_font(14, bold=True)
    small_font = get_font(14)

    clean_desc = sanitize_text(description)
    wrapped_desc = textwrap.fill(clean_desc, width=48)
    desc_line_h = 23
    desc_h = (wrapped_desc.count("\n") + 1) * desc_line_h

    # ---------------------------------------------------------------
    # 1) Photo + glowing glass card overlaid at the bottom of the photo
    # ---------------------------------------------------------------
    photo = Image.fromarray(image_rgb).convert("RGBA")
    photo = ImageOps.fit(photo, (card_w, int(card_w * photo.height / photo.width)), Image.LANCZOS)

    greeting_text = sanitize_text(name).strip()
    greeting_h = 30 if greeting_text else 0

    margin = 18
    inner_pad = 22
    label_h, gap, title_h, tags_h = 20, 8, 40, 38

    overlay_h = inner_pad * 2 + greeting_h + label_h + gap + title_h + gap + desc_h + gap + tags_h
    ox0, oy0 = margin, photo.height - overlay_h - margin
    ox1, oy1 = card_w - margin, photo.height - margin

    overlay = Image.new("RGBA", photo.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle(
        [ox0, oy0, ox1, oy1], radius=20,
        fill=(24, 14, 34, 215), outline=(199, 130, 255, 160), width=2,
    )
    photo = Image.alpha_composite(photo, overlay)
    draw = ImageDraw.Draw(photo)

    tx, ty = ox0 + inner_pad, oy0 + inner_pad
    if greeting_text:
        draw.text((tx, ty), f"Hi, {greeting_text}!", font=greeting_font, fill=(255, 255, 255, 255))
        ty += greeting_h
    draw.text((tx, ty), "YOUR VIBE", font=label_font, fill=(224, 182, 255, 255))
    ty += label_h + gap
    draw.text((tx, ty), sanitize_text(vibe), font=title_font, fill=(255, 255, 255, 255))
    ty += title_h + gap
    draw.multiline_text((tx, ty), wrapped_desc, font=desc_font, fill=(230, 225, 235, 255), spacing=6)
    ty += desc_h + gap

    tagx = tx
    for tag in tags:
        tw = draw.textlength(tag, font=tag_font) + 22
        draw.rounded_rectangle(
            [tagx, ty, tagx + tw, ty + 30], radius=15,
            outline=(214, 158, 255, 255), width=2,
        )
        draw.text((tagx + 11, ty + 6), tag, font=tag_font, fill=(241, 222, 254, 255))
        tagx += tw + 10

    photo_rgb = photo.convert("RGB")

    # ---------------------------------------------------------------
    # 2) Why / Match / Other Vibes / Palette / Visual Energy panels,
    #    same rounded-card language and order as the web page
    # ---------------------------------------------------------------
    panel_fill = (23, 16, 31)
    panel_outline = (60, 48, 70)
    section_gap = 16
    panel_pad = 20

    # -- Why This Vibe? --
    wrapped_why = textwrap.fill(sanitize_text(why_text), width=62)
    why_line_h = 21
    why_h = (wrapped_why.count("\n") + 1) * why_line_h
    why_panel_h = panel_pad * 2 + 22 + why_h

    # -- Vibe Match --
    match_pct_int = int(round(match_pct))
    match_caption = (
        "How closely your photo's visual tone matches this aesthetic "
        "profile - not a measure of personality, accuracy, or attractiveness."
    )
    wrapped_caption = textwrap.fill(match_caption, width=70)
    caption_line_h = 17
    caption_h = (wrapped_caption.count("\n") + 1) * caption_line_h
    match_panel_h = panel_pad * 2 + 22 + 40 + 10 + 8 + caption_h

    # -- Your Other Vibes (only if there's a meaningful runner-up) --
    secondary_panel_h = panel_pad * 2 + 22 + len(secondary) * 27 if secondary else 0

    # -- Palette / Visual Energy (unchanged sizing, palette now 2 text lines) --
    palette_panel_h = panel_pad * 2 + 22 + 80
    energy_panel_h = panel_pad * 2 + 22 + len(energy) * 27

    extra_h = (
        section_gap + why_panel_h
        + section_gap + match_panel_h
        + (section_gap + secondary_panel_h if secondary else 0)
        + section_gap + palette_panel_h
        + section_gap + energy_panel_h
        + section_gap
    )
    canvas = Image.new("RGB", (card_w, photo_rgb.height + extra_h), (10, 7, 15))
    canvas.paste(photo_rgb, (0, 0))
    draw = ImageDraw.Draw(canvas)

    y = photo_rgb.height + section_gap

    # --- Why This Vibe? panel ---
    draw.rounded_rectangle(
        [margin, y, card_w - margin, y + why_panel_h],
        radius=16, fill=panel_fill, outline=panel_outline,
    )
    draw.text((margin + panel_pad, y + panel_pad), "WHY THIS VIBE?", font=label_font, fill=(224, 182, 255))
    draw.multiline_text(
        (margin + panel_pad, y + panel_pad + 26), wrapped_why,
        font=small_font, fill=(220, 215, 225), spacing=6,
    )
    y += why_panel_h + section_gap

    # --- Vibe Match panel ---
    draw.rounded_rectangle(
        [margin, y, card_w - margin, y + match_panel_h],
        radius=16, fill=panel_fill, outline=panel_outline,
    )
    draw.text((margin + panel_pad, y + panel_pad), "VIBE MATCH", font=label_font, fill=(224, 182, 255))
    my = y + panel_pad + 26
    draw.text((margin + panel_pad, my), f"{match_pct_int}%", font=match_font, fill=(255, 255, 255))
    my += 40
    bar_x2 = margin + panel_pad
    bar_w2 = card_w - bar_x2 - margin - panel_pad
    draw.rounded_rectangle([bar_x2, my, bar_x2 + bar_w2, my + 10], radius=5, fill=(50, 40, 60))
    fill_w2 = int(bar_w2 * match_pct_int / 100)
    if fill_w2 > 0:
        draw.rounded_rectangle([bar_x2, my, bar_x2 + fill_w2, my + 10], radius=5, fill=(199, 130, 255))
    my += 10 + 8
    draw.multiline_text((margin + panel_pad, my), wrapped_caption, font=small_font, fill=(190, 185, 195), spacing=5)
    y += match_panel_h + section_gap

    # --- Your Other Vibes panel ---
    if secondary:
        draw.rounded_rectangle(
            [margin, y, card_w - margin, y + secondary_panel_h],
            radius=16, fill=panel_fill, outline=panel_outline,
        )
        draw.text((margin + panel_pad, y + panel_pad), "YOUR OTHER VIBES", font=label_font, fill=(224, 182, 255))
        sy = y + panel_pad + 26
        s_bar_x = margin + panel_pad + 170
        s_bar_w = card_w - s_bar_x - margin - panel_pad - 46
        for item in secondary:
            s_vibe = sanitize_text(item["profile"]["vibe"])
            s_pct = int(round(item["match_pct"]))
            draw.text((margin + panel_pad, sy), s_vibe, font=small_font, fill=(220, 215, 225))
            draw.text((s_bar_x + s_bar_w + 12, sy), f"{s_pct}%", font=small_font, fill=(220, 215, 225))
            draw.rounded_rectangle([s_bar_x, sy + 3, s_bar_x + s_bar_w, sy + 13], radius=6, fill=(50, 40, 60))
            s_fill_w = int(s_bar_w * s_pct / 100)
            if s_fill_w > 0:
                draw.rounded_rectangle([s_bar_x, sy + 3, s_bar_x + s_fill_w, sy + 13], radius=6, fill=(199, 130, 255))
            sy += 27
        y += secondary_panel_h + section_gap

    # --- Palette panel (dot + semantic label + hex) ---
    draw.rounded_rectangle(
        [margin, y, card_w - margin, y + palette_panel_h],
        radius=16, fill=panel_fill, outline=panel_outline,
    )
    draw.text((margin + panel_pad, y + panel_pad), "YOUR PALETTE", font=label_font, fill=(224, 182, 255))
    px = margin + panel_pad
    py = y + panel_pad + 26
    for p in palette:
        draw.ellipse([px, py, px + 44, py + 44], fill=p["rgb"], outline=(255, 255, 255))
        draw.text((px, py + 50), p.get("label", ""), font=small_font, fill=(170, 165, 180))
        draw.text((px, py + 66), p["hex"], font=small_font, fill=(190, 185, 195))
        px += 70
    y += palette_panel_h + section_gap

    # --- Visual energy panel (bar + qualitative word) ---
    draw.rounded_rectangle(
        [margin, y, card_w - margin, y + energy_panel_h],
        radius=16, fill=panel_fill, outline=panel_outline,
    )
    draw.text((margin + panel_pad, y + panel_pad), "YOUR VISUAL ENERGY", font=label_font, fill=(224, 182, 255))
    ey = y + panel_pad + 26
    bar_x = margin + panel_pad + 190
    bar_w = card_w - bar_x - margin - panel_pad - 46
    for label, value in energy.items():
        word = _energy_label(label, value)
        draw.text((margin + panel_pad, ey), f"{label} · {word}", font=small_font, fill=(220, 215, 225))
        draw.text((bar_x + bar_w + 12, ey), f"{value}%", font=small_font, fill=(220, 215, 225))
        draw.rounded_rectangle([bar_x, ey + 3, bar_x + bar_w, ey + 13], radius=6, fill=(50, 40, 60))
        fill_w = int(bar_w * value / 100)
        if fill_w > 0:
            draw.rounded_rectangle([bar_x, ey + 3, bar_x + fill_w, ey + 13], radius=6, fill=(199, 130, 255))
        ey += 27

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def extract_visual_dimensions(image_rgb):
    """Pure visual measurement of the PHOTO - no personality claims, no
    face-shape input. This is the single source of truth used by both the
    Vibe Engine (matching) and the Visual Energy display."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hue = float(np.mean(hsv[:, :, 0]))          # 0-179 in OpenCV
    saturation = float(np.mean(hsv[:, :, 1]))   # 0-255
    brightness = float(np.mean(hsv[:, :, 2]))   # 0-255

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    contrast = float(np.std(gray))              # 0-~128

    brightness_norm = float(np.clip(brightness / 255.0, 0.0, 1.0))
    saturation_norm = float(np.clip(saturation / 255.0, 0.0, 1.0))
    contrast_norm = float(np.clip(contrast / 128.0, 0.0, 1.0))

    # Hue is meaningless once a photo is near-grayscale (very low
    # saturation) - without this, dark/monochrome photos get a "random"
    # warm/cool reading purely from color noise. Fade warmth toward
    # neutral (0) as saturation drops, instead of trusting hue blindly.
    raw_warmth = float(np.cos(np.radians(hue * 2)))
    warmth = raw_warmth * saturation_norm

    return {
        "brightness": brightness,
        "saturation": saturation,
        "contrast": contrast,
        "brightness_norm": brightness_norm,
        "saturation_norm": saturation_norm,
        "contrast_norm": contrast_norm,
        "warmth": warmth,
    }


# Max possible squared distance in the 4D normalized space (b, s, c each
# span 1.0; warmth spans 2.0 and carries a 0.7 weight) -> used to turn a
# raw distance into an absolute, honest 0-100% match score.
_VIBE_MATCH_D_MAX = 1.0 + 1.0 + 1.0 + 0.7 * (2.0 ** 2)


def score_all_vibes(dims):
    """Score every vibe profile against the photo's measured dimensions.
    Returns all profiles sorted by match_pct, highest first."""
    scored = []
    for profile in VIBE_PROFILES:
        d = (
            (dims["brightness_norm"] - profile["b"]) ** 2
            + (dims["saturation_norm"] - profile["s"]) ** 2
            + (dims["contrast_norm"] - profile["c"]) ** 2
            + 0.7 * (dims["warmth"] - profile["w"]) ** 2
        )
        match_pct = max(0.0, min(100.0, 100.0 * (1 - d / _VIBE_MATCH_D_MAX)))
        scored.append({"profile": profile, "match_pct": match_pct})

    scored.sort(key=lambda x: x["match_pct"], reverse=True)
    return scored


def _describe_brightness(b_norm):
    if b_norm < 0.35:
        return "dim, moody lighting"
    if b_norm < 0.65:
        return "balanced lighting"
    return "bright, luminous lighting"


def _describe_saturation(s_norm):
    if s_norm < 0.35:
        return "muted, understated colors"
    if s_norm < 0.65:
        return "balanced color richness"
    return "bold, saturated colors"


def _describe_contrast(c_norm):
    if c_norm < 0.35:
        return "soft, gentle contrast"
    if c_norm < 0.65:
        return "balanced contrast"
    return "sharp, dramatic contrast"


def _describe_warmth(warmth):
    if warmth < -0.25:
        return "a cool color tone"
    if warmth <= 0.25:
        return "a neutral color tone"
    return "a warm color tone"


def generate_why_explanation(dims, profile):
    """Build a short, evidence-based sentence from the actual measured
    dimensions - always about the PHOTO, never about the person."""
    diffs = {
        "brightness": abs(dims["brightness_norm"] - profile["b"]),
        "saturation": abs(dims["saturation_norm"] - profile["s"]),
        "contrast": abs(dims["contrast_norm"] - profile["c"]),
        "warmth": abs(dims["warmth"] - profile["w"]),
    }
    # The 2 dimensions where the photo is closest to this profile's target
    # are the ones that actually earned the match - lead with those.
    top_two = sorted(diffs, key=diffs.get)[:2]

    phrase_map = {
        "brightness": _describe_brightness(dims["brightness_norm"]),
        "saturation": _describe_saturation(dims["saturation_norm"]),
        "contrast": _describe_contrast(dims["contrast_norm"]),
        "warmth": _describe_warmth(dims["warmth"]),
    }
    phrase_one, phrase_two = (phrase_map[d] for d in top_two)
    mood = " and ".join(t.lower() for t in profile["tags"][:2])

    return f"Your photo has {phrase_one} and {phrase_two}, creating a {mood} atmosphere."


VIBE_PROFILES = [
    # target values are normalized 0-1 (brightness, saturation, contrast),
    # warmth is -1 (cool) .. +1 (warm)
    dict(vibe="🌸 The Soft Soul", b=0.75, s=0.25, c=0.35, w=0.0,
         description="Soft energy, calm presence, and a touch of quiet elegance.",
         tags=["Calm", "Elegant", "Warm"]),
    dict(vibe="🔥 The Bold One", b=0.55, s=0.65, c=0.78, w=0.3,
         description="Strong energy, expressive style, and a presence that stands out.",
         tags=["Bold", "Expressive", "Confident"]),
    dict(vibe="🌙 The Dreamer", b=0.30, s=0.30, c=0.40, w=-0.2,
         description="A little mysterious, introspective, and probably lost in your own thoughts.",
         tags=["Dreamy", "Thoughtful", "Mysterious"]),
    dict(vibe="☀️ The Spark", b=0.70, s=0.78, c=0.55, w=0.4,
         description="Bright energy, playful vibes, and a personality that feels alive.",
         tags=["Playful", "Bright", "Energetic"]),
    dict(vibe="🌿 The Wanderer", b=0.55, s=0.45, c=0.45, w=0.0,
         description="Curious, grounded, and always ready for something new.",
         tags=["Curious", "Grounded", "Adventurous"]),
    dict(vibe="🌊 The Calm Wave", b=0.55, s=0.32, c=0.32, w=-0.6,
         description="Cool, steady, and quietly reassuring — the person everyone relaxes around.",
         tags=["Serene", "Steady", "Cool-toned"]),
    dict(vibe="🍂 The Warm Ember", b=0.50, s=0.55, c=0.45, w=0.7,
         description="Cozy, familiar warmth with a comforting, homey kind of glow.",
         tags=["Cozy", "Genuine", "Down-to-earth"]),
    dict(vibe="⚡ The Electric", b=0.65, s=0.88, c=0.85, w=0.2,
         description="High-voltage energy and striking presence — impossible to ignore.",
         tags=["Vivid", "Intense", "Magnetic"]),
    dict(vibe="🖤 The Minimalist", b=0.55, s=0.12, c=0.72, w=0.0,
         description="Clean lines, quiet confidence, and a 'less is more' kind of style.",
         tags=["Sleek", "Composed", "Understated"]),
    dict(vibe="🌷 The Romantic", b=0.80, s=0.45, c=0.28, w=0.6,
         description="Soft, warm, and a little dreamy — effortlessly charming.",
         tags=["Charming", "Gentle", "Warm-hued"]),
    dict(vibe="🧊 The Cool Breeze", b=0.80, s=0.18, c=0.28, w=-0.7,
         description="Crisp, airy, and refreshingly easygoing.",
         tags=["Fresh", "Airy", "Effortless"]),
    dict(vibe="🌑 The Enigma", b=0.18, s=0.35, c=0.68, w=-0.1,
         description="Dramatic and a little guarded — the kind of person people want to figure out.",
         tags=["Dramatic", "Guarded", "Intriguing"]),
    dict(vibe="🌻 The Golden Hour", b=0.78, s=0.72, c=0.48, w=0.8,
         description="Radiant, sunny, and full of warmth — a natural glow that's hard to fake.",
         tags=["Radiant", "Sunny", "Magnetic"]),
    dict(vibe="🕊️ The Serene One", b=0.82, s=0.18, c=0.22, w=0.0,
         description="Light, peaceful, and unhurried — carries a calm that puts others at ease.",
         tags=["Peaceful", "Light", "Unhurried"]),
]


# =========================================================
# MAIN
# =========================================================

if "stage" not in st.session_state:
    st.session_state.stage = "input"
if "widget_key" not in st.session_state:
    st.session_state.widget_key = 0

# ---------------------------------------------------------
# STAGE: INPUT (upload/camera -> scanning animation)
# ---------------------------------------------------------
if st.session_state.stage == "input":

    user_name = st.text_input(
        "Your name (optional)",
        placeholder="Your name (optional)",
        label_visibility="collapsed",
        key=f"name_{st.session_state.widget_key}",
    )

    tab_upload, tab_camera = st.tabs(["📤 Upload Photo", "📷 Use Camera"])

    uploaded_file = None
    with tab_upload:
        file_from_upload = st.file_uploader(
            "Choose a photo", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed", key=f"uploader_{st.session_state.widget_key}",
        )
        if file_from_upload is not None:
            uploaded_file = file_from_upload

    with tab_camera:
        file_from_camera = st.camera_input(
            "Take a photo", label_visibility="collapsed",
            key=f"camera_{st.session_state.widget_key}",
        )
        if file_from_camera is not None:
            uploaded_file = file_from_camera

    if uploaded_file is not None:
        image_rgb = load_image(uploaded_file)
        height, width = image_rgb.shape[:2]

        # Single placeholder reused for: original photo -> scanning GIF
        photo_area = st.empty()
        photo_area.image(image_rgb, caption="Your photo", width="content")

        if st.button("✨ Generate My Vibe", use_container_width=True):

            with st.spinner("Reading facial structure..."):
                detector = load_detector()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                result = detector.detect(mp_image)
                face_landmarks_list = result.face_landmarks

            if len(face_landmarks_list) == 0:
                st.error("I couldn't detect a clear face. Try a front-facing photo.")
            else:
                landmarks = face_landmarks_list[0]

                points = pixel_points(landmarks, SELECTED_POINTS, width, height)
                all_points = pixel_points(landmarks, range(len(landmarks)), width, height)
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]

                bounds = (
                    max(0, min(xs) - 35),
                    min(width - 1, max(xs) + 35),
                    max(0, min(ys) - 35),
                    min(height - 1, max(ys) + 35),
                )

                rng = np.random.default_rng(42)
                face_left, face_right, face_top, face_bottom = bounds
                sparkle_points = [
                    (int(rng.integers(face_left, face_right + 1)), int(rng.integers(face_top, face_bottom + 1)))
                    for _ in range(12)
                ]

                # --- Render every frame in memory (NO widget updates in this loop) ---
                build_progress = st.progress(0, text="Rendering scan animation...")
                frames = []
                for i in range(TOTAL_FRAMES):
                    raw_progress = (i + 1) / TOTAL_FRAMES
                    smooth = raw_progress * raw_progress * (3 - 2 * raw_progress)
                    frame = build_frame(
                        image_rgb, landmarks, points, bounds,
                        smooth, raw_progress, sparkle_points,
                    )
                    frames.append(frame)
                    build_progress.progress(int(raw_progress * 100))
                build_progress.empty()

                # --- Show the whole animation as ONE smooth GIF ---
                gif_buf = make_gif(frames, FRAME_DURATION_MS)
                photo_area.image(gif_buf.getvalue())

                # Let it play through once
                time.sleep((TOTAL_FRAMES * FRAME_DURATION_MS) / 1000 + 0.3)

                # --- Run the Vibe Engine and hand off to the result stage ---
                dims = extract_visual_dimensions(image_rgb)
                scored_vibes = score_all_vibes(dims)

                primary = scored_vibes[0]
                secondary = [v for v in scored_vibes[1:] if v["match_pct"] >= 30][:3]
                why_text = generate_why_explanation(dims, primary["profile"])

                palette = extract_palette(image_rgb)
                energy = compute_energy(
                    dims["brightness"], dims["saturation"], dims["contrast"], dims["warmth"]
                )

                st.session_state.result = dict(
                    image_rgb=image_rgb,
                    name=user_name,
                    vibe=primary["profile"]["vibe"],
                    description=primary["profile"]["description"],
                    tags=primary["profile"]["tags"],
                    match_pct=primary["match_pct"],
                    why=why_text,
                    secondary=secondary,
                    palette=palette,
                    energy=energy,
                    brightness=dims["brightness"],
                    saturation=dims["saturation"],
                    contrast=dims["contrast"],
                )
                st.session_state.stage = "result"
                st.rerun()

# ---------------------------------------------------------
# STAGE: RESULT (vibe card + palette + energy + actions)
# ---------------------------------------------------------
elif st.session_state.stage == "result":
    data = st.session_state.result

    st.markdown(
        render_vibe_overlay(
            data["image_rgb"], data["vibe"], data["description"], data["tags"],
            name=data.get("name", ""),
        ),
        unsafe_allow_html=True,
    )
    st.markdown(render_why_html(data["why"]), unsafe_allow_html=True)
    st.markdown(render_match_html(data["match_pct"]), unsafe_allow_html=True)
    if data["secondary"]:
        st.markdown(render_secondary_html(data["secondary"]), unsafe_allow_html=True)
    st.markdown(render_palette_html(data["palette"]), unsafe_allow_html=True)
    st.markdown(render_energy_html(data["energy"]), unsafe_allow_html=True)

    st.caption(
        "Just for fun — this is a playful visual interpretation, "
        "not a real personality analysis."
    )

    share_png = compose_share_card(
        data["image_rgb"], data["vibe"], data["description"], data["tags"],
        data["why"], data["match_pct"], data["secondary"],
        data["palette"], data["energy"],
        name=data.get("name", ""),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇ Save Vibe Card",
            data=share_png,
            file_name="my_vibe_card.png",
            mime="image/png",
            use_container_width=True,
        )
    with col2:
        if st.button("↻ Try Another Photo", use_container_width=True):
            st.session_state.stage = "input"
            st.session_state.widget_key += 1
            del st.session_state["result"]
            st.rerun()

    with st.expander("See the analysis"):
        st.write(f"Brightness: {data['brightness']:.2f}")
        st.write(f"Saturation: {data['saturation']:.2f}")
        st.write(f"Contrast: {data['contrast']:.2f}")
        