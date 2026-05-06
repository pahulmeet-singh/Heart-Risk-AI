import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import io
import base64

# ── PDF imports ──────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Risk Assessment",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL ASSETS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model         = joblib.load('heart_disease_model.pkl')
    scaler        = joblib.load('heart_scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    'age':         'Age',
    'sex':         'Sex',
    'cp':          'Chest Pain Type',
    'trestbps':    'Resting Blood Pressure',
    'chol':        'Serum Cholesterol',
    'fbs':         'High Fasting Blood Sugar',
    'restecg':     'Resting ECG',
    'thalach':     'Max Heart Rate',
    'exang':       'Exercise-Induced Pain',
    'slope':       'ST Slope',
    'ca':          'Major Vessels',
    'thal':        'Thalassemia',
    'oldpeak_log': 'ST Depression', 
}

CP_LABELS    = {0: "Typical Angina",    1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}
THAL_LABELS  = {1: "Fixed Defect",      2: "Normal",          3: "Reversible Defect"}
ECG_LABELS   = {0: "Normal",            1: "ST-T Abnormality", 2: "LV Hypertrophy"}
SLOPE_LABELS = {0: "Upsloping",         1: "Flat",             2: "Downsloping"}

# Design tokens
C_BG    = "#070d1a"
C_CARD  = "#0d1525"
C_BORD  = "#1a2e50"
C_TEXT  = "#e2e8f0"
C_MUTED = "#64748b"
C_ACCNT = "#00d4ff"
C_LOW   = "#10b981"
C_MID   = "#f59e0b"
C_HIGH  = "#ef4444"

def risk_color(p):  return C_HIGH if p >= 60 else (C_MID if p >= 30 else C_LOW)
def risk_label(p):  return "HIGH RISK"     if p >= 60 else ("MODERATE RISK" if p >= 30 else "LOW RISK")

# ─────────────────────────────────────────────────────────────────────────────
#  CSS INJECTION (Dense block to prevent Markdown bleed)
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    css = """<style>
    *, *::before, *::after { box-sizing: border-box; }
    html, body, .stApp, [data-testid="stApp"] { background: #070d1a !important; font-family: 'DM Sans', sans-serif !important; color: #e2e8f0 !important; }
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stHeader"], header { visibility: hidden !important; height: 0 !important; overflow: hidden !important; }
    [data-testid="stSidebar"] { display: none !important; }
    section[data-testid="stMain"] { padding-top: 0 !important; }
    .block-container { padding: 0 2rem 5rem 2rem !important; max-width: 980px !important; margin: 0 auto !important; }
    h1,h2,h3,h4,p,span { font-family: 'DM Sans', sans-serif !important; color: #e2e8f0 !important; }
    
    .stButton > button { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; letter-spacing: 0.3px !important; border-radius: 10px !important; transition: all 0.22s ease !important; font-size: 0.88rem !important; width: 100% !important; }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, #00b4d8, #00d4ff) !important; color: #070d1a !important; border: none !important; box-shadow: 0 0 18px #00d4ff28 !important; padding: 0.65rem 1.8rem !important; }
    .stButton > button[kind="primary"]:hover { box-shadow: 0 0 30px #00d4ff50 !important; transform: translateY(-1px) !important; }
    .stButton > button[kind="secondary"] { background: transparent !important; color: #64748b !important; border: 1px solid #1a2e50 !important; padding: 0.65rem 1.8rem !important; }
    .stButton > button[kind="secondary"]:hover { border-color: #00d4ff !important; color: #00d4ff !important; }
    
    .stDownloadButton > button { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; border-radius: 10px !important; background: linear-gradient(135deg, #00b4d8, #00d4ff) !important; color: #070d1a !important; border: none !important; box-shadow: 0 0 18px #00d4ff28 !important; transition: all 0.22s ease !important; width: 100% !important; }
    .stDownloadButton > button:hover { box-shadow: 0 0 30px #00d4ff50 !important; transform: translateY(-1px) !important; }
    
    .stNumberInput input { background: #0d1525 !important; color: #e2e8f0 !important; border: 1px solid #1a2e50 !important; border-radius: 8px !important; font-family: 'DM Mono', monospace !important; font-size: 1.1rem !important; font-weight: 500 !important; }
    .stNumberInput input:focus { border-color: #00d4ff !important; box-shadow: 0 0 0 2px #00d4ff25 !important; }
    .stNumberInput > label, .stSlider > label, .stRadio > label { color: #64748b !important; font-size: 0.78rem !important; font-weight: 600 !important; letter-spacing: 1px !important; text-transform: uppercase !important; }
    .stNumberInput [data-testid="stNumberInputContainer"] button { background: #0d1525 !important; border-color: #1a2e50 !important; color: #64748b !important; }
    
    [data-testid="stSlider"] > div > div > div > div { background: #00d4ff !important; }
    [data-testid="stSlider"] [data-testid="stTickBar"] { background: transparent !important; }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] { display: flex !important; flex-wrap: wrap !important; gap: 8px !important; }
    div[data-testid="stRadio"] label { flex: 1 !important; min-width: 110px !important; background: #0d1525 !important; border: 1.5px solid #1a2e50 !important; border-radius: 10px !important; padding: 12px 14px !important; cursor: pointer !important; transition: all 0.18s ease !important; color: #64748b !important; font-size: 0.82rem !important; font-weight: 500 !important; text-transform: none !important; letter-spacing: 0 !important; line-height: 1.3 !important; }
    div[data-testid="stRadio"] label:hover { border-color: #00d4ff70 !important; background: #0a182a !important; color: #e2e8f0 !important; }
    div[data-testid="stRadio"] label:has(input:checked) { border-color: #00d4ff !important; background: #00d4ff16 !important; color: #00d4ff !important; box-shadow: 0 0 14px #00d4ff20 !important; }
    div[data-testid="stRadio"] label input { display: none !important; }
    div[data-testid="stRadio"] label > div:first-child { display: none !important; }
    
    [data-testid="stTable"] table { background: #0d1525 !important; border-radius: 12px !important; overflow: hidden !important; }
    [data-testid="stTable"] th { background: #111d35 !important; color: #64748b !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 1px !important; border-bottom: 1px solid #1a2e50 !important; padding: 10px 14px !important; }
    [data-testid="stTable"] td { color: #e2e8f0 !important; border-bottom: 1px solid #1a2e5050 !important; background: transparent !important; padding: 9px 14px !important; }
    [data-testid="stExpander"] { background: #0d1525 !important; border: 1px solid #1a2e50 !important; border-radius: 12px !important; }
    [data-testid="stExpander"] summary { color: #64748b !important; }
    [data-testid="stExpander"] p, [data-testid="stExpander"] li { color: #94a3b8 !important; }
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #070d1a; }
    ::-webkit-scrollbar-thumb { background: #1a2e50; border-radius: 3px; }
    hr { border-color: #1a2e50 !important; }
    </style>"""
    st.write(css, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = dict(
        step=0,
        age=50, sex="Male",
        cp="Typical Angina (cp=0)",
        trestbps=120, chol=240, thalach=150, oldpeak=1.0,
        fbs="No", restecg="Normal", exang="No",
        slope="Flat", ca=0, thal="Normal",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
#  PREPROCESS + RAW INPUT
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(raw: dict):
    # Log transformation required by your specific .pkl files
    df = pd.DataFrame([raw])
    df['oldpeak_log'] = np.log1p(df['oldpeak'])
    df.drop('oldpeak', axis=1, inplace=True)
    df = df[model_columns]
    scaled = scaler.transform(df)
    return scaled, df

def build_raw() -> dict:
    sex_map   = {"Male": 1, "Female": 0}
    cp_map    = {"Typical Angina (cp=0)": 0, "Atypical Angina (cp=1)": 1,
                 "Non-anginal Pain (cp=2)": 2, "Asymptomatic (cp=3)": 3}
    fbs_map   = {"Yes": 1, "No": 0}
    ecg_map   = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
    exang_map = {"Yes": 1, "No": 0}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map  = {"Fixed Defect": 1, "Normal": 2, "Reversible Defect": 3}
    return {
        'age':      st.session_state.age,
        'sex':      sex_map[st.session_state.sex],
        'cp':       cp_map[st.session_state.cp],
        'trestbps': st.session_state.trestbps,
        'chol':     st.session_state.chol,
        'fbs':      fbs_map[st.session_state.fbs],
        'restecg':  ecg_map[st.session_state.restecg],
        'thalach':  st.session_state.thalach,
        'exang':    exang_map[st.session_state.exang],
        'oldpeak':  float(st.session_state.oldpeak),
        'slope':    slope_map[st.session_state.slope],
        'ca':       int(st.session_state.ca),
        'thal':     thal_map[st.session_state.thal],
    }

# ─────────────────────────────────────────────────────────────────────────────
#  STEP PROGRESS INDICATOR
# ─────────────────────────────────────────────────────────────────────────────
def render_progress(current: int):
    steps = ["Personal", "Vitals", "Cardiac"]
    html = '<div style="display:flex;align-items:flex-start;max-width:380px;margin:24px auto 40px auto;">'
    for i, label in enumerate(steps):
        n = i + 1
        done    = n < current
        active  = n == current
        bg      = C_ACCNT  if (done or active) else "transparent"
        bd      = C_ACCNT  if (done or active) else C_BORD
        tc      = "#070d1a" if (done or active) else C_MUTED
        lc      = C_TEXT   if active else (C_ACCNT if done else C_MUTED)
        icon    = "✓"      if done  else str(n)
        glow    = f"box-shadow:0 0 16px {C_ACCNT}55;" if active else ""

        html += f"""
        <div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;gap:6px;">
          <div style="width:36px;height:36px;border-radius:50%;background:{bg};border:2px solid {bd};
                      display:flex;align-items:center;justify-content:center;
                      font-size:13px;font-weight:700;color:{tc};{glow}">{icon}</div>
          <span style="font-size:10px;color:{lc};font-weight:600;letter-spacing:0.5px;">{label}</span>
        </div>"""
        if i < len(steps) - 1:
            lc2 = C_ACCNT if done else C_BORD
            html += f'<div style="flex:1;height:2px;background:{lc2};margin:17px 4px 0 4px;border-radius:2px;"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ANIMATED GAUGE
# ─────────────────────────────────────────────────────────────────────────────
def render_gauge(risk_pct: float):
    color     = risk_color(risk_pct)
    arc_total = 408.41                         
    glow_blur = max(3, int(risk_pct / 16))

    gauge_html = f"""<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  html,body{{margin:0;padding:0;background-color: #0d1525; display:flex;flex-direction:column;align-items:center;overflow:hidden;}}
  .risk-label{{font-family:'DM Sans',sans-serif;font-size:14px;font-weight:700;letter-spacing:3px;
               color:{color};text-transform:uppercase;text-shadow:0 0 18px {color}88;margin-top:-4px;}}
  .risk-sub{{font-family:'DM Sans',sans-serif;font-size:10px;color:#3a4a60;letter-spacing:2px;
             text-transform:uppercase;margin-top:5px;}}
</style></head><body>
<svg viewBox="0 0 320 200" width="320" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#10b981"/>
      <stop offset="42%"  stop-color="#f59e0b"/>
      <stop offset="100%" stop-color="#ef4444"/>
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="{glow_blur}" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="sg" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <path d="M 30 155 A 130 130 0 0 1 290 155" fill="none" stroke="#0a1628" stroke-width="22" stroke-linecap="round"/>
  <path d="M 30 155 A 130 130 0 0 1 290 155" fill="none" stroke="url(#g)" stroke-width="22" stroke-linecap="round" opacity="0.1"/>

  <path id="arc" d="M 30 155 A 130 130 0 0 1 290 155"
        fill="none" stroke="url(#g)" stroke-width="18" stroke-linecap="round"
        stroke-dasharray="{arc_total:.2f}" stroke-dashoffset="{arc_total:.2f}"
        filter="url(#glow)"
        style="transition:stroke-dashoffset 1.9s cubic-bezier(0.34,1.56,0.64,1)"/>

  <g id="ndl" style="transform-origin:160px 155px;transform:rotate(-90deg); transition:transform 1.9s cubic-bezier(0.34,1.56,0.64,1)">
    <rect x="158.8" y="44" width="2.4" height="116" rx="1.2" fill="white" opacity="0.9"/>
    <circle cx="160" cy="50" r="5.5" fill="{color}" filter="url(#glow)" opacity="0.7"/>
    <circle cx="160" cy="50" r="3"   fill="{color}"/>
  </g>

  <circle cx="160" cy="155" r="11" fill="#070d1a" stroke="white" stroke-width="2.5"/>
  <circle cx="160" cy="155" r="4.5" fill="{color}" filter="url(#sg)"/>

  <text id="pct" x="160" y="118" text-anchor="middle" font-size="46" fill="white"
        font-family="'DM Mono',monospace" font-weight="500" filter="url(#sg)">0%</text>

  <text x="16"  y="178" text-anchor="middle" font-size="8.5" fill="#10b981" font-family="DM Sans" font-weight="600">LOW</text>
  <text x="160" y="26"  text-anchor="middle" font-size="8.5" fill="#f59e0b" font-family="DM Sans" font-weight="600">MOD</text>
  <text x="304" y="178" text-anchor="middle" font-size="8.5" fill="#ef4444" font-family="DM Sans" font-weight="600">HIGH</text>

  <line x1="30"  y1="149" x2="30"  y2="161" stroke="#2a3d5a" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="290" y1="149" x2="290" y2="161" stroke="#2a3d5a" stroke-width="1.5" stroke-linecap="round"/>
</svg>

<div class="risk-label">{risk_label(risk_pct)}</div>
<div class="risk-sub">Clinical Risk Estimate</div>

<script>
  const riskVal  = {risk_pct:.2f};
  const total    = {arc_total:.2f};
  const targetOff = total * (1 - riskVal / 100);
  const targetRot = -90 + (riskVal / 100) * 180;

  setTimeout(() => {{
    document.getElementById('arc').style.strokeDashoffset = targetOff;
    document.getElementById('ndl').style.transform = `rotate(${{targetRot}}deg)`;

    let t0 = null, dur = 1900;
    function easeOutElastic(x) {{
      const c4 = (2 * Math.PI) / 4.5;
      return x === 0 ? 0 : x === 1 ? 1 : Math.pow(2, -10*x) * Math.sin((x*10 - 0.75) * c4) + 1;
    }}
    function frame(ts) {{
      if (!t0) t0 = ts;
      const p = Math.min((ts - t0) / dur, 1);
      const v = riskVal * easeOutElastic(p);
      document.getElementById('pct').textContent = v.toFixed(1) + '%';
      if (p < 1) requestAnimationFrame(frame);
      else document.getElementById('pct').textContent = riskVal.toFixed(1) + '%';
    }}
    requestAnimationFrame(frame);
  }}, 180);
</script>
</body></html>"""
    
    b64 = base64.b64encode(gauge_html.encode('utf-8')).decode('utf-8')
    st.markdown(f'<iframe src="data:text/html;base64,{b64}" width="100%" height="295" style="border:none;"></iframe>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE CONTRIBUTION CHART
# ─────────────────────────────────────────────────────────────────────────────
def render_contribution_chart(scaled_data):
    weights       = model.coef_[0]
    contributions = -(scaled_data[0] * weights)
    contrib_df    = pd.DataFrame({
        'Feature':      [FEATURE_LABELS[c] for c in model_columns],
        'Contribution': contributions,
    }).sort_values('Contribution', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor('#0d1525')
    ax.set_facecolor('#0d1525')

    bar_colors = ['#ef4444' if v > 0 else '#10b981' for v in contrib_df['Contribution']]

    for i, (v, c) in enumerate(zip(contrib_df['Contribution'], bar_colors)):
        ax.barh(i, v, height=0.55, color=c, alpha=0.15, edgecolor='none')

    ax.barh(range(len(contrib_df)), contrib_df['Contribution'],
            height=0.55, color=bar_colors, edgecolor='none', zorder=3)

    ax.axvline(0, color='#2a3d5a', lw=1.2, zorder=4)
    ax.set_yticks(range(len(contrib_df)))
    ax.set_yticklabels(contrib_df['Feature'], fontsize=8.5, color='#94a3b8')
    ax.tick_params(axis='x', labelsize=7.5, colors='#4a5568', length=0)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel("Contribution to Disease Risk", fontsize=8, color='#4a5568', labelpad=8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(axis='x', color='#1a2e50', linewidth=0.5, linestyle='--', alpha=0.6, zorder=0)

    red_p   = mpatches.Patch(facecolor='#ef4444', label='↑ Increases risk', edgecolor='none')
    green_p = mpatches.Patch(facecolor='#10b981', label='↓ Reduces risk',   edgecolor='none')
    ax.legend(handles=[red_p, green_p], fontsize=7.5, frameon=False,
              labelcolor='#94a3b8', loc='lower right')

    plt.tight_layout(pad=0.6)
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  PDF GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(risk_pct: float, raw: dict, scaled_data) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=14*mm, bottomMargin=14*mm)
    W   = A4[0] - 40*mm
    story = []

    hex_c   = risk_color(risk_pct).lstrip('#')
    r,g,b   = int(hex_c[0:2],16)/255, int(hex_c[2:4],16)/255, int(hex_c[4:6],16)/255
    rc      = colors.Color(r, g, b)
    navy    = colors.Color(0.04, 0.07, 0.12)
    card_bg = colors.Color(0.07, 0.10, 0.18)
    alt_bg  = colors.Color(0.075, 0.105, 0.19)
    lt      = colors.Color(0.88, 0.91, 0.94)
    mt      = colors.Color(0.39, 0.45, 0.55)
    bord    = colors.Color(0.10, 0.18, 0.31)

    def p(txt, sz=10, bold=False, col=None, align=TA_LEFT, sa=4):
        c  = col or lt
        fn = 'Helvetica-Bold' if bold else 'Helvetica'
        s  = ParagraphStyle('_', fontName=fn, fontSize=sz, textColor=c, alignment=align, spaceAfter=sa*mm, leading=sz*1.45)
        return Paragraph(txt, s)

    base_ts = [
        ('TOPPADDING',    (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 10), ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('GRID',          (0,0), (-1,-1), 0.4, bord),
    ]

    ht = Table([[p("🫀  CardioAI", 19, bold=True, col=lt), p("Clinical Risk Assessment Report", 9, col=mt)]], colWidths=[W*0.55, W*0.45])
    ht.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), navy), ('TOPPADDING',    (0,0), (-1,-1), 14), ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING',   (0,0), (-1,-1), 16), ('RIGHTPADDING',  (0,0), (-1,-1), 16), ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story += [ht, Spacer(1, 5*mm)]
    story.append(p(f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}", 8, col=mt, sa=3))
    story.append(Spacer(1, 4*mm))

    rb = Table([[
        p(f"{risk_pct:.1f}%", 38, bold=True, col=rc, align=TA_CENTER),
        p(f"<b>{risk_label(risk_pct)}</b><br/><br/>Statistical probability of heart disease based on 13 clinical features.", 10, col=lt),
    ]], colWidths=[W*0.28, W*0.72])
    rb.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), card_bg), ('TOPPADDING',   (0,0), (-1,-1), 18), ('BOTTOMPADDING', (0,0), (-1,-1), 18),
        ('LEFTPADDING',  (0,0), (-1,-1), 18), ('RIGHTPADDING',  (0,0), (-1,-1), 18), ('BOX',          (0,0), (-1,-1), 1.5, bord),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'), ('LINEAFTER',    (0,0), (0,-1),  2,   rc),
    ]))
    story += [rb, Spacer(1, 5*mm)]

    if risk_pct >= 60:   rec = "Consult a cardiologist immediately. Detailed cardiac diagnostics are strongly advised."
    elif risk_pct >= 30: rec = "Schedule a physician follow-up. Further evaluation is recommended."
    else:                rec = "Maintain a healthy lifestyle. Continue routine annual check-ups."
    rct = Table([[p(f"Recommendation: {rec}", 9, col=lt)]], colWidths=[W])
    rct.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), colors.Color(r*0.25, g*0.25, b*0.25, 0.4)),
        ('TOPPADDING',    (0,0), (-1,-1), 10), ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING',   (0,0), (-1,-1), 14), ('RIGHTPADDING',  (0,0), (-1,-1), 14), ('LINEBEFORE',    (0,0), (0,-1),  3,   rc),
    ]))
    story += [rct, Spacer(1, 6*mm)]

    story += [p("Patient Data", 11, bold=True, col=lt, sa=2), HRFlowable(width=W, thickness=0.5, color=bord, spaceAfter=4*mm)]
    sex_s  = "Male" if raw['sex'] else "Female"
    vitals = [
        ["Parameter",      "Value",                  "Parameter",       "Value"],
        ["Age",             f"{raw['age']} yrs",      "Sex",              sex_s],
        ["Resting BP",      f"{raw['trestbps']} mmHg","Cholesterol",      f"{raw['chol']} mg/dl"],
        ["Max Heart Rate",  f"{raw['thalach']} bpm",  "ST Depression",    str(raw['oldpeak'])],
        ["Chest Pain",      CP_LABELS[raw['cp']],     "Exercise Angina",  "Yes" if raw['exang'] else "No"],
        ["Resting ECG",     ECG_LABELS[raw['restecg']],"High Fasting BS", "Yes" if raw['fbs'] else "No"],
        ["ST Slope",        SLOPE_LABELS[raw['slope']],"Major Vessels",   str(raw['ca'])],
        ["Thalassemia",     THAL_LABELS[raw['thal']], "",                 ""],
    ]
    cw = W/4
    vt = Table(vitals, colWidths=[cw*1.25, cw*0.75, cw*1.25, cw*0.75])
    vt.setStyle(TableStyle(base_ts + [
        ('BACKGROUND',    (0,0), (-1, 0), navy), ('FONTNAME',      (0,0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE',      (0,0), (-1,-1), 8.5),
        ('TEXTCOLOR',     (0,0), (-1, 0), mt), ('TEXTCOLOR',     (0,1), (-1,-1), lt),
        ('TEXTCOLOR',     (0,1), (0,-1),  mt), ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (2,1), (2,-1),  mt), ('FONTNAME', (2,1), (2,-1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [card_bg, alt_bg]),
    ]))
    story += [vt, Spacer(1, 6*mm)]

    story += [p("Top Risk Drivers", 11, bold=True, col=lt, sa=2), HRFlowable(width=W, thickness=0.5, color=bord, spaceAfter=4*mm)]
    contribs = -(scaled_data[0] * model.coef_[0])
    cdf = pd.DataFrame({'Feature': [FEATURE_LABELS[c] for c in model_columns], 'Contribution': contribs}).sort_values('Contribution', ascending=False)
    dd = [["#", "Feature", "Score", "Effect"]]
    for i, (_, row) in enumerate(cdf.head(6).iterrows(), 1):
        dd.append([str(i), row['Feature'], f"{abs(row['Contribution']):.3f}", "↑ Increases risk" if row['Contribution'] > 0 else "↓ Reduces risk"])
    dt = Table(dd, colWidths=[W*0.07, W*0.44, W*0.18, W*0.31])
    dt.setStyle(TableStyle(base_ts + [
        ('BACKGROUND',    (0,0), (-1, 0), navy), ('FONTNAME',      (0,0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE',      (0,0), (-1,-1), 8.5),
        ('TEXTCOLOR',     (0,0), (-1, 0), mt), ('TEXTCOLOR',     (0,1), (-1,-1), lt),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [card_bg, alt_bg]), ('ALIGN',         (2,0), (2,-1), 'CENTER'),
    ]))
    story += [dt, Spacer(1, 6*mm)]

    story += [p("Model Information", 11, bold=True, col=lt, sa=2), HRFlowable(width=W, thickness=0.5, color=bord, spaceAfter=4*mm)]
    mi = [
        ["Algorithm",       "Logistic Regression"], ["Dataset",         "Cleveland Heart Disease — 302 samples"],
        ["Test Accuracy",   "~85%"], ["AUC-ROC",         "~0.90"],
        ["Target Encoding", "target=0 → Disease | target=1 → Healthy"],
    ]
    mt2 = Table(mi, colWidths=[W*0.32, W*0.68])
    mt2.setStyle(TableStyle(base_ts + [
        ('FONTSIZE',      (0,0), (-1,-1), 8.5), ('TEXTCOLOR',     (0,0), (0,-1), mt), ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (1,0), (1,-1), lt), ('ROWBACKGROUNDS',(0,0), (-1,-1), [card_bg, alt_bg]),
    ]))
    story += [mt2, Spacer(1, 7*mm)]

    disc = Table([[p("Disclaimer: This report is generated by an AI model for educational purposes (INT428 AI Essentials). It is NOT a medical diagnosis. Always consult a qualified healthcare professional for clinical decisions.", 7.5, col=mt)]], colWidths=[W])
    disc.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), colors.Color(0.06,0.06,0.1)), ('BOX',           (0,0), (-1,-1), 0.5, bord),
        ('TOPPADDING',    (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 12), ('RIGHTPADDING',  (0,0), (-1,-1), 12),
    ]))
    story.append(disc)

    doc.build(story)
    return buf.getvalue()

# ═════════════════════════════════════════════════════════════════════════════
#  PAGES
# ═════════════════════════════════════════════════════════════════════════════

def page_welcome():
    st.markdown(f"""
    <div style="text-align:center;padding:64px 0 8px;">
      <div style="font-size:60px;line-height:1;margin-bottom:18px; filter:drop-shadow(0 0 20px #ef444488);">🫀</div>
      <h1 style="font-size:2.8rem;font-weight:800;letter-spacing:-1px;margin:0; background:linear-gradient(135deg,#00d4ff 0%,#ffffff 65%); -webkit-background-clip:text;-webkit-text-fill-color:transparent; background-clip:text;">CardioAI</h1>
    <p style="font-size:1rem;color:{C_MUTED};max-width:440px;margin:12px auto 0;line-height:1.75;">A <strong style="color:#94a3b8;">Logistic Regression</strong> model trained on <strong style="color:#94a3b8;">4500 patient records</strong> gives you a statistical estimate of heart disease probability in under a minute.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    stats = [("~85%","Test Accuracy"),("~0.90","AUC-ROC"),("4500","Training Patients"),("13","Clinical Features")]
    cols  = st.columns(4)
    for col, (val, lbl) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div style="background:{C_CARD};border:1px solid {C_BORD};border-radius:14px; padding:22px 10px;text-align:center; transition:border-color 0.2s;">
              <div style="font-size:1.7rem;font-weight:700;color:{C_ACCNT}; font-family:'DM Mono',monospace;">{val}</div>
              <div style="font-size:0.68rem;color:{C_MUTED};margin-top:5px; letter-spacing:1.2px;text-transform:uppercase;font-weight:600;">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    _, c, _ = st.columns([1.2, 1, 1.2])
    with c:
        if st.button("Begin Assessment  →", type="primary"):
            st.session_state.step = 1
            st.rerun()

    st.markdown(f'<p style="text-align:center;font-size:0.72rem;color:#2a3d5a;margin-top:24px;">INT428 — AI Essentials &nbsp;·&nbsp; For educational purposes only</p>', unsafe_allow_html=True)

def page_step1():
    render_progress(1)
    st.markdown("<h2 style='text-align:center;'>Personal Information</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:{C_MUTED};margin-bottom:30px;'>Let's start with the basics about the patient.</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"<p style='font-size:0.72rem;font-weight:600;letter-spacing:1px;color:{C_MUTED};text-transform:uppercase;'>Age (years)</p>", unsafe_allow_html=True)
        age = st.slider("age_sl", 20, 80, st.session_state.age, label_visibility="collapsed")
        st.session_state.age = age

        st.write("")
        st.markdown(f"<p style='font-size:0.72rem;font-weight:600;letter-spacing:1px;color:{C_MUTED};text-transform:uppercase;'>Biological Sex</p>", unsafe_allow_html=True)
        sex = st.radio("sex_rad", ["Male", "Female"], index=0 if st.session_state.sex == "Male" else 1, horizontal=True, label_visibility="collapsed")
        st.session_state.sex = sex

        st.write("")
        if st.button("Next: Vital Signs  →", type="primary"):
            st.session_state.step = 2; st.rerun()

def page_step2():
    render_progress(2)
    st.markdown("<h2 style='text-align:center;'>Vital Signs</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:{C_MUTED};margin-bottom:30px;'>Basic clinical measurements from the patient's record.</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([0.5, 3, 0.5])
    with c2:
        row1_1, row1_2 = st.columns(2)
        with row1_1:
            st.markdown(f"<p style='font-size:0.72rem;font-weight:600;letter-spacing:1px;color:{C_MUTED};text-transform:uppercase;'>Resting Blood Pressure <span style='color:#334155'>(mm Hg)</span></p>", unsafe_allow_html=True)
            trestbps = st.number_input("bp", 80, 200, st.session_state.trestbps, label_visibility="collapsed")
            st.session_state.trestbps = trestbps
        with row1_2:
            st.markdown(f"<p style='font-size:0.72rem;font-weight:600;letter-spacing:1px;color:{C_MUTED};text-transform:uppercase;'>Serum Cholesterol <span style='color:#334155'>(mg/dl)</span></p>", unsafe_allow_html=True)
            chol = st.number_input("ch", 100, 600, st.session_state.chol, label_visibility="collapsed")
            st.session_state.chol = chol

        st.write("")
        row2_1, row2_2 = st.columns(2)
        with row2_1:
            st.markdown(f"<p style='font-size:0.72rem;font-weight:600;letter-spacing:1px;color:{C_MUTED};text-transform:uppercase;'>Max Heart Rate <span style='color:#334155'>(bpm)</span></p>", unsafe_allow_html=True)
            thalach = st.number_input("hr", 60, 220, st.session_state.thalach, label_visibility="collapsed")
            st.session_state.thalach = thalach
        with row2_2:
            st.markdown(f"<p style='font-size:0.72rem;font-weight:600;letter-spacing:1px;color:{C_MUTED};text-transform:uppercase;'>ST Depression <span style='color:#334155'>(oldpeak)</span></p>", unsafe_allow_html=True)
            oldpeak = st.number_input("op", 0.0, 6.2, float(st.session_state.oldpeak), step=0.1, label_visibility="collapsed")
            st.session_state.oldpeak = oldpeak

        st.write("")
        st.markdown(f"<p style='font-size:0.72rem;font-weight:600;letter-spacing:1px;color:{C_MUTED};text-transform:uppercase;'>Fasting Blood Sugar &gt; 120 mg/dl?</p>", unsafe_allow_html=True)
        fbs = st.radio("fbs_r", ["No", "Yes"], index=0 if st.session_state.fbs == "No" else 1, horizontal=True, label_visibility="collapsed")
        st.session_state.fbs = fbs

        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("←  Back", type="secondary"):
                st.session_state.step = 1; st.rerun()
        with b2:
            if st.button("Next: Cardiac Findings  →", type="primary"):
                st.session_state.step = 3; st.rerun()

def page_step3():
    render_progress(3)
    st.markdown("<h2 style='text-align:center;'>Cardiac Examination</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:{C_MUTED};margin-bottom:30px;'>Clinical findings from the cardiac workup.</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([0.5, 3, 0.5])
    with c2:
        st.markdown(f'<p style="font-size:0.7rem;font-weight:600;letter-spacing:1.5px;color:{C_MUTED};text-transform:uppercase;">Chest Pain Type</p>', unsafe_allow_html=True)
        cp_opts = ["Typical Angina (cp=0)", "Atypical Angina (cp=1)", "Non-anginal Pain (cp=2)", "Asymptomatic (cp=3)"]
        cp = st.radio("cp_rad", cp_opts, index=cp_opts.index(st.session_state.cp) if st.session_state.cp in cp_opts else 0, horizontal=True, label_visibility="collapsed")
        st.session_state.cp = cp

        st.write("")
        row1_1, row1_2 = st.columns(2)
        with row1_1:
            st.markdown(f'<p style="font-size:0.7rem;font-weight:600;letter-spacing:1.5px;color:{C_MUTED};text-transform:uppercase;">Exercise-Induced Chest Pain?</p>', unsafe_allow_html=True)
            exang = st.radio("ex_rad", ["No", "Yes"], index=0 if st.session_state.exang == "No" else 1, horizontal=True, label_visibility="collapsed")
            st.session_state.exang = exang
        with row1_2:
            st.markdown(f'<p style="font-size:0.7rem;font-weight:600;letter-spacing:1.5px;color:{C_MUTED};text-transform:uppercase;">Resting ECG Result</p>', unsafe_allow_html=True)
            ecg_opts = ["Normal", "ST-T Abnormality", "LV Hypertrophy"]
            restecg = st.radio("ecg_rad", ecg_opts, index=ecg_opts.index(st.session_state.restecg) if st.session_state.restecg in ecg_opts else 0, horizontal=True, label_visibility="collapsed")
            st.session_state.restecg = restecg

        st.write("")
        row2_1, row2_2, row2_3 = st.columns(3)
        with row2_1:
            st.markdown(f'<p style="font-size:0.7rem;font-weight:600;letter-spacing:1.5px;color:{C_MUTED};text-transform:uppercase;">ST Segment Slope</p>', unsafe_allow_html=True)
            slope_opts = ["Upsloping", "Flat", "Downsloping"]
            slope = st.radio("slope_rad", slope_opts, index=slope_opts.index(st.session_state.slope) if st.session_state.slope in slope_opts else 1, label_visibility="collapsed")
            st.session_state.slope = slope
        with row2_2:
            st.markdown(f'<p style="font-size:0.7rem;font-weight:600;letter-spacing:1.5px;color:{C_MUTED};text-transform:uppercase;">Major Vessels (0–3)</p>', unsafe_allow_html=True)
            ca = st.radio("ca_rad", [0, 1, 2, 3], index=int(st.session_state.ca), label_visibility="collapsed", horizontal=True)
            st.session_state.ca = ca
        with row2_3:
            st.markdown(f'<p style="font-size:0.7rem;font-weight:600;letter-spacing:1.5px;color:{C_MUTED};text-transform:uppercase;">Thalassemia Status</p>', unsafe_allow_html=True)
            thal_opts = ["Fixed Defect", "Normal", "Reversible Defect"]
            thal = st.radio("thal_rad", thal_opts, index=thal_opts.index(st.session_state.thal) if st.session_state.thal in thal_opts else 1, label_visibility="collapsed")
            st.session_state.thal = thal

        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("←  Back", type="secondary"):
                st.session_state.step = 2; st.rerun()
        with b2:
            if st.button("🔍  Run Analysis", type="primary"):
                st.session_state.step = 4; st.rerun()

def page_results():
    raw         = build_raw()
    scaled, _   = preprocess(raw)
    risk_pct    = model.predict_proba(scaled)[:, 0][0] * 100
    color       = risk_color(risk_pct)

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center; padding:22px 0 26px;border-bottom:1px solid {C_BORD};margin-bottom:28px;">
      <div>
        <h2 style="margin:0;font-size:1.5rem;">Assessment Complete</h2>
        <p style="margin:4px 0 0;color:{C_MUTED};font-size:0.82rem;">{datetime.now().strftime('%B %d, %Y — %H:%M')}</p>
      </div>
      <div style="background:{color}18;border:1px solid {color}55;border-radius:10px; padding:8px 22px;font-size:0.78rem;font-weight:700;color:{color}; letter-spacing:2px;text-transform:uppercase;">
        {risk_label(risk_pct)}
      </div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1.35])
    with left:
        st.markdown(f"<p style='font-size:0.68rem;font-weight:600;letter-spacing:2px;color:{C_MUTED};text-transform:uppercase;text-align:center;margin-bottom:0;'>Risk Score</p>", unsafe_allow_html=True)
        render_gauge(risk_pct)

        if risk_pct >= 60:
            ico, txt, rb, rbd = "🚨", "Consult a cardiologist immediately.", f"{C_HIGH}12", f"{C_HIGH}45"
        elif risk_pct >= 30:
            ico, txt, rb, rbd = "⚠️", "Schedule a physician follow-up.",     f"{C_MID}12",  f"{C_MID}45"
        else:
            ico, txt, rb, rbd = "✅", "Healthy lifestyle + routine check-ups.", f"{C_LOW}12", f"{C_LOW}45"

        st.markdown(f"""
        <div style="background:{rb};border:1px solid {rbd};border-radius:12px; padding:16px 20px;text-align:center;margin-top:4px;">
          <div style="font-size:1.5rem;">{ico}</div>
          <p style="font-size:0.83rem;color:{C_TEXT};margin:6px 0 0;line-height:1.55;">{txt}</p>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"<p style='font-size:0.68rem;font-weight:600;letter-spacing:2px;color:{C_MUTED};text-transform:uppercase;margin-bottom:12px;'>Patient Summary</p>", unsafe_allow_html=True)

        vitals_display = [
            ("Age",           f"{raw['age']} yrs"),
            ("Sex",           "Male" if raw['sex'] else "Female"),
            ("Chest Pain",    CP_LABELS[raw['cp']]),
            ("Resting BP",    f"{raw['trestbps']} mm Hg"),
            ("Cholesterol",   f"{raw['chol']} mg/dl"),
            ("Max HR",        f"{raw['thalach']} bpm"),
            ("ST Depression", str(raw['oldpeak'])),
            ("Exer. Angina",  "Yes" if raw['exang'] else "No"),
            ("ECG",           ECG_LABELS[raw['restecg']]),
            ("ST Slope",      SLOPE_LABELS[raw['slope']]),
            ("Vessels",       str(raw['ca'])),
            ("Thalassemia",   THAL_LABELS[raw['thal']]),
        ]
        rows = ""
        for i, (k, v) in enumerate(vitals_display):
            bg = C_CARD if i%2==0 else "rgba(255,255,255,0.018)"
            rows += f'<div style="display:flex;justify-content:space-between;align-items:center; padding:8px 16px;background:{bg};border-bottom:1px solid {C_BORD}38;"><span style="font-size:0.78rem;color:{C_MUTED};font-weight:500;">{k}</span><span style="font-size:0.82rem;color:{C_TEXT};font-weight:600;">{v}</span></div>'
        st.markdown(f'<div style="border:1px solid {C_BORD};border-radius:12px;overflow:hidden;">{rows}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.68rem;font-weight:600;letter-spacing:2px;color:{C_MUTED};text-transform:uppercase;margin-bottom:12px;'>Feature Contributions — What's Driving This Score?</p>", unsafe_allow_html=True)
    render_contribution_chart(scaled)

    st.markdown("<br>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("← Start Over", type="secondary"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
    with a2:
        if st.button("← Edit Inputs", type="secondary"):
            st.session_state.step = 3; st.rerun()
    with a3:
        pdf_data = generate_pdf(risk_pct, raw, scaled)
        st.download_button(
            label="📄  Download PDF Report",
            data=pdf_data,
            file_name=f"CardioAI_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

# ═════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════
def main():
    inject_css()
    init_state()
    {0: page_welcome, 1: page_step1, 2: page_step2,
     3: page_step3,  4: page_results}.get(
        st.session_state.step, page_welcome)()

if __name__ == "__main__":
    main()