"""
AMP Discovery Pipeline — Streamlit Web Application
====================================================
Purpose:
    Interactive web interface for antimicrobial peptide
    prediction and explainability.

Design:
    Dark forest green sidebar. Clean white main area.
    Color-coded amino acids. SHAP waterfall chart.
    Radar chart. Multi-model consensus. Downloadable report.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from src.feature_engineering import (
    calculate_physicochemical_features,
    calculate_aa_composition,
    is_valid_sequence
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AMP Discovery | Peptide Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;1,400&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: #F7F9F7;
    color: #1C2B1E;
}

.main { background: #F7F9F7; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1300px; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1C2B1E !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.7) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* Radio buttons in sidebar */
[data-testid="stSidebar"] .stRadio label {
    background: transparent;
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 0.85rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    color: rgba(255,255,255,0.6) !important;
    transition: all 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.9) !important;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #1C2B1E 0%, #2D5A35 60%, #3A7044 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '🧬';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.12;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #fff;
    margin: 0 0 0.4rem;
    position: relative; z-index: 1;
}
.hero p {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.65);
    margin: 0 0 1.2rem;
    max-width: 550px;
    line-height: 1.6;
    position: relative; z-index: 1;
}
.hero-badges {
    display: flex; gap: 8px; flex-wrap: wrap;
    position: relative; z-index: 1;
}
.hbadge {
    background: rgba(255,255,255,0.12);
    border: 0.5px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.85);
    font-weight: 500;
}

/* Cards */
.card {
    background: #ffffff;
    border: 0.5px solid #D8E8DA;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(28,43,30,0.06);
}
.card-label {
    font-size: 0.72rem;
    color: #5A8C5E;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* Result */
.result-amp {
    background: linear-gradient(135deg, #EAF4EC, #F7FAF7);
    border: 1.5px solid #4CAF60;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.result-non-amp {
    background: linear-gradient(135deg, #FCEEF0, #FBF7F7);
    border: 1.5px solid #C0606A;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.pred-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    margin: 0.4rem 0;
}
.pred-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 500;
    line-height: 1;
}
.pred-sublabel {
    font-size: 0.78rem;
    color: #7A9A7C;
    margin-top: 4px;
}

/* Sequence display */
.seq-box {
    background: #1C2B1E;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    word-break: break-all;
    line-height: 2;
    margin: 0.5rem 0;
}
.aa-pos { color: #6FCFA0; font-weight: 500; }
.aa-neg { color: #E07B7B; font-weight: 500; }
.aa-hph { color: #A8D5B0; }
.aa-oth { color: rgba(255,255,255,0.45); }

/* Chips */
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.75rem; }
.chip {
    font-size: 0.72rem;
    padding: 4px 12px;
    border-radius: 20px;
    background: #EEF6EF;
    border: 0.5px solid #C4DEC6;
    color: #2D5A35;
    font-weight: 500;
}

/* SHAP bars */
.shap-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 7px;
}
.shap-feat {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #4A6B4D;
    width: 100px;
    flex-shrink: 0;
}
.shap-track {
    flex: 1;
    height: 7px;
    background: #F0F5F1;
    border-radius: 4px;
    overflow: hidden;
}
.shap-fill-pos {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #2D7A3A, #6FCFA0);
}
.shap-fill-neg {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #C0606A, #E07B7B);
}
.shap-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    width: 48px;
    text-align: right;
}
.shap-val-pos { color: #2D7A3A; }
.shap-val-neg { color: #C0606A; }

/* Model bars */
.model-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.model-name {
    font-size: 0.78rem;
    color: #4A6B4D;
    width: 120px;
    flex-shrink: 0;
}
.model-track {
    flex: 1;
    height: 10px;
    background: #F0F5F1;
    border-radius: 5px;
    overflow: hidden;
}
.model-fill {
    height: 100%;
    border-radius: 5px;
    background: linear-gradient(90deg, #1C2B1E, #4CAF60);
}
.model-fill-low {
    height: 100%;
    border-radius: 5px;
    background: linear-gradient(90deg, #C0606A, #E07B7B);
}
.model-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    width: 40px;
    text-align: right;
}

/* Info / warning boxes */
.info-box {
    background: #EAF4EC;
    border-left: 3px solid #2D7A3A;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.75rem 0;
    font-size: 0.83rem;
    color: #2D5A35;
    line-height: 1.6;
}
.warn-box {
    background: #FEF3E2;
    border-left: 3px solid #E8923A;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.75rem 0;
    font-size: 0.83rem;
    color: #92570A;
    line-height: 1.6;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1C2B1E, #2D7A3A);
    color: white !important;
    border: none;
    border-radius: 25px;
    padding: 0.5rem 1.75rem;
    font-family: 'Outfit', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(45,122,58,0.35);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #EEF6EF;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #5A8C5E;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: white;
    color: #1C2B1E;
}

/* Textarea */
.stTextArea textarea {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    border: 1.5px solid #D8E8DA;
    border-radius: 10px;
    background: #FAFCFA;
    color: #1C2B1E;
    letter-spacing: 0.04em;
}
.stTextArea textarea:focus {
    border-color: #2D7A3A;
    box-shadow: 0 0 0 3px rgba(45,122,58,0.1);
}

/* Progress */
.stProgress > div > div {
    background: linear-gradient(90deg, #1C2B1E, #4CAF60);
    border-radius: 10px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    color: #8AB08C;
    font-size: 0.75rem;
    border-top: 0.5px solid #D8E8DA;
    margin-top: 2rem;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

POSITIVE_AA = set('KRH')
NEGATIVE_AA = set('DE')
HYDROPHOBIC_AA = set('AVILMFYW')

EXAMPLE_SEQUENCES = {
    "Magainin-2 (frog skin AMP)": "GIGKFLHSAKKFGKAFVGEIMNS",
    "LL-37 fragment (human AMP)": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Indolicidin (bovine AMP)": "ILPWKWPWWPWRR",
    "Synthetic cationic AMP": "KLLLKWLLKWLKK",
    "Non-AMP control": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLSQTQARLNMVMVYRDGDGAM",
}

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    models = {}
    files = {
        'XGBoost': 'models/xgboost.pkl',
        'Random Forest': 'models/random_forest.pkl',
        'Logistic Regression': 'models/logistic_regression.pkl',
    }
    for name, path in files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    feature_names = joblib.load('models/feature_names.pkl')
    return models, feature_names

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model.named_steps['model'])

# ============================================================
# CORE FUNCTIONS
# ============================================================

def predict(sequence, pipeline, feature_names):
    seq = sequence.upper().strip()
    feats = {}
    feats.update(calculate_physicochemical_features(seq))
    feats.update(calculate_aa_composition(seq))
    vec = np.array([feats.get(f, 0.0) for f in feature_names]).reshape(1, -1)
    scaler = pipeline.named_steps['scaler']
    model  = pipeline.named_steps['model']
    vec_s  = scaler.transform(vec)
    prob   = model.predict_proba(vec_s)[0][1]
    return {
        'prediction': 'AMP' if prob >= 0.5 else 'non-AMP',
        'amp_probability': float(prob),
        'features': feats,
        'vec_scaled': vec_s,
    }

def get_shap(vec_scaled, explainer, feature_names):
    vals  = explainer.shap_values(vec_scaled)[0]
    pairs = list(zip(feature_names, vals))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs

def color_seq(seq):
    out = ''
    for aa in seq:
        if aa in POSITIVE_AA:
            out += f'<span class="aa-pos">{aa}</span>'
        elif aa in NEGATIVE_AA:
            out += f'<span class="aa-neg">{aa}</span>'
        elif aa in HYDROPHOBIC_AA:
            out += f'<span class="aa-hph">{aa}</span>'
        else:
            out += f'<span class="aa-oth">{aa}</span>'
    return out

# ============================================================
# PLOTS
# ============================================================

def plot_shap_bar(contributions, top_n=12):
    top    = contributions[:top_n]
    feats  = [c[0] for c in top]
    values = [c[1] for c in top]
    colors = ['#2D7A3A' if v > 0 else '#C0606A' for v in values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F7F9F7')

    bars = ax.barh(range(len(feats)), values,
                   color=colors, alpha=0.85, height=0.55,
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=9,
                       fontfamily='monospace', color='#2D5A35')
    ax.invert_yaxis()
    ax.axvline(x=0, color='#1C2B1E', linewidth=0.8, alpha=0.25)
    ax.set_xlabel('SHAP Value  (→ AMP  |  ← non-AMP)',
                  fontsize=9, color='#5A8C5E')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis='x', alpha=0.12, linestyle='--', color='#2D5A35')

    for bar, val in zip(bars, values):
        ax.text(val + (0.004 if val >= 0 else -0.004),
                bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=7.5, color='#4A6B4D',
                fontfamily='monospace')

    amp_p   = mpatches.Patch(color='#2D7A3A', label='→ AMP signal')
    noamp_p = mpatches.Patch(color='#C0606A', label='→ non-AMP signal')
    ax.legend(handles=[amp_p, noamp_p], fontsize=8,
              framealpha=0.95, edgecolor='#D8E8DA')
    plt.tight_layout()
    return fig


def plot_radar(features):
    cats = ['Charge', 'Hydrophobicity',
            'Positive\nFraction', 'Hydrophobic\nFraction',
            'Charge\nDensity']
    raw  = [
        features.get('charge', 0),
        features.get('hydrophobicity', 0),
        features.get('positive_fraction', 0),
        features.get('hydrophobic_fraction', 0),
        features.get('charge_density', 0),
    ]
    ranges = [(-5, 15), (-2, 2), (0, 0.5), (0, 0.8), (-0.3, 0.5)]
    norm = [max(0, min(1, (v - mn)/(mx - mn)))
            for v, (mn, mx) in zip(raw, ranges)]

    N      = len(cats)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    norm   += norm[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5),
                           subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F7F9F7')
    ax.plot(angles, norm, 'o-', lw=2,
            color='#2D7A3A', markersize=5)
    ax.fill(angles, norm, alpha=0.2, color='#4CAF60')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=8, color='#2D5A35')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25','0.50','0.75','1.0'],
                       size=6.5, color='#8AB08C')
    ax.grid(color='#C8E0CA', linewidth=0.5)
    ax.spines['polar'].set_visible(False)
    plt.tight_layout()
    return fig


def plot_consensus(seq, models, feature_names):
    names = []
    probs = []
    for name, pipeline in models.items():
        r = predict(seq, pipeline, feature_names)
        names.append(name)
        probs.append(r['amp_probability'])

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F7F9F7')

    colors = ['#2D7A3A' if p >= 0.5 else '#C0606A' for p in probs]
    bars   = ax.bar(names, probs, color=colors,
                    alpha=0.85, width=0.45,
                    edgecolor='white', linewidth=1)
    ax.axhline(y=0.5, color='#E8923A', lw=1.5,
               linestyle='--', alpha=0.8,
               label='Decision threshold (0.5)')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('AMP Probability', fontsize=9, color='#5A8C5E')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.12, linestyle='--', color='#2D5A35')
    ax.legend(fontsize=8, framealpha=0.95, edgecolor='#D8E8DA')

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.03,
                f'{prob:.3f}', ha='center', va='bottom',
                fontsize=9.5, fontweight='600',
                color='#1C2B1E', fontfamily='monospace')
    plt.tight_layout()
    return fig

# ============================================================
# SIDEBAR
# ============================================================

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:1rem 0 1.25rem;'>
          <div style='font-size:2.5rem;'>🧬</div>
          <div style='font-family:Playfair Display,serif;
                      font-size:1.05rem; color:#A8D5B0;
                      margin-top:6px;'>
            AMP Discovery
          </div>
          <div style='font-size:0.7rem; color:rgba(255,255,255,0.35);
                      margin-top:3px;'>
            Peptide Intelligence Platform
          </div>
        </div>
        <hr style='border:0.5px solid rgba(255,255,255,0.08);
                   margin-bottom:1rem;'>
        """, unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["🔬  Predict",
             "📊  Dataset Info",
             "🧠  Model Performance",
             "📖  About"],
            label_visibility="collapsed"
        )

        st.markdown("""
        <hr style='border:0.5px solid rgba(255,255,255,0.08);
                   margin:1rem 0;'>
        <div style='font-size:0.72rem; color:rgba(255,255,255,0.35);
                    line-height:2; padding:0 0.25rem;'>
          <span style='color:#6FCFA0; font-weight:500;'>
            Model Stack
          </span><br>
          XGBoost · Random Forest<br>
          Logistic Regression<br><br>
          <span style='color:#6FCFA0; font-weight:500;'>
            Explainability
          </span><br>
          SHAP TreeExplainer<br><br>
          <span style='color:#6FCFA0; font-weight:500;'>
            Dataset
          </span><br>
          1,586 sequences<br>
          UniProt SwissProt<br>
          10 biological sources<br><br>
          <span style='color:#6FCFA0; font-weight:500;'>
            Best Model
          </span><br>
          XGBoost AUC-ROC: 0.9725
        </div>
        <hr style='border:0.5px solid rgba(255,255,255,0.08);
                   margin:1rem 0;'>
        <div style='font-size:0.68rem; color:rgba(255,255,255,0.25);
                    text-align:center; line-height:1.7;'>
          ⚠️ Research prototype only.<br>
          Not for clinical use.<br><br>
          RVCE Bengaluru · 2026<br>
          Biotech Engineering
        </div>
        """, unsafe_allow_html=True)

    return page

# ============================================================
# PAGE: PREDICT
# ============================================================

def page_predict(models, feature_names):

    st.markdown("""
    <div class="hero">
      <h1>Antimicrobial Peptide Intelligence</h1>
      <p>Paste any peptide sequence to predict antimicrobial activity,
         understand the biological reasoning, and explore feature-level
         explanations powered by SHAP.</p>
      <div class="hero-badges">
        <span class="hbadge">🧬 XGBoost · AUC 0.97</span>
        <span class="hbadge">⚡ SHAP Explainability</span>
        <span class="hbadge">🌍 Open Research</span>
        <span class="hbadge">🔬 1,586 Training Sequences</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input Row ──
    col_in, col_tip = st.columns([3, 2], gap="large")

    with col_in:
        st.markdown("""
        <div style='font-size:0.8rem; color:#5A8C5E; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.06em;
                    margin-bottom:6px;'>
          🔬 Paste Peptide Sequence
        </div>
        """, unsafe_allow_html=True)

        seq_input = st.text_area(
            "seq",
            placeholder="Example: GIGKFLHSAKKFGKAFVGEIMNS",
            height=90,
            label_visibility="collapsed"
        )

        c1, c2 = st.columns([1, 2])
        with c1:
            go = st.button("Analyze →", use_container_width=True)
        with c2:
            ex = st.selectbox(
                "ex",
                ["Load an example sequence..."] +
                list(EXAMPLE_SEQUENCES.keys()),
                label_visibility="collapsed"
            )

        if ex != "Load an example sequence...":
            seq_input = EXAMPLE_SEQUENCES[ex]
            st.markdown(f"""
            <div class="info-box">
              ✅ Loaded: <b>{ex}</b>
            </div>
            """, unsafe_allow_html=True)

    with col_tip:
        st.markdown("""
        <div class="card" style='height:100%;'>
          <div class="card-label">💡 How To Use</div>
          <div style='font-size:0.83rem; color:#4A6B4D;
                      line-height:1.9;'>
            <b>1.</b> Paste sequence in single-letter code<br>
            <b>2.</b> Click <b>Analyze →</b><br>
            <b>3.</b> View AMP probability + confidence<br>
            <b>4.</b> Explore SHAP tab for WHY<br>
            <b>5.</b> Check model consensus<br>
            <b>6.</b> Download feature report<br><br>
            <span style='color:#5A8C5E; font-size:0.78rem;'>
              Valid: A C D E F G H I K L M N P Q R S T V W Y<br>
              Length: 10–100 amino acids
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Run Prediction ──
    if not (go or ex != "Load an example sequence..."):
        st.markdown("""
        <div style='text-align:center; padding:3rem;
                    color:#8AB08C; font-size:0.9rem;'>
          🧬 Enter a peptide sequence above to begin analysis
        </div>
        """, unsafe_allow_html=True)
        return

    if not seq_input or not seq_input.strip():
        st.markdown('<div class="warn-box">⚠️ Please enter a sequence.</div>',
                    unsafe_allow_html=True)
        return

    seq = seq_input.upper().strip().replace(" ","").replace("\n","")

    if not is_valid_sequence(seq):
        st.markdown(f"""
        <div class="warn-box">
          ⚠️ Invalid sequence — length {len(seq)} aa
          (valid: 10–100), or contains non-standard characters.
        </div>
        """, unsafe_allow_html=True)
        return

    with st.spinner("Running prediction + SHAP analysis..."):
        result    = predict(seq, models['XGBoost'], feature_names)
        explainer = load_explainer(models['XGBoost'])
        contribs  = get_shap(result['vec_scaled'],
                             explainer, feature_names)

    st.markdown(
        "<hr style='border:0.5px solid #D8E8DA; margin:1.25rem 0;'>",
        unsafe_allow_html=True
    )

    # ── Results: 3 columns ──
    r1, r2, r3 = st.columns([1.1, 1.4, 1], gap="large")

    # Column 1 — Prediction result
    with r1:
        prob   = result['amp_probability']
        is_amp = result['prediction'] == 'AMP'
        cls    = "result-amp" if is_amp else "result-non-amp"
        clr    = "#2D7A3A" if is_amp else "#C0606A"
        emoji  = "✅" if is_amp else "❌"

        st.markdown(f"""
        <div class="{cls}">
          <div style='font-size:0.72rem; color:#8AB08C;
                      text-transform:uppercase; letter-spacing:0.08em;'>
            Prediction
          </div>
          <div class="pred-label" style='color:{clr};'>
            {emoji} {result['prediction']}
          </div>
          <div class="pred-prob" style='color:{clr};'>
            {prob*100:.1f}%
          </div>
          <div class="pred-sublabel">AMP probability</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(float(prob))

        conf = "Very High" if prob > 0.9 or prob < 0.1 else \
               "High" if prob > 0.75 or prob < 0.25 else \
               "Moderate — borderline sequence"
        st.markdown(f"""
        <div style='text-align:center; margin-top:6px;
                    font-size:0.78rem; color:#5A8C5E;'>
          Confidence: <b>{conf}</b>
        </div>
        """, unsafe_allow_html=True)

    # Column 2 — Sequence properties
    with r2:
        feats  = result['features']
        colored = color_seq(seq)

        st.markdown(f"""
        <div class="card">
          <div class="card-label">🔭 Sequence Properties</div>
          <div class="seq-box">{colored}</div>
          <div class="chips">
            <span class="chip">📏 {int(feats.get('length', len(seq)))} aa</span>
            <span class="chip">⚡ Charge: {feats.get('charge', 0):.2f}</span>
            <span class="chip">💧 Hydro: {feats.get('hydrophobicity', 0):.3f}</span>
            <span class="chip">🎯 pI: {feats.get('isoelectric_point', 0):.2f}</span>
            <span class="chip">+ {feats.get('positive_fraction', 0)*100:.1f}% positive</span>
            <span class="chip">◎ {feats.get('hydrophobic_fraction', 0)*100:.1f}% hydrophobic</span>
          </div>
          <div style='margin-top:10px; font-size:0.72rem;
                      color:#8AB08C; line-height:1.8;'>
            🟢 <span style='color:#6FCFA0;'>Positive (K,R,H)</span>
            &nbsp;·&nbsp;
            🔴 <span style='color:#E07B7B;'>Negative (D,E)</span>
            &nbsp;·&nbsp;
            <span style='color:#A8D5B0;'>Hydrophobic</span>
            &nbsp;·&nbsp;
            <span style='color:rgba(100,100,100,0.6);'>Other</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Column 3 — Radar chart
    with r3:
        fig_radar = plot_radar(feats)
        st.pyplot(fig_radar, use_container_width=True)
        plt.close()

    # ── Tabs ──
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs([
        "🧠  SHAP Explanation",
        "🔬  Model Consensus",
        "📋  Full Feature Report"
    ])

    with tab1:
        st.markdown("""
        <div class="info-box">
          <b>What is SHAP?</b> Each bar shows how much a feature pushed
          the prediction toward AMP (green) or non-AMP (red). This reveals
          exactly WHY the model made this decision — making it actionable
          for researchers who want to modify the peptide.
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2 = st.columns([3, 2], gap="large")

        with sc1:
            fig_shap = plot_shap_bar(contribs, top_n=12)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close()

        with sc2:
            st.markdown("""
            <div style='font-size:0.78rem; font-weight:600;
                        color:#2D7A3A; margin-bottom:8px;'>
              ↑ Driving toward AMP
            </div>
            """, unsafe_allow_html=True)

            pos = [(f,v) for f,v in contribs if v > 0][:6]
            max_pos = max([abs(v) for _,v in pos], default=1)
            for feat, val in pos:
                w = int(abs(val)/max_pos * 100)
                st.markdown(f"""
                <div class="shap-row">
                  <div class="shap-feat">{feat}</div>
                  <div class="shap-track">
                    <div class="shap-fill-pos" style="width:{w}%"></div>
                  </div>
                  <div class="shap-val shap-val-pos">{val:+.3f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style='font-size:0.78rem; font-weight:600;
                        color:#C0606A; margin:12px 0 8px;'>
              ↓ Driving toward non-AMP
            </div>
            """, unsafe_allow_html=True)

            neg = [(f,v) for f,v in contribs if v < 0][:6]
            max_neg = max([abs(v) for _,v in neg], default=1)
            for feat, val in neg:
                w = int(abs(val)/max_neg * 100)
                st.markdown(f"""
                <div class="shap-row">
                  <div class="shap-feat">{feat}</div>
                  <div class="shap-track">
                    <div class="shap-fill-neg" style="width:{w}%"></div>
                  </div>
                  <div class="shap-val shap-val-neg">{val:+.3f}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="info-box">
          <b>Model Consensus:</b> Three independent models evaluate the
          same sequence. If all three agree, confidence is higher.
          Disagreement flags borderline sequences worth wet lab testing.
        </div>
        """, unsafe_allow_html=True)

        fig_con = plot_consensus(seq, models, feature_names)
        st.pyplot(fig_con, use_container_width=True)
        plt.close()

        votes   = 0
        avg     = 0
        all_p   = {}
        for name, pipeline in models.items():
            r = predict(seq, pipeline, feature_names)
            all_p[name] = r['amp_probability']
            avg += r['amp_probability']
            if r['prediction'] == 'AMP':
                votes += 1
        avg /= len(models)

        verdict = "Strong AMP — all models agree" if votes == 3 else \
                  "Strong non-AMP — all models agree" if votes == 0 else \
                  "Borderline — mixed model signals, wet lab advised"

        st.markdown(f"""
        <div class="card">
          <div class="card-label">Consensus Summary</div>
          <div style='font-size:0.88rem; color:#4A6B4D; line-height:2;'>
            <b>Verdict:</b>
            <span style='color:#1C2B1E; font-weight:600;'>
              {verdict}
            </span><br>
            <b>Model vote:</b> {votes}/3 predict AMP<br>
            <b>Average probability:</b>
            <code style='background:#EEF6EF; padding:2px 8px;
                         border-radius:6px; font-size:0.82rem;'>
              {avg:.4f}
            </code>
          </div>
        </div>
        """, unsafe_allow_html=True)

        for name, p in all_p.items():
            is_a = p >= 0.5
            fill = "model-fill" if is_a else "model-fill-low"
            st.markdown(f"""
            <div class="model-row">
              <div class="model-name">{name}</div>
              <div class="model-track">
                <div class="{fill}" style="width:{int(p*100)}%"></div>
              </div>
              <div class="model-val"
                   style="color:{'#2D7A3A' if is_a else '#C0606A'};">
                {p:.3f}
              </div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("**Complete Feature Values for This Sequence**")

        feat_rows = []
        for k, v in result['features'].items():
            cat = 'Physicochemical' if k in [
                'charge','hydrophobicity','isoelectric_point','length',
                'positive_fraction','negative_fraction',
                'hydrophobic_fraction','charge_density',
                'positive_count','hydrophobic_count'
            ] else 'Amino Acid Composition'
            feat_rows.append({
                'Feature': k,
                'Value': round(v, 6),
                'Category': cat
            })

        feat_df = pd.DataFrame(feat_rows)
        st.dataframe(
            feat_df, use_container_width=True, hide_index=True,
            column_config={
                "Value": st.column_config.NumberColumn(format="%.6f"),
            }
        )

        csv = feat_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Feature Report (CSV)",
            data=csv,
            file_name=f"amp_features_{seq[:10]}.csv",
            mime="text/csv"
        )

# ============================================================
# PAGE: DATASET
# ============================================================

def page_dataset():
    st.markdown("""
    <div style='font-family:Playfair Display,serif; font-size:1.8rem;
                color:#1C2B1E; margin-bottom:0.25rem;'>
      Dataset Overview
    </div>
    <div style='color:#8AB08C; margin-bottom:1.5rem; font-size:0.88rem;'>
      UniProt SwissProt — manually curated protein sequences
    </div>
    """, unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    for col, (val, label, icon) in zip(
        [m1,m2,m3,m4],
        [("1,586","Total Sequences","🧬"),
         ("793","AMP Sequences","✅"),
         ("793","non-AMP Sequences","❌"),
         ("10","Biological Sources","🔬")]
    ):
        with col:
            st.markdown(f"""
            <div class="card" style='text-align:center;'>
              <div style='font-size:1.75rem;'>{icon}</div>
              <div style='font-family:Playfair Display,serif;
                          font-size:1.75rem; color:#2D7A3A;'>
                {val}
              </div>
              <div style='font-size:0.75rem; color:#8AB08C;'>
                {label}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-label">🌿 Positive Class — AMP Sources</div>
      <div style='font-size:0.85rem; color:#4A6B4D; line-height:2;'>
        <b>UniProt KW-0929</b> — Gold standard manually verified AMPs<br>
        <b>Defensins</b> — Cysteine-rich immune AMPs<br>
        <b>Bacteriocins (KW-0078)</b> — Bacteria-derived AMPs<br>
        <b>Antibiotic peptides (KW-0045)</b> — Broad antibiotic activity<br>
        <b>Cathelicidins</b> — Mammalian innate immune defense
      </div>
    </div>
    <div class="card">
      <div class="card-label">⚖️ Negative Class — non-AMP Sources</div>
      <div style='font-size:0.85rem; color:#4A6B4D; line-height:2;'>
        <b>Neuropeptides (KW-0547)</b> — Signaling peptides, no antimicrobial activity<br>
        <b>Hormones (KW-0134)</b> — Insulin, glucagon fragments<br>
        <b>Growth Factors (KW-0349)</b> — EGF, FGF, IGF fragments<br>
        <b>Transcription regulators (KW-0167)</b> — Regulatory short peptides
      </div>
    </div>
    <div class="warn-box">
      <b>Known Limitation:</b> Defensins represent ~20% of positive
      sequences, causing a mild cysteine bias. Version 1.1 will
      apply family-balanced sampling to correct this.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: MODEL PERFORMANCE
# ============================================================

def page_model():
    st.markdown("""
    <div style='font-family:Playfair Display,serif; font-size:1.8rem;
                color:#1C2B1E; margin-bottom:0.25rem;'>
      Model Performance
    </div>
    <div style='color:#8AB08C; margin-bottom:1.5rem; font-size:0.88rem;'>
      5-Fold Stratified Cross Validation Results
    </div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame({
        'Model': ['XGBoost','Random Forest','Logistic Regression'],
        'Accuracy': [0.9168, 0.9124, 0.8373],
        'AUC-ROC':  [0.9725, 0.9705, 0.9003],
        'Precision':[0.9261, 0.9164, 0.8430],
        'Recall':   [0.9066, 0.9079, 0.8297],
        'F1 Score': [0.9158, 0.9119, 0.8358],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="card" style='margin-top:1rem;'>
      <div class="card-label">📖 Understanding The Metrics</div>
      <div style='font-size:0.85rem; color:#4A6B4D; line-height:2;'>
        <b>AUC-ROC 0.9725</b> — Near-perfect separation.
        0.5 = random, 1.0 = perfect.<br>
        <b>Accuracy 91.7%</b> — Correctly classifies 917/1000
        unseen peptides.<br>
        <b>Precision 92.6%</b> — When model says AMP, correct 93%
        of the time.<br>
        <b>Recall 90.7%</b> — Catches 91% of all real AMPs.<br>
        <b>F1 0.916</b> — Balanced precision + recall.
      </div>
    </div>
    <div class="info-box">
      All results from 5-Fold Stratified Cross Validation —
      an honest, unbiased performance estimate on unseen data.
      No data leakage. No cherry-picked results.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: ABOUT
# ============================================================

def page_about():
    st.markdown("""
    <div style='font-family:Playfair Display,serif; font-size:1.8rem;
                color:#1C2B1E; margin-bottom:1.25rem;'>
      About This Project
    </div>

    <div class="card">
      <div class="card-label">🌍 The Problem</div>
      <div style='font-size:0.88rem; color:#4A6B4D; line-height:1.9;'>
        Antimicrobial resistance kills 1.3 million people annually.
        By 2050 that number reaches 10 million — more than cancer.
        The antibiotic pipeline is failing. Antimicrobial peptides
        offer a promising alternative: bacteria cannot easily develop
        resistance to their physical membrane-disruption mechanism.
      </div>
    </div>

    <div class="card">
      <div class="card-label">🔬 What We Built</div>
      <div style='font-size:0.88rem; color:#4A6B4D; line-height:1.9;'>
        An end-to-end ML pipeline that predicts AMP activity from
        sequence alone. SHAP explainability reveals the biological
        reasoning behind each prediction — making results actionable
        for researchers, not just a black-box yes/no answer.
      </div>
    </div>

    <div class="card">
      <div class="card-label">⚗️ Technical Stack</div>
      <div style='font-size:0.88rem; color:#4A6B4D; line-height:2;'>
        <b>Data:</b> UniProt SwissProt —
        1,586 manually verified sequences<br>
        <b>Features:</b> 30 physicochemical +
        amino acid composition features<br>
        <b>Models:</b> XGBoost · Random Forest ·
        Logistic Regression<br>
        <b>Explainability:</b> SHAP TreeExplainer<br>
        <b>Validation:</b> 5-Fold Stratified Cross Validation<br>
        <b>Deployment:</b> Streamlit + HuggingFace Spaces
      </div>
    </div>

    <div class="warn-box">
      <b>Disclaimer:</b> Research prototype for computational
      biomarker discovery. Not validated for clinical use.
      All predictions require experimental validation.
    </div>

    <div class="card">
      <div class="card-label">👨‍🔬 Built By</div>
      <div style='font-size:0.88rem; color:#4A6B4D; line-height:1.9;'>
        Final Year Biotechnology Engineering Student<br>
        RV College of Engineering, Bengaluru, India · 2026
      </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN
# ============================================================

def main():
    try:
        models, feature_names = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Run src/model_training.py first.")
        return

    page = sidebar()

    if "Predict" in page:
        page_predict(models, feature_names)
    elif "Dataset" in page:
        page_dataset()
    elif "Model" in page:
        page_model()
    elif "About" in page:
        page_about()

    st.markdown("""
    <div class="footer">
      AMP Discovery Pipeline · XGBoost + SHAP ·
      Research Prototype · Not for clinical use<br>
      RVCE Bengaluru · Biotechnology Engineering · 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()