import html
import re
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tiki Book Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("data/tiki_books_labeled.csv")
REVIEWS_PATH = Path("data/tiki_books_reviews.csv")
MODEL_COMPARISON_PATH = Path("data/model_comparison.csv")
CLASSIFICATION_REPORT_PATH = Path("data/classification_report.csv")
MODEL_PATH = Path("models/product_performance_model.pkl")
ENCODER_PATH = Path("models/label_encoder.pkl")
CONFUSION_MATRIX_PATH = Path("images/confusion_matrix.png")
FEATURE_IMPORTANCE_PATH = Path("images/feature_importance.png")
SENTIMENT_SIMPLE_PATH = Path("data/tiki_books_sentiment.csv")
SENTIMENT_PHOBERT_PATH = Path("data/tiki_books_sentiment_phobert.csv")


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    header[data-testid="stHeader"] {
        background: transparent;
        height: 0rem;
    }

    /* ================= GLOBAL ================= */

    .stApp {
        background: linear-gradient(160deg, #EFF6FF 0%, #F8FAFF 35%, #F5F0FF 70%, #FFF8F0 100%);
        color: #0F172A;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 850 !important;
    }

    p, span, label, div {
        color: #0F172A;
    }

    /* ================= SIDEBAR ================= */

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFF 100%);
        border-right: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    .sidebar-logo {
        width: 62px;
        height: 62px;
        border-radius: 20px;
        background: linear-gradient(135deg, #0B63F6 0%, #7C3AED 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        margin-bottom: 14px;
        box-shadow: 0 8px 22px rgba(11, 99, 246, 0.32);
    }

    .sidebar-title {
        font-size: 22px !important;
        font-weight: 900 !important;
        color: #0B63F6 !important;
        line-height: 1.2 !important;
        margin-bottom: 8px !important;
    }

    .sidebar-subtitle {
        color: #64748B !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        max-width: 210px;
        margin-bottom: 18px;
    }

    .sidebar-line {
        height: 2px;
        background: linear-gradient(90deg, #0B63F6, #8B5CF6, #10B981);
        margin: 20px 0 18px 0 !important;
        border-radius: 999px;
        opacity: 0.5;
    }

    section[data-testid="stSidebar"] {
        min-width: 270px !important;
        max-width: 270px !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding: 28px 18px 24px 18px;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 13px;
        font-weight: 800;
        color: #94A3B8 !important;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #FFFFFF;
        border: 1.5px solid #E2E8F0;
        border-radius: 14px;
        padding: 11px 13px !important;
        margin-bottom: 7px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        transition: all 0.2s ease;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: linear-gradient(135deg, #EFF6FF, #F5F0FF);
        border-color: #C4B5FD;
        transform: translateX(3px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.12);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #334155 !important;
        margin: 0 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #0B63F6 0%, #7C3AED 100%) !important;
        border-color: transparent !important;
        box-shadow: 0 8px 22px rgba(11, 99, 246, 0.3), 0 4px 10px rgba(124, 58, 237, 0.2) !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 850 !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label input {
        accent-color: #0B63F6;
    }

    /* ================= HERO CARD ================= */

    .hero-card {
        background: linear-gradient(135deg, #0B1F5E 0%, #0B63F6 55%, #7C3AED 100%);
        border: none;
        border-radius: 28px;
        padding: 34px 40px;
        box-shadow: 0 20px 60px rgba(11, 99, 246, 0.28), 0 8px 24px rgba(124, 58, 237, 0.18);
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        position: relative;
        overflow: hidden;
    }

    .hero-card::before {
        content: '';
        position: absolute;
        top: -80px; right: -80px;
        width: 280px; height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.07), transparent 65%);
        pointer-events: none;
    }

    .hero-card::after {
        content: '';
        position: absolute;
        bottom: -60px; left: 30%;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.05), transparent 65%);
        pointer-events: none;
    }

    .hero-left {
        display: flex;
        align-items: center;
        gap: 22px;
        position: relative;
        z-index: 1;
    }

    .hero-icon {
        width: 80px;
        height: 80px;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        font-size: 40px;
        border: 1px solid rgba(255,255,255,0.25);
        flex-shrink: 0;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 900;
        letter-spacing: -0.035em;
        color: #FFFFFF !important;
        margin: 0 0 8px 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.82) !important;
        font-size: 16px;
        line-height: 1.5;
        max-width: 680px;
        margin: 0;
    }

    .hero-art {
        min-width: 140px;
        height: 120px;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 64px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: inset 0 2px 12px rgba(255,255,255,0.08);
        position: relative;
        z-index: 1;
    }

    /* ================= METRIC CARDS ================= */

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        min-height: 118px;
        display: flex;
        gap: 15px;
        align-items: center;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0B63F6, #8B5CF6);
        border-radius: 20px 20px 0 0;
    }

    .metric-card.green::before  { background: linear-gradient(90deg, #10B981, #059669); }
    .metric-card.red::before    { background: linear-gradient(90deg, #EF4444, #DC2626); }
    .metric-card.yellow::before { background: linear-gradient(90deg, #F59E0B, #D97706); }
    .metric-card.purple::before { background: linear-gradient(90deg, #8B5CF6, #7C3AED); }

    .metric-icon {
        width: 54px;
        height: 54px;
        border-radius: 17px;
        background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
        color: #0B63F6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        flex-shrink: 0;
    }

    .metric-icon.green  { background: linear-gradient(135deg, #ECFDF5, #D1FAE5); color: #10B981; }
    .metric-icon.red    { background: linear-gradient(135deg, #FEF2F2, #FECACA); color: #EF4444; }
    .metric-icon.yellow { background: linear-gradient(135deg, #FFFBEB, #FDE68A); color: #D97706; }
    .metric-icon.purple { background: linear-gradient(135deg, #FAF5FF, #EDE9FE); color: #7C3AED; }

    .metric-title {
        font-size: 12px;
        color: #64748B;
        font-weight: 700;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .metric-value {
        font-size: 23px;
        color: #0B63F6;
        font-weight: 900;
        line-height: 1.15;
    }

    .metric-value.green  { color: #10B981; }
    .metric-value.red    { color: #EF4444; }
    .metric-value.yellow { color: #D97706; }
    .metric-value.purple { color: #7C3AED; }

    .metric-subtitle {
        color: #94A3B8;
        font-size: 12px;
        margin-top: 5px;
    }

    /* ================= PAGE HEADER ================= */

    .page-header-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 24px;
        padding: 28px 32px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }

    .page-header-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0B63F6, #8B5CF6, #10B981);
    }

    .page-title {
        font-size: 36px;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: #0F172A !important;
        margin: 0 0 8px 0;
    }

    .page-subtitle {
        color: #64748B;
        font-size: 16px;
        line-height: 1.5;
        margin: 0;
    }

    .section-head {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0B63F6;
        border-radius: 18px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        margin: 18px 0 12px 0;
    }

    .section-title {
        font-size: 20px;
        font-weight: 850;
        color: #0F172A;
        margin: 0;
    }

    .section-subtitle {
        color: #64748B;
        font-size: 13px;
        margin-top: 4px;
    }

    /* ================= LABEL CARDS ================= */

    .label-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: transform 0.18s ease;
    }

    .label-card:hover {
        transform: translateX(3px);
    }

    .label-left {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #0F172A;
        font-weight: 700;
        font-size: 14px;
    }

    .label-icon {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
        color: #0B63F6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }

    .label-value {
        color: #0B63F6;
        font-size: 22px;
        font-weight: 900;
    }

    /* ================= INFO & MINI CARDS ================= */

    .info-card {
        background: linear-gradient(135deg, #EFF6FF, #F0F9FF);
        border: 1px solid #BFDBFE;
        border-radius: 18px;
        padding: 16px 18px;
        color: #1E3A8A;
        font-size: 15px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        margin-top: 15px;
    }

    .mini-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .mini-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }

    .mini-card-title {
        font-size: 16px;
        font-weight: 850;
        color: #0F172A;
        margin-bottom: 8px;
    }

    .mini-card-text {
        color: #475569;
        font-size: 14px;
        line-height: 1.6;
    }

    /* ================= CONTENT & FEATURE GRID ================= */

    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin-top: 16px;
    }

    .feature-box {
        background: linear-gradient(135deg, #F8FBFF, #F0F4FF);
        border: 1px solid #DBEAFE;
        border-radius: 18px;
        padding: 20px;
        transition: transform 0.2s ease;
    }

    .feature-box:hover {
        transform: translateY(-2px);
    }

    .feature-title {
        font-size: 15px;
        font-weight: 850;
        color: #0B63F6;
        margin-bottom: 8px;
    }

    .feature-text {
        font-size: 14px;
        color: #475569;
        line-height: 1.6;
    }

    /* ================= TABLE ================= */

    .table-wrap {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        overflow: auto;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
        margin-top: 12px;
    }

    table.light-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        background: #FFFFFF;
    }

    table.light-table thead th {
        background: linear-gradient(135deg, #0B63F6 0%, #2563EB 100%);
        color: #FFFFFF !important;
        padding: 13px 16px;
        text-align: left;
        font-weight: 700;
        border-bottom: none;
        white-space: nowrap;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    table.light-table tbody td {
        color: #0F172A;
        padding: 12px 16px;
        border-bottom: 1px solid #F1F5F9;
        vertical-align: middle;
    }

    table.light-table tbody tr:nth-child(even) {
        background: #F8FAFC;
    }

    table.light-table tbody tr:hover {
        background: #EFF6FF;
    }

    table.light-table a {
        color: #0B63F6;
        text-decoration: none;
        font-weight: 700;
        padding: 3px 10px;
        background: #EFF6FF;
        border-radius: 8px;
        font-size: 12px;
        border: 1px solid #BFDBFE;
    }

    table.light-table a:hover {
        background: #DBEAFE;
        text-decoration: none;
    }

    /* ================= TIMELINE ================= */

    .timeline-box {
        background: transparent;
        padding: 4px 0;
    }

    .timeline-step {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
        align-items: flex-start;
    }

    .step-number {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0B63F6, #7C3AED);
        color: #FFFFFF !important;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 16px;
        flex-shrink: 0;
        box-shadow: 0 6px 16px rgba(11, 99, 246, 0.3);
    }

    .step-content-title {
        font-weight: 850;
        color: #0F172A;
        margin-bottom: 5px;
        font-size: 15px;
    }

    .step-content-text {
        color: #64748B;
        font-size: 14px;
        line-height: 1.5;
    }

    /* ================= STAT PILL & MODEL HIGHLIGHT ================= */

    .stat-pill {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
        color: #0B63F6 !important;
        font-weight: 800;
        font-size: 12px;
        margin: 3px 4px 3px 0;
        border: 1px solid #BFDBFE;
    }

    .model-highlight {
        background: linear-gradient(135deg, #0B1F5E 0%, #0B63F6 60%, #7C3AED 100%);
        border-radius: 22px;
        padding: 26px;
        box-shadow: 0 16px 40px rgba(11, 99, 246, 0.28);
        color: white;
        position: relative;
        overflow: hidden;
    }

    .model-highlight::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 160px; height: 160px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.07), transparent 70%);
    }

    .model-highlight * {
        color: white !important;
    }

    .model-highlight-title {
        font-size: 13px;
        font-weight: 800;
        opacity: 0.8;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .model-highlight-value {
        font-size: 26px;
        font-weight: 900;
        line-height: 1.25;
    }

    .model-highlight-text {
        font-size: 13px;
        opacity: 0.8;
        margin-top: 12px;
        line-height: 1.55;
    }

    /* ================= FORM INPUTS ================= */

    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
    }

    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #0B63F6 !important;
        box-shadow: 0 0 0 3px rgba(11, 99, 246, 0.1) !important;
    }

    /* ================= BUTTONS ================= */

    .stFormSubmitButton > button,
    .stFormSubmitButton button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stFormSubmitButton"] button,
    .stButton > button,
    .stButton button,
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #0B63F6 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 14px !important;
        min-height: 48px !important;
        padding: 0.75rem 1.6rem !important;
        font-size: 15px !important;
        font-weight: 850 !important;
        box-shadow: 0 8px 20px rgba(11, 99, 246, 0.28), 0 4px 10px rgba(124, 58, 237, 0.2) !important;
        transition: all 0.2s ease !important;
    }

    .stFormSubmitButton > button *,
    .stFormSubmitButton button *,
    div[data-testid="stFormSubmitButton"] > button *,
    div[data-testid="stFormSubmitButton"] button *,
    .stButton > button *,
    .stButton button *,
    button[kind="primary"] *,
    button[data-testid="baseButton-primary"] *,
    button[data-testid="baseButton-secondary"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        font-weight: 850 !important;
    }

    .stFormSubmitButton > button:hover,
    .stFormSubmitButton button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    .stButton > button:hover,
    .stButton button:hover,
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #0755D5 0%, #6D28D9 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(11, 99, 246, 0.35), 0 6px 14px rgba(124, 58, 237, 0.25) !important;
    }

    .stFormSubmitButton > button:hover *,
    .stFormSubmitButton button:hover *,
    div[data-testid="stFormSubmitButton"] > button:hover *,
    div[data-testid="stFormSubmitButton"] button:hover *,
    .stButton > button:hover *,
    .stButton button:hover *,
    button[kind="primary"]:hover *,
    button[data-testid="baseButton-primary"]:hover *,
    button[data-testid="baseButton-secondary"]:hover * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    .stFormSubmitButton > button:focus,
    .stFormSubmitButton > button:active,
    div[data-testid="stFormSubmitButton"] button:focus,
    div[data-testid="stFormSubmitButton"] button:active,
    .stButton button:focus,
    .stButton button:active {
        background: linear-gradient(135deg, #064BC0 0%, #5B21B6 100%) !important;
        color: #FFFFFF !important;
        outline: 3px solid rgba(11, 99, 246, 0.2) !important;
    }

    /* ================= RESPONSIVE ================= */

    @media (max-width: 900px) {
        .hero-card { flex-direction: column; align-items: flex-start; }
        .hero-art { width: 100%; min-width: 0; }
        .hero-title { font-size: 30px; }
    }

    @media (max-width: 1000px) {
        .feature-grid { grid-template-columns: 1fr; }
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

# Ghi chú: Lấy thời gian sửa file để cache biết khi nào cần tải lại dữ liệu.
def file_mtime(path: Path) -> float:
    """Return file modified time so Streamlit cache refreshes when CSV/model changes."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


# Ghi chú: Đọc dữ liệu sản phẩm đã gán nhãn từ file CSV.
@st.cache_data(show_spinner=False)
def load_products(_mtime: float) -> pd.DataFrame:
    """Load labeled products. _mtime forces cache invalidation when file changes."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH)


# Ghi chú: Đọc bảng so sánh hiệu năng giữa các mô hình.
@st.cache_data(show_spinner=False)
def load_model_comparison(_mtime: float) -> pd.DataFrame:
    if not MODEL_COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_COMPARISON_PATH)


# Ghi chú: Đọc báo cáo phân loại và chuẩn hóa cột nhãn nếu cần.
@st.cache_data(show_spinner=False)
def load_classification_report(_mtime: float) -> pd.DataFrame:
    if not CLASSIFICATION_REPORT_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(CLASSIFICATION_REPORT_PATH)

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "label"})

    return df


# Ghi chú: Đếm số dòng đánh giá và số sản phẩm có đánh giá trong file crawl.
@st.cache_data(show_spinner=False)
def load_review_stats(path_str: str, _mtime: float) -> tuple[int, int]:
    """Read raw crawl CSV stats without writing or mutating data."""
    path = Path(path_str)
    if not path.exists():
        return 0, 0

    try:
        reviews_df = pd.read_csv(path)
    except Exception:
        return 0, 0

    total_review_rows = len(reviews_df)
    unique_review_products = (
        reviews_df["product_id"].nunique()
        if "product_id" in reviews_df.columns
        else 0
    )
    return total_review_rows, int(unique_review_products)




# Ghi chú: Đọc và ghép sentiment_simple + PhoBERT để so sánh theo nhãn sản phẩm.
@st.cache_data(show_spinner=False)
def load_sentiment_data(_mtime_simple: float, _mtime_phobert: float) -> pd.DataFrame:
    """Trả về DataFrame có cột: product_label, sentiment_simple, sentiment_phobert."""
    if not SENTIMENT_SIMPLE_PATH.exists():
        return pd.DataFrame()

    simple_df = pd.read_csv(SENTIMENT_SIMPLE_PATH)
    if "sentiment_score" not in simple_df.columns or "product_label" not in simple_df.columns:
        return pd.DataFrame()

    result = simple_df[["product_id", "product_label", "sentiment_score"]].rename(
        columns={"sentiment_score": "sentiment_simple"}
    )

    if SENTIMENT_PHOBERT_PATH.exists():
        phobert_df = pd.read_csv(SENTIMENT_PHOBERT_PATH)
        if "sentiment_score" in phobert_df.columns:
            result = result.merge(
                phobert_df[["product_id", "sentiment_score"]].rename(
                    columns={"sentiment_score": "sentiment_phobert"}
                ),
                on="product_id",
                how="left",
            )
            result["sentiment_phobert"] = result["sentiment_phobert"].fillna(0)

    return result


# Ghi chú: Bổ sung thuộc tính còn thiếu để model sklearn cũ chạy được trên môi trường mới.
def patch_sklearn_tree_model(model):
    """Fix sklearn pickle compatibility for tree models on Streamlit Cloud.

    Some models were trained with a different scikit-learn version. On newer
    versions, tree estimators may expect the attribute `monotonic_cst`.
    Old pickled DecisionTree estimators may not have it, causing prediction to fail.
    """
    # Ghi chú: Sửa từng cây quyết định nếu thiếu thuộc tính tương thích.
    def patch_one(est):
        try:
            cls_name = est.__class__.__name__
            if cls_name in {
                "DecisionTreeClassifier",
                "DecisionTreeRegressor",
                "ExtraTreeClassifier",
                "ExtraTreeRegressor",
            }:
                if not hasattr(est, "monotonic_cst"):
                    est.monotonic_cst = None
        except Exception:
            pass

    if model is None:
        return model

    patch_one(model)

    # RandomForest / ExtraTrees store many decision trees in estimators_
    try:
        estimators = getattr(model, "estimators_", [])
        for est in estimators:
            if isinstance(est, (list, tuple)):
                for sub_est in est:
                    patch_one(sub_est)
            else:
                patch_one(est)
    except Exception:
        pass

    # Some wrappers keep the real estimator here
    try:
        patch_one(getattr(model, "estimator_", None))
    except Exception:
        pass

    return model


# Ghi chú: Tải model dự đoán và bộ mã hóa nhãn từ thư mục models.
@st.cache_resource(show_spinner=False)
def load_model_and_encoder(_model_mtime: float, _encoder_mtime: float):
    """Load model/encoder. mtimes force refresh after retraining."""
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    model = patch_sklearn_tree_model(model)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder


# ============================================================
# HELPERS
# ============================================================

COLUMN_NAME_MAP = {
    "product_id": "Mã sản phẩm",
    "product_name": "Tên sản phẩm",
    "price": "Giá bán",
    "rating": "Điểm đánh giá",
    "review_count": "Số đánh giá",
    "sold_count": "Lượt bán",
    "comment_count": "Số bình luận",
    "avg_comment_rating": "Điểm bình luận TB",
    "positive_ratio": "Tỷ lệ tích cực",
    "neutral_ratio": "Tỷ lệ trung tính",
    "negative_ratio": "Tỷ lệ tiêu cực",
    "estimated_revenue": "Doanh thu ước tính",
    "product_label": "Nhãn sản phẩm",
    "product_url": "Đường dẫn",
    "suggestion": "Gợi ý cải thiện",
    "Model": "Mô hình",
    "model": "Mô hình",
    "Accuracy": "Độ chính xác",
    "accuracy": "Độ chính xác",
    "Precision": "Precision",
    "precision": "Precision",
    "Recall": "Recall",
    "recall": "Recall",
    "F1-score": "F1-score",
    "f1-score": "F1-score",
    "support": "Số mẫu",
    "label": "Nhãn",
}


# Ghi chú: Chuyển giá trị thành chuỗi HTML an toàn để tránh lỗi hiển thị.
def safe_html(value) -> str:
    if pd.isna(value):
        return ""
    return html.escape(str(value))


# Ghi chú: Định dạng số nguyên có dấu phẩy phân tách hàng nghìn.
def format_number(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return "0"


# Ghi chú: Định dạng giá trị tiền tệ theo đơn vị đồng Việt Nam.
def format_vnd(value) -> str:
    try:
        return f"{int(float(value)):,}đ"
    except Exception:
        return "0đ"


# Ghi chú: Định dạng số thập phân theo số chữ số mong muốn.
def format_float(value, digits=2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "0.00"


# Ghi chú: Lấy một chỉ số cụ thể trong bảng báo cáo đánh giá mô hình.
def get_report_value(report_df, label_name, col_candidates, default=0):
    if report_df.empty or "label" not in report_df.columns:
        return default

    rows = report_df[report_df["label"].astype(str).str.lower() == str(label_name).lower()]
    if rows.empty:
        return default

    row = rows.iloc[0]
    for col in col_candidates:
        if col in report_df.columns and pd.notna(row.get(col)):
            return row.get(col)

    return default


# Ghi chú: Hiển thị tiêu đề chính và mô tả ngắn cho từng trang.
def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-header-card">
            <div class="page-title">{safe_html(title)}</div>
            <p class="page-subtitle">{safe_html(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Ghi chú: Hiển thị tiêu đề cho từng khu vực nội dung trong dashboard.
def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-title">{safe_html(title)}</div>
            <div class="section-subtitle">{safe_html(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Ghi chú: Tạo thẻ chỉ số gồm biểu tượng, tiêu đề, giá trị và mô tả.
# accent: "" (xanh dương mặc định) | "green" | "red"
def metric_card(icon: str, title: str, value: str, subtitle: str = "", accent: str = "") -> None:
    accent_cls = f" {accent}" if accent in ("green", "red") else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon{accent_cls}">{icon}</div>
            <div>
                <div class="metric-title">{safe_html(title)}</div>
                <div class="metric-value{accent_cls}">{safe_html(value)}</div>
                <div class="metric-subtitle">{safe_html(subtitle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Ghi chú: Tạo thẻ thông tin nhỏ với màu nhấn ở viền trái.
def mini_card(title: str, text: str, accent: str = "#2563EB") -> None:
    st.markdown(
        f"""
        <div class="mini-card" style="border-left: 5px solid {accent};">
            <div class="mini-card-title">{safe_html(title)}</div>
            <div class="mini-card-text">{safe_html(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Ghi chú: Hiển thị một thẻ thống kê số lượng theo nhãn — màu sắc riêng từng nhãn.
_LABEL_STYLE = {
    "Best Seller":             {"bg": "#EFF6FF", "border": "#BFDBFE", "icon_bg": "#DBEAFE", "color": "#0B63F6", "icon": "🏆"},
    "High Potential":          {"bg": "#ECFDF5", "border": "#A7F3D0", "icon_bg": "#D1FAE5", "color": "#059669", "icon": "🚀"},
    "Premium / Niche Quality": {"bg": "#FAF5FF", "border": "#DDD6FE", "icon_bg": "#EDE9FE", "color": "#7C3AED", "icon": "💎"},
    "Normal":                  {"bg": "#F8FAFC", "border": "#E2E8F0", "icon_bg": "#F1F5F9", "color": "#64748B", "icon": "📘"},
    "Needs Improvement":       {"bg": "#FEF2F2", "border": "#FECACA", "icon_bg": "#FEE2E2", "color": "#DC2626", "icon": "⚠️"},
}

def render_label_card(label: str, value: int) -> None:
    s = _LABEL_STYLE.get(label, {"bg": "#F8FAFC", "border": "#E2E8F0", "icon_bg": "#EFF6FF", "color": "#0B63F6", "icon": "🏷️"})
    st.markdown(
        f"""
        <div class="label-card" style="background:{s['bg']};border-color:{s['border']};">
            <div class="label-left">
                <div class="label-icon" style="background:{s['icon_bg']};color:{s['color']};">{s['icon']}</div>
                <div style="color:{s['color']};font-weight:700;">{safe_html(label)}</div>
            </div>
            <div class="label-value" style="color:{s['color']};">{format_number(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Ghi chú: Chuẩn hóa đường dẫn sản phẩm thành URL đầy đủ khi có thể.
def normalize_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    url = url.strip()
    if url.startswith("http"):
        return url
    if url.startswith("/"):
        return "https://tiki.vn" + url
    return url


# Ghi chú: Hiển thị DataFrame thành bảng HTML nhẹ, có định dạng cột dễ đọc.
def render_light_table(df: pd.DataFrame, columns=None, max_rows: int = 30) -> None:
    if df is None or df.empty:
        st.info("Không có dữ liệu để hiển thị.")
        return

    display_df = df.copy()

    if columns:
        existing_cols = [col for col in columns if col in display_df.columns]
        display_df = display_df[existing_cols]

    display_df = display_df.head(max_rows).copy()

    money_cols = {"price", "estimated_revenue"}
    ratio_cols = {"positive_ratio", "neutral_ratio", "negative_ratio"}

    headers = "".join(
        f"<th>{safe_html(COLUMN_NAME_MAP.get(col, col))}</th>"
        for col in display_df.columns
    )

    html_rows = []

    for _, row in display_df.iterrows():
        cells = []
        for col in display_df.columns:
            val = row[col]

            if col in money_cols:
                cell = format_vnd(val)
            elif col in ratio_cols:
                cell = format_float(val, 2)
            elif col == "product_url":
                url = normalize_url(str(val))
                if url:
                    cell = f'<a href="{safe_html(url)}" target="_blank">Xem trên Tiki</a>'
                else:
                    cell = ""
            elif isinstance(val, float):
                cell = format_float(val, 2)
            else:
                cell = safe_html(val)

            cells.append(f"<td>{cell}</td>")

        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <div class="table-wrap">
        <table class="light-table">
            <thead>
                <tr>{headers}</tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# Ghi chú: Đếm số sản phẩm theo từng nhãn hiệu quả.
def label_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "product_label" not in df.columns:
        return pd.Series(dtype=int)

    order = [
        "Normal",
        "Needs Improvement",
        "Premium / Niche Quality",
        "Best Seller",
        "High Potential",
    ]

    counts = df["product_label"].value_counts()
    return counts.reindex(order).fillna(0).astype(int)


# Ghi chú: Tạo biểu đồ cột thể hiện số lượng sản phẩm theo nhãn — mỗi nhãn một màu riêng.
def make_bar_chart(counts: pd.Series):
    chart_df = counts.reset_index()
    chart_df.columns = ["Nhãn sản phẩm", "Số lượng"]

    _label_colors = {
        "Normal":                  "#64748B",
        "Needs Improvement":       "#EF4444",
        "Premium / Niche Quality": "#8B5CF6",
        "Best Seller":             "#0B63F6",
        "High Potential":          "#10B981",
    }
    bar_colors = [_label_colors.get(lbl, "#0B63F6") for lbl in chart_df["Nhãn sản phẩm"]]

    fig = go.Figure(go.Bar(
        x=chart_df["Nhãn sản phẩm"],
        y=chart_df["Số lượng"],
        text=chart_df["Số lượng"],
        textposition="outside",
        textfont=dict(color="#334155", size=14),
        marker=dict(color=bar_colors, line_width=0, opacity=0.88),
        cliponaxis=False,
    ))

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#0F172A",
        height=430,
        margin=dict(l=50, r=20, t=40, b=80),
        showlegend=False,
        bargap=0.38,
        xaxis=dict(
            title="Nhãn sản phẩm",
            tickfont=dict(color="#334155", size=12),
            title_font=dict(color="#64748B", size=13),
            showgrid=False,
            linecolor="#E2E8F0",
        ),
        yaxis=dict(
            title="Số lượng sản phẩm",
            tickfont=dict(color="#64748B", size=12),
            title_font=dict(color="#64748B", size=13),
            gridcolor="#F1F5F9",
            linecolor="#E2E8F0",
        ),
    )

    return fig

# Ghi chú: Tạo biểu đồ donut thể hiện tỷ trọng sản phẩm theo nhãn — màu sắc theo nhãn chuẩn.
def make_donut_chart(counts: pd.Series):
    chart_df = counts.reset_index()
    chart_df.columns = ["Nhãn", "Số lượng"]

    _label_colors = {
        "Normal":                  "#64748B",
        "Needs Improvement":       "#EF4444",
        "Premium / Niche Quality": "#8B5CF6",
        "Best Seller":             "#0B63F6",
        "High Potential":          "#10B981",
    }
    colors = [_label_colors.get(lbl, "#94A3B8") for lbl in chart_df["Nhãn"]]

    fig = go.Figure(go.Pie(
        labels=chart_df["Nhãn"],
        values=chart_df["Số lượng"],
        hole=0.52,
        marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
        textinfo="percent+label",
        textfont=dict(size=13, color="#0F172A"),
        hovertemplate="<b>%{label}</b><br>Số lượng: %{value:,}<br>Tỷ lệ: %{percent}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="#0F172A",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(
            font=dict(size=13, color="#334155"),
            bgcolor="rgba(255,255,255,0.9)",
        ),
        annotations=[dict(
            text="<b>Nhãn<br>SP</b>",
            x=0.5, y=0.5,
            font_size=15,
            font_color="#64748B",
            showarrow=False,
        )],
    )

    return fig


# Ghi chú: Lấy thông tin mô hình tốt nhất dựa trên F1-score.
def get_best_model_info(model_df: pd.DataFrame):
    if model_df.empty:
        return "Random Forest Classifier", 0.9815, 0.9823, 0.9817

    lower_map = {c.lower().strip(): c for c in model_df.columns}

    model_col = lower_map.get("model", model_df.columns[0])
    acc_col = lower_map.get("accuracy")
    pre_col = lower_map.get("precision")
    f1_col = lower_map.get("f1-score") or lower_map.get("f1_score") or lower_map.get("f1")

    if f1_col is None:
        return "Random Forest Classifier", 0.9815, 0.9823, 0.9817

    best_idx = model_df[f1_col].astype(float).idxmax()
    best_row = model_df.loc[best_idx]

    best_model = str(best_row[model_col])
    acc = float(best_row[acc_col]) if acc_col else 0.0
    pre = float(best_row[pre_col]) if pre_col else 0.0
    f1 = float(best_row[f1_col])

    return best_model, acc, pre, f1


# Ghi chú: Chọn các cột quan trọng để hiển thị trong bảng sản phẩm.
def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "product_name",
        "price",
        "rating",
        "review_count",
        "sold_count",
        "comment_count",
        "estimated_revenue",
        "product_label",
        "product_url",
    ]
    return df[[c for c in keep_cols if c in df.columns]].copy()


# Ghi chú: Hiển thị biểu đồ Plotly và tự xử lý khác biệt phiên bản Streamlit.
def show_plotly(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# Ghi chú: Hiển thị ảnh và tự xử lý khác biệt phiên bản Streamlit.
def show_image(path):
    try:
        st.image(str(path), width="stretch")
    except TypeError:
        st.image(str(path), use_container_width=True)


TIKI_PRODUCT_API = "https://tiki.vn/api/v2/products/{product_id}"
TIKI_REVIEW_API = "https://tiki.vn/api/v2/reviews"
TIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://tiki.vn/",
}
MODEL_FEATURE_COLUMNS = [
    "price",
    "review_count",
    "comment_count",
    "avg_comment_rating",
    "neutral_ratio",
    "sentiment_score",
]


# Ghi chú: Trích mã sản phẩm Tiki từ link hoặc chuỗi người dùng nhập.
def extract_product_id(url_or_text):
    text = str(url_or_text or "").strip()
    if not text:
        return None

    for pattern in (r"-p(\d+)\.html", r"/p/(\d+)", r"(\d+)"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


# Ghi chú: Chuyển dữ liệu lượt bán từ nhiều định dạng về số nguyên.
def parse_sold_count(value):
    if value is None:
        return 0

    if isinstance(value, dict):
        raw_value = value.get("value")
        if raw_value not in (None, ""):
            return parse_sold_count(raw_value)
        return parse_sold_count(value.get("text"))

    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()
    if not text:
        return 0

    match = re.search(r"(\d+(?:[.,]\d+)?)\s*([km])?", text)
    if not match:
        return 0

    number_text = match.group(1)
    suffix = match.group(2) or ""

    if suffix in {"k", "m"}:
        number = float(number_text.replace(",", "."))
        multiplier = 1000 if suffix == "k" else 1_000_000
        return int(number * multiplier)

    digits = re.sub(r"\D", "", number_text)
    return int(digits) if digits else 0


# Ghi chú: Lấy lượt bán từ dữ liệu sản phẩm Tiki bằng nhiều tên trường có thể có.
def extract_sold_count(product):
    for key in ("quantity_sold", "all_time_quantity_sold", "sold_count", "order_count"):
        value = product.get(key)
        if value in (None, "", {}):
            continue
        return parse_sold_count(value)

    quantity_sold = product.get("quantity_sold")
    if isinstance(quantity_sold, dict):
        value = quantity_sold.get("value")
        if value not in (None, ""):
            return parse_sold_count(value)
        text = quantity_sold.get("text")
        if text not in (None, ""):
            return parse_sold_count(text)

    return parse_sold_count(product.get("sold"))


# Ghi chú: Gọi API Tiki để lấy thông tin chi tiết của một sản phẩm.
def fetch_tiki_product(product_id):
    try:
        response = requests.get(
            TIKI_PRODUCT_API.format(product_id=product_id),
            headers=TIKI_HEADERS,
            timeout=12,
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    try:
        product = response.json()
    except ValueError:
        return None

    return product if isinstance(product, dict) else None


# Ghi chú: Gọi API Tiki để lấy danh sách đánh giá của một sản phẩm.
def fetch_tiki_reviews(product_id, limit=20):
    params = {
        "product_id": product_id,
        "limit": limit,
        "page": 1,
        "include": "comments,contribute_info,attribute_vote_summary",
        "sort": "score|desc,id|desc,stars|all",
    }

    try:
        response = requests.get(TIKI_REVIEW_API, headers=TIKI_HEADERS, params=params, timeout=12)
    except requests.RequestException:
        return []

    if response.status_code != 200:
        return []

    try:
        data = response.json()
    except ValueError:
        return []

    reviews = data.get("data", [])
    return reviews if isinstance(reviews, list) else []


# Ghi chú: Chuyển giá trị về số thực, nếu lỗi thì dùng giá trị mặc định.
def _to_float(value, default=0):
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Ghi chú: Phân tích đánh giá để tính số bình luận, điểm trung bình, tỷ lệ cảm xúc và sentiment_score.
def analyze_reviews(reviews):
    if not reviews:
        return {
            "comment_count": 0,
            "avg_comment_rating": 0,
            "positive_ratio": 0,
            "neutral_ratio": 0,
            "negative_ratio": 0,
            "sentiment_score": 0.0,
        }

    ratings = []
    comment_count = 0

    for review in reviews:
        content = str(review.get("content") or "").strip()
        title = str(review.get("title") or "").strip()
        if content or title:
            comment_count += 1

        rating = _to_float(review.get("rating"), None)
        if rating is not None:
            ratings.append(rating)

    if not ratings:
        return {
            "comment_count": comment_count,
            "avg_comment_rating": 0,
            "positive_ratio": 0,
            "neutral_ratio": 0,
            "negative_ratio": 0,
            "sentiment_score": 0.0,
        }

    total = len(ratings)
    positive_count = sum(1 for r in ratings if r >= 4)
    neutral_count  = sum(1 for r in ratings if r == 3)
    negative_count = sum(1 for r in ratings if r <= 2)

    # sentiment_score = positive_ratio - negative_ratio ∈ [-1, 1]
    sentiment_score = (positive_count - negative_count) / total

    return {
        "comment_count": comment_count,
        "avg_comment_rating": sum(ratings) / total,
        "positive_ratio": positive_count / total,
        "neutral_ratio": neutral_count / total,
        "negative_ratio": negative_count / total,
        "sentiment_score": sentiment_score,
    }


# Ghi chú: Tạo bộ đặc trưng đầu vào cho model từ thông tin sản phẩm và đánh giá.
# Trả về (input_df, display_info):
#   input_df    — DataFrame 6 cột khớp MODEL_FEATURE_COLUMNS để đưa vào model
#   display_info — dict chứa thêm rating, sold_count, estimated_revenue để hiển thị UI
def build_features_from_product(product, reviews):
    price        = _to_float(product.get("price") or product.get("list_price") or 0, 0)
    rating       = _to_float(product.get("rating_average") or product.get("rating") or 0, 0)
    review_count = int(_to_float(product.get("review_count") or 0, 0))
    sold_count   = extract_sold_count(product)
    review_stats = analyze_reviews(reviews)

    model_values = {
        "price":              price,
        "review_count":       review_count,
        "comment_count":      review_stats["comment_count"],
        "avg_comment_rating": review_stats["avg_comment_rating"],
        "neutral_ratio":      review_stats["neutral_ratio"],
        "sentiment_score":    review_stats["sentiment_score"],
    }

    input_df = pd.DataFrame(
        [[model_values[col] for col in MODEL_FEATURE_COLUMNS]],
        columns=MODEL_FEATURE_COLUMNS,
    )

    display_info = {
        "price":              price,
        "rating":             rating,
        "review_count":       review_count,
        "sold_count":         sold_count,
        "comment_count":      review_stats["comment_count"],
        "avg_comment_rating": review_stats["avg_comment_rating"],
        "neutral_ratio":      review_stats["neutral_ratio"],
        "sentiment_score":    review_stats["sentiment_score"],
        "estimated_revenue":  price * sold_count,
    }

    return input_df, display_info


# Ghi chú: Tạo URL sản phẩm Tiki đầy đủ từ dữ liệu API hoặc mã sản phẩm.
def build_tiki_product_url(product, product_id):
    url_path = product.get("url_path")
    if url_path:
        url_path = str(url_path).strip()
        if url_path.startswith("http"):
            return url_path
        return f"https://tiki.vn/{url_path.lstrip('/')}"
    return f"https://tiki.vn/p/{product_id}"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-logo">📚</div>
    <div class="sidebar-title">Tiki Book Intelligence</div>
    <div class="sidebar-subtitle">
        DASHBOARD PHÂN TÍCH HIỆU QUẢ SẢN PHẨM SÁCH TRÊN TIKI 
    </div>
    <div class="sidebar-line"></div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Chọn trang",
    [
        "🏠 Tổng quan",
        "🗄️ Dữ liệu sản phẩm",
        "🏷️ Gán nhãn sản phẩm",
        "🧠 Mô hình dự đoán",
        "📊 Đánh giá mô hình",
        "🔎 Dự đoán sản phẩm",
        "💡 Gợi ý cải thiện",
        "🚩 Kết luận",
    ],
)


# ============================================================
# LOAD DATA
# ============================================================

df = load_products(file_mtime(DATA_PATH))
model_df = load_model_comparison(file_mtime(MODEL_COMPARISON_PATH))
report_df = load_classification_report(file_mtime(CLASSIFICATION_REPORT_PATH))
sentiment_df = load_sentiment_data(
    file_mtime(SENTIMENT_SIMPLE_PATH),
    file_mtime(SENTIMENT_PHOBERT_PATH),
)
total_review_rows, unique_review_products = load_review_stats(
    str(REVIEWS_PATH),
    file_mtime(REVIEWS_PATH),
)
total_rows = total_review_rows if total_review_rows > 0 else len(df)


if df.empty:
    page_header(
        "Chưa có dữ liệu",
        "Không tìm thấy file data/tiki_books_labeled.csv. Hãy chạy bước gán nhãn trước.",
    )
    st.code("python label_products.py", language="bash")
    st.stop()

counts = label_counts(df)
total_products = len(df)
total_revenue = df["estimated_revenue"].sum() if "estimated_revenue" in df.columns else 0
best_model_name, best_acc, best_precision, best_f1 = get_best_model_info(model_df)


# ============================================================
# PAGES
# ============================================================

if page == "🏠 Tổng quan":
    has_phobert = SENTIMENT_PHOBERT_PATH.exists()
    sentiment_badge = (
        '<span style="background:#10B981;color:#fff;border-radius:999px;padding:3px 10px;font-size:13px;font-weight:800;margin-left:10px;">PhoBERT ✓</span>'
        if has_phobert else
        '<span style="background:#F59E0B;color:#fff;border-radius:999px;padding:3px 10px;font-size:13px;font-weight:800;margin-left:10px;">Sentiment Simple</span>'
    )

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-left">
                <div class="hero-icon">📚</div>
                <div>
                    <h1 class="hero-title">Tiki Book Intelligence {sentiment_badge}</h1>
                    <p class="hero-subtitle">
                        Hệ thống thông minh phân tích & dự đoán 
hiệu quả sản phẩm sách trên Tiki.
                    </p>
                </div>
            </div>
            <div class="hero-art">📈</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("🗃️", "Tổng dòng dữ liệu crawl", format_number(total_rows), "Số dòng reviews trong file crawl", accent="")

    with c2:
        metric_card("📦", "Tổng sản phẩm", format_number(total_products), "Sản phẩm duy nhất", accent="purple")

    with c3:
        metric_card("💰", "Doanh thu ước tính", format_vnd(total_revenue), "Giá × số đã bán", accent="green")

    with c4:
        metric_card("🏆", "Mô hình tốt nhất", best_model_name, f"Accuracy {format_float(best_acc * 100, 2)}% | F1 {format_float(best_f1, 4)}", accent="yellow")

    left, right = st.columns([2.1, 1], gap="large")

    with left:
        section_header("📊 Phân bố nhãn sản phẩm")
        show_plotly(make_bar_chart(counts))

    with right:
        section_header("🏷️ Số lượng theo nhãn")
        for label, value in counts.items():
            render_label_card(label, int(value))

    # Pipeline hiện tại
    pipeline_phobert_style = "color:#10B981;font-weight:800;" if has_phobert else "color:#94A3B8;"
    pipeline_phobert_label = "sentiment_phobert.py ✓" if has_phobert else "sentiment_phobert.py (chưa có)"
    st.markdown(
        f"""
        <div class="content-card" style="margin-top:8px;">
            <div class="section-title">🔄 Pipeline hiện tại</div>
            <p style="color:#475569;font-size:14px;margin:10px 0 14px 0;">
                Dữ liệu được xử lý qua 5 bước từ crawl đến dashboard.
            </p>
            <div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;font-size:13px;font-weight:700;">
                <span class="stat-pill">crawl_tiki_books.py</span>
                <span style="color:#94A3B8;">→</span>
                <span class="stat-pill">label_products.py</span>
                <span style="color:#94A3B8;">→</span>
                <span class="stat-pill">sentiment_simple.py ✓</span>
                <span style="color:#94A3B8;">→</span>
                <span class="stat-pill" style="{pipeline_phobert_style}">{pipeline_phobert_label}</span>
                <span style="color:#94A3B8;">→</span>
                <span class="stat-pill">train_product_classifier.py</span>
            </div>
            <p style="color:#475569;font-size:13px;margin-top:14px;">
                ℹ️ <b>Doanh thu ước tính</b> = giá bán × lượt bán. Đây là chỉ số tham khảo, không phải doanh thu thực tế.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


elif page == "🗄️ Dữ liệu sản phẩm":
    page_header(
        "Dữ liệu sản phẩm",
        "Bảng dữ liệu sản phẩm đã được làm sạch, tổng hợp và gán nhãn hiệu quả.",
    )

    comment_series = df["comment_count"] if "comment_count" in df.columns else pd.Series([0] * len(df))
    comment_count = int((comment_series > 0).sum())
    no_comment_count = total_products - comment_count

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("📦", "Tổng sản phẩm", format_number(total_products), "Sản phẩm duy nhất")

    with c2:
        metric_card("💬", "Có bình luận", format_number(comment_count), "Sản phẩm có comment")

    with c3:
        metric_card("📭", "Không bình luận", format_number(no_comment_count), "Sản phẩm chưa có comment")

    search = st.text_input(
        "Tìm kiếm tên sách",
        placeholder="Nhập tên sách cần tìm..."
    )

    selected_label = st.radio(
        "Chọn nhãn sản phẩm",
        [
            "Tất cả",
            "Best Seller",
            "High Potential",
            "Premium / Niche Quality",
            "Normal",
            "Needs Improvement",
        ],
        horizontal=True,
        key="product_label_filter",
    )

    show_df = df.copy()
    if search and "product_name" in show_df.columns:
        show_df = show_df[
            show_df["product_name"].astype(str).str.contains(search, case=False, na=False)
        ]

    if selected_label != "Tất cả" and "product_label" in show_df.columns:
        show_df = show_df[show_df["product_label"] == selected_label]

    section_header(
        "📋 Danh sách sản phẩm",
        f"Đang hiển thị {len(show_df):,} sản phẩm phù hợp. Bảng bên dưới hiển thị tối đa 50 dòng đầu tiên."
    )
    render_light_table(prepare_display_df(show_df), max_rows=50)


elif page == "🏷️ Gán nhãn sản phẩm":
    page_header(
        "Gán nhãn sản phẩm",
        "Mỗi sản phẩm được gán vào một nhóm hiệu quả dựa trên điểm tổng hợp từ giá, đánh giá, lượt bán và phản hồi.",
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        mini_card("Best Seller", "Bán chạy, doanh thu và lượt bán cao.", "#0B63F6")

    with c2:
        mini_card("High Potential", "Có tiềm năng tăng trưởng tốt.", "#10B981")

    with c3:
        mini_card("Premium / Niche", "Phù hợp phân khúc cao cấp hoặc ngách.", "#8B5CF6")

    with c4:
        mini_card("Normal", "Hiệu quả ở mức trung bình.", "#64748B")

    with c5:
        mini_card("Needs Improvement", "Cần cải thiện hiệu quả hoặc phản hồi.", "#EF4444")

    left, right = st.columns([2.1, 1], gap="large")

    with left:
        section_header("🍩 Tỷ lệ nhãn sản phẩm")
        show_plotly(make_donut_chart(counts))

    with right:
        section_header("📌 Thống kê nhãn")
        for label, value in counts.items():
            render_label_card(label, int(value))

    section_header("📋 Danh sách sản phẩm đã gán nhãn")
    render_light_table(prepare_display_df(df), max_rows=40)

elif page == "🧠 Mô hình dự đoán":
    page_header(
        "Mô hình dự đoán",
        "So sánh các mô hình học máy và lựa chọn mô hình tốt nhất để dự đoán nhãn hiệu quả sản phẩm.",
    )

    top_left, top_right = st.columns([1.15, 2], gap="large")

    with top_left:
        st.markdown(
            f"""
            <div class="model-highlight">
                <div class="model-highlight-title">🏆 Mô hình tốt nhất</div>
                <div class="model-highlight-value">{safe_html(best_model_name)}</div>
                <div class="model-highlight-text">
                    Mô hình được chọn dựa trên F1-score weighted cao nhất, giúp cân bằng giữa Precision và Recall.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("🎯", "Accuracy", format_float(best_acc, 4), "Độ chính xác tổng thể")
        with c2:
            metric_card("📌", "Precision", format_float(best_precision, 4), "Độ chính xác dự đoán")
        with c3:
            metric_card("⚖️", "F1-score", format_float(best_f1, 4), "Chỉ số cân bằng")

    st.markdown(
        """
        <div class="content-card">
            <div class="section-title">🧠 Cách mô hình hoạt động</div>
            <div class="feature-grid">
                <div class="feature-box">
                    <div class="feature-title">1. Đầu vào dữ liệu</div>
                    <div class="feature-text">
                        Hệ thống sử dụng 6 đặc trưng: giá bán (price), số đánh giá
                        (review_count), số bình luận (comment_count), điểm bình luận trung bình
                        (avg_comment_rating), tỷ lệ trung tính (neutral_ratio) và điểm cảm xúc
                        PhoBERT (sentiment_score).
                    </div>
                </div>
                <div class="feature-box">
                    <div class="feature-title">2. Học từ nhãn sản phẩm</div>
                    <div class="feature-text">
                        Mô hình học mối quan hệ giữa đặc trưng sản phẩm và các nhãn như
                        Best Seller, High Potential, Normal hoặc Needs Improvement.
                    </div>
                </div>
                <div class="feature-box">
                    <div class="feature-title">3. Dự đoán sản phẩm mới</div>
                    <div class="feature-text">
                        Khi nhập thông tin sản phẩm mới, mô hình dự đoán nhóm hiệu quả
                        để hỗ trợ phân tích và ra quyết định.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("📊 Bảng so sánh mô hình", "So sánh hiệu năng giữa Decision Tree, Random Forest và XGBoost.")

    if model_df.empty:
        st.warning("Chưa có file model_comparison.csv. Hãy chạy: python train_product_classifier.py")
    else:
        render_light_table(model_df, max_rows=10)

    st.markdown(
        """
        <div class="content-card">
            <div class="section-title">📌 Ý nghĩa các chỉ số đánh giá</div>
            <p style="color:#475569; line-height:1.6; margin-top:10px;">
                <span class="stat-pill">Accuracy</span>
                Cho biết tỷ lệ dự đoán đúng trên toàn bộ tập kiểm tra.
                <br>
                <span class="stat-pill">Precision</span>
                Cho biết khi mô hình dự đoán một nhãn, khả năng dự đoán đó đúng là bao nhiêu.
                <br>
                <span class="stat-pill">Recall</span>
                Cho biết mô hình tìm được bao nhiêu mẫu đúng trong từng nhãn.
                <br>
                <span class="stat-pill">F1-score</span>
                Là chỉ số cân bằng giữa Precision và Recall, phù hợp khi dữ liệu có nhiều nhãn.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

   
elif page == "📊 Đánh giá mô hình":
    page_header(
        "Đánh giá mô hình",
        "Quan sát ma trận nhầm lẫn, mức độ quan trọng đặc trưng và báo cáo phân loại.",
    )

    if not report_df.empty and "label" in report_df.columns:
        accuracy_value = get_report_value(report_df, "accuracy", ["f1-score", "F1-score"], 0)
        macro_f1 = get_report_value(report_df, "macro avg", ["f1-score", "F1-score"], 0)
        weighted_f1 = get_report_value(report_df, "weighted avg", ["f1-score", "F1-score"], 0)
        total_support = get_report_value(report_df, "weighted avg", ["support", "Số mẫu"], 0)

        if not total_support:
            total_support = get_report_value(report_df, "macro avg", ["support", "Số mẫu"], 0)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("🧪", "Tổng mẫu test", format_number(total_support), "Số mẫu dùng đánh giá")

        with c2:
            metric_card("🎯", "Accuracy", format_float(accuracy_value, 4), "Tỷ lệ dự đoán đúng", accent="green")

        with c3:
            metric_card("📊", "Macro F1-score", format_float(macro_f1, 4), "Trung bình đều các lớp", accent="green")

        with c4:
            metric_card("⚖️", "Weighted F1-score", format_float(weighted_f1, 4), "Có xét số lượng từng lớp", accent="green")

    left, right = st.columns(2, gap="large")

    with left:
        section_header("🧩 Confusion Matrix")
        if CONFUSION_MATRIX_PATH.exists():
            show_image(CONFUSION_MATRIX_PATH)
        else:
            st.warning("Thiếu confusion_matrix.png. Hãy chạy: python train_product_classifier.py")

    with right:
        section_header("📈 Feature Importance")
        if FEATURE_IMPORTANCE_PATH.exists():
            show_image(FEATURE_IMPORTANCE_PATH)
        else:
            st.warning("Thiếu feature_importance.png. Hãy chạy: python train_product_classifier.py")

    # ── So sánh Sentiment: Simple vs PhoBERT ──────────────────────────
    section_header(
        "🧪 So sánh Sentiment: Simple vs PhoBERT",
        "Điểm sentiment trung bình theo từng nhãn sản phẩm giữa hai phương pháp phân tích cảm xúc."
    )

    label_order = ["Best Seller", "Premium / Niche Quality", "High Potential", "Normal", "Needs Improvement"]

    if sentiment_df.empty or "product_label" not in sentiment_df.columns:
        st.info("Chưa có dữ liệu sentiment. Hãy chạy: python sentiment_simple.py")
    else:
        has_phobert_col = "sentiment_phobert" in sentiment_df.columns

        # Tính trung bình theo nhãn
        agg_cols = {"sentiment_simple": "mean"}
        if has_phobert_col:
            agg_cols["sentiment_phobert"] = "mean"

        avg_by_label = (
            sentiment_df.groupby("product_label")
            .agg(agg_cols)
            .reindex(label_order)
            .fillna(0)
            .reset_index()
        )

        if has_phobert_col:
            fig_sent = go.Figure()
            fig_sent.add_trace(go.Bar(
                name="Sentiment Simple (từ điển)",
                x=avg_by_label["product_label"],
                y=avg_by_label["sentiment_simple"].round(4),
                marker_color="#94A3B8",
                marker_line_color="#64748B",
                marker_line_width=1,
                text=avg_by_label["sentiment_simple"].round(3),
                textposition="outside",
                textfont=dict(color="#64748B", size=12),
            ))
            fig_sent.add_trace(go.Bar(
                name="PhoBERT (vinai/phobert-base)",
                x=avg_by_label["product_label"],
                y=avg_by_label["sentiment_phobert"].round(4),
                marker_color="#0B63F6",
                marker_line_color="#0755D5",
                marker_line_width=1,
                text=avg_by_label["sentiment_phobert"].round(3),
                textposition="outside",
                textfont=dict(color="#0B63F6", size=12),
            ))
            fig_sent.update_layout(
                barmode="group",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font_color="#0F172A",
                height=430,
                margin=dict(l=60, r=20, t=50, b=90),
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.03,
                    xanchor="center", x=0.5,
                    font=dict(size=13, color="#0F172A"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#E2E8F0",
                    borderwidth=1,
                ),
                yaxis=dict(
                    title="Điểm Sentiment (0 = tiêu cực, 1 = tích cực)",
                    title_font=dict(size=13, color="#475569"),
                    tickfont=dict(color="#475569"),
                    gridcolor="#E2E8F0",
                    range=[0, 1.15],
                ),
                xaxis=dict(
                    title="Nhãn sản phẩm",
                    title_font=dict(size=13, color="#475569"),
                    tickfont=dict(color="#0F172A", size=12),
                    showgrid=False,
                ),
            )
            show_plotly(fig_sent)

            # Bảng chi tiết
            avg_display = avg_by_label.copy()
            avg_display.columns = ["Nhãn sản phẩm", "Sentiment Simple", "PhoBERT"]
            avg_display["Chênh lệch"] = (avg_display["PhoBERT"] - avg_display["Sentiment Simple"]).round(4)
            avg_display["Sentiment Simple"] = avg_display["Sentiment Simple"].round(4)
            avg_display["PhoBERT"] = avg_display["PhoBERT"].round(4)
            st.dataframe(avg_display.set_index("Nhãn sản phẩm"), use_container_width=True)
        else:
            # Chỉ có sentiment_simple
            fig_sent = px.bar(
                avg_by_label,
                x="product_label",
                y="sentiment_simple",
                text=avg_by_label["sentiment_simple"].round(3),
                color_discrete_sequence=["#94A3B8"],
                template="plotly_white",
                labels={"product_label": "Nhãn sản phẩm", "sentiment_simple": "Sentiment Simple"},
            )
            fig_sent.update_traces(textposition="outside")
            fig_sent.update_layout(height=380, paper_bgcolor="white", plot_bgcolor="white")
            show_plotly(fig_sent)
            st.info("Chưa có file sentiment_phobert.csv — chạy trên Colab để thêm cột PhoBERT.")

    # ── So sánh hiệu năng model trước/sau PhoBERT ─────────────────────
    section_header(
        "📈 Tác động của PhoBERT lên hiệu năng mô hình",
        "So sánh Accuracy và F1-score trước (Sentiment Simple) và sau (PhoBERT) khi thay thế sentiment_score."
    )

    perf_before = {
        "Mô hình": ["Decision Tree", "Random Forest", "XGBoost"],
        "Accuracy (Simple)": [0.7935, 0.7932, 0.8061],
        "F1 (Simple)": [0.7546, 0.7716, 0.7603],
        "Accuracy (PhoBERT)": [0.7932, 0.7952, 0.8081],
        "F1 (PhoBERT)": [0.7552, 0.7736, 0.7620],
    }
    perf_df = pd.DataFrame(perf_before)
    perf_df["ΔAccuracy"] = (perf_df["Accuracy (PhoBERT)"] - perf_df["Accuracy (Simple)"]).round(4)
    perf_df["ΔF1"] = (perf_df["F1 (PhoBERT)"] - perf_df["F1 (Simple)"]).round(4)

    st.dataframe(perf_df.set_index("Mô hình"), use_container_width=True)

    # Đọc động từ model_comparison.csv để tránh hardcode số liệu
    _mc = pd.read_csv("data/model_comparison.csv")
    _best_f1_row = _mc.loc[_mc["F1-score"].idxmax()]
    _best_acc_row = _mc.loc[_mc["Accuracy"].idxmax()]
    _best_f1_name = _best_f1_row["Model"].replace(" Classifier", "")
    _best_f1_val = f"{_best_f1_row['F1-score']:.4f}"
    _best_acc_name = _best_acc_row["Model"].replace(" Classifier", "")
    _best_acc_val = f"{_best_acc_row['Accuracy'] * 100:.2f}%"
    st.markdown(
        f"""
        <div class="info-card">
            🧠 <b>PhoBERT</b> cải thiện nhẹ F1-score ở cả 3 mô hình (~0.002 điểm).
            {_best_f1_name} đạt F1 cao nhất (<b>{_best_f1_val}</b>), {_best_acc_name} dẫn đầu về Accuracy (<b>{_best_acc_val}</b>).
            sentiment_score (PhoBERT) chiếm 16% feature importance — đứng thứ 4 sau price, avg_comment_rating, review_count.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Classification Report ─────────────────────────────────────────
    section_header(
        "📄 Classification Report",
        "Bảng precision, recall, F1-score và số mẫu test theo từng nhãn."
    )
    if report_df.empty:
        st.warning("Chưa có classification_report.csv. Hãy chạy: python train_product_classifier.py")
    else:
        render_light_table(report_df, max_rows=30)


elif page == "🔎 Dự đoán sản phẩm":
    page_header(
        "Dự đoán sản phẩm từ link Tiki",
        "Dán link sản phẩm Tiki để hệ thống tự lấy dữ liệu public, tạo feature và dự đoán nhãn hiệu quả.",
    )

    model, encoder = load_model_and_encoder(file_mtime(MODEL_PATH), file_mtime(ENCODER_PATH))

    if model is None or encoder is None:
        st.warning("Chưa tìm thấy model. Hãy chạy: python train_product_classifier.py")
    else:
        # Hướng dẫn lấy link
        st.markdown(
            """
            <div class="content-card" style="margin-bottom:16px;">
                <div class="section-title" style="font-size:17px;margin-bottom:10px;">📌 Cách lấy link sản phẩm Tiki</div>
                <div style="display:flex;gap:24px;flex-wrap:wrap;">
                    <div style="flex:1;min-width:220px;">
                        <div style="font-weight:800;color:#0F172A;margin-bottom:6px;">Bước 1</div>
                        <div style="color:#475569;font-size:14px;line-height:1.6;">
                            Truy cập <b>tiki.vn</b>, tìm kiếm sản phẩm sách cần phân tích.
                        </div>
                    </div>
                    <div style="flex:1;min-width:220px;">
                        <div style="font-weight:800;color:#0F172A;margin-bottom:6px;">Bước 2</div>
                        <div style="color:#475569;font-size:14px;line-height:1.6;">
                            Mở trang sản phẩm, sao chép toàn bộ URL từ thanh địa chỉ trình duyệt.
                        </div>
                    </div>
                    <div style="flex:1;min-width:220px;">
                        <div style="font-weight:800;color:#0F172A;margin-bottom:6px;">Bước 3</div>
                        <div style="color:#475569;font-size:14px;line-height:1.6;">
                            Dán vào ô bên dưới và nhấn <b>Phân tích và dự đoán</b>.
                        </div>
                    </div>
                </div>
                <div style="margin-top:14px;padding:10px 14px;background:#F8FBFF;border-radius:12px;border:1px solid #DBEAFE;">
                    <div style="font-size:13px;font-weight:700;color:#64748B;margin-bottom:6px;">Ví dụ link hợp lệ:</div>
                    <div style="font-size:13px;color:#0B63F6;font-family:monospace;line-height:2;">
                        https://tiki.vn/dac-nhan-tam-p8873163.html<br>
                        https://tiki.vn/p/8873163<br>
                        8873163
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tiki_link = st.text_input(
            "Nhập link sản phẩm Tiki",
            placeholder="Dán link sản phẩm Tiki, ví dụ: https://tiki.vn/...-p316018.html",
        )

        submitted = st.button(
            "Phân tích và dự đoán",
            type="primary",
            use_container_width=False,
        )

        if submitted:
            product_id = extract_product_id(tiki_link)
            if not product_id:
                st.error("Không tìm thấy mã sản phẩm trong link.")
            else:
                with st.spinner("Đang lấy dữ liệu sản phẩm từ Tiki..."):
                    product = fetch_tiki_product(product_id)

                if not product:
                    st.error("Không lấy được dữ liệu sản phẩm từ Tiki.")
                else:
                    with st.spinner("Đang lấy đánh giá và tạo feature dự đoán..."):
                        reviews = fetch_tiki_reviews(product_id, limit=20)
                        input_df, disp = build_features_from_product(product, reviews)

                    try:
                        pred = model.predict(input_df)[0]
                        label = encoder.inverse_transform([pred])[0]
                    except Exception as e:
                        st.error(f"Lỗi khi dự đoán: {e}")
                        st.info(
                            "Nếu lỗi liên quan đến sklearn/monotonic_cst, hãy chạy lại "
                            "train_product_classifier.py rồi push lại file models/product_performance_model.pkl."
                        )
                    else:
                        product_name = product.get("name") or "Không rõ tên sản phẩm"
                        product_url = build_tiki_product_url(product, product_id)
                        explanations = {
                            "Best Seller": "Sản phẩm có khả năng bán chạy, hiệu quả cao.",
                            "High Potential": "Sản phẩm có tiềm năng tăng trưởng tốt.",
                            "Premium / Niche Quality": "Sản phẩm phù hợp phân khúc cao cấp hoặc ngách.",
                            "Normal": "Sản phẩm có hiệu quả ở mức trung bình.",
                            "Needs Improvement": "Sản phẩm cần cải thiện chỉ số bán hàng hoặc phản hồi.",
                        }

                        _pred_style = _LABEL_STYLE.get(label, {
                            "bg": "#F8FAFC", "border": "#E2E8F0",
                            "icon_bg": "#EFF6FF", "color": "#0B63F6", "icon": "🏷️"
                        })
                        _label_gradient = {
                            "Best Seller":             "linear-gradient(135deg,#0B1F5E,#0B63F6)",
                            "High Potential":          "linear-gradient(135deg,#064E3B,#10B981)",
                            "Premium / Niche Quality": "linear-gradient(135deg,#3B1C8C,#8B5CF6)",
                            "Normal":                  "linear-gradient(135deg,#334155,#64748B)",
                            "Needs Improvement":       "linear-gradient(135deg,#7F1D1D,#EF4444)",
                        }.get(label, "linear-gradient(135deg,#0B63F6,#8B5CF6)")
                        _label_shadow = {
                            "Best Seller":             "rgba(11,99,246,0.3)",
                            "High Potential":          "rgba(16,185,129,0.3)",
                            "Premium / Niche Quality": "rgba(139,92,246,0.3)",
                            "Normal":                  "rgba(100,116,139,0.25)",
                            "Needs Improvement":       "rgba(239,68,68,0.3)",
                        }.get(label, "rgba(11,99,246,0.3)")

                        st.markdown(
                            f"""
                            <div style="
                                background:{_label_gradient};
                                border-radius:26px;
                                padding:30px 36px;
                                box-shadow:0 20px 50px {_label_shadow};
                                margin-bottom:22px;
                                position:relative;
                                overflow:hidden;
                            ">
                                <div style="position:absolute;top:-50px;right:-50px;width:200px;height:200px;
                                    border-radius:50%;background:radial-gradient(circle,rgba(255,255,255,0.07),transparent 65%);"></div>
                                <div style="display:flex;align-items:center;gap:18px;margin-bottom:14px;">
                                    <div style="font-size:48px;background:rgba(255,255,255,0.15);
                                        border-radius:18px;width:72px;height:72px;display:flex;
                                        align-items:center;justify-content:center;
                                        border:1px solid rgba(255,255,255,0.2);">{_pred_style['icon']}</div>
                                    <div>
                                        <div style="font-size:13px;font-weight:700;color:rgba(255,255,255,0.7);
                                            text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">
                                            ✅ KẾT QUẢ DỰ ĐOÁN
                                        </div>
                                        <div style="font-size:30px;font-weight:900;color:#FFFFFF;
                                            line-height:1.2;text-shadow:0 2px 8px rgba(0,0,0,0.2);">
                                            {safe_html(label)}
                                        </div>
                                    </div>
                                </div>
                                <div style="font-size:17px;font-weight:700;color:rgba(255,255,255,0.95);
                                    margin-bottom:8px;line-height:1.5;">
                                    {safe_html(product_name)}
                                </div>
                                <div style="font-size:15px;color:rgba(255,255,255,0.75);
                                    margin-bottom:16px;line-height:1.5;">
                                    {safe_html(explanations.get(label, "Không có mô tả."))}
                                </div>
                                <a href="{safe_html(product_url)}" target="_blank"
                                    style="display:inline-block;background:rgba(255,255,255,0.2);
                                        color:#FFFFFF;font-weight:700;font-size:14px;
                                        padding:8px 20px;border-radius:999px;
                                        border:1px solid rgba(255,255,255,0.3);
                                        text-decoration:none;">
                                    🔗 Xem sản phẩm trên Tiki
                                </a>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        _label_accent = {
                            "Best Seller": "", "High Potential": "green",
                            "Premium / Niche Quality": "purple",
                            "Normal": "", "Needs Improvement": "red",
                        }.get(label, "")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            metric_card("💰", "Giá bán", format_vnd(disp["price"]), "Giá từ API Tiki")
                        with c2:
                            metric_card("⭐", "Rating", format_float(disp["rating"], 2), "Điểm đánh giá", accent="yellow")
                        with c3:
                            metric_card(_pred_style["icon"], "Nhãn dự đoán", label, "Kết quả từ mô hình", accent=_label_accent)

                        c4, c5, c6 = st.columns(3)
                        with c4:
                            metric_card("📝", "Số đánh giá", format_number(disp["review_count"]), "review_count")
                        with c5:
                            metric_card("📦", "Lượt bán", format_number(disp["sold_count"]), "sold_count", accent="purple")
                        with c6:
                            metric_card("💬", "Số bình luận lấy được", format_number(disp["comment_count"]), "20 review đầu")

                        c7, c8, c9 = st.columns(3)
                        with c7:
                            metric_card(
                                "📈",
                                "Doanh thu ước tính",
                                format_vnd(disp["estimated_revenue"]),
                                "Giá × lượt bán",
                                accent="green",
                            )
                        with c8:
                            metric_card(
                                "😊",
                                "Sentiment Score",
                                format_float(disp["sentiment_score"], 3),
                                "positive_ratio − negative_ratio",
                                accent="green" if disp["sentiment_score"] >= 0 else "red",
                            )
                        with c9:
                            metric_card("🔗", "Mã sản phẩm", product_id, "Trích từ link Tiki")

                        section_header("📄 Features đưa vào model (6 cột)")
                        render_light_table(input_df, max_rows=1)

        # Card giải thích 5 nhãn — luôn hiển thị bên dưới form
        st.markdown(
            """
            <div class="content-card" style="margin-top:24px;">
                <div class="section-title" style="font-size:17px;margin-bottom:14px;">🏷️ Ý nghĩa 5 nhãn sản phẩm</div>
                <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;flex-wrap:wrap;">
                    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:14px;padding:14px;">
                        <div style="font-size:20px;margin-bottom:6px;">🏆</div>
                        <div style="font-weight:800;color:#0B63F6;font-size:13px;margin-bottom:5px;">Best Seller</div>
                        <div style="color:#475569;font-size:12px;line-height:1.5;">Bán chạy, lượt bán cao, doanh thu vượt trội, rating tốt.</div>
                    </div>
                    <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:14px;padding:14px;">
                        <div style="font-size:20px;margin-bottom:6px;">🚀</div>
                        <div style="font-weight:800;color:#10B981;font-size:13px;margin-bottom:5px;">High Potential</div>
                        <div style="color:#475569;font-size:12px;line-height:1.5;">Có tín hiệu tích cực nhưng chưa đủ review hoặc lượt bán.</div>
                    </div>
                    <div style="background:#FAF5FF;border:1px solid #E9D5FF;border-radius:14px;padding:14px;">
                        <div style="font-size:20px;margin-bottom:6px;">💎</div>
                        <div style="font-weight:800;color:#8B5CF6;font-size:13px;margin-bottom:5px;">Premium / Niche</div>
                        <div style="color:#475569;font-size:12px;line-height:1.5;">Giá cao, rating tốt, phù hợp phân khúc cao cấp hoặc ngách.</div>
                    </div>
                    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:14px;padding:14px;">
                        <div style="font-size:20px;margin-bottom:6px;">📘</div>
                        <div style="font-weight:800;color:#64748B;font-size:13px;margin-bottom:5px;">Normal</div>
                        <div style="color:#475569;font-size:12px;line-height:1.5;">Hiệu quả trung bình, ổn định nhưng chưa nổi bật.</div>
                    </div>
                    <div style="background:#FFF5F5;border:1px solid #FECACA;border-radius:14px;padding:14px;">
                        <div style="font-size:20px;margin-bottom:6px;">⚠️</div>
                        <div style="font-weight:800;color:#EF4444;font-size:13px;margin-bottom:5px;">Needs Improvement</div>
                        <div style="color:#475569;font-size:12px;line-height:1.5;">Rating thấp hoặc tỷ lệ phản hồi tiêu cực cao, cần tối ưu.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


elif page == "💡 Gợi ý cải thiện":
    page_header(
        "Gợi ý cải thiện",
        "Tập trung vào nhóm sản phẩm cần cải thiện và đưa ra hành động đề xuất.",
    )

    if "product_label" not in df.columns:
        st.warning("Không có cột product_label trong dữ liệu.")
    else:
        improve_df = df[df["product_label"] == "Needs Improvement"].copy()

        if improve_df.empty:
            st.info("Không có sản phẩm thuộc nhóm Needs Improvement.")
        else:
            avg_rating = improve_df["rating"].mean() if "rating" in improve_df.columns else 0
            avg_negative = improve_df["negative_ratio"].mean() if "negative_ratio" in improve_df.columns else 0

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card("💡", "Sản phẩm cần cải thiện", format_number(len(improve_df)), "Thuộc nhóm Needs Improvement", accent="red")

            with c2:
                metric_card("⭐", "Rating trung bình", format_float(avg_rating, 2), "Trung bình nhóm", accent="red")

            with c3:
                metric_card("⚠️", "Tỷ lệ tiêu cực TB", format_float(avg_negative, 2), "Trung bình nhóm", accent="red")

            # Ghi chú: Tạo gợi ý cải thiện dựa trên rating, tỷ lệ tiêu cực và lượt bán.
            def make_suggestion(row):
                rating_value = row.get("rating", 0)
                negative_value = row.get("negative_ratio", 0)
                sold_value = row.get("sold_count", 0)

                if negative_value >= 0.35:
                    return "Kiểm tra phản hồi tiêu cực và cải thiện mô tả/chất lượng sản phẩm."
                if rating_value > 0 and rating_value < 3.5:
                    return "Cải thiện chất lượng sản phẩm hoặc nội dung hiển thị."
                if sold_value <= 10:
                    return "Tăng hiển thị, tối ưu tiêu đề, hình ảnh và chiến lược quảng bá."
                return "Theo dõi thêm phản hồi và tối ưu nội dung bán hàng."

            improve_df["suggestion"] = improve_df.apply(make_suggestion, axis=1)

            cols = [
                "product_name",
                "price",
                "rating",
                "review_count",
                "sold_count",
                "negative_ratio",
                "estimated_revenue",
                "suggestion",
                "product_url",
            ]

            section_header("📋 Danh sách sản phẩm cần cải thiện")
            render_light_table(improve_df, columns=cols, max_rows=50)


elif page == "🚩 Kết luận":
    page_header(
        "Kết luận",
        "Tổng hợp kết quả đạt được, hạn chế còn tồn tại và hướng phát triển tiếp theo của hệ thống.",
    )

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown(
            """
            <div class="mini-card" style="border-left: 6px solid #10B981;">
                <div class="mini-card-title">✅ Kết quả đạt được</div>
                <ul class="mini-card-text">
                    <li>Xây dựng dashboard phân tích dữ liệu sách Tiki trực quan.</li>
                    <li>Tổng hợp được dữ liệu sản phẩm, đánh giá, lượt bán và doanh thu ước tính.</li>
                    <li>Gán nhãn sản phẩm thành 5 nhóm hiệu quả khác nhau.</li>
                    <li>Huấn luyện mô hình học máy để dự đoán nhãn sản phẩm.</li>
                    <li>Có biểu đồ, bảng đánh giá mô hình và chức năng dự đoán sản phẩm mới.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="mini-card" style="border-left: 6px solid #F59E0B;">
                <div class="mini-card-title">⚠️ Hạn chế</div>
                <ul class="mini-card-text">
                    <li>Dữ liệu phụ thuộc vào API public nên có thể thiếu một số trường.</li>
                    <li>Nhãn sản phẩm được tạo theo quy tắc nghiệp vụ, chưa phải nhãn chuyên gia.</li>
                    <li>Một số sản phẩm chưa có bình luận hoặc lượt bán đầy đủ.</li>
                    <li>Chưa phân tích sâu nội dung cảm xúc tiếng Việt trong bình luận.</li>
                    <li>Chưa đánh giá xu hướng thay đổi theo thời gian.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="mini-card" style="border-left: 6px solid #0B63F6;">
                <div class="mini-card-title">🚀 Hướng phát triển</div>
                <ul class="mini-card-text">
                    <li>Mở rộng dữ liệu với nhiều nhóm sách và nhiều thời điểm crawl khác nhau.</li>
                    <li>Bổ sung phân tích cảm xúc tiếng Việt từ nội dung bình luận.</li>
                    <li>Xây dựng module dự báo xu hướng bán hàng theo thời gian.</li>
                    <li>Tối ưu giao diện dashboard để có thể triển khai online.</li>
                    <li>Kết hợp thêm dữ liệu người bán, khuyến mãi và tồn kho nếu có.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Bảng hành trình cải thiện model
    st.markdown(
        """
        <div class="content-card" style="margin-top:20px;">
            <div class="section-title">📈 Hành trình cải thiện mô hình</div>
            <p style="color:#64748B;font-size:14px;margin:8px 0 16px 0;">
                Ba giai đoạn phát triển — từ mô hình ban đầu (label leakage) đến khi tích hợp PhoBERT.
            </p>
            <div class="table-wrap">
                <table class="light-table">
                    <thead>
                        <tr>
                            <th>Giai đoạn</th>
                            <th>Phương pháp gán nhãn</th>
                            <th>Sentiment</th>
                            <th>Best Accuracy</th>
                            <th>Best F1 (Weighted)</th>
                            <th>Ghi chú</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><b>1 — Ban đầu</b></td>
                            <td>Quantile thủ công</td>
                            <td>Không có</td>
                            <td style="color:#EF4444;font-weight:800;">99.1%</td>
                            <td style="color:#EF4444;font-weight:800;">0.9910</td>
                            <td>⚠️ Label leakage — accuracy giả</td>
                        </tr>
                        <tr>
                            <td><b>2 — HDBSCAN + Fix leakage</b></td>
                            <td>HDBSCAN (24 cụm tự động)</td>
                            <td>Sentiment Simple</td>
                            <td style="color:#F59E0B;font-weight:800;">80.61%</td>
                            <td style="color:#F59E0B;font-weight:800;">0.7716</td>
                            <td>✅ Kết quả thực, không leakage</td>
                        </tr>
                        <tr style="background:#F0FDF4;">
                            <td><b>3 — PhoBERT NLP</b></td>
                            <td>HDBSCAN (24 cụm tự động)</td>
                            <td>PhoBERT (vinai)</td>
                            <td style="color:#10B981;font-weight:800;">80.81%</td>
                            <td style="color:#10B981;font-weight:800;">0.7736</td>
                            <td>✅ Sentiment tiếng Việt chất lượng cao</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <p style="color:#64748B;font-size:13px;margin-top:10px;">
                * Giai đoạn 1 dùng <code>sold_count</code>, <code>rating</code>, <code>estimated_revenue</code> vừa để gán nhãn vừa để train → model chỉ học thuộc quy tắc, không có giá trị thực tế.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-card">
            <div class="section-title">📌 Tổng kết chung</div>
            <p style="color:#475569; line-height:1.7; font-size:15px; margin-top:10px;">
                Hệ thống <b>Tiki Book Intelligence</b> giúp chuyển dữ liệu sản phẩm sách từ dạng thô
                thành các thông tin có ý nghĩa hơn, bao gồm phân nhóm hiệu quả, đánh giá mô hình,
                dự đoán sản phẩm mới và gợi ý cải thiện. Đây là nền tảng phù hợp để phát triển
                thành một công cụ hỗ trợ phân tích kinh doanh sách trên sàn thương mại điện tử.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="content-card">
            <div class="section-title">🧭 Quy trình hệ thống</div>
            <div class="timeline-box">
                <div class="timeline-step">
                    <div class="step-number">1</div>
                    <div>
                        <div class="step-content-title">Thu thập dữ liệu</div>
                        <div class="step-content-text">Crawl dữ liệu sản phẩm, đánh giá, giá bán, lượt bán và đường dẫn từ Tiki.</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="step-number">2</div>
                    <div>
                        <div class="step-content-title">Làm sạch và tổng hợp</div>
                        <div class="step-content-text">Chuẩn hóa dữ liệu, gộp theo sản phẩm và tạo các đặc trưng cấp sản phẩm.</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="step-number">3</div>
                    <div>
                        <div class="step-content-title">Gán nhãn hiệu quả</div>
                        <div class="step-content-text">Phân sản phẩm thành 5 nhóm như Best Seller, High Potential, Normal và Needs Improvement.</div>
                    </div>
                </div>
                <div class="timeline-step">
                    <div class="step-number">4</div>
                    <div>
                        <div class="step-content-title">Huấn luyện và dự đoán</div>
                        <div class="step-content-text">So sánh các mô hình học máy và sử dụng mô hình tốt nhất để dự đoán sản phẩm mới.</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
