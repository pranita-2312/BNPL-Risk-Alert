
import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# BNPL RISK INTELLIGENCE
# Modern fintech risk-monitoring dashboard
# ============================================================

st.set_page_config(
    page_title="BNPL Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_PATH / "data"
MODEL_PATH = PROJECT_PATH / "models"

# The resolver also accepts the uploaded "(1)" filenames during setup.
ROOT_PATH = Path(__file__).resolve().parent


def resolve_asset(folder: Path, clean_name: str, uploaded_name: str | None = None) -> Path:
    candidates = [
        folder / clean_name,
        folder / (uploaded_name or ""),
        ROOT_PATH / clean_name,
        ROOT_PATH / (uploaded_name or ""),
        Path("/mnt/data") / clean_name,
        Path("/mnt/data") / (uploaded_name or ""),
    ]
    for path in candidates:
        if str(path) and path.exists():
            return path
    return folder / clean_name


HISTORY_FILE = resolve_asset(
    DATA_PATH,
    "SIMULATED_BNPL_Customer_History.xlsx",
    "SIMULATED_BNPL_Customer_History(2).xlsx",
)

MAIN_DATA_FILE = resolve_asset(
    DATA_PATH,
    "bnpl_dataset (Buy Now, Pay Later (BNPL) Dataset) Practice.xlsx",
    "bnpl_dataset (Buy Now, Pay Later (BNPL) Dataset) Practice(2).xlsx",
)


# ============================================================
# CONSTANTS
# ============================================================

KEEP = [
    "Customer_Age",
    "Annual_Income",
    "Credit_Score",
    "Purchase_Amount",
    "Gender",
    "Purchase_Category",
    "BNPL_Provider",
]

CLUSTER_FEATURES = [
    "Customer_Age",
    "Annual_Income",
    "Credit_Score",
    "Purchase_Amount",
]

CATEGORIES = [
    "Electronics",
    "Fashion",
    "Groceries",
    "Travel",
    "Beauty",
    "Home & Furniture",
]

PROVIDERS = ["Klarna", "Afterpay", "Sezzle", "Affirm"]
GENDERS = ["Male", "Female", "Non-Binary"]

PAGE_NAMES = [
    "Dashboard",
    "Assessment",
    "Risk Monitor",
    "Analytics",
    "Trust Center",
]

DEMO_PROFILES = {
    "Safe customer": {
        "Customer_ID": "CUST_1003",
        "Customer_Age": 45,
        "Annual_Income": 85000,
        "Credit_Score": 720,
        "Purchase_Amount": 400,
        "Gender": "Female",
        "Purchase_Category": "Fashion",
        "BNPL_Provider": "Afterpay",
        "Shared_Device_Accounts": 1,
        "Shared_Payment_Accounts": 1,
        "Account_Age_Days": 400,
        "Identity_Mismatch": False,
        "Unusual_Device": False,
        "First_Party_Fraud_Signal": False,
    },
    "High-risk customer": {
        "Customer_ID": "CUST_1011",
        "Customer_Age": 24,
        "Annual_Income": 28000,
        "Credit_Score": 380,
        "Purchase_Amount": 1800,
        "Gender": "Male",
        "Purchase_Category": "Electronics",
        "BNPL_Provider": "Klarna",
        "Shared_Device_Accounts": 8,
        "Shared_Payment_Accounts": 5,
        "Account_Age_Days": 14,
        "Identity_Mismatch": True,
        "Unusual_Device": True,
        "First_Party_Fraud_Signal": False,
    },
    "Medium-risk customer": {
        "Customer_ID": "CUST_1021",
        "Customer_Age": 32,
        "Annual_Income": 52000,
        "Credit_Score": 520,
        "Purchase_Amount": 950,
        "Gender": "Male",
        "Purchase_Category": "Travel",
        "BNPL_Provider": "Affirm",
        "Shared_Device_Accounts": 2,
        "Shared_Payment_Accounts": 1,
        "Account_Age_Days": 180,
        "Identity_Mismatch": False,
        "Unusual_Device": False,
        "First_Party_Fraud_Signal": False,
    },
    "Early-warning customer": {
        "Customer_ID": "CUST_1016",
        "Customer_Age": 29,
        "Annual_Income": 41000,
        "Credit_Score": 490,
        "Purchase_Amount": 1200,
        "Gender": "Female",
        "Purchase_Category": "Beauty",
        "BNPL_Provider": "Sezzle",
        "Shared_Device_Accounts": 3,
        "Shared_Payment_Accounts": 2,
        "Account_Age_Days": 90,
        "Identity_Mismatch": False,
        "Unusual_Device": True,
        "First_Party_Fraud_Signal": False,
    },
}


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #070817;
    --panel: rgba(20, 24, 50, .82);
    --panel-2: rgba(28, 31, 67, .70);
    --line: rgba(126, 139, 207, .22);
    --text: #F5F3FF;
    --muted: #9CA6C7;
    --cyan: #49F2FF;
    --purple: #A970FF;
    --pink: #FF55D9;
    --green: #59F5C9;
    --yellow: #FFD66E;
    --red: #FF6B81;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(73,242,255,.13), transparent 22%),
        radial-gradient(circle at 78% 0%, rgba(169,112,255,.16), transparent 25%),
        radial-gradient(circle at 88% 75%, rgba(255,85,217,.08), transparent 24%),
        linear-gradient(135deg, #050611 0%, #090B1B 50%, #060714 100%);
    color: var(--text);
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    display: block !important;
    background: transparent !important;
    height: 3rem !important;
}

header[data-testid="stHeader"] button {
    color: #B8C4FF !important;
}

header[data-testid="stHeader"] button:hover {
    color: #49F2FF !important;
}

.block-container {
    max-width: 100% !important;
    padding: 1.6rem 2rem 3rem 2rem !important;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(
            circle at 15% 8%,
            rgba(73, 242, 255, 0.16),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 35%,
            rgba(169, 112, 255, 0.18),
            transparent 35%
        ),
        radial-gradient(
            circle at 45% 85%,
            rgba(255, 85, 217, 0.08),
            transparent 32%
        ),
        linear-gradient(
            160deg,
            #111A36 0%,
            #0D122A 45%,
            #0A0C20 100%
        ) !important;

    border-right: 1px solid rgba(126, 139, 207, 0.30) !important;

    box-shadow:
        8px 0 35px rgba(0, 0, 0, 0.18),
        inset -1px 0 rgba(73, 242, 255, 0.05) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

/* Responsive main content when sidebar opens/closes */
[data-testid="stSidebar"][aria-expanded="true"] ~ section[data-testid="stMain"] {
    width: calc(100% - 280px) !important;
}

[data-testid="stSidebar"][aria-expanded="false"] ~ section[data-testid="stMain"] {
    width: 100% !important;
}

[data-testid="stSidebar"] label {
    color: var(--muted) !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(10,13,32,.75) !important;
    color: var(--text) !important;
    border: 1px solid rgba(126,139,207,.24) !important;
    border-radius: 10px !important;
}

.stButton > button {
    border-radius: 11px !important;
    min-height: 42px;
    font-weight: 600 !important;
    border: 1px solid rgba(126,139,207,.24) !important;

    background:
        linear-gradient(
            135deg,
            rgba(31,38,78,.92),
            rgba(22,27,59,.92)
        ) !important;

    color: #F5F3FF !important;

    box-shadow:
        inset 0 1px rgba(255,255,255,.035),
        0 5px 16px rgba(0,0,0,.12) !important;

    transition: all .18s ease !important;
}

.stButton > button:hover {
    border-color: rgba(73,242,255,.50) !important;
    color: #FFFFFF !important;

    background:
        linear-gradient(
            135deg,
            rgba(38,49,91,.98),
            rgba(29,34,73,.98)
        ) !important;

    box-shadow:
        0 0 18px rgba(73,242,255,.10),
        inset 0 1px rgba(255,255,255,.05) !important;

    transform: translateX(2px);
}

.nav-button button {
    text-align: left !important;
}

.hero {
    background:
        linear-gradient(120deg, rgba(22,27,58,.94), rgba(30,22,59,.78));
    border: 1px solid rgba(126,139,207,.25);
    border-radius: 20px;
    padding: 22px 26px;
    box-shadow: 0 18px 50px rgba(0,0,0,.25);
    margin-bottom: 18px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 700;
    color: var(--text);
}

.hero-sub {
    color: var(--muted);
    font-size: 12px;
    margin-top: 5px;
}

.hero-right {
    color: var(--muted);
    font-size: 10px;
    text-align: right;
    line-height: 1.8;
}

.dashboard-head {
    display:flex; align-items:flex-end; justify-content:space-between; gap:20px;
    margin: 4px 0 18px;
}
.dashboard-head-title {
    font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700;
    letter-spacing:-.02em;
}
.dashboard-head-sub { color:var(--muted); font-size:12px; margin-top:5px; }
.dashboard-chip {
    display:inline-flex; align-items:center; gap:7px; padding:7px 11px;
    border:1px solid rgba(73,242,255,.22); border-radius:999px;
    background:rgba(73,242,255,.06); color:#BDEFF5; font-size:10px; font-weight:700;
}
.donut-card {
    background:linear-gradient(145deg, rgba(24,29,61,.88), rgba(12,15,35,.84));
    border:1px solid rgba(126,139,207,.21); border-radius:18px; padding:18px;
    box-shadow:inset 0 1px rgba(255,255,255,.035),0 14px 35px rgba(0,0,0,.18);
}
.section-box {
    background: linear-gradient(145deg, rgba(24,29,61,.88), rgba(12,15,35,.84));
    border: 1px solid rgba(126,139,207,.21);
    border-radius: 18px;
    padding: 15px 18px;
    margin-bottom: 10px;
    box-shadow: inset 0 1px rgba(255,255,255,.035), 0 14px 35px rgba(0,0,0,.18);
}

.section-box-title {
    color: #F5F3FF;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.circle-stat {
    width:86px; height:86px; border-radius:50%; display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    background:radial-gradient(circle at 35% 30%,rgba(169,112,255,.28),rgba(20,24,50,.95) 64%);
    border:1px solid rgba(169,112,255,.35); box-shadow:0 0 24px rgba(169,112,255,.12);
}
.circle-stat-value { font-family:'Space Grotesk',sans-serif; font-size:20px; font-weight:700; }
.circle-stat-label { color:var(--muted); font-size:8px; text-transform:uppercase; letter-spacing:.1em; }
.risk-legend-row { display:flex; align-items:center; justify-content:space-between; padding:9px 0; border-bottom:1px solid rgba(126,139,207,.12); font-size:12px; }
.risk-legend-row:last-child { border-bottom:none; }
.risk-dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:8px; }
.page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 700;
    margin: 6px 0 3px;
}

.page-subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 18px;
}

.kpi {
    background: linear-gradient(145deg, rgba(24,29,61,.90), rgba(13,16,37,.84));
    border: 1px solid rgba(126,139,207,.22);
    border-radius: 16px;
    padding: 16px 17px;
    min-height: 108px;
    box-shadow: inset 0 1px rgba(255,255,255,.035), 0 14px 35px rgba(0,0,0,.18);
}

.kpi-label {
    color: var(--muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.kpi-value {
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 700;
    margin-top: 8px;
}

.kpi-sub {
    color: var(--muted);
    font-size: 10px;
    margin-top: 4px;
}

.card {
    background: linear-gradient(145deg, rgba(24,29,61,.82), rgba(12,15,35,.82));
    border: 1px solid rgba(126,139,207,.21);
    border-radius: 17px;
    padding: 18px;
    box-shadow: inset 0 1px rgba(255,255,255,.035), 0 14px 35px rgba(0,0,0,.18);
    margin-bottom: 16px;
}

.card-title {
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.profile-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
}

.big-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 36px;
    font-weight: 700;
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 800;
    border: 1px solid currentColor;
}

.badge-low { color: var(--green); background: rgba(89,245,201,.10); }
.badge-medium { color: var(--yellow); background: rgba(255,214,110,.10); }
.badge-high { color: var(--pink); background: rgba(255,85,217,.10); }
.badge-clear { color: var(--green); background: rgba(89,245,201,.10); }
.badge-watch { color: var(--yellow); background: rgba(255,214,110,.10); }
.badge-alert { color: var(--pink); background: rgba(255,85,217,.10); }

.alert-card {
    border-radius: 13px;
    padding: 13px 15px;
    margin: 7px 0;
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(126,139,207,.16);
}

.metric-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 9px 0;
    border-bottom: 1px solid rgba(126,139,207,.13);
    color: #DADCF0;
    font-size: 12px;
}

.metric-row:last-child {
    border-bottom: none;
}

.sidebar-brand {
    padding: 7px 5px 18px;
}

.sidebar-brand-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 19px;
    font-weight: 700;
}

.sidebar-brand-sub {
    color: var(--muted);
    font-size: 10px;
    margin-top: 3px;
}

.sidebar-section {
    color: #697394;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: .15em;
    text-transform: uppercase;
    margin: 18px 5px 7px;
}

.small-muted {
    color: var(--muted);
    font-size: 10px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(126,139,207,.18);
    border-radius: 12px;
    overflow: hidden;
}

hr {
    border-color: rgba(126,139,207,.16) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL / DATA LOADING
# ============================================================

def model_path(clean_name: str, uploaded_name: str) -> Path:
    return resolve_asset(MODEL_PATH, clean_name, uploaded_name)


@st.cache_resource
def load_models():
    preprocessor = joblib.load(model_path("preprocessor.pkl", "preprocessor(1).pkl"))
    lgb_model = joblib.load(model_path("lgb_model.pkl", "lgb_model(1).pkl"))
    lr_model = joblib.load(model_path("lr_model.pkl", "lr_model(1).pkl"))
    kmeans = joblib.load(model_path("kmeans.pkl", "kmeans(1).pkl"))
    cluster_scaler = joblib.load(model_path("kmeans_scaler.pkl", "kmeans_scaler(1).pkl"))
    segment_labels = joblib.load(model_path("segment_labels.pkl", "segment_labels(1).pkl"))
    top_risk_factors = joblib.load(
        model_path("top_risk_factors.pkl", "top_risk_factors(1).pkl")
    )
    final_risk_config = joblib.load(
        model_path("final_risk_config.pkl", "final_risk_config(1).pkl")
    )
    fraud_config = joblib.load(
        model_path("fraud_config.pkl", "fraud_config(1).pkl")
    )
    return (
        preprocessor,
        lgb_model,
        lr_model,
        kmeans,
        cluster_scaler,
        segment_labels,
        top_risk_factors,
        final_risk_config,
        fraud_config,
    )


@st.cache_data
def load_main_dataset():
    return pd.read_excel(
        MAIN_DATA_FILE,
        sheet_name="bnpl_dataset (Buy Now, Pay Late",
    )


@st.cache_data
def load_history():
    df = pd.read_excel(
        HISTORY_FILE,
        sheet_name="Transaction History",
        header=2,
    )
    df.columns = (
        df.columns.astype(str)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .str.replace(" ", "_")
    )
    return df


@st.cache_data
def load_behaviour_table():
    df = pd.read_excel(
        HISTORY_FILE,
        sheet_name="Customer Behavioural Features",
        header=2,
    )
    df.columns = (
        df.columns.astype(str)
        .str.replace("\n", " ", regex=False)
        .str.strip()
        .str.replace(" ", "_")
    )
    if "Customer_ID" not in df.columns and "Customer_ID" not in df.columns:
        for col in df.columns:
            if str(col).lower().replace("_", " ") == "customer id":
                df = df.rename(columns={col: "Customer_ID"})
                break
    return df


models_error = None
try:
    (
        preprocessor,
        lgb_model,
        lr_model,
        kmeans,
        cluster_scaler,
        SEGMENT_LABELS,
        TOP_RISK_FACTORS,
        FINAL_RISK_CONFIG,
        FRAUD_CONFIG,
    ) = load_models()
    main_df = load_main_dataset()
    history_df = load_history()
    behaviour_df = load_behaviour_table()
    MODELS_READY = True
except Exception as exc:
    MODELS_READY = False
    models_error = str(exc)
    main_df = None
    history_df = None
    behaviour_df = None


# ============================================================
# CORE MODEL LOGIC
# ============================================================

def classify_risk(default_probability):
    if default_probability < FINAL_RISK_CONFIG["risk_low"]:
        return "LOW"
    if default_probability < FINAL_RISK_CONFIG["risk_medium"]:
        return "MEDIUM"
    return "HIGH"


def calculate_credit_limit(annual_income, risk):
    base_limit = (annual_income / 12) * 0.20
    risk_factor = {"LOW": 1.00, "MEDIUM": 0.60, "HIGH": 0.30}.get(risk, 0.30)
    return round(max(1000, min(base_limit * risk_factor, 15000)), 0)


def predict_customer(customer_data):
    customer_df = pd.DataFrame([customer_data])
    encoded = preprocessor.transform(customer_df[KEEP])

    # LightGBM default probability
    lgb_default_index = list(lgb_model.classes_).index("Defaulted")
    lgb_probs = lgb_model.predict_proba(encoded)[0]
    lgb_default = float(lgb_probs[lgb_default_index])

    # Logistic Regression baseline
    lr_default_index = list(lr_model.classes_).index("Defaulted")
    lr_probs = lr_model.predict_proba(encoded)[0]
    lr_default = float(lr_probs[lr_default_index])

    # Risk level from model probability
    risk = classify_risk(lgb_default)

    # Risk score shown to the user
    risk_score = round(
        (
            lgb_default * 50 + (1 - customer_data["Credit_Score"] / 850) * 30 + min(
                (customer_data["Purchase_Amount"]
                 / max(customer_data["Annual_Income"], 1)) * 100,
                20,
            )
        ),
        1,
    )

    risk_score = float(np.clip(risk_score, 0, 100))

    credit_limit = calculate_credit_limit(
        customer_data["Annual_Income"],
        risk
    )

    # K-Means behaviour segment
    cluster_input = [[customer_data[f] for f in CLUSTER_FEATURES]]

    segment_num = int(
        kmeans.predict(
            cluster_scaler.transform(cluster_input)
        )[0]
    )

    segment_label = SEGMENT_LABELS[segment_num]

    clean_factors = [
        str(f)
        .replace("numeric__", "")
        .replace("categorical__", "")
        .replace("_", " ")
        .title()
        for f in TOP_RISK_FACTORS[:8]
    ]

    return {
        "default_probability": lgb_default,
        "lr_default_probability": lr_default,
        "risk_level": risk,
        "risk_score": risk_score,
        "credit_limit": credit_limit,
        "segment": segment_label,
        "segment_number": segment_num,
        "top_risk_factors": clean_factors,
    }


def fraud_check(customer):
    score = 0
    alerts = []

    if customer["Shared_Device_Accounts"] > 5:
        score += FRAUD_CONFIG["shared_device_score"]
        alerts.append("Device linked to multiple customer accounts")

    if customer["Shared_Payment_Accounts"] > 3:
        score += FRAUD_CONFIG["shared_payment_score"]
        alerts.append("Payment instrument linked to multiple customer accounts")

    if customer["Account_Age_Days"] < 30:
        score += FRAUD_CONFIG["new_account_score"]
        alerts.append("Very new account")

    if customer["Identity_Mismatch"]:
        score += FRAUD_CONFIG["identity_mismatch_score"]
        alerts.append("Identity signal mismatch")

    if customer["Unusual_Device"]:
        score += FRAUD_CONFIG["unusual_device_score"]
        alerts.append("Unusual device activity")

    if customer["First_Party_Fraud_Signal"]:
        score += FRAUD_CONFIG["first_party_fraud_score"]
        alerts.append("Potential first-party fraud behaviour")

    if score >= FRAUD_CONFIG["suspicious_threshold"]:
        status = "SUSPICIOUS"
    elif score >= FRAUD_CONFIG["review_threshold"]:
        status = "REVIEW"
    else:
        status = "CLEAR"

    return {"fraud_score": score, "status": status, "alerts": alerts}


def early_warning_check(customer_id):
    if history_df is None:
        return {
            "alert_status": "No History",
            "alert_triggers": [],
            "alert_message": "Transaction history is unavailable.",
            "behavioural_summary": {},
        }

    ch = history_df[
        history_df["Customer_ID"].astype(str).str.strip()
        == str(customer_id).strip()
    ].copy()

    if ch.empty:
        return {
            "alert_status": "No History",
            "alert_triggers": [],
            "alert_message": "No transaction history available for this customer.",
            "behavioural_summary": {},
        }

    ch["Transaction_Date"] = pd.to_datetime(
        ch["Transaction_Date"], errors="coerce"
    )
    ch = ch.sort_values("Transaction_Date")

    total_transactions = len(ch)
    statuses = ch["Repayment_Status"].astype(str).str.lower()

    on_time_rate = statuses.eq("paid on time").mean()
    late_rate = statuses.eq("late payment").mean()
    default_rate = statuses.eq("defaulted").mean()

    avg_days = pd.to_numeric(ch["Days_to_Repay"], errors="coerce").mean()
    avg_purchase = pd.to_numeric(ch["Purchase_Amount"], errors="coerce").mean()

    amounts = pd.to_numeric(ch["Purchase_Amount"], errors="coerce").dropna().values
    recent = amounts[-min(5, len(amounts)):] if len(amounts) else np.array([])
    spend_trend = (
        float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
        if len(recent) >= 2
        else 0.0
    )

    recent_bad = int(
        statuses.tail(3).isin(["late payment", "defaulted"]).sum()
    )

    if ch["Transaction_Date"].notna().any():
        days_since = max(
            0,
            (
                pd.Timestamp.today().normalize()
                - ch["Transaction_Date"].max().normalize()
            ).days,
        )
    else:
        days_since = 0

    triggers = []
    conditions_met = 0

    if spend_trend > 20:
        conditions_met += 1
        triggers.append(f"Spending is rising (trend slope: +{spend_trend:.1f})")

    if on_time_rate < 0.60:
        conditions_met += 1
        triggers.append(f"On-time payment rate is low ({on_time_rate:.0%})")

    if recent_bad >= 1:
        conditions_met += 1
        triggers.append(f"{recent_bad} bad payment(s) in last 3 transactions")

    if conditions_met >= 2:
        status = "ALERT"
        message = "Multiple warning signals. Consider reducing credit limit."
    elif conditions_met == 1:
        status = "WATCH"
        message = "One warning signal detected. Continue monitoring."
    else:
        status = "CLEAR"
        message = "No behavioural warning signals detected."

    return {
        "alert_status": status,
        "alert_triggers": triggers,
        "alert_message": message,
        "behavioural_summary": {
            "total_transactions": total_transactions,
            "avg_purchase_amount": round(float(avg_purchase), 2),
            "on_time_rate": round(float(on_time_rate), 3),
            "late_payment_rate": round(float(late_rate), 3),
            "default_rate": round(float(default_rate), 3),
            "avg_days_to_repay": round(float(avg_days), 1),
            "recent_spend_trend": round(float(spend_trend), 2),
            "days_since_last_purchase": int(days_since),
            "recent_bad_payments_last3": recent_bad,
        },
    }


# ============================================================
# PORTFOLIO ANALYTICS
# ============================================================

@st.cache_data(show_spinner=False)
def build_portfolio_predictions(df):
    if not MODELS_READY:
        return pd.DataFrame()

    X = df[KEEP].copy()
    encoded = preprocessor.transform(X)

    lgb_default_index = list(lgb_model.classes_).index("Defaulted")
    probabilities = lgb_model.predict_proba(encoded)[:, lgb_default_index]

    result = pd.DataFrame({
        "Transaction_ID": df["Transaction_ID"].astype(str).values,
        "Default_Probability": probabilities,
        "Repayment_Status": df["Repayment_Status"].astype(str).values,
        "Purchase_Amount": pd.to_numeric(df["Purchase_Amount"], errors="coerce").values,
        "Annual_Income": pd.to_numeric(df["Annual_Income"], errors="coerce").values,
        "Credit_Score": pd.to_numeric(df["Credit_Score"], errors="coerce").values,
        "Customer_Age": pd.to_numeric(df["Customer_Age"], errors="coerce").values,
        "Purchase_Category": df["Purchase_Category"].astype(str).values,
        "BNPL_Provider": df["BNPL_Provider"].astype(str).values,
    })

    result["Risk_Level"] = np.select(
        [
            result["Default_Probability"] < FINAL_RISK_CONFIG["risk_low"],
            result["Default_Probability"] < FINAL_RISK_CONFIG["risk_medium"],
        ],
        ["LOW", "MEDIUM"],
        default="HIGH",
    )

    result["Credit_Exposure"] = result.apply(
        lambda r: calculate_credit_limit(r["Annual_Income"], r["Risk_Level"]),
        axis=1,
    )

    return result


# ============================================================
# UI HELPERS
# ============================================================

def risk_badge(level):
    cls = {
        "LOW": "badge-low",
        "MEDIUM": "badge-medium",
        "HIGH": "badge-high",
    }.get(level, "badge")
    return f'<span class="badge {cls}">{level} RISK</span>'


def status_badge(status):
    cls = {
        "CLEAR": "badge-clear",
        "WATCH": "badge-watch",
        "ALERT": "badge-alert",
        "REVIEW": "badge-watch",
        "SUSPICIOUS": "badge-alert",
    }.get(status, "badge")
    return f'<span class="badge {cls}">{status}</span>'


def kpi(label, value, sub="", accent=None):
    color = accent or "#F5F3FF"
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color};">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_title(text):
    st.markdown(f'<div class="card-title">{text}</div>', unsafe_allow_html=True)


def plot_layout(fig, height=310):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="#DADCF0",
            family="Times New Roman",
            size=16,
        ),

        legend=dict(
            font=dict(
                color="#DADCF0",
                family="Times New Roman",
                size=14,
            )
        ),

        xaxis=dict(
            gridcolor="rgba(126,139,207,.10)",
            zeroline=False,
            tickfont=dict(
                family="Times New Roman",
                size=15,
            ),
            title_font=dict(
                family="Times New Roman",
                size=16,
            ),
        ),

        yaxis=dict(
            gridcolor="rgba(126,139,207,.10)",
            zeroline=False,
            tickfont=dict(
                family="Times New Roman",
                size=15,
            ),
            title_font=dict(
                family="Times New Roman",
                size=16,
            ),
        ),
    )

    return fig

# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "customer_id" not in st.session_state:
    st.session_state.customer_id = ""

if "demo_profile" not in st.session_state:
    st.session_state.demo_profile = "Safe customer"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">🛡 BNPL Risk</div>
            <div class="sidebar-brand-sub">Risk intelligence console</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    nav_icons = {
        "Dashboard",
        "Assessment",
        "Risk Monitor",
        "Analytics",
        "Trust Center",
    }

    for page in PAGE_NAMES:
        if st.button(
            page,
            key=f"nav_{page}",
            use_container_width=True,
        ):
            st.session_state.page = page
            st.rerun()

    st.markdown('<div class="sidebar-section">Quick Demo</div>', unsafe_allow_html=True)

    demo_choice = st.selectbox(
        "Sample customer",
        ["Select a demo..."] + list(DEMO_PROFILES.keys()),
        key="demo_choice",
    )

    if st.button("Launch Quick Demo", use_container_width=True):
        if demo_choice != "Select a demo...":
            st.session_state.demo_profile = demo_choice
            st.session_state.customer_id = DEMO_PROFILES[demo_choice]["Customer_ID"]
            st.session_state.page = "Assessment"
            st.rerun()

    st.markdown('<div class="sidebar-section">Customer</div>', unsafe_allow_html=True)

    st.text_input(
        "Customer ID",
        key="customer_id",
        placeholder="e.g. CUST_1006",
    )

    st.markdown(
        '<div class="small-muted">Synthetic customer history: CUST_1001 to CUST_1050</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="small-muted">
            Prototype environment<br>
            Synthetic data only<br>
            No live customer information
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# GLOBAL HEADER
# ============================================================

if not MODELS_READY:
    st.error(
        "The application could not load the trained model files. "
        "Keep the model files inside the project's `models` folder."
    )
    st.code(models_error or "Unknown model-loading error")
    st.stop()


# ============================================================
# DASHBOARD
# ============================================================

def page_dashboard():
    st.markdown(
        """
        <div class="dashboard-head">
            <div>
                <div class="dashboard-head-title">Risk Command Center</div>
                <div class="dashboard-head-sub">Portfolio risk, exposure and customer behaviour at a glance.</div>
            </div>
            <div class="dashboard-chip"><span style="color:#59F5C9">●</span> LIVE-STYLE PORTFOLIO VIEW</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    portfolio = build_portfolio_predictions(main_df)

    total = len(main_df)
    defaults = int((main_df["Repayment_Status"] == "Defaulted").sum())
    late = int((main_df["Repayment_Status"] == "Late Payment").sum())
    observed_default = defaults / total if total else 0

    high = int((portfolio["Risk_Level"] == "HIGH").sum())
    medium = int((portfolio["Risk_Level"] == "MEDIUM").sum())
    low = int((portfolio["Risk_Level"] == "LOW").sum())

    high_exposure = float(
        portfolio.loc[portfolio["Risk_Level"] == "HIGH", "Credit_Exposure"].sum()
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Total Transactions", f"{total:,}", "50K training transactions")
    with c2:
        kpi("Observed Defaults", f"{observed_default:.1%}", f"{defaults:,} defaulted", "#FF6B81")
    with c3:
        kpi("High Risk", f"{high:,}", f"{high/total:.1%} of portfolio", "#FF55D9")
    with c4:
        kpi("Late Payments", f"{late:,}", "Observed portfolio", "#FFD66E")
    with c5:
        kpi("High-Risk Exposure", f"₹{high_exposure/1e6:.1f}M", "Recommended credit limits", "#49F2FF")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown(
            '<div class="section-box"><div class="section-box-title">Risk Distribution</div></div>',
            unsafe_allow_html=True,
        )

        low_pct = (low / total) * 100 if total else 0
        medium_pct = (medium / total) * 100 if total else 0
        high_pct = (high / total) * 100 if total else 0

        components.html(
            f"""
            <html>
            <head>
                <style>
                    * {{
                        box-sizing: border-box;
                    }}

                    body {{
                        margin: 0;
                        background: transparent;
                        font-family: Inter, Arial, sans-serif;
                        color: #F5F3FF;
                    }}

                    .risk-wrapper {{
                        width: 100%;
                        height: 330px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        gap: 35px;
                    }}

                    .rings {{
                        position: relative;
                        width: 260px;
                        height: 260px;
                        flex-shrink: 0;
                    }}

                    svg {{
                        width: 260px;
                        height: 260px;
                        transform: rotate(-90deg);
                        overflow: visible;
                    }}

                    .ring-bg {{
                        fill: none;
                        stroke: rgba(126,139,207,.10);
                        stroke-width: 14;
                    }}

                    .ring {{
                        fill: none;
                        stroke-width: 14;
                        stroke-linecap: round;
                        stroke-dasharray: 0 1000;
                        animation: drawRing 1.8s ease-out forwards;
                    }}

                    .low {{
                        stroke: #59F5C9;
                        animation-delay: .15s;
                    }}

                    .medium {{
                        stroke: #FFD66E;
                        animation-delay: .35s;
                    }}

                    .high {{
                        stroke: #FF55D9;
                        animation-delay: .55s;
                    }}

                    @keyframes drawRing {{
                        from {{
                            stroke-dasharray: 0 1000;
                        }}
                        to {{
                            stroke-dasharray: var(--target) 1000;
                        }}
                    }}

                    .center {{
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        text-align: center;
                        width: 120px;
                    }}

                    .center-value {{
                        font-size: 31px;
                        font-weight: 700;
                        color: #F5F3FF;
                    }}

                    .center-label {{
                        margin-top: 5px;
                        font-size: 9px;
                        letter-spacing: .12em;
                        color: #9CA6C7;
                    }}

                    .legend {{
                        min-width: 155px;
                    }}

                    .legend-row {{
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        padding: 9px 0;
                        border-bottom: 1px solid rgba(126,139,207,.12);
                        font-size: 12px;
                    }}

                    .legend-row:last-child {{
                        border-bottom: none;
                    }}

                    .legend-left {{
                        display: flex;
                        align-items: center;
                        gap: 8px;
                    }}

                    .dot {{
                        width: 9px;
                        height: 9px;
                        border-radius: 50%;
                    }}

                    .low-dot {{
                        background: #59F5C9;
                    }}

                    .medium-dot {{
                        background: #FFD66E;
                    }}

                    .high-dot {{
                        background: #FF55D9;
                    }}

                    .percentage {{
                        color: #9CA6C7;
                        font-size: 10px;
                        margin-top: 2px;
                        text-align: right;
                    }}
                </style>
            </head>

            <body>
                <div class="risk-wrapper">

                    <div class="rings">

                        <svg viewBox="0 0 260 260">

                            <!-- Background rings -->
                            <circle
                                class="ring-bg"
                                cx="130"
                                cy="130"
                                r="105"
                            />

                            <circle
                                class="ring-bg"
                                cx="130"
                                cy="130"
                                r="82"
                            />

                            <circle
                                class="ring-bg"
                                cx="130"
                                cy="130"
                                r="59"
                            />

                            <!-- Animated rings -->
                            <circle
                                class="ring low"
                                cx="130"
                                cy="130"
                                r="105"
                                pathLength="100"
                                style="--target:{low_pct};"
                            />

                            <circle
                                class="ring medium"
                                cx="130"
                                cy="130"
                                r="82"
                                pathLength="100"
                                style="--target:{medium_pct};"
                            />

                            <circle
                                class="ring high"
                                cx="130"
                                cy="130"
                                r="59"
                                pathLength="100"
                                style="--target:{high_pct};"
                            />

                        </svg>

                        <div class="center">
                            <div class="center-value">{total/1000:.0f}K</div>
                            <div class="center-label">CUSTOMERS</div>
                        </div>

                    </div>

                    <div class="legend">

                        <div class="legend-row">
                            <div class="legend-left">
                                <span class="dot low-dot"></span>
                                <strong>Low</strong>
                            </div>
                            <strong>{low:,}</strong>
                        </div>
                        <div class="percentage">{low_pct:.1f}%</div>

                        <div class="legend-row">
                            <div class="legend-left">
                                <span class="dot medium-dot"></span>
                                <strong>Medium</strong>
                            </div>
                            <strong>{medium:,}</strong>
                        </div>
                        <div class="percentage">{medium_pct:.1f}%</div>

                        <div class="legend-row">
                            <div class="legend-left">
                                <span class="dot high-dot"></span>
                                <strong>High</strong>
                            </div>
                            <strong>{high:,}</strong>
                        </div>
                        <div class="percentage">{high_pct:.1f}%</div>

                    </div>

                </div>
            </body>
            </html>
            """,
            height=340,
            scrolling=False,
        )
    
    with right:
        st.markdown(
        '<div class="section-box"><div class="section-box-title">Portfolio Snapshot</div></div>',
        unsafe_allow_html=True,)
        snapshot_rows = [
            ("Predicted high risk", f"{high/total:.1%}", "#FF55D9"),
            ("Predicted medium risk", f"{medium/total:.1%}", "#FFD66E"),
            ("Predicted low risk", f"{low/total:.1%}", "#59F5C9"),
            ("Observed default rate", f"{observed_default:.1%}", "#FF6B81"),
            ("Observed late-payment rate", f"{late/total:.1%}", "#49F2FF"),
        ]
        for label, value, color in snapshot_rows:
            st.markdown(
                f'<div class="metric-row"><span>{label}</span><strong style="color:{color}">{value}</strong></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div style="display:flex;gap:12px;margin-top:18px;align-items:center">', unsafe_allow_html=True)
        st.markdown(f'<div class="circle-stat"><div class="circle-stat-value">{high/total:.0%}</div><div class="circle-stat-label">High Risk</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div><div style="font-size:12px;font-weight:700;color:#F5F3FF">Exposure under watch</div><div style="font-size:22px;font-weight:700;color:#49F2FF;margin-top:4px">₹{high_exposure/1e6:.1f}M</div><div style="font-size:10px;color:#9CA6C7;margin-top:3px">Recommended credit limits</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    left, right = st.columns(2, vertical_alignment="top")

    with left:
        st.markdown(
            '<div class="section-box"><div class="section-box-title">Default Probability</div></div>',
            unsafe_allow_html=True,
        )

        counts, bin_edges = np.histogram(
            portfolio["Default_Probability"],
            bins=30,
            range=(0, 1),
        )

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=counts,
                mode="lines",
                line=dict(
                    color="#49F2FF",
                    width=3,
                    shape="spline",
                ),
                fill="tozeroy",
                fillcolor="rgba(73,242,255,.08)",
                hovertemplate=(
                    "Default Probability: %{x:.1%}"
                    "<br>Transactions: %{y:,}"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_xaxes(
            title="Default probability",
            tickformat=".0%",
            range=[0, 1],
        )

        fig.update_yaxes(
            title="Transactions",
        )

        st.plotly_chart(
            plot_layout(fig, 300),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        st.markdown(
        '<div class="section-box"><div class="section-box-title">Risk by BNPL Provider</div></div>',
        unsafe_allow_html=True,)
        provider_risk = (
            portfolio.groupby("BNPL_Provider")
            .agg(
                Default_Probability=("Default_Probability", "mean"),
                Transactions=("Transaction_ID", "count"),
            )
            .reset_index()
        )
        provider_risk["Default_Probability"] *= 100
        fig = px.bar(
            provider_risk,
            x="BNPL_Provider",
            y="Default_Probability",
            text="Default_Probability",
        )
        fig.update_traces(marker_color="#49C8FFFF", texttemplate="%{text:.1f}%")
        fig.update_yaxes(title="Average predicted default probability (%)")
        st.plotly_chart(plot_layout(fig, 300), use_container_width=True)

    st.markdown(
        '<div class="section-box"><div class="section-box-title">Recent Portfolio Risk Alerts</div></div>',
        unsafe_allow_html=True,)

    top = portfolio.nlargest(8, "Default_Probability")[
        [
            "Transaction_ID",
            "Default_Probability",
            "Risk_Level",
            "Credit_Score",
            "Purchase_Amount",
            "BNPL_Provider",
        ]
    ].copy()

    top["Default_Probability"] = top["Default_Probability"].map(lambda x: f"{x:.1%}")
    top["Purchase_Amount"] = top["Purchase_Amount"].map(lambda x: f"₹{x:,.0f}")
    top["Credit_Score"] = top["Credit_Score"].round().astype(int)

    st.dataframe(top, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CUSTOMER PROFILE HELPERS
# ============================================================

def customer_from_id(customer_id):
    customer_id = str(customer_id).strip()

    # --------------------------------------------------------
    # 1. Keep the four curated demo profiles unchanged
    # --------------------------------------------------------
    if customer_id in [p["Customer_ID"] for p in DEMO_PROFILES.values()]:
        for profile in DEMO_PROFILES.values():
            if profile["Customer_ID"] == customer_id:
                return dict(profile), "demo profile"

    # --------------------------------------------------------
    # 2. Build a customer-specific profile from history
    # --------------------------------------------------------
    if history_df is not None:
        rows = history_df[
            history_df["Customer_ID"].astype(str).str.strip()
            == customer_id
        ].copy()

        if not rows.empty:

            # Use the customer's own transaction history
            purchase_amounts = pd.to_numeric(
                rows["Purchase_Amount"],
                errors="coerce"
            ).dropna()

            purchase = (
                float(purchase_amounts.mean())
                if not purchase_amounts.empty
                else 500.0
            )

            category = str(
                rows["Purchase_Category"].mode().iloc[0]
            ) if not rows["Purchase_Category"].dropna().empty else CATEGORIES[0]

            provider = str(
                rows["BNPL_Provider"].mode().iloc[0]
            ) if not rows["BNPL_Provider"].dropna().empty else PROVIDERS[0]

            if category not in CATEGORIES:
                category = CATEGORIES[0]

            if provider not in PROVIDERS:
                provider = PROVIDERS[0]

            # ------------------------------------------------
            # Behaviour statistics for this customer
            # ------------------------------------------------
            statuses = (
                rows["Repayment_Status"]
                .astype(str)
                .str.lower()
            )

            on_time_rate = statuses.eq("paid on time").mean()
            late_rate = statuses.eq("late payment").mean()
            default_rate = statuses.eq("defaulted").mean()

            total_transactions = len(rows)

            # ------------------------------------------------
            # Create a deterministic customer-specific profile
            # ------------------------------------------------

            # Age varies by customer ID
            numeric_id = int(
                customer_id.replace("CUST_", "")
            )

            age = 21 + (numeric_id % 30)

            # Income starts from a customer-specific base
            # and is adjusted by repayment behaviour
            income = (
                30000
                + (numeric_id % 21) * 3000
            )

            if default_rate >= 0.30:
                income *= 0.75
            elif default_rate <= 0.05 and on_time_rate >= 0.70:
                income *= 1.25

            income = int(round(income / 1000) * 1000)

            # Credit score is behaviour-sensitive
            credit_score = (
                500
                + int(on_time_rate * 220)
                - int(default_rate * 250)
                - int(late_rate * 80)
            )

            credit_score = int(
                np.clip(credit_score, 300, 850)
            )

            # Deterministic gender assignment
            gender = GENDERS[numeric_id % len(GENDERS)]

            # ------------------------------------------------
            # Customer-specific fraud indicators
            # ------------------------------------------------
            shared_device_accounts = (
                6 if default_rate >= 0.30 else
                3 if late_rate >= 0.30 else
                1
            )

            shared_payment_accounts = (
                4 if default_rate >= 0.30 else
                2 if late_rate >= 0.30 else
                1
            )

            account_age_days = max(
                30,
                420 - total_transactions * 15
            )

            unusual_device = bool(
                late_rate >= 0.30 or default_rate >= 0.20
            )

            identity_mismatch = bool(
                default_rate >= 0.40
            )

            first_party_fraud_signal = bool(
                default_rate >= 0.50
                and purchase > 1000
            )

            profile = {
    "Customer_ID": customer_id,
    "Customer_Age": age,
    "Annual_Income": income,
    "Credit_Score": credit_score,
    "Purchase_Amount": int(round(purchase)),
    "Gender": gender,
    "Purchase_Category": category,
    "BNPL_Provider": provider,

    # Fraud signals
    "Shared_Device_Accounts": shared_device_accounts,
    "Shared_Payment_Accounts": shared_payment_accounts,
    "Account_Age_Days": account_age_days,
    "Identity_Mismatch": identity_mismatch,
    "Unusual_Device": unusual_device,
    "First_Party_Fraud_Signal": first_party_fraud_signal,
}

            return profile, "customer history"

    return None, None


def render_customer_assessment(customer, source_label):
    ml = predict_customer(customer)
    fraud = fraud_check(customer)
    early = early_warning_check(customer["Customer_ID"])
    bs = early["behavioural_summary"]

    st.markdown(
        f"""
        <div class="card">
            <div class="small-muted">CUSTOMER PROFILE · {source_label.upper()}</div>
            <div class="profile-name">{customer["Customer_ID"]}</div>
            <div class="small-muted">
                {customer["Purchase_Category"]} · {customer["BNPL_Provider"]} ·
                Credit score {customer["Credit_Score"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Default Probability", f"{ml['default_probability']:.1%}", "LightGBM", "#49F2FF")
    with c2:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Risk Level</div>'
            f'<div style="margin-top:12px;">{risk_badge(ml["risk_level"])}</div>'
            f'<div class="kpi-sub">Score {ml["risk_score"]}/100</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        kpi("Credit Limit", f"₹{ml['credit_limit']:,.0f}", "Recommended")
    with c4:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Fraud Status</div>'
            f'<div style="margin-top:12px;">{status_badge(fraud["status"])}</div>'
            f'<div class="kpi-sub">Score {fraud["fraud_score"]}/140</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">Early Warning</div>'
            f'<div style="margin-top:12px;">{status_badge(early["alert_status"])}</div>'
            f'<div class="kpi-sub">Behaviour monitor</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown(
        '<div class="section-box"><div class="section-box-title">Risk Score</div></div>',
        unsafe_allow_html=True,)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=ml["risk_score"],
                number={"suffix": "/100", "font": {"size": 32, "color": "#F5F3FF"}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#59617F",
                        "tickfont": {"color": "#9CA6C7"},
                    },
                    "bar": {
                        "color": (
                            "#59F5C9" if ml["risk_score"] < 35
                            else "#FFD66E" if ml["risk_score"] < 65
                            else "#FF55D9"
                        )
                    },
                    "bgcolor": "rgba(20,24,50,.7)",
                    "steps": [
                        {"range": [0, 35], "color": "rgba(89,245,201,.10)"},
                        {"range": [35, 65], "color": "rgba(255,214,110,.10)"},
                        {"range": [65, 100], "color": "rgba(255,85,217,.10)"},
                    ],
                },
            )
        )
        st.plotly_chart(plot_layout(fig, 270), use_container_width=True)

        st.markdown(
            f"""
            <div class="metric-row"><span>Risk Level</span><strong>{ml["risk_level"]}</strong></div>
<div class="metric-row"><span>Default Probability</span><strong>{ml["default_probability"]:.1%}</strong></div>
            <div class="metric-row"><span>Behaviour segment</span><strong>{ml["segment"]}</strong></div>
            <div class="metric-row"><span>Logistic Regression</span><strong>{ml["lr_default_probability"]:.1%}</strong></div>
            <div class="metric-row"><span>Purchase amount</span><strong>₹{customer["Purchase_Amount"]:,.0f}</strong></div>
            <div class="metric-row"><span>Annual income</span><strong>₹{customer["Annual_Income"]:,.0f}</strong></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(
        '<div class="section-box"><div class="section-box-title">Risk Explanation</div></div>',
        unsafe_allow_html=True,)

        st.markdown(
            f"""
            <div class="metric-row"><span>Behaviour Segment</span><strong>{ml["segment"]}</strong></div>
            <div class="metric-row"><span>Credit Score</span><strong>{customer["Credit_Score"]}</strong></div>
            <div class="metric-row"><span>Purchase Amount</span><strong>₹{customer["Purchase_Amount"]:,.0f}</strong></div>
            <div class="metric-row"><span>Customer Age</span><strong>{customer["Customer_Age"]}</strong></div>
            <div class="metric-row"><span>Recommended Credit Limit</span><strong>₹{ml["credit_limit"]:,.0f}</strong></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        card_title("Top LightGBM Risk Factors")

        for i, factor in enumerate(ml["top_risk_factors"], 1):
            st.markdown(
                f"""
                <div class="metric-row">
                    <span>{i}. {factor}</span>
                    <span style="color:#49F2FF;">{"▰" * max(1, 8-i)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    if bs:
        left, right = st.columns([1.1, 1])

        with left:
            st.markdown(
        '<div class="section-box"><div class="section-box-title">Behaviour Monitoring</div></div>',
        unsafe_allow_html=True,)

            behaviour_metrics = [
                ("Transactions", f'{bs["total_transactions"]}'),
                ("On-time rate", f'{bs["on_time_rate"]:.0%}'),
                ("Late-payment rate", f'{bs["late_payment_rate"]:.0%}'),
                ("Default rate", f'{bs["default_rate"]:.0%}'),
                ("Average repayment", f'{bs["avg_days_to_repay"]:.1f} days'),
                ("Spend trend", f'{bs["recent_spend_trend"]:+.1f}'),
                ("Bad payments, last 3", f'{bs["recent_bad_payments_last3"]}'),
            ]

            for label, value in behaviour_metrics:
                st.markdown(
                    f'<div class="metric-row"><span>{label}</span><strong>{value}</strong></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown(
        '<div class="section-box"><div class="section-box-title">Early Warning Signals</div></div>',
        unsafe_allow_html=True,)

            if early["alert_triggers"]:
                for trigger in early["alert_triggers"]:
                    st.markdown(
                        f'<div class="alert-card">⚠️ {trigger}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="alert-card">🟢 No active behavioural warning signals.</div>',
                    unsafe_allow_html=True,
                )

            st.info(early["alert_message"])

            st.markdown(
                f'<div style="margin-top:10px;">{status_badge(early["alert_status"])}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    if fraud["alerts"]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_title("Active Fraud Signals")
        for alert in fraud["alerts"]:
            st.markdown(
                f'<div class="alert-card">🚨 {alert}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ASSESSMENT
# ============================================================

def page_assessment():
    st.markdown('<div class="page-title">Customer Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Search a customer ID or launch a Quick Demo to run the complete risk workflow.</div>',
        unsafe_allow_html=True,
    )

    customer_id = st.session_state.customer_id

    profile, source = customer_from_id(customer_id)

    if profile is None:
        st.markdown(
            """
            <div class="card" style="text-align:center;padding:38px;">
                <div style="font-size:36px;">◎</div>
                <div class="profile-name">Customer not found</div>
                <div class="small-muted">
                    Try CUST_1001 to CUST_1050 or use Quick Demo.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Allow the user to override the model inputs.
    with st.expander("Customer inputs", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            profile["Customer_Age"] = st.number_input(
                "Age", 18, 80, int(profile["Customer_Age"]), 
                key=f"assess_age_{customer_id}"
            )
        with c2:
            profile["Annual_Income"] = st.number_input(
                "Annual income",
                10000,
                300000,
                int(profile["Annual_Income"]),
                step=1000,
                key=f"assess_income_{customer_id}",
            )
        with c3:
            profile["Credit_Score"] = st.number_input(
                "Credit score",
                300,
                850,
                int(profile["Credit_Score"]),
                key=f"assess_credit_{customer_id}",
            )
        with c4:
            profile["Purchase_Amount"] = st.number_input(
                "Purchase amount",
                20,
                10000,
                int(profile["Purchase_Amount"]),
                step=50,
                key=f"assess_purchase_{customer_id}",
            )

    render_customer_assessment(profile, source)


# ============================================================
# RISK MONITOR
# ============================================================

def page_risk_monitor():
    st.markdown('<div class="page-title">Risk Monitor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Surface portfolio transactions and synthetic customer histories requiring attention.</div>',
        unsafe_allow_html=True,
    )

    portfolio = build_portfolio_predictions(main_df)

    f1, f2, f3 = st.columns(3)
    with f1:
        risk_filter = st.selectbox(
            "Risk level",
            ["All", "HIGH", "MEDIUM", "LOW"],
            key="monitor_risk",
        )
    with f2:
        provider_filter = st.selectbox(
            "Provider",
            ["All"] + sorted(portfolio["BNPL_Provider"].unique().tolist()),
            key="monitor_provider",
        )
    with f3:
        min_prob = st.slider(
            "Minimum default probability",
            0.0,
            1.0,
            0.35,
            0.05,
            key="monitor_prob",
        )

    filtered = portfolio.copy()

    if risk_filter != "All":
        filtered = filtered[filtered["Risk_Level"] == risk_filter]
    if provider_filter != "All":
        filtered = filtered[filtered["BNPL_Provider"] == provider_filter]

    filtered = filtered[filtered["Default_Probability"] >= min_prob]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Attention Items", f"{len(filtered):,}", "Current filters", "#FF55D9")
    with c2:
        kpi(
            "High Risk",
            f"{(filtered['Risk_Level'] == 'HIGH').sum():,}",
            "Portfolio transactions",
            "#FF55D9",
        )
    with c3:
        kpi(
            "Demo Profiles",
            "4",
            "Synthetic demo profiles",
            "#FFD66E",
        )
    with c4:
        kpi(
            "Early Warning",
            "Live",
            "Customer history engine",
            "#49F2FF",
        )

    st.markdown('<div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-box"><div class="section-box-title">Transaction Risk Monitor</div></div>',
        unsafe_allow_html=True,)

    display = filtered.nlargest(25, "Default_Probability").copy()
    display["Default Probability"] = display["Default_Probability"].map(lambda x: f"{x:.1%}")
    display["Purchase Amount"] = display["Purchase_Amount"].map(lambda x: f"₹{x:,.0f}")
    display["Credit Score"] = display["Credit_Score"].round().astype(int)

    display = display[
        [
            "Transaction_ID",
            "Default Probability",
            "Risk_Level",
            "Credit Score",
            "Purchase Amount",
            "Purchase_Category",
            "BNPL_Provider",
        ]
    ].rename(columns={
        "Transaction_ID": "Transaction",
        "Risk_Level": "Risk",
        "Purchase_Category": "Category",
        "BNPL_Provider": "Provider",
    })

    st.dataframe(display, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
            '<div class="section-box"><div class="section-box-title">Synthetic Customer Early-Warning Monitor</div></div>',
            unsafe_allow_html=True,)

    rows = []
    for cid in sorted(history_df["Customer_ID"].astype(str).unique()):
        ew = early_warning_check(cid)
        bs = ew["behavioural_summary"]
        rows.append({
            "Customer ID": cid,
            "Early Warning": ew["alert_status"],
            "On-Time Rate": bs.get("on_time_rate", np.nan),
            "Default Rate": bs.get("default_rate", np.nan),
            "Spend Trend": bs.get("recent_spend_trend", np.nan),
            "Bad Payments Last 3": bs.get("recent_bad_payments_last3", np.nan),
        })

    monitor_df = pd.DataFrame(rows)
    monitor_df["On-Time Rate"] = monitor_df["On-Time Rate"].map(
        lambda x: f"{x:.0%}" if pd.notna(x) else "N/A"
    )
    monitor_df["Default Rate"] = monitor_df["Default Rate"].map(
        lambda x: f"{x:.0%}" if pd.notna(x) else "N/A"
    )

    st.dataframe(
        monitor_df[monitor_df["Early Warning"] != "CLEAR"],
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ANALYTICS
# ============================================================

def page_analytics():
    st.markdown('<div class="page-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Interactive portfolio views across default probability, providers, categories and customer behaviour.</div>',
        unsafe_allow_html=True,
    )

    portfolio = build_portfolio_predictions(main_df)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
                    '<div class="section-box"><div class="section-box-title">Risk by Credit Score</div></div>',
                    unsafe_allow_html=True,)
        score_bins = [299, 399, 499, 599, 699, 799, 850]
        labels = ["300–399", "400–499", "500–599", "600–699", "700–799", "800+"]
        temp = portfolio.copy()
        temp["Score Band"] = pd.cut(
            temp["Credit_Score"],
            bins=score_bins,
            labels=labels,
            include_lowest=True,
        )
        score_chart = (
            temp.groupby("Score Band", observed=False)["Default_Probability"]
            .mean()
            .reset_index()
        )
        score_chart["Default_Probability"] *= 100

        fig = px.line(
            score_chart,
            x="Score Band",
            y="Default_Probability",
            markers=True,
        )
        fig.update_traces(line_color="#49F2FF")
        fig.update_yaxes(title="Predicted default probability (%)")
        st.plotly_chart(plot_layout(fig, 320), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-box"><div class="section-box-title">Default Risk by Category</div></div>',unsafe_allow_html=True,)
        cat = (
            portfolio.groupby("Purchase_Category")["Default_Probability"]
            .mean()
            .reset_index()
        )
        cat["Default_Probability"] *= 100
        cat = cat.sort_values("Default_Probability", ascending=True)

        fig = px.bar(
            cat,
            x="Default_Probability",
            y="Purchase_Category",
            orientation="h",
        )
        fig.update_traces(marker_color="#A970FF")
        fig.update_xaxes(title="Average predicted default probability (%)")
        fig.update_yaxes(title="")
        st.plotly_chart(plot_layout(fig, 320), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-box"><div class="section-box-title">Purchase Amount vs Default Probability</div></div>',unsafe_allow_html=True,)
        sample = portfolio.sample(min(2500, len(portfolio)), random_state=42)
        fig = px.scatter(
            sample,
            x="Purchase_Amount",
            y="Default_Probability",
            color="Risk_Level",
            hover_data=["Credit_Score", "BNPL_Provider"],
            color_discrete_map={
                "LOW": "#59F5C9",
                "MEDIUM": "#FFD66E",
                "HIGH": "#FF55D9",
            },
            opacity=.65,
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(plot_layout(fig, 330), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-box"><div class="section-box-title">Provider Exposure</div></div>',unsafe_allow_html=True,)
        provider = (
            portfolio.groupby("BNPL_Provider")
            .agg(
                Transactions=("Transaction_ID", "count"),
                Avg_Default_Probability=("Default_Probability", "mean"),
            )
            .reset_index()
        )
        provider["Avg_Default_Probability"] *= 100

        fig = px.bar(
            provider,
            x="BNPL_Provider",
            y="Transactions",
            color="Avg_Default_Probability",
            color_continuous_scale=["#59F5C9", "#FFD66E", "#FF55D9"],
        )
        fig.update_yaxes(title="Transactions")
        st.plotly_chart(plot_layout(fig, 330), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# TRUST CENTER
# ============================================================

def page_trust_center():
    st.markdown('<div class="page-title">Trust Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Transparency, explainability, model governance and responsible-use information.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Model Transparency</div>
                <div class="big-number">3</div>
                <div class="small-muted">Core decision layers</div>
                <div class="metric-row"><span>Default prediction</span><strong>LightGBM</strong></div>
                <div class="metric-row"><span>Baseline</span><strong>Logistic Regression</strong></div>
                <div class="metric-row"><span>Behaviour</span><strong>K-Means</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Data Transparency</div>
                <div class="big-number">50K</div>
                <div class="small-muted">BNPL training transactions</div>
                <div class="metric-row"><span>Customer history</span><strong>500 records</strong></div>
                <div class="metric-row"><span>Customer IDs</span><strong>50 synthetic</strong></div>
                <div class="metric-row"><span>Live PII</span><strong>None</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Responsible AI</div>
                <div class="big-number">✓</div>
                <div class="small-muted">Decision-support prototype</div>
                <div class="metric-row"><span>Explainability</span><strong>Top risk factors</strong></div>
                <div class="metric-row"><span>Human review</span><strong>Recommended</strong></div>
                <div class="metric-row"><span>Data type</span><strong>Synthetic</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-box"><div class="section-box-title">Risk Methodology</div></div>',unsafe_allow_html=True,)
        st.markdown(
            """
            <div class="metric-row"><span>Default probability</span><strong>LightGBM</strong></div>
            <div class="metric-row"><span>Baseline comparison</span><strong>Logistic Regression</strong></div>
            <div class="metric-row"><span>Risk bands</span><strong>&lt;15% · 15–35% · ≥35%</strong></div>
            <div class="metric-row"><span>Behaviour segmentation</span><strong>K-Means</strong></div>
            <div class="metric-row"><span>Fraud layer</span><strong>Rule engine</strong></div>
            <div class="metric-row"><span>Early warning</span><strong>Behavioural signals</strong></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-box"><div class="section-box-title">Explainability and Monitoring</div></div>',unsafe_allow_html=True,)
        st.markdown(
            """
            <div class="alert-card">Top LightGBM factors are shown for every assessment.</div>
            <div class="alert-card">Portfolio analytics expose changes in predicted risk.</div>
            <div class="alert-card">Fraud and early-warning signals are shown separately from default prediction.</div>
            <div class="alert-card">The prototype uses synthetic data and should not be treated as a production credit decision system.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-box"><div class="section-box-title">Security and Privacy</div></div>',unsafe_allow_html=True,)
    st.markdown(
        """
        This prototype does not require live customer PII. The supplied customer history is
        explicitly synthetic. Model outputs should be reviewed with appropriate business,
        compliance and human-decision controls before any real-world deployment.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ROUTER
# ============================================================

if st.session_state.page == "Dashboard":
    page_dashboard()
elif st.session_state.page == "Assessment":
    page_assessment()
elif st.session_state.page == "Risk Monitor":
    page_risk_monitor()
elif st.session_state.page == "Analytics":
    page_analytics()
elif st.session_state.page == "Trust Center":
    page_trust_center()
