
import json
from pathlib import Path
from html import escape
from textwrap import dedent

import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AutoNAS-RL | AI Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

REPORT_PATH = ARTIFACT_DIR / "final_experiment_report.json"


# =========================================================
# 2. LOAD EXPERIMENT DATA
# =========================================================

@st.cache_data
def load_report():

    if not REPORT_PATH.exists():
        return {}

    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


report = load_report()

validation = report.get("validation_results", {})

fixed = validation.get("fixed_cnn", {})
rl = validation.get("rl_searched_cnn", {})

best_config = report.get("best_config", {})

best_reward = report.get("best_reward", 0.0)


# =========================================================
# 3. GLOBAL CSS
# =========================================================

st.markdown(dedent("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0B1020;
    color: #E5E7EB;
}

header[data-testid="stHeader"] {
    background: rgba(11, 16, 32, 0.95);
}

.block-container {
    width: 100%;
    max-width: 1600px;
    padding: 2rem 3rem 3rem 3rem;
}

[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263247;
}

[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1rem;
}

/* HERO */

.hero {
    background: linear-gradient(
        115deg,
        #172554 0%,
        #172033 55%,
        #211C46 100%
    );
    border: 1px solid #34446B;
    border-radius: 18px;
    padding: 38px;
    margin-bottom: 30px;
}

.hero-label {
    color: #93C5FD;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    color: #F8FAFC;
    font-size: 40px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-top: 12px;
}

.hero-description {
    color: #B6C4DD;
    font-size: 14px;
    line-height: 1.9;
    max-width: 850px;
    margin-top: 12px;
}

.status-pill {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 30px;
    background: #12352D;
    color: #6EE7B7;
    font-size: 10px;
    font-weight: 700;
    border: 1px solid #21634F;
    letter-spacing: 1px;
}

/* SECTION HEADINGS */

.section-heading {
    color: #F1F5F9;
    font-size: 22px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 5px;
}

.section-description {
    color: #94A3B8;
    font-size: 13px;
    margin-bottom: 18px;
}

/* METRIC CARDS */

.metric-card {
    background: #141D30;
    border: 1px solid #29364D;
    border-radius: 15px;
    padding: 22px;
    min-height: 135px;
    transition: all 0.2s ease;
}

.metric-card:hover {
    border-color: #647CB5;
    transform: translateY(-2px);
}

.metric-label {
    color: #94A3B8;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #F8FAFC;
    font-size: 27px;
    font-weight: 800;
    margin-top: 13px;
}

.metric-caption {
    color: #64748B;
    font-size: 11px;
    margin-top: 7px;
}

/* INFORMATION CARDS */

.info-card {
    background: #141D30;
    border: 1px solid #29364D;
    border-radius: 15px;
    padding: 23px;
    min-height: 110px;
    margin-bottom: 12px;
}

.info-card h3 {
    color: #E2E8F0;
    font-size: 17px;
    font-weight: 700;
}

.info-card p {
    color: #94A3B8;
    font-size: 13px;
    line-height: 1.8;
}

/* STREAMLIT METRICS */

div[data-testid="stMetric"] {
    background: #141D30;
    border: 1px solid #29364D;
    border-radius: 13px;
    padding: 17px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC;
}

/* JSON */

div[data-testid="stJson"] {
    border: 1px solid #29364D;
    border-radius: 12px;
    overflow: hidden;
}

/* BUTTONS */

.stButton > button {
    border-radius: 9px;
}

/* FOOTER */

.footer {
    color: #64748B;
    font-size: 11px;
    text-align: center;
    padding: 20px 0 5px 0;
}

</style>
"""), unsafe_allow_html=True)


# =========================================================
# 4. HTML COMPONENT HELPERS
# =========================================================

def render_metric_card(label, value, caption):

    st.html(dedent(f"""
    <div class="metric-card">
        <div class="metric-label">{escape(str(label))}</div>
        <div class="metric-value">{escape(str(value))}</div>
        <div class="metric-caption">{escape(str(caption))}</div>
    </div>
    """))


def render_info_card(title, description):

    st.html(dedent(f"""
    <div class="info-card">
        <h3>{escape(str(title))}</h3>
        <p>{escape(str(description))}</p>
    </div>
    """))


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        "<h2 style='color:#F8FAFC;'>🧠 AutoNAS-RL</h2>",
        unsafe_allow_html=True
    )

    st.caption("HARDWARE-AWARE NAS PLATFORM")

    st.divider()

    st.markdown("##### RESEARCH PLATFORM")

    st.page_link(
        "app.py",
        label="Overview",
        icon="🏠"
    )

    st.page_link(
        "pages/1_Architecture_Search.py",
        label="Architecture Search",
        icon="🧠"
    )

    st.page_link(
        "pages/2_Model_Benchmarking.py",
        label="Model Benchmarking",
        icon="📊"
    )

    st.page_link(
        "pages/3_Hardware_Efficiency.py",
        label="Hardware Efficiency",
        icon="⚡"
    )

    st.page_link(
        "pages/4_Prediction_Lab.py",
        label="Prediction Lab",
        icon="🔬"
    )

    st.page_link(
        "pages/5_Experiment_Reports.py",
        label="Experiment Reports",
        icon="📁"
    )

    st.divider()

    st.markdown("##### EXPERIMENT DETAILS")

    st.caption("Dataset")
    st.markdown("**Fashion-MNIST**")

    st.caption("Search Algorithm")
    st.markdown("**Tabular Q-Learning**")

    st.caption("Framework")
    st.markdown("**TensorFlow 2.21**")

    st.caption("Application")
    st.markdown("**Streamlit 1.67**")

    st.divider()

    st.caption("RESEARCH PROTOTYPE · V1.0")


# =========================================================
# 6. HERO SECTION
# =========================================================

st.html(dedent("""
<div class="hero">

    <div class="hero-label">
        AI RESEARCH & EXPERIMENTATION PLATFORM
    </div>

    <div class="hero-title">
        AutoNAS-RL
    </div>

    <div class="hero-description">
        A hardware-aware Convolutional Neural Network
        architecture search framework powered by
        Reinforcement Learning.

        <br><br>

        Explore architecture selection, model performance,
        computational efficiency and experimental results
        through an interactive research environment.

        <br><br>

        <span class="status-pill">
            EXPERIMENTS COMPLETED
        </span>
    </div>

</div>
"""))


# =========================================================
# 7. RESEARCH OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-heading">Research Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Final validation metrics from the completed experiment.'
    '</div>',
    unsafe_allow_html=True
)


if fixed and rl:

    col1, col2, col3, col4 = st.columns(4)

    parameter_reduction = (
        1 - rl["parameters"] / fixed["parameters"]
    ) * 100

    render_metric_card(
        "RL Model Accuracy",
        f"{rl['accuracy'] * 100:.2f}%",
        "Validation accuracy"
    )

    render_metric_card(
        "Best Search Reward",
        f"{best_reward:.4f}",
        "Highest recorded candidate reward"
    )

    render_metric_card(
        "Parameter Reduction",
        f"{parameter_reduction:.2f}%",
        "Compared with fixed CNN"
    )

    render_metric_card(
        "Mean Inference Latency",
        f"{rl['mean_latency_ms']:.2f} ms",
        "Single-image inference"
    )

else:

    st.warning(
        "Experiment metrics could not be loaded. "
        "Please verify final_experiment_report.json."
    )


# =========================================================
# 8. MODEL COMPARISON
# =========================================================

st.markdown(
    '<div class="section-heading">Model Comparison</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Comparison between the conventional CNN baseline '
    'and the Q-learning-selected architecture.'
    '</div>',
    unsafe_allow_html=True
)


if fixed and rl:

    left, right = st.columns(2, gap="large")

    with left:

        render_info_card(
            "Fixed CNN Baseline",
            "Manually configured convolutional neural network "
            "used as the baseline architecture."
        )

        st.metric(
            "Validation Accuracy",
            f"{fixed['accuracy'] * 100:.2f}%"
        )

        st.metric(
            "Macro F1",
            f"{fixed['f1'] * 100:.2f}%"
        )

        st.metric(
            "Parameters",
            f"{fixed['parameters']:,}"
        )

        st.metric(
            "Mean Latency",
            f"{fixed['mean_latency_ms']:.3f} ms"
        )

    with right:

        render_info_card(
            "RL-Searched CNN",
            "Compact CNN architecture selected through "
            "a Q-learning-controlled candidate search."
        )

        st.metric(
            "Validation Accuracy",
            f"{rl['accuracy'] * 100:.2f}%"
        )

        st.metric(
            "Macro F1",
            f"{rl['f1'] * 100:.2f}%"
        )

        st.metric(
            "Parameters",
            f"{rl['parameters']:,}"
        )

        st.metric(
            "Mean Latency",
            f"{rl['mean_latency_ms']:.3f} ms"
        )


# =========================================================
# 9. SELECTED ARCHITECTURE
# =========================================================

st.markdown(
    '<div class="section-heading">Selected Architecture</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Best candidate under the defined weighted reward function.'
    '</div>',
    unsafe_allow_html=True
)


if best_config:

    config_col, result_col = st.columns([1.25, 0.75], gap="large")

    with config_col:

        
        st.markdown("### Selected CNN Architecture")

        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric("Conv 1 Filters", best_config["filters1"])
            st.metric("Kernel Size", f'{best_config["kernel_size"]} × {best_config["kernel_size"]}')
            st.metric("Pooling 1", best_config["pooling1"].title())

        with p2:
            st.metric("Conv 2 Filters", best_config["filters2"])
            st.metric("Dense Units", best_config["dense_units"])
            st.metric("Pooling 2", best_config["pooling2"].title())

        with p3:
            st.metric("Dropout 1", best_config["dropout1"])
            st.metric("Dropout 2", best_config["dropout2"])
            st.metric("Learning Rate", best_config["learning_rate"])


    with result_col:

        render_info_card(
            "Search Result",
            "The selected architecture uses a compact "
            "two-block convolutional design with max and "
            "average pooling."
        )

        st.metric(
            "Selected Candidate",
            f"#{report.get('best_candidate_number', 2)}"
        )

        st.metric(
            "Weighted Reward",
            f"{best_reward:.4f}"
        )

        st.metric(
            "Parameter Count",
            f"{rl.get('parameters', 0):,}"
        )


# =========================================================
# 10. FINAL TEST RESULTS
# =========================================================

test_results = report.get("test_results", {})

if test_results:

    st.markdown(
        '<div class="section-heading">Final Test Evaluation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Evaluation on the held-out Fashion-MNIST test set.'
        '</div>',
        unsafe_allow_html=True
    )

    test_fixed = test_results.get("fixed_cnn", {})
    test_rl = test_results.get("rl_searched_cnn", {})

    if test_fixed and test_rl:

        t1, t2 = st.columns(2, gap="large")

        with t1:

            st.markdown("#### Fixed CNN")

            st.metric(
                "Test Accuracy",
                f"{test_fixed['accuracy'] * 100:.2f}%"
            )

            st.metric(
                "Macro F1",
                f"{test_fixed['f1'] * 100:.2f}%"
            )

        with t2:

            st.markdown("#### RL-Searched CNN")

            st.metric(
                "Test Accuracy",
                f"{test_rl['accuracy'] * 100:.2f}%"
            )

            st.metric(
                "Macro F1",
                f"{test_rl['f1'] * 100:.2f}%"
            )


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()

st.html(dedent("""
<div class="footer">
    AutoNAS-RL · Hardware-Aware Neural Architecture Search
    <br><br>
    Fashion-MNIST · TensorFlow · Q-Learning · Streamlit
    <br>
    Research Prototype v1.0
</div>
"""))
