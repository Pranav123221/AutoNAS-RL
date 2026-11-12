
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Hardware Efficiency | AutoNAS-RL",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# THEME
# =========================================================

st.markdown("""
<style>
.stApp { background:#0B1020; }
.block-container { max-width:1500px; padding:2rem 3rem; }

.hero {
    background:linear-gradient(110deg,#172554,#211C46);
    border:1px solid #34446B;
    border-radius:16px;
    padding:28px;
    margin-bottom:25px;
}
.hero h1 { color:#F8FAFC; font-size:32px; }
.hero p { color:#BFDBFE; line-height:1.8; }

div[data-testid="stMetric"] {
    background:#141D30;
    border:1px solid #29364D;
    border-radius:12px;
    padding:16px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="hero">
    <div style="color:#93C5FD;font-size:11px;font-weight:700;
                letter-spacing:2px;">
        HARDWARE-AWARE MODEL ANALYSIS
    </div>
    <h1>⚡ Hardware Efficiency</h1>
    <p>
        Analyze the computational footprint and inference
        performance of the fixed and reinforcement-learning-
        searched CNN architectures.
    </p>
</div>
""")

# =========================================================
# EXPERIMENTAL DATA
# =========================================================

data = pd.DataFrame([
    {
        "Model": "Fixed CNN",
        "Parameters": 422026,
        "Training Time (s)": 58.69,
        "Mean Latency (ms)": 21.510,
        "P95 Latency (ms)": 38.044,
        "Validation Accuracy": 0.9337,
        "Validation F1": 0.9333,
        "Test Accuracy": 0.9257,
        "Test F1": 0.9251
    },
    {
        "Model": "RL-Searched CNN",
        "Parameters": 106058,
        "Training Time (s)": 45.17,
        "Mean Latency (ms)": 14.024,
        "P95 Latency (ms)": 21.305,
        "Validation Accuracy": 0.9250,
        "Validation F1": 0.9245,
        "Test Accuracy": 0.9142,
        "Test F1": 0.9133
    }
])

fixed = data.iloc[0]
rl = data.iloc[1]

parameter_reduction = (
    (fixed["Parameters"] - rl["Parameters"])
    / fixed["Parameters"] * 100
)

mean_latency_reduction = (
    (fixed["Mean Latency (ms)"] - rl["Mean Latency (ms)"])
    / fixed["Mean Latency (ms)"] * 100
)

p95_reduction = (
    (fixed["P95 Latency (ms)"] - rl["P95 Latency (ms)"])
    / fixed["P95 Latency (ms)"] * 100
)

training_reduction = (
    (fixed["Training Time (s)"] - rl["Training Time (s)"])
    / fixed["Training Time (s)"] * 100
)

# =========================================================
# KPI SECTION
# =========================================================

st.markdown("## Efficiency Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Parameter Reduction",
    f"{parameter_reduction:.2f}%"
)

c2.metric(
    "Mean Latency Reduction",
    f"{mean_latency_reduction:.2f}%"
)

c3.metric(
    "P95 Latency Reduction",
    f"{p95_reduction:.2f}%"
)

c4.metric(
    "Training Time Reduction",
    f"{training_reduction:.2f}%"
)

st.divider()

# =========================================================
# PARAMETER COMPARISON
# =========================================================

st.markdown("## 1. Model Complexity")

fig_params = px.bar(
    data,
    x="Model",
    y="Parameters",
    color="Model",
    text="Parameters",
    title="Trainable and Non-trainable Parameter Footprint",
    template="plotly_dark"
)

fig_params.update_layout(
    showlegend=False,
    paper_bgcolor="#0B1020",
    plot_bgcolor="#141D30"
)

st.plotly_chart(
    fig_params,
    use_container_width=True
)

# =========================================================
# INFERENCE LATENCY
# =========================================================

st.divider()

st.markdown("## 2. Inference Latency Analysis")

latency_type = st.radio(
    "Select latency measurement",
    ["Mean Latency", "P95 Latency"],
    horizontal=True
)

latency_column = (
    "Mean Latency (ms)"
    if latency_type == "Mean Latency"
    else "P95 Latency (ms)"
)

fig_latency = px.bar(
    data,
    x="Model",
    y=latency_column,
    color="Model",
    text=latency_column,
    title=f"{latency_type} Comparison",
    template="plotly_dark"
)

fig_latency.update_layout(
    showlegend=False,
    paper_bgcolor="#0B1020",
    plot_bgcolor="#141D30"
)

st.plotly_chart(
    fig_latency,
    use_container_width=True
)

st.caption(
    "Latency was measured using warm-up inference runs followed "
    "by repeated single-image predictions on the same experimental setup."
)

# =========================================================
# ACCURACY VS EFFICIENCY
# =========================================================

st.divider()

st.markdown("## 3. Accuracy vs Computational Cost")

fig_tradeoff = px.scatter(
    data,
    x="Mean Latency (ms)",
    y="Test Accuracy",
    size="Parameters",
    color="Model",
    text="Model",
    hover_data=[
        "Parameters",
        "P95 Latency (ms)",
        "Test F1"
    ],
    title="Test Accuracy vs Mean Inference Latency",
    template="plotly_dark"
)

fig_tradeoff.update_layout(
    yaxis=dict(tickformat=".1%"),
    paper_bgcolor="#0B1020",
    plot_bgcolor="#141D30"
)

st.plotly_chart(
    fig_tradeoff,
    use_container_width=True
)

st.info(
    "The RL-Searched CNN occupies a lower-latency, lower-parameter "
    "region, while the Fixed CNN achieves higher test accuracy. "
    "This represents an accuracy-efficiency trade-off."
)

# =========================================================
# TRAINING COST
# =========================================================

st.divider()

st.markdown("## 4. Training Cost")

fig_training = px.bar(
    data,
    x="Model",
    y="Training Time (s)",
    color="Model",
    text="Training Time (s)",
    title="CNN Training Time",
    template="plotly_dark"
)

fig_training.update_layout(
    showlegend=False,
    paper_bgcolor="#0B1020",
    plot_bgcolor="#141D30"
)

st.plotly_chart(
    fig_training,
    use_container_width=True
)

# =========================================================
# FULL COMPARISON TABLE
# =========================================================

st.divider()

st.markdown("## 5. Detailed Hardware Comparison")

st.dataframe(
    data,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# ENGINEERING INTERPRETATION
# =========================================================

st.divider()

st.markdown("## 6. Engineering Interpretation")

st.markdown(f"""
**Observed results:**

- Parameter count reduced by **{parameter_reduction:.2f}%**.
- Mean single-image latency reduced by **{mean_latency_reduction:.2f}%**.
- P95 single-image latency reduced by **{p95_reduction:.2f}%**.
- CNN training time reduced by **{training_reduction:.2f}%**.
- Final test accuracy changed from **92.57% to 91.42%**.

The RL-selected architecture is smaller and faster in this
measured environment, but sacrifices some classification accuracy.
""")

st.warning(
    "These measurements are specific to the hardware, runtime, "
    "batching and inference protocol used in this experiment. "
    "They should not be interpreted as universal deployment benchmarks."
)

# =========================================================
# EXPORT
# =========================================================

st.divider()

st.markdown("## Export Hardware Report")

csv_bytes = data.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Hardware Metrics CSV",
    data=csv_bytes,
    file_name="autonas_rl_hardware_metrics.csv",
    mime="text/csv"
)

st.caption(
    "AutoNAS-RL | Hardware Efficiency | Experimental Analysis"
)
