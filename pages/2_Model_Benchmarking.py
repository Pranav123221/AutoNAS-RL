
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Model Benchmarking | AutoNAS-RL",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# THEME
# ---------------------------------------------------------

st.markdown("""
<style>
.stApp { background: #0B1020; }
.block-container { padding: 2rem 3rem; max-width: 1500px; }
.hero {
    background: linear-gradient(110deg,#172554,#211C46);
    border: 1px solid #34446B;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 25px;
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

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.html("""
<div class="hero">
    <div style="color:#93C5FD;font-size:11px;font-weight:700;
                letter-spacing:2px;">EXPERIMENTAL PERFORMANCE ANALYSIS</div>
    <h1>📊 Model Benchmarking</h1>
    <p>
        Evaluate traditional machine learning baselines and
        convolutional neural networks using recorded experimental
        results from the AutoNAS-RL research workflow.
    </p>
</div>
""")

# ---------------------------------------------------------
# RECORDED VALIDATION RESULTS
# ---------------------------------------------------------

validation_data = pd.DataFrame([
    {
        "Model": "Random Forest",
        "Accuracy": 0.8867,
        "Macro Precision": 0.8862,
        "Macro Recall": 0.8867,
        "Macro F1": 0.8854,
        "Training Time (s)": 142.19,
        "Evaluation Split": "Validation"
    },
    {
        "Model": "XGBoost",
        "Accuracy": 0.9138,
        "Macro Precision": 0.9137,
        "Macro Recall": 0.9138,
        "Macro F1": 0.9136,
        "Training Time (s)": 1003.64,
        "Evaluation Split": "Validation"
    },
    {
        "Model": "Soft Voting",
        "Accuracy": 0.9082,
        "Macro Precision": 0.9079,
        "Macro Recall": 0.9082,
        "Macro F1": 0.9078,
        "Training Time (s)": None,
        "Evaluation Split": "Validation"
    },
    {
        "Model": "Fixed CNN",
        "Accuracy": 0.9337,
        "Macro Precision": 0.9334,
        "Macro Recall": 0.9337,
        "Macro F1": 0.9333,
        "Training Time (s)": 58.69,
        "Evaluation Split": "Validation"
    },
    {
        "Model": "RL-Searched CNN",
        "Accuracy": 0.9250,
        "Macro Precision": 0.9253,
        "Macro Recall": 0.9250,
        "Macro F1": 0.9245,
        "Training Time (s)": 45.17,
        "Evaluation Split": "Validation"
    }
])

# ---------------------------------------------------------
# FINAL TEST RESULTS
# ---------------------------------------------------------

test_data = pd.DataFrame([
    {
        "Model": "Fixed CNN",
        "Accuracy": 0.9257,
        "Macro Precision": 0.9253,
        "Macro Recall": 0.9257,
        "Macro F1": 0.9251
    },
    {
        "Model": "RL-Searched CNN",
        "Accuracy": 0.9142,
        "Macro Precision": 0.9141,
        "Macro Recall": 0.9142,
        "Macro F1": 0.9133
    }
])

# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------

st.markdown("## Research Benchmark Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Models Evaluated", "5")
c2.metric("Best Validation Accuracy", "93.37%", "Fixed CNN")
c3.metric("Best Test Accuracy", "92.57%", "Fixed CNN")
c4.metric("RL-CNN Parameters", "106,058")

st.divider()

# ---------------------------------------------------------
# VALIDATION COMPARISON
# ---------------------------------------------------------

st.markdown("## 1. Validation Performance")
st.caption(
    "All five models evaluated on the same 6,000-image validation split."
)

metric_choice = st.selectbox(
    "Select evaluation metric",
    ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
)

fig = px.bar(
    validation_data,
    x="Model",
    y=metric_choice,
    color="Model",
    text=validation_data[metric_choice].apply(
        lambda x: f"{x:.2%}"
    ),
    title=f"{metric_choice} — Validation Set",
    template="plotly_dark"
)

fig.update_layout(
    yaxis=dict(range=[0.80, 1.0], tickformat=".0%"),
    showlegend=False,
    paper_bgcolor="#0B1020",
    plot_bgcolor="#141D30"
)

st.plotly_chart(fig, use_container_width=True)

display_validation = validation_data.copy()

for col in [
    "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"
]:
    display_validation[col] = (
        display_validation[col] * 100
    ).round(2).astype(str) + "%"

display_validation["Training Time (s)"] = (
    display_validation["Training Time (s)"].round(2)
)

st.dataframe(
    display_validation,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# TEST SET COMPARISON
# ---------------------------------------------------------

st.divider()
st.markdown("## 2. Final Test Set Comparison")

st.info(
    "The untouched 10,000-image test set was evaluated after "
    "model selection. These are the final reported test metrics."
)

test_fig = px.bar(
    test_data,
    x="Model",
    y="Accuracy",
    color="Model",
    text=test_data["Accuracy"].apply(lambda x: f"{x:.2%}"),
    title="Final Test Accuracy",
    template="plotly_dark"
)

test_fig.update_layout(
    yaxis=dict(range=[0.85, 0.95], tickformat=".0%"),
    showlegend=False,
    paper_bgcolor="#0B1020",
    plot_bgcolor="#141D30"
)

st.plotly_chart(test_fig, use_container_width=True)

test_display = test_data.copy()

for col in [
    "Accuracy", "Macro Precision", "Macro Recall", "Macro F1"
]:
    test_display[col] = (
        test_display[col] * 100
    ).round(2).astype(str) + "%"

st.dataframe(
    test_display,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# CNN EFFICIENCY COMPARISON
# ---------------------------------------------------------

st.divider()
st.markdown("## 3. CNN Computational Efficiency")

cnn_efficiency = pd.DataFrame([
    {
        "Model": "Fixed CNN",
        "Parameters": 422026,
        "Training Time (s)": 58.69,
        "Mean Latency (ms)": 21.510,
        "P95 Latency (ms)": 38.044
    },
    {
        "Model": "RL-Searched CNN",
        "Parameters": 106058,
        "Training Time (s)": 45.17,
        "Mean Latency (ms)": 14.024,
        "P95 Latency (ms)": 21.305
    }
])

e1, e2, e3 = st.columns(3)

e1.metric(
    "Parameter Reduction",
    "74.87%"
)

e2.metric(
    "Mean Latency Reduction",
    "34.80%"
)

e3.metric(
    "P95 Latency Reduction",
    "44.00%"
)

efficiency_metric = st.selectbox(
    "Compare computational metric",
    [
        "Parameters",
        "Training Time (s)",
        "Mean Latency (ms)",
        "P95 Latency (ms)"
    ]
)

eff_fig = px.bar(
    cnn_efficiency,
    x="Model",
    y=efficiency_metric,
    color="Model",
    text=efficiency_metric,
    title=f"CNN Comparison — {efficiency_metric}",
    template="plotly_dark"
)

eff_fig.update_layout(
    showlegend=False,
    paper_bgcolor="#0B1020",
    plot_bgcolor="#141D30"
)

st.plotly_chart(eff_fig, use_container_width=True)

st.dataframe(
    cnn_efficiency,
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# RESEARCH INTERPRETATION
# ---------------------------------------------------------

st.divider()
st.markdown("## 4. Experimental Findings")

st.markdown("""
- The Fixed CNN achieved the highest recorded validation and test accuracy.
- The RL-Searched CNN uses substantially fewer parameters.
- The RL-Searched CNN also achieved lower measured mean and P95 inference latency.
- The RL-selected architecture demonstrates a measurable accuracy-efficiency trade-off.
- Soft Voting did not outperform the standalone XGBoost baseline in this experiment.
""")

st.warning(
    "Interpretation: The RL-Searched CNN is more compact and faster "
    "in the measured setup, but it does not outperform the Fixed CNN "
    "in classification accuracy. Results are specific to this dataset, "
    "candidate space, training setup and hardware."
)

# ---------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------

st.divider()

st.markdown("## Export Benchmark Results")

csv_data = validation_data.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Validation Benchmark CSV",
    data=csv_data,
    file_name="autonas_rl_validation_benchmark.csv",
    mime="text/csv"
)

test_csv = test_data.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Final Test Results CSV",
    data=test_csv,
    file_name="autonas_rl_test_benchmark.csv",
    mime="text/csv"
)

st.caption(
    "AutoNAS-RL · Model Benchmarking · Recorded Experimental Results"
)
