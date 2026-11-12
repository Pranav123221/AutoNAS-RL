
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.services.artifact_service import (
    get_candidate_results,
    get_candidate_configs,
    get_q_table,
    get_search_history
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Architecture Search | AutoNAS-RL",
    page_icon="🧠",
    layout="wide"
)


# =====================================================
# CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background: #0B1020;
}

.block-container {
    width: 100%;
    max-width: 1600px;
    padding: 2rem 3rem 3rem 3rem;
}

.search-hero {
    background: linear-gradient(
        115deg,
        #172554,
        #172033,
        #211C46
    );
    border: 1px solid #34446B;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 25px;
}

.search-hero h1 {
    color: #F8FAFC;
    font-size: 34px;
    font-weight: 800;
}

.search-hero p {
    color: #A5B4CF;
    font-size: 14px;
    line-height: 1.7;
}

.section-heading {
    color: #F1F5F9;
    font-size: 22px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 12px;
}

.metric-card {
    background: #141D30;
    border: 1px solid #29364D;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
}

.metric-value {
    color: #F8FAFC;
    font-size: 27px;
    font-weight: 800;
    margin-top: 10px;
}

.info-panel {
    background: #141D30;
    border: 1px solid #29364D;
    border-radius: 15px;
    padding: 22px;
}

div[data-testid="stMetric"] {
    background: #141D30;
    border: 1px solid #29364D;
    border-radius: 14px;
    padding: 16px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD EXPERIMENT DATA
# =====================================================

try:

    candidates = get_candidate_results()
    configs = get_candidate_configs()
    q_table = get_q_table()
    history = get_search_history()

except Exception as error:

    st.error(f"Unable to load experiment artifacts: {error}")
    st.stop()



# =========================================================
# HERO SECTION
# =========================================================

st.html("""
<div class="prediction-hero">

    <div style="
        color:#93C5FD;
        font-size:11px;
        font-weight:700;
        letter-spacing:2px;
    ">
        REINFORCEMENT LEARNING RESEARCH
    </div>

    <h1 style="
        color:#F8FAFC;
        font-size:36px;
        font-weight:800;
        margin-top:15px;
    ">
        Neural Architecture Search
    </h1>

    <p style="
        color:#B6C4DD;
        font-size:14px;
        line-height:1.9;
        max-width:850px;
    ">
        Explore CNN architecture candidates evaluated by
        the Q-learning-controlled search process.

        <br><br>

        Inspect architecture configurations, compare measured
        performance and examine the stored Q-table.
    </p>

    <div style="
        display:inline-block;
        background:#12352D;
        color:#6EE7B7;
        border:1px solid #21634F;
        border-radius:20px;
        padding:7px 13px;
        font-size:10px;
        font-weight:700;
        letter-spacing:1px;
    ">
        12 CANDIDATES EVALUATED
    </div>

</div>
""")


# =====================================================
# TOP METRICS
# =====================================================

best = candidates.loc[candidates["reward"].idxmax()]

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Candidates Evaluated",
    len(candidates)
)

m2.metric(
    "Best Candidate",
    f"#{int(best['candidate_id'])}"
)

m3.metric(
    "Best Reward",
    f"{best['reward']:.4f}"
)

m4.metric(
    "Recorded Episodes",
    len(history)
)


# =====================================================
# CANDIDATE EXPLORER
# =====================================================

st.markdown(
    '<div class="section-heading">Architecture Explorer</div>',
    unsafe_allow_html=True
)

st.caption(
    "Select any evaluated candidate to inspect its configuration and metrics."
)

candidate_ids = candidates["candidate_id"].astype(int).tolist()

selected_id = st.selectbox(
    "Select Candidate",
    candidate_ids,
    index=candidate_ids.index(int(best["candidate_id"]))
)


# Support list or dictionary configuration formats
if isinstance(configs, dict):

    selected_config = configs.get(
        str(selected_id),
        configs.get(selected_id)
    )

else:

    selected_config = configs[selected_id - 1]


if selected_config is None:

    st.error("Configuration for this candidate was not found.")
    st.stop()


selected_row = candidates[
    candidates["candidate_id"] == selected_id
].iloc[0]


left, right = st.columns([1, 1])


with left:

    st.markdown("#### Selected CNN Architecture")

    st.html(f"""
    <div style="background:#141D30;border:1px solid #29364D;
                border-radius:14px;padding:18px;text-align:center;">

        <div style="background:#12352D;padding:14px;border-radius:9px;">
            <b style="color:#F8FAFC;">Input Image</b>
            <p style="color:#93C5FD;margin:5px;">28 × 28 × 1</p>
        </div>

        <div style="color:#60A5FA;font-size:22px;">↓</div>

        <div style="background:#172554;padding:14px;border-radius:9px;">
            <b style="color:#F8FAFC;">Convolution Block 1</b>
            <p style="color:#BFDBFE;margin:5px;">
                Conv2D · {selected_config['filters1']} Filters ·
                {selected_config['kernel_size']}×{selected_config['kernel_size']}
            </p>
            <p style="color:#BFDBFE;margin:5px;">
                BatchNorm → ReLU → {selected_config['pooling1'].title()} Pooling
            </p>
        </div>

        <div style="color:#60A5FA;font-size:22px;">↓</div>

        <div style="background:#172554;padding:14px;border-radius:9px;">
            <b style="color:#F8FAFC;">Convolution Block 2</b>
            <p style="color:#BFDBFE;margin:5px;">
                Conv2D · {selected_config['filters2']} Filters ·
                {selected_config['kernel_size']}×{selected_config['kernel_size']}
            </p>
            <p style="color:#BFDBFE;margin:5px;">
                BatchNorm → ReLU → {selected_config['pooling2'].title()} Pooling
            </p>
        </div>

        <div style="color:#60A5FA;font-size:22px;">↓</div>

        <div style="background:#172554;padding:14px;border-radius:9px;">
            <b style="color:#F8FAFC;">Flatten → Dropout</b>
            <p style="color:#BFDBFE;margin:5px;">
                Dropout: {selected_config['dropout1']}
            </p>
        </div>

        <div style="color:#60A5FA;font-size:22px;">↓</div>

        <div style="background:#172554;padding:14px;border-radius:9px;">
            <b style="color:#F8FAFC;">Dense Layer</b>
            <p style="color:#BFDBFE;margin:5px;">
                {selected_config['dense_units']} Units ·
                Dropout: {selected_config['dropout2']}
            </p>
        </div>

        <div style="color:#60A5FA;font-size:22px;">↓</div>

        <div style="background:#31204D;padding:14px;border-radius:9px;">
            <b style="color:#F8FAFC;">Softmax Output</b>
            <p style="color:#D8B4FE;margin:5px;">10 Classes</p>
        </div>

    </div>
    """)

    st.markdown("##### Configuration Parameters")

   
st.markdown("##### Architecture Parameters")

param1, param2, param3 = st.columns(3)

with param1:
    st.metric("Conv 1 Filters", selected_config["filters1"])
    st.metric("Kernel Size", f'{selected_config["kernel_size"]} × {selected_config["kernel_size"]}')
    st.metric("Pooling 1", selected_config["pooling1"].title())

with param2:
    st.metric("Conv 2 Filters", selected_config["filters2"])
    st.metric("Dense Units", selected_config["dense_units"])
    st.metric("Pooling 2", selected_config["pooling2"].title())

with param3:
    st.metric("Dropout 1", selected_config["dropout1"])
    st.metric("Dropout 2", selected_config["dropout2"])
    st.metric("Learning Rate", selected_config["learning_rate"])



with right:

    st.markdown("#### Candidate Performance")

    p1, p2 = st.columns(2)

    p1.metric(
        "Accuracy",
        f"{selected_row['accuracy'] * 100:.2f}%"
    )

    p2.metric(
        "Macro F1",
        f"{selected_row['macro_f1'] * 100:.2f}%"
    )

    p3, p4 = st.columns(2)

    p3.metric(
        "Mean Latency",
        f"{selected_row['mean_latency_ms']:.3f} ms"
    )

    p4.metric(
        "Parameters",
        f"{int(selected_row['parameters']):,}"
    )

    st.metric(
        "Weighted Reward",
        f"{selected_row['reward']:.4f}"
    )


st.divider()


# =====================================================
# REWARD RANKING
# =====================================================

st.markdown(
    '<div class="section-heading">Reward Ranking</div>',
    unsafe_allow_html=True
)

ranked = candidates.sort_values(
    "reward",
    ascending=True
)

fig_reward = px.bar(
    ranked,
    x="reward",
    y="candidate_id",
    orientation="h",
    color="reward",
    color_continuous_scale="Blues",
    text="reward",
    labels={
        "reward": "Weighted Reward",
        "candidate_id": "Candidate ID"
    },
    template="plotly_dark"
)

fig_reward.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_reward.update_layout(
    height=480,
    paper_bgcolor="#0B1020",
    plot_bgcolor="#141D30",
    font=dict(color="#CBD5E1"),
    coloraxis_showscale=False,
    margin=dict(l=20, r=50, t=30, b=30)
)

st.plotly_chart(
    fig_reward,
    use_container_width=True
)


# =====================================================
# ACCURACY VS LATENCY
# =====================================================

st.markdown(
    '<div class="section-heading">Accuracy vs Latency</div>',
    unsafe_allow_html=True
)

st.caption(
    "Bubble size represents parameter count. "
    "Color represents the weighted search reward."
)

fig_scatter = px.scatter(
    candidates,
    x="mean_latency_ms",
    y="accuracy",
    size="parameters",
    color="reward",
    hover_name="candidate_id",
    hover_data=[
        "macro_f1",
        "parameters",
        "reward"
    ],
    color_continuous_scale="Viridis",
    labels={
        "mean_latency_ms": "Mean Latency (ms)",
        "accuracy": "Validation Accuracy",
        "reward": "Reward"
    },
    template="plotly_dark"
)

fig_scatter.update_layout(
    height=480,
    paper_bgcolor="#0B1020",
    plot_bgcolor="#141D30",
    font=dict(color="#CBD5E1")
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)


# =====================================================
# Q TABLE HEATMAP
# =====================================================

st.markdown(
    '<div class="section-heading">Q-Learning State-Action Table</div>',
    unsafe_allow_html=True
)

st.caption(
    "Stored Q-values from the completed experiment."
)

state_names = [
    f"State {i}"
    for i in range(q_table.shape[0])
]

action_names = [
    f"Candidate {i + 1}"
    for i in range(q_table.shape[1])
]

fig_q = go.Figure(
    data=go.Heatmap(
        z=q_table,
        x=action_names,
        y=state_names,
        colorscale="Viridis",
        colorbar=dict(title="Q-value"),
        hovertemplate=(
            "%{y}<br>"
            "%{x}<br>"
            "Q-value: %{z:.5f}"
            "<extra></extra>"
        )
    )
)

fig_q.update_layout(
    height=390,
    paper_bgcolor="#0B1020",
    plot_bgcolor="#141D30",
    font=dict(color="#CBD5E1"),
    margin=dict(l=20, r=20, t=40, b=40)
)

st.plotly_chart(
    fig_q,
    use_container_width=True
)


# =====================================================
# COMPLETE CANDIDATE TABLE
# =====================================================

st.markdown(
    '<div class="section-heading">Candidate Performance Database</div>',
    unsafe_allow_html=True
)

display_df = candidates.copy()

display_df["accuracy"] = (
    display_df["accuracy"] * 100
).round(2)

display_df["macro_f1"] = (
    display_df["macro_f1"] * 100
).round(2)

display_df["mean_latency_ms"] = (
    display_df["mean_latency_ms"]
).round(3)

display_df["reward"] = (
    display_df["reward"]
).round(4)

st.dataframe(
    display_df.sort_values(
        "reward",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# =====================================================
# SEARCH HISTORY
# =====================================================

st.markdown(
    '<div class="section-heading">Search Episode History</div>',
    unsafe_allow_html=True
)

if isinstance(history, list) and history:

    if all(isinstance(item, dict) for item in history):

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.write(history)

else:

    st.info(
        "Search history is available but does not contain "
        "tabular records."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AutoNAS-RL | Q-Learning-Controlled CNN Architecture Search"
)

st.caption(
    "All results are loaded from saved artifacts. "
    "No new training or Q-table updates are performed here."
)
