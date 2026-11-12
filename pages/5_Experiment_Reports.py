
import streamlit as st
import pandas as pd
import json
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Experiment Reports | AutoNAS-RL",
    page_icon="📑",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

REPORT_PATH = ARTIFACT_DIR / "final_experiment_report.json"
SUMMARY_PATH = ARTIFACT_DIR / "experiment_summary.json"
CANDIDATE_PATH = ARTIFACT_DIR / "candidate_results.csv"
CONFIG_PATH = ARTIFACT_DIR / "candidate_configs.json"


# =========================================================
# STYLING
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #0B1020;
}

.block-container {
    max-width: 1500px;
    padding: 2rem 3rem;
}

.hero {
    background: linear-gradient(110deg, #172554, #211C46);
    border: 1px solid #34446B;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #F8FAFC;
    font-size: 32px;
}

.hero p {
    color: #BFDBFE;
    line-height: 1.8;
}

div[data-testid="stMetric"] {
    background: #141D30;
    border: 1px solid #29364D;
    border-radius: 12px;
    padding: 16px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="hero">

    <div style="
        color:#93C5FD;
        font-size:11px;
        font-weight:700;
        letter-spacing:2px;
    ">
        RESEARCH DOCUMENTATION & EXPERIMENT TRACKING
    </div>

    <h1>📑 Experiment Reports</h1>

    <p>
        Explore recorded experiment outcomes, architecture
        configurations, candidate evaluations and final
        model performance. Download original research artifacts.
    </p>

</div>
""")


# =========================================================
# ARTIFACT LOADERS
# =========================================================

def load_json(path):

    if not path.exists():
        return None

    try:

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as error:

        st.error(f"Could not read {path.name}: {error}")
        return None


def load_csv(path):

    if not path.exists():
        return None

    try:

        return pd.read_csv(path)

    except Exception as error:

        st.error(f"Could not read {path.name}: {error}")
        return None


# =========================================================
# LOAD EXPERIMENT ARTIFACTS
# =========================================================

report = load_json(REPORT_PATH)
summary = load_json(SUMMARY_PATH)

candidate_df = load_csv(CANDIDATE_PATH)

candidate_configs = load_json(CONFIG_PATH)


# =========================================================
# ARTIFACT AVAILABILITY
# =========================================================

st.markdown("## Artifact Availability")

artifact_status = [
    ("Final Experiment Report", REPORT_PATH),
    ("Experiment Summary", SUMMARY_PATH),
    ("Candidate Results", CANDIDATE_PATH),
    ("Candidate Configurations", CONFIG_PATH)
]

status_cols = st.columns(4)

for col, (name, path) in zip(status_cols, artifact_status):

    with col:

        st.metric(
            name,
            "Available" if path.exists() else "Missing"
        )


st.divider()


# =========================================================
# RESEARCH SUMMARY
# =========================================================

st.markdown("## Research Summary")

if isinstance(report, dict) and report:

    best_candidate_id = report.get(
        "best_candidate_id",
        report.get("best_candidate", "N/A")
    )

    best_reward = report.get("best_reward")

    fixed_test = report.get("fixed_test_results", {})
    rl_test = report.get("rl_test_results", {})

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Candidates Evaluated",
        len(candidate_df) if candidate_df is not None else "N/A"
    )

    r2.metric(
        "Selected Candidate",
        str(best_candidate_id)
    )

    r3.metric(
        "Best Search Reward",
        f"{best_reward:.4f}"
        if isinstance(best_reward, (int, float))
        else "N/A"
    )

    r4.metric(
        "Search Space",
        "12 Architectures"
    )

    st.divider()

    # -----------------------------------------------------
    # FINAL TEST EVALUATION
    # -----------------------------------------------------

    st.markdown("## Final Test Evaluation")

    if fixed_test and rl_test:

        test_table = pd.DataFrame([
            {
                "Model": "Fixed CNN",
                "Accuracy": fixed_test.get("accuracy"),
                "Macro Precision": fixed_test.get("precision"),
                "Macro Recall": fixed_test.get("recall"),
                "Macro F1": fixed_test.get("f1")
            },
            {
                "Model": "RL-Searched CNN",
                "Accuracy": rl_test.get("accuracy"),
                "Macro Precision": rl_test.get("precision"),
                "Macro Recall": rl_test.get("recall"),
                "Macro F1": rl_test.get("f1")
            }
        ])

        display_table = test_table.copy()

        for col in [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1"
        ]:

            display_table[col] = (
                pd.to_numeric(
                    display_table[col],
                    errors="coerce"
                ) * 100
            ).round(2).map(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            )

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Final test evaluation details are not available "
            "in the saved report."
        )

    # -----------------------------------------------------
    # SELECTED ARCHITECTURE
    # -----------------------------------------------------

    st.divider()

    st.markdown("## Selected Architecture Configuration")

    best_config = report.get("best_config", {})

    if isinstance(best_config, dict) and best_config:

        config_cols = st.columns(3)

        config_items = list(best_config.items())

        for index, (key, value) in enumerate(config_items):

            with config_cols[index % 3]:

                st.metric(
                    key.replace("_", " ").title(),
                    str(value)
                )

    else:

        st.info("Selected architecture configuration unavailable.")

else:

    st.warning(
        "Final experiment report was not found. "
        "Check the artifacts directory."
    )


# =========================================================
# CANDIDATE EVALUATION RECORDS
# =========================================================

st.divider()

st.markdown("## Candidate Evaluation Records")

if candidate_df is not None and not candidate_df.empty:

    candidate_df = candidate_df.copy()

    st.caption(
        f"{len(candidate_df)} candidate evaluation records loaded "
        "from the saved experiment results."
    )

    # Remove unnecessary unnamed CSV index columns
    unnamed_columns = [
        col for col in candidate_df.columns
        if str(col).startswith("Unnamed:")
    ]

    candidate_df = candidate_df.drop(
        columns=unnamed_columns,
        errors="ignore"
    )

    st.dataframe(
        candidate_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # NUMERIC METRIC EXPLORER
    # -----------------------------------------------------

    numeric_columns = candidate_df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        st.markdown("### Candidate Performance Explorer")

        selected_metric = st.selectbox(
            "Select candidate evaluation metric",
            numeric_columns
        )

        # Use actual candidate identifiers when available.
        # Otherwise, use row-order labels without modifying
        # the original experiment data.

        id_column = next(
            (
                col for col in [
                    "candidate_id",
                    "candidate",
                    "id"
                ]
                if col in candidate_df.columns
            ),
            None
        )

        if id_column is not None:

            candidate_labels = (
                candidate_df[id_column].astype(str)
            )

        else:

            candidate_labels = pd.Series(
                [
                    f"Candidate {i+1}"
                    for i in range(len(candidate_df))
                ],
                index=candidate_df.index
            )

        chart_df = pd.DataFrame({
            "Candidate": candidate_labels.values,
            selected_metric: pd.to_numeric(
                candidate_df[selected_metric],
                errors="coerce"
            ).values
        })

        chart_df = chart_df.dropna(
            subset=[selected_metric]
        )

        if not chart_df.empty:

            st.bar_chart(
                chart_df.set_index("Candidate"),
                use_container_width=True
            )

        else:

            st.info(
                "No valid numeric values are available for this metric."
            )

    else:

        st.info(
            "No numeric candidate metrics found in the CSV."
        )

else:

    st.warning(
        "Candidate results CSV not found or contains no records."
    )


# =========================================================
# CANDIDATE CONFIGURATIONS
# =========================================================

st.divider()

st.markdown("## Candidate Architecture Configurations")

if isinstance(candidate_configs, list):

    config_display = pd.DataFrame(candidate_configs)

    st.dataframe(
        config_display,
        use_container_width=True,
        hide_index=True
    )

elif isinstance(candidate_configs, dict):

    st.json(
        candidate_configs,
        expanded=False
    )

else:

    st.warning(
        "Candidate configuration file not found or unsupported."
    )


# =========================================================
# EXPERIMENT METADATA
# =========================================================

st.divider()

st.markdown("## Experiment Metadata")

if summary is not None:

    with st.expander("View Saved Experiment Summary"):

        st.json(
            summary,
            expanded=True
        )

else:

    st.info(
        "Experiment summary is not available."
    )


# =========================================================
# DOWNLOAD CENTER
# =========================================================

st.divider()

st.markdown("## Research Artifact Download Center")

st.caption(
    "Download the original saved files used in this dashboard."
)

download_files = [
    ("Final Experiment Report", REPORT_PATH, "json"),
    ("Experiment Summary", SUMMARY_PATH, "json"),
    ("Candidate Results", CANDIDATE_PATH, "csv"),
    ("Candidate Configurations", CONFIG_PATH, "json"),
    ("Search History", ARTIFACT_DIR / "search_history.pkl", "pkl"),
    ("Q-Table", ARTIFACT_DIR / "q_table.npy", "npy")
]

download_cols = st.columns(2)

for index, (label, path, extension) in enumerate(download_files):

    with download_cols[index % 2]:

        if path.exists():

            with open(path, "rb") as file:

                file_bytes = file.read()

            mime_types = {
                "json": "application/json",
                "csv": "text/csv",
                "pkl": "application/octet-stream",
                "npy": "application/octet-stream"
            }

            st.download_button(
                f"⬇ Download {label}",
                data=file_bytes,
                file_name=path.name,
                mime=mime_types[extension],
                key=f"download_{extension}_{index}",
                use_container_width=True
            )

        else:

            st.button(
                f"{label} — Unavailable",
                disabled=True,
                use_container_width=True,
                key=f"unavailable_{index}"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AutoNAS-RL | Experiment Reports | Reproducible Research Artifacts"
)
