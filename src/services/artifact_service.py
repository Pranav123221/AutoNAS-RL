
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "artifacts"


def artifact_path(filename):
    return ARTIFACT_DIR / filename


@st.cache_data
def load_json(filename):
    path = artifact_path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {filename}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_candidate_results():
    path = artifact_path("candidate_results.csv")

    if not path.exists():
        raise FileNotFoundError("candidate_results.csv not found")

    df = pd.read_csv(path)

    required_columns = {
        "candidate_id",
        "accuracy",
        "macro_f1",
        "mean_latency_ms",
        "parameters",
        "reward"
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing candidate result columns: {sorted(missing)}"
        )

    return df


@st.cache_data
def load_q_table():
    path = artifact_path("q_table.npy")

    if not path.exists():
        raise FileNotFoundError("q_table.npy not found")

    return np.load(path)


@st.cache_data
def load_pickle(filename):
    path = artifact_path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {filename}")

    with open(path, "rb") as file:
        return pickle.load(file)


def get_experiment_report():
    return load_json("final_experiment_report.json")


def get_candidate_results():
    return load_candidate_results()


def get_q_table():
    return load_q_table()


def get_search_history():
    return load_pickle("search_history.pkl")


def get_candidate_configs():
    return load_json("candidate_configs.json")
