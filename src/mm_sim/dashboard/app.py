"""Streamlit dashboard entry point.

Run: `uv run streamlit run src/mm_sim/dashboard/app.py`

Streamlit auto-discovers pages under `pages/` and shows them in the
sidebar. This file is the landing page.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from mm_sim.dashboard import loader

DEFAULT_EXPERIMENTS_DIR = Path("experiments")

st.set_page_config(
    page_title="mm-sim dashboard",
    layout="wide",
)

st.title("mm-sim experiment dashboard")

st.markdown(
    """
    Explore experiment artifacts interactively. Use the sidebar to pick
    a page:

    - **Single Run** — drill into one scenario/version
    - **Compare Scenarios** — overlay metrics across scenarios in a season
    """
)

exp_dir_str = st.sidebar.text_input(
    "experiments directory",
    value=str(DEFAULT_EXPERIMENTS_DIR),
    help="Path to the top-level experiments/ directory.",
)
st.session_state["experiments_dir"] = exp_dir_str

seasons = loader.cached_list_seasons(exp_dir_str)
if not seasons:
    st.warning(
        f"No experiments found under `{exp_dir_str}`. "
        "Run `just scenarios` to generate some."
    )
else:
    st.subheader("Available seasons")
    for season in seasons:
        scenarios = loader.cached_list_scenarios(exp_dir_str, season)
        st.markdown(f"**{season}** — {len(scenarios)} scenarios")
