"""
Streamlit dashboard for the FinOps Optimization Agent.

A thin presentation layer over the FastAPI backend: it performs no business
logic itself, only HTTP calls, matching the separation-of-concerns principle
used in the backend (API communication kept out of the "components").

Run with: streamlit run frontend/dashboard.py
(requires the FastAPI backend running at BACKEND_URL, default http://localhost:8000)
"""

import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="FinOps Optimization Agent", layout="wide")
st.title("☁️ FinOps Optimization Agent")
st.caption("Autonomous cloud cost right-sizing powered by forecasting + a tool-calling agent")


def _get(path: str, **params):
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def _post(path: str, json_body: dict):
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json=json_body, timeout=60)
        resp.raise_for_status()
        return resp.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


instances, err = _get("/api/v1/instances")

if err:
    st.error(f"Could not reach backend at {BACKEND_URL}: {err}")
    st.stop()

if not instances:
    st.info("No instances found yet. Start the backend once to trigger auto-seeding, then refresh.")
    st.stop()

st.subheader("Fleet Overview")
df = pd.DataFrame(instances)
df["criticality"] = df["tags"].apply(lambda t: t.get("criticality", "non-production"))
st.dataframe(df[["instance_id", "name", "instance_type", "status", "criticality"]], use_container_width=True)

st.divider()
st.subheader("Inspect an Instance")
selected = st.selectbox("Instance", options=[i["instance_id"] for i in instances])

col1, col2 = st.columns(2)

with col1:
    summary, err = _get(f"/api/v1/usage/{selected}/summary")
    st.markdown("**Usage Summary (trailing window)**")
    if err:
        st.error(err)
    elif summary["sample_count"] == 0:
        st.warning("No usage samples available for this instance.")
    else:
        st.metric("Avg CPU %", summary["avg_cpu_pct"])
        st.metric("Max CPU %", summary["max_cpu_pct"])
        st.metric("Avg Memory %", summary["avg_memory_pct"])

with col2:
    forecast, err = _get(f"/api/v1/forecast/{selected}")
    st.markdown("**7-Day CPU Forecast**")
    if err:
        st.error(err)
    elif not forecast.get("points"):
        st.warning("Not enough history to forecast yet.")
    else:
        fdf = pd.DataFrame(forecast["points"]).set_index("timestamp")
        st.line_chart(fdf["predicted_cpu_pct"])
        st.caption(f"Method: {forecast['method']} | Forecast avg: {forecast['forecast_avg_cpu_pct']}%")

st.divider()
st.subheader("Run the Optimization Agent")
dry_run = st.toggle("Dry run (recommended)", value=True)
scope = st.radio("Scope", ["This instance only", "Entire fleet"], horizontal=True)

if st.button("Run Agent", type="primary"):
    with st.spinner("Evaluating usage, forecasting, and reasoning about actions..."):
        body = {"instance_id": None if scope == "Entire fleet" else selected, "dry_run": dry_run}
        result, err = _post("/api/v1/agent/optimize", body)

    if err:
        st.error(f"Agent run failed: {err}")
    elif not result["actions"]:
        st.info("No actions were proposed.")
    else:
        st.success(
            f"Agent source: {result['agent_source']} | "
            f"Estimated monthly savings: ${result['total_estimated_monthly_savings']}"
        )
        st.dataframe(pd.DataFrame(result["actions"]), use_container_width=True)

