import json
import os


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = []

cells.append(md("""
# FinOps Optimization Agent: Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkJishan04/cloudcost-sentinel/blob/main/notebooks/finops_agent_colab.ipynb)

This notebook clones the repository, installs it, and lets you **test and run the whole project in the browser**: unit tests, the evaluation harness, the FastAPI backend, forecasting, and the optimization agent.

**How to use:** click **Runtime > Run all**, or run the cells top to bottom with `Shift + Enter`.
No API key is needed. By default the agent runs on the deterministic rule-based policy engine.

| Step | What happens |
|---|---|
| 1 | Clone the repo and install dependencies |
| 2 | Run the pytest suite and the evaluation harness |
| 3 | Start the FastAPI backend |
| 4 | Inspect the fleet, usage and a 7-day forecast |
| 5 | Run the agent (dry run, then real execution) |
| 6 | Demonstrate the production-shutdown guardrail |
| 7 | *(Optional)* Use a real LLM, or open the Streamlit dashboard |
"""))

cells.append(md("## 1. Clone the repository and install dependencies"))

cells.append(code('''
import os
import subprocess
import sys

REPO_URL = "https://github.com/SkJishan04/cloudcost-sentinel.git"
REPO_DIR = "/content/cloudcost-sentinel"

if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "-q"], check=True)
else:
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)

# Environment passed to every subprocess (tests, evaluation, API server).
# LLM_PROVIDER=none keeps the project fully runnable with no API key.
ENV = {
    **os.environ,
    "LLM_PROVIDER": "none",
    "DATABASE_URL": "sqlite:///./finops.db",
    "AUTO_SEED": "true",
    "DRY_RUN_DEFAULT": "true",
}


def run(cmd, tail=None):
    """Run a shell command inside the repo and print its output."""
    proc = subprocess.run(cmd, shell=True, cwd=REPO_DIR, env=ENV, capture_output=True, text=True)
    output = (proc.stdout + proc.stderr).strip().splitlines()
    print("\\n".join(output[-tail:] if tail else output))
    print(f"\\n[exit code: {proc.returncode}]")
    return proc.returncode


print("Repository ready at", REPO_DIR)
'''))

cells.append(code('''
# Installs the exact pinned versions from requirements.txt (about 1-2 minutes).
run(f"{sys.executable} -m pip install -q -r requirements.txt", tail=15)
'''))

cells.append(md("""
> If the install step reports a dependency error, use **Runtime > Restart session** and run the notebook again from the top.
"""))

cells.append(md("## 2. Run the test suite and the evaluation harness"))

cells.append(code('''
# Unit, integration and guardrail tests (isolated in-memory database, no API key).
run(f"{sys.executable} -m pytest -q -p no:cacheprovider", tail=25)
'''))

cells.append(code('''
# Scores the agent against the 8-case golden dataset and writes evaluation/report.json.
import json
import pandas as pd

run(f"{sys.executable} -m evaluation.evaluate_agent", tail=5)

with open(os.path.join(REPO_DIR, "evaluation", "report.json")) as fh:
    report = json.load(fh)

print(f"\\nAccuracy: {report['accuracy']:.0%}  ({report['correct']}/{report['total_cases']} cases correct)")
pd.DataFrame(report["cases"])[["instance_id", "expected", "predicted", "correct", "agent_source"]]
'''))

cells.append(md("## 3. Start the FastAPI backend"))

cells.append(code('''
import time
import requests

BASE = "http://localhost:8000"
server = globals().get("server")


def start_server(fresh_db=True):
    """(Re)start uvicorn in the background. A fresh DB reseeds the demo fleet."""
    global server
    if server is not None and server.poll() is None:
        server.terminate()
        server.wait(timeout=10)
    if fresh_db:
        for name in ("finops.db",):
            path = os.path.join(REPO_DIR, name)
            if os.path.exists(path):
                os.remove(path)
    log = open(os.path.join(REPO_DIR, "server.log"), "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=REPO_DIR, env=ENV, stdout=log, stderr=subprocess.STDOUT,
    )
    for _ in range(90):
        try:
            health = requests.get(f"{BASE}/health", timeout=2).json()
            print("Backend is up:", health)
            return
        except requests.RequestException:
            time.sleep(1)
    print("Backend did not start. Last log lines:")
    print(open(os.path.join(REPO_DIR, "server.log")).read()[-2000:])


start_server()
'''))

cells.append(md("""
The interactive API docs (Swagger UI) run inside the Colab VM at `http://localhost:8000/docs`.
They are not directly reachable from your browser, so the cells below call the API from Python instead.
"""))

cells.append(md("## 4. Explore the fleet, usage and forecast"))

cells.append(code('''
instances = requests.get(f"{BASE}/api/v1/instances").json()
fleet = pd.DataFrame(instances)
fleet["criticality"] = fleet["tags"].apply(lambda t: t.get("criticality", "non-production"))
fleet[["instance_id", "name", "instance_type", "status", "criticality"]]
'''))

cells.append(code('''
# Trailing usage summary for every instance.
summaries = [
    requests.get(f"{BASE}/api/v1/usage/{i['instance_id']}/summary").json() for i in instances
]
pd.DataFrame(summaries)[["instance_id", "avg_cpu_pct", "max_cpu_pct", "min_cpu_pct", "avg_memory_pct", "sample_count"]]
'''))

cells.append(code('''
# 7-day Holt-Winters CPU forecast with the model's backtest error.
import matplotlib.pyplot as plt

INSTANCE_ID = "i-prod-api-01"   # try: i-staging-web-01, i-prod-db-01, i-prod-worker-01

forecast = requests.get(f"{BASE}/api/v1/forecast/{INSTANCE_ID}").json()
fdf = pd.DataFrame(forecast["points"])
fdf["timestamp"] = pd.to_datetime(fdf["timestamp"])

plt.figure(figsize=(10, 3.5))
plt.plot(fdf["timestamp"], fdf["predicted_cpu_pct"], color="#2e9e5b")
plt.title(f"{INSTANCE_ID}: 7-day CPU forecast")
plt.ylabel("Predicted CPU %")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Method: {forecast['method']} | Forecast avg: {forecast['forecast_avg_cpu_pct']}% | Backtest MAE: {forecast['backtest_mae']}")
'''))

cells.append(md("## 5. Run the optimization agent"))

cells.append(md("**Dry run** only proposes actions. Nothing is changed."))

cells.append(code('''
resp = requests.post(f"{BASE}/api/v1/agent/optimize", json={"dry_run": True}).json()

actions = pd.DataFrame(resp["actions"])
print(f"Agent source: {resp['agent_source']} | Instances evaluated: {resp['evaluated_instances']}")
print(f"Total estimated monthly savings: ${resp['total_estimated_monthly_savings']}")
actions[["instance_id", "action_type", "previous_instance_type", "new_instance_type",
         "estimated_monthly_savings", "status", "dry_run"]]
'''))

cells.append(code('''
# Why did the agent decide that? Every action stores numeric reasoning (audit trail).
for _, row in actions.iterrows():
    print(f"{row['instance_id']:<20} {row['action_type']:<10} {row['reasoning']}\\n")
'''))

cells.append(md("""
**Real execution** (`dry_run=False`) applies the actions to the simulated cloud provider.
Afterwards the idle non-production instances show as `stopped`.
"""))

cells.append(code('''
resp = requests.post(f"{BASE}/api/v1/agent/optimize", json={"dry_run": False}).json()
print(f"Executed. Total monthly savings: ${resp['total_estimated_monthly_savings']}")

after = pd.DataFrame(requests.get(f"{BASE}/api/v1/instances").json())
after[["instance_id", "instance_type", "status"]]
'''))

cells.append(code('''
# Reset the demo: restarts the backend with a fresh database and a freshly seeded fleet.
start_server(fresh_db=True)
'''))

cells.append(md("""
## 6. Guardrail demo: production instances can never be shut down

The guardrail lives in the **tool layer**, so it holds even if an LLM decides to propose a shutdown.
"""))

cells.append(code('''
guardrail_demo = """
from app.agent.tools import propose_shutdown
from app.core.exceptions import GuardrailViolationError

try:
    propose_shutdown.func("i-prod-api-01", "demo: attempt to shut down a production instance", True)
    print("UNEXPECTED: the shutdown was allowed")
except GuardrailViolationError as exc:
    print("BLOCKED by guardrail:", exc.message)
"""
subprocess.run([sys.executable, "-c", guardrail_demo], cwd=REPO_DIR, env=ENV)
'''))

cells.append(md("""
## 7. Optional extras

### 7a. Use a real LLM agent

Set `USE_LLM = True` and paste your key when prompted. The key is kept in memory only and is never written to disk or committed.
The agent then runs as a LangGraph ReAct agent and falls back to the rule-based engine if the LLM fails.
"""))

cells.append(code('''
USE_LLM = False          # set to True to enable
PROVIDER = "anthropic"   # "anthropic" or "openai"

if USE_LLM:
    from getpass import getpass

    ENV["LLM_PROVIDER"] = PROVIDER
    ENV[f"{PROVIDER.upper()}_API_KEY"] = getpass(f"{PROVIDER} API key: ")
    if PROVIDER == "openai":
        ENV["LLM_MODEL"] = "gpt-4o-mini"
    start_server(fresh_db=True)
    resp = requests.post(f"{BASE}/api/v1/agent/optimize", json={"instance_id": "i-staging-web-01", "dry_run": True}).json()
    print("Agent source:", resp["agent_source"])
    print(resp["actions"][0]["reasoning"])
else:
    print("Skipped (USE_LLM = False). The rule-based engine is in use.")
'''))

cells.append(md("""
### 7b. Open the Streamlit dashboard

Needs a free [ngrok](https://dashboard.ngrok.com/get-started/your-authtoken) auth token. Set `USE_DASHBOARD = True` to enable.
"""))

cells.append(code('''
USE_DASHBOARD = False    # set to True to enable

if USE_DASHBOARD:
    from getpass import getpass

    run(f"{sys.executable} -m pip install -q pyngrok", tail=3)
    from pyngrok import ngrok

    ngrok.set_auth_token(getpass("ngrok auth token: "))
    ui_env = {**ENV, "BACKEND_URL": BASE}
    ui_log = open(os.path.join(REPO_DIR, "streamlit.log"), "w")
    ui = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/dashboard.py",
         "--server.port", "8501", "--server.headless", "true"],
        cwd=REPO_DIR, env=ui_env, stdout=ui_log, stderr=subprocess.STDOUT,
    )
    time.sleep(8)
    print("Dashboard URL:", ngrok.connect(8501, "http").public_url)
else:
    print("Skipped (USE_DASHBOARD = False).")
'''))

cells.append(md("## 8. Clean up"))

cells.append(code('''
if server is not None and server.poll() is None:
    server.terminate()
    print("Backend stopped.")
'''))

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "finops_agent_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/finops_agent_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Created notebooks/finops_agent_colab.ipynb with", len(cells), "cells")