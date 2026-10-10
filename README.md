# FinOps Optimization Agent

<!--
IMAGE PLACEHOLDER 1 (hero banner)
Save your image as: docs/images/banner.png  (suggested size: 1600x400)
Then delete this comment block and uncomment the line below:

<p align="center"><img src="docs/images/banner.png" alt="FinOps Optimization Agent banner" width="100%"></p>
-->

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-ReAct%20Agent-1C3C3C" alt="LangGraph">
  <img src="https://img.shields.io/badge/SQLAlchemy-2.0-D71F00" alt="SQLAlchemy 2.0">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Tests-pytest-0A9EDC?logo=pytest&logoColor=white" alt="pytest">
</p>

An autonomous, tool-calling LLM agent that forecasts cloud compute utilization
and proposes (or executes) cost-optimizing resize/shutdown actions — a
production-shaped prototype of a corporate FinOps automation system.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Motivation](#motivation)
- [Target Users](#target-users)
- [Key Features](#key-features)
- [Architecture](#architecture)
  - [System Workflow](#system-workflow)
- [AI/ML/GenAI Methodology](#aimlgenai-methodology)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Environment Variables](#environment-variables)
- [Running Locally](#running-locally)
- [Usage Examples](#usage-examples)
  - [API Usage](#api-usage)
  - [Dashboard Walkthrough](#dashboard-walkthrough)
- [Results](#results)
- [Testing](#testing)
- [Evaluation Methodology](#evaluation-methodology)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

## Overview

The FinOps Optimization Agent watches the utilization of a fleet of cloud
compute instances, forecasts the next 7 days of CPU demand, and decides, under
an explicit policy, whether each instance should be **resized**, **shut down**
or **left alone**. Every decision is explainable, auditable and safe to preview
with `dry_run`.

| | |
|---|---|
| **Forecasting** | Holt-Winters seasonal time-series model with a moving-average fallback |
| **Decision-making** | LangGraph ReAct tool-calling agent, or a deterministic rule-based engine when no API key is set |
| **Safety** | Guardrails enforced at the tool layer: production instances can never be shut down |
| **Auditability** | Every decision is stored with numeric reasoning and estimated monthly savings |
| **Evaluation** | Golden, hand-labeled dataset with per-case accuracy reporting |
| **Interfaces** | FastAPI REST API (auto-generated OpenAPI docs) and a Streamlit dashboard |

```mermaid
flowchart LR
    A["Cloud usage and<br/>billing metrics"] --> B["Time-series<br/>forecast"]
    B --> C["ReAct agent<br/>with guardrails"]
    C --> D["Resize / Shutdown<br/>proposal or action"]
    D --> E[("Audit trail")]
```

## Problem Statement

Engineering teams routinely over-provision cloud compute and forget to scale
it back down. Enterprises lose meaningful budget every month to idle or
oversized EC2-style instances because right-sizing requires someone to
continuously correlate utilization history, forecast near-term demand, and
weigh that against the blast radius of touching a production system — work
that rarely happens manually at scale.

```mermaid
flowchart LR
    P1["Teams provision<br/>servers manually"] --> P2["No operational<br/>budget tracking"]
    P2 --> P3["Idle or oversized<br/>instances keep running"]
    P3 --> P4["Recurring cloud<br/>overspend"]
```

## Motivation

This project applies Operations Research and Managerial Accounting ideas
(cost-center management and operational-expenditure control) to cloud
infrastructure, a discipline known as **FinOps**. Right-sizing is a
decision-under-uncertainty problem: act too late and money is wasted, act too
aggressively and a production system breaks. The goal is to show how
forecasting, an LLM agent and hard safety rules can work together so that
cost decisions are fast, explainable and bounded by policy.

## Target Users

- **Platform/DevOps/SRE teams** who own a cloud budget and want automated,
policy-bound right-sizing recommendations.
- **FinOps/engineering leadership** who need auditable, explainable cost
optimization decisions rather than an opaque "auto-scaler."

## Key Features

| Area | What it does |
|---|---|
| **Usage telemetry** | Synthetic but realistic hourly usage telemetry generation (idle, business-hours, steady-high and spiky profiles) standing in for a CloudWatch-style billing/metrics feed. |
| **Forecasting** | Holt-Winters seasonal time-series forecasting of CPU utilization, with automatic fallback to a moving-average model when history is too short, and self-reported backtest MAE. |
| **LLM agent** | A **LangGraph ReAct tool-calling agent** that reasons over usage summaries and forecasts, then calls `propose_resize` / `propose_shutdown` tools against a simulated cloud control plane. |
| **Rule-based engine** | A **deterministic rule-based policy engine** implementing the exact same FinOps policy — used automatically when no LLM API key is configured, and used as the ground-truth baseline in the evaluation harness. |
| **Guardrails** | **Guardrails enforced at the tool layer** (not just the prompt): production-tagged instances can never be shut down, regardless of what the LLM decides. |
| **Safe execution** | `dry_run` semantics end-to-end: every action can be previewed before execution. |
| **Evaluation** | An evaluation harness that scores agent decisions against a golden, hand-labeled dataset and reports accuracy. |
| **Dashboard** | A Streamlit dashboard for fleet overview, forecasts, and triggering agent runs. |


## Architecture

<!--
IMAGE PLACEHOLDER 2 (architecture illustration)
Save your image as: docs/images/architecture.png  (suggested size: 1400x800)
Then delete this comment block and uncomment the line below:

<p align="center"><img src="docs/images/architecture.png" alt="System architecture illustration" width="90%"></p>
-->

```mermaid
flowchart LR
    subgraph DL["Data Layer"]
        B["Synthetic Billing/Usage Generator"] --> DB[("SQLite/Postgres")]
    end

    subgraph IL["Intelligence Layer"]
        DB --> F["Holt-Winters Forecasting Service"]
        F --> A["LangGraph ReAct Agent / Rule-Based Fallback"]
        DB --> A
    end

    subgraph AL["Action Layer"]
        A -->|"propose_resize / propose_shutdown tools"| G["Guardrails"]
        G --> CP["Simulated Cloud Provider"]
        CP --> DB
    end

    subgraph IF["Interfaces"]
        API["FastAPI REST API"] --> A
        API --> F
        API --> DB
        UI["Streamlit Dashboard"] --> API
    end

    classDef data fill:#e8f1fb,stroke:#3b82c4,color:#0b2a4a;
    classDef intel fill:#eaf7ee,stroke:#2e9e5b,color:#0d3a1f;
    classDef action fill:#fdf0e3,stroke:#d9822b,color:#4a2a05;
    classDef iface fill:#f3eafb,stroke:#8a4fc4,color:#2e0f4a;
    class B,DB data;
    class F,A intel;
    class G,CP action;
    class API,UI iface;
```

### System Workflow

1. Usage telemetry (hourly CPU/memory/network) accumulates per instance.
2. On an optimize request, the forecasting service fits a seasonal
Holt-Winters model (or falls back to a moving average) to project CPU
utilization over the next 7 days.
3. The agent — LLM ReAct if `LLM_PROVIDER` is configured with a key,
otherwise the deterministic rule-based engine — evaluates trailing +
forecasted CPU against policy thresholds and the instance's
`criticality` tag.
4. The agent calls a tool to propose (or, if `dry_run=False`, execute) a
`resize` or `shutdown`. Tools enforce guardrails independently of the
agent's reasoning (e.g. production instances can never be shut down).
5. Every decision is persisted as an `AgentAction` row with its numeric
reasoning and estimated monthly savings, giving a full audit trail.

```mermaid
sequenceDiagram
    autonumber
    actor U as User / Dashboard
    participant API as FastAPI
    participant AG as Agent (LLM or rule-based)
    participant FC as Forecasting Service
    participant TL as Tools + Guardrails
    participant CP as Simulated Cloud Provider
    participant DB as Database

    U->>API: POST /api/v1/agent/optimize
    API->>AG: run_agent_optimization(instance_id, dry_run)
    AG->>DB: read usage summary
    AG->>FC: forecast next 7 days of CPU
    FC-->>AG: forecast average + backtest MAE
    AG->>TL: propose_resize / propose_shutdown
    TL->>TL: enforce guardrails
    alt dry_run = false and guardrails pass
        TL->>CP: execute resize / shutdown
    end
    TL->>DB: persist AgentAction (reasoning + savings)
    API-->>U: actions + total estimated monthly savings
```

### Data Model

```mermaid
erDiagram
    CLOUD_INSTANCE ||--o{ USAGE_METRIC : "has hourly"
    CLOUD_INSTANCE ||--o{ AGENT_ACTION : "receives"

    CLOUD_INSTANCE {
        int id PK
        string instance_id UK
        string name
        string instance_type
        string region
        string provider
        enum status
        json tags
    }
    USAGE_METRIC {
        int id PK
        int instance_id FK
        datetime timestamp
        float cpu_utilization_pct
        float memory_utilization_pct
        float network_in_mb
        float network_out_mb
    }
    AGENT_ACTION {
        int id PK
        int instance_id FK
        enum action_type
        string previous_instance_type
        string new_instance_type
        text reasoning
        float forecasted_avg_cpu
        float estimated_monthly_savings
        bool dry_run
        enum status
        string agent_source
    }
```


## AI/ML/GenAI Methodology

- **Forecasting**: Holt-Winters exponential smoothing with additive trend and
24-hour seasonality (`statsmodels`), chosen over Prophet to avoid a heavy
native build toolchain for hourly, short-horizon operational forecasting.
Backtested with a held-out 24-hour window and reported as MAE.
- **Agent orchestration**: `langgraph.prebuilt.create_react_agent` gives the
LLM a fixed toolset (`list_all_instances`, `get_instance_usage_summary`,
`get_forecast`, `propose_resize`, `propose_shutdown`) and a system prompt
encoding the FinOps policy; the LLM must call the data tools before
proposing an action rather than inferring numbers.
- **Hallucination mitigation**: the LLM never has authority to fabricate
utilization numbers — all figures come from tool calls against the real
(simulated) database — and destructive actions are re-validated by
guardrails in the tool layer, not trusted from the LLM's text output.
- **Reliability**: LLM tool invocations are wrapped in retry-with-backoff
(`tenacity`); on repeated failure, the run transparently falls back to the
rule-based policy engine so the API never hard-fails.
- **Evaluation**: `evaluation/evaluate_agent.py` runs the active agent against
a golden dataset of 8 labeled scenarios spanning all four policy branches
(production/non-production × idle/steady) and reports accuracy plus a
per-case breakdown, written to `evaluation/report.json`.

### Decision Policy

Both the LLM agent and the rule-based engine implement the same policy. An
instance is **underutilized** when both its trailing average CPU and its
forecasted average CPU are below `UNDERUTILIZED_CPU_THRESHOLD`.

```mermaid
flowchart TD
    S(["Evaluate instance"]) --> D1{"Forecast CPU above 80%<br/>of capacity?"}
    D1 -->|Yes| R1["RESIZE up to next larger tier"]
    D1 -->|No| D2{"Underutilized?<br/>avg CPU and forecast CPU<br/>below threshold"}
    D2 -->|No| N["NO_ACTION"]
    D2 -->|Yes| D3{"Criticality tag"}
    D3 -->|production| R2["RESIZE down to next smaller tier"]
    D3 -->|non-production| X["SHUTDOWN"]

    classDef ok fill:#eaf7ee,stroke:#2e9e5b,color:#0d3a1f;
    classDef warn fill:#fdf0e3,stroke:#d9822b,color:#4a2a05;
    classDef neutral fill:#eef0f3,stroke:#6b7280,color:#1f2937;
    class R1,R2 ok;
    class X warn;
    class N neutral;
```

> **Guardrail:** a `shutdown` proposed for a production-tagged instance is
> rejected at the tool layer and logged as `REJECTED`, regardless of what the
> LLM decided. Production instances can only ever be resized.

### Graceful Degradation

```mermaid
flowchart LR
    R(["Optimize request"]) --> Q{"LLM provider and<br/>API key configured?"}
    Q -->|No| RB["Rule-based policy engine"]
    Q -->|Yes| L["LangGraph ReAct agent<br/>with retry and backoff"]
    L -->|success| OUT(["Actions returned"])
    L -->|repeated failure| RB
    RB --> OUT
```

## Technology Stack

| Layer               | Technology                                         | Why                                                                 |
| ------------------- | -------------------------------------------------- | ------------------------------------------------------------------- |
| API                 | FastAPI + Pydantic v2                              | Async-ready, typed request/response validation, auto OpenAPI docs   |
| ORM/DB              | SQLAlchemy 2.0 (SQLite default, Postgres-ready)    | Zero-config local dev; swap `DATABASE_URL` for production           |
| Forecasting         | statsmodels (Holt-Winters)                         | Lightweight, pure-Python seasonal forecasting suited to hourly data |
| Agent orchestration | LangGraph (`create_react_agent`) + LangChain tools | Standard tool-calling ReAct pattern, model-agnostic                 |
| LLM providers       | Anthropic / OpenAI (optional)                      | Pluggable via `LLM_PROVIDER`; app runs fully without either         |
| Retry/reliability   | `tenacity`                                         | Exponential backoff around LLM tool invocations                     |
| Frontend            | Streamlit                                          | Fast, professional dashboard without a separate JS build            |
| Testing             | pytest                                             | Unit, integration and guardrail tests                               |

## Project Structure

```text
cloudcost-sentinel/
├── app/
│   ├── main.py                # FastAPI app, lifespan seeding, routers
│   ├── config.py              # Environment-driven settings
│   ├── db/                    # Engine, session, ORM models, init
│   ├── schemas/               # Pydantic request/response models
│   ├── services/              # Billing sim, forecasting, cost math, cloud provider
│   ├── agent/                 # Prompts, tools, rule-based engine, LangGraph orchestration
│   └── api/routes/            # instances, usage, forecast, agent endpoints
├── scripts/seed_data.py       # Demo fleet seeding
├── evaluation/                # Golden dataset + evaluation harness
├── frontend/dashboard.py      # Streamlit UI
└── tests/                     # Unit + integration tests
```

```mermaid
flowchart TD
    API["api/routes<br/>HTTP layer"] --> SVC["services<br/>billing, forecasting, cost, cloud provider"]
    API --> AGT["agent<br/>prompts, tools, rule-based, LangGraph"]
    AGT --> SVC
    SVC --> DBL["db<br/>engine, session, ORM models"]
    API --> SCH["schemas<br/>Pydantic models"]
    CFG["config.py<br/>environment settings"] -.-> API
    CFG -.-> SVC
    CFG -.-> AGT
```

Routes stay thin, business logic lives in `services/`, and the agent's tools are
thin wrappers around the same services, so the LLM agent and the rule-based
engine share one implementation of the underlying cost and forecasting logic.

## Setup Instructions

**Prerequisites:** Python 3.10 or newer.

```bash
git clone https://github.com/SkJishan04/cloudcost-sentinel.git
cd cloudcost-sentinel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

> **Windows:** activate the virtual environment with `.venv\Scripts\activate`
> and copy the env file with `copy .env.example .env`.

By default `LLM_PROVIDER=none`, so the app runs immediately on the
deterministic rule-based policy with no API key required. To use a real LLM
agent, set `LLM_PROVIDER=anthropic` (or `openai`) and the matching API key in `.env`.

## Environment Variables

| Variable                               | Description                             | Default                      |
| -------------------------------------- | --------------------------------------- | ---------------------------- |
| `DATABASE_URL`                         | SQLAlchemy connection string            | `sqlite:///./finops.db`      |
| `LLM_PROVIDER`                         | `anthropic` \| `openai` \| `none`       | `none`                       |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Provider API key                        | unset                        |
| `LLM_MODEL`                            | Model name for the selected provider    | `claude-3-5-sonnet-20241022` |
| `UNDERUTILIZED_CPU_THRESHOLD`          | % below which CPU is "underutilized"    | `15.0`                       |
| `FORECAST_HORIZON_HOURS`               | Forecast horizon                        | `168` (7 days)               |
| `HISTORY_LOOKBACK_DAYS`                | Usage history window                    | `14`                         |
| `DRY_RUN_DEFAULT`                      | Default `dry_run` for `/agent/optimize` | `true`                       |
| `AUTO_SEED`                            | Seed demo fleet on empty DB at startup  | `true`                       |
| `APP_NAME`                             | Application display name                | `FinOps Optimization Agent`  |
| `ENV`                                  | `development` \| `test` \| `production` | `development`                |
| `LOG_LEVEL`                            | Logging verbosity                       | `INFO`                       |
| `CORS_ORIGINS`                         | Comma-separated allowed origins         | `*`                          |

Secrets such as API keys are read only from environment variables and are never
stored in source code. `.env` is git-ignored.

## Running Locally

```bash
uvicorn app.main:app --reload
# API docs: http://localhost:8000/docs
```

In a second terminal (with the virtual environment activated):

```bash
streamlit run frontend/dashboard.py
```

On first start the database is created and a demo fleet of 6 instances with 14
days of synthetic hourly usage is seeded automatically (`AUTO_SEED=true`).

