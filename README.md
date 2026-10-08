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