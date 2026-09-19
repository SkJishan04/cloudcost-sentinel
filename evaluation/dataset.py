"""
Golden evaluation dataset.

Each case pins an instance's tags and usage profile to a known-correct
expected action under the FinOps policy in app/agent/prompts.py. This is the
ground truth the agent (rule-based or LLM) is scored against in
evaluate_agent.py, analogous to a retrieval/QA eval set in a RAG project.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    instance_id: str
    name: str
    instance_type: str
    criticality: str
    usage_profile: str
    seed: int
    expected_action: str  # "resize" | "shutdown" | "no_action"


GOLDEN_DATASET: list[EvalCase] = [
    EvalCase("eval-prod-idle-01", "eval-prod-idle", "m5.xlarge", "production", "idle", 101, "resize"),
    EvalCase("eval-prod-idle-02", "eval-prod-idle-2", "t3.xlarge", "production", "idle", 102, "resize"),
    EvalCase("eval-nonprod-idle-01", "eval-nonprod-idle", "t3.large", "non-production", "idle", 103, "shutdown"),
    EvalCase("eval-nonprod-idle-02", "eval-nonprod-idle-2", "m5.xlarge", "non-production", "idle", 104, "shutdown"),
    EvalCase("eval-prod-steady-01", "eval-prod-steady", "m5.2xlarge", "production", "steady_high", 105, "no_action"),
    EvalCase("eval-nonprod-steady-01", "eval-nonprod-steady", "t3.large", "non-production", "steady_high", 106, "no_action"),
    EvalCase("eval-prod-business-01", "eval-prod-business", "t3.medium", "production", "business_hours", 107, "no_action"),
    EvalCase("eval-nonprod-business-01", "eval-nonprod-business", "t3.small", "non-production", "business_hours", 108, "no_action"),
]
