# -*- coding: utf-8 -*-
"""
Agentic RAG 包。

Phase 0/1 已落地：显式 State（state.py）。
后续模块（router / evaluator / controller / executor / stopping / synthesizer / planner）
将围绕 AgentState 逐步填充。
"""

from .state import (
    AgentState,
    Requirement,
    Evidence,
    EvidenceSource,
    RequirementStatusItem,
    Gap,
    Action,
    Query,
    Budget,
    RequirementStatus,
    ActionType,
    RetrievalTool,
    IMPORTANCE_HIGH,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_LATENCY_MS,
)
from .retriever import retrieve, route
from .evaluator import evaluate
from .controller import choose_action
from .executor import execute
from .stopping import should_stop, update_no_progress
from .synthesizer import synthesize, collect_validated
from .router import classify_complexity
from .planner import plan
from .agent import run_agentic

__all__ = [
    "AgentState",
    "Requirement",
    "Evidence",
    "EvidenceSource",
    "RequirementStatusItem",
    "Gap",
    "Action",
    "Query",
    "Budget",
    "RequirementStatus",
    "ActionType",
    "RetrievalTool",
    "IMPORTANCE_HIGH",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_MAX_LATENCY_MS",
    "retrieve",
    "route",
    "evaluate",
    "choose_action",
    "execute",
    "should_stop",
    "update_no_progress",
    "synthesize",
    "collect_validated",
    "classify_complexity",
    "plan",
    "run_agentic",
]
