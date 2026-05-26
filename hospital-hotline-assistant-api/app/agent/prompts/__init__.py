"""Prompt-building helpers for the triage agent.

Prompts live in their own subpackage so they can grow independently of
the ADK agent definition. The triage system prompt is built dynamically
from the five-level JSON source of truth.
"""

from app.agent.prompts.triage_system import (
    TRIAGE_SYSTEM_PROMPT,
    build_triage_system_prompt,
)

__all__ = ["TRIAGE_SYSTEM_PROMPT", "build_triage_system_prompt"]
