"""Demo of the secure PII-collection signalling tools."""

import json

from app.agent.tools.pii_tools import (
    PII_FIELDS,
    acknowledge_pii_completion,
    trigger_pii_collection,
)
from app.agent.tools import get_triage_tools


def show(title: str, payload: dict) -> None:
    print(f"--- {title} ---")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()


print(f"PII_FIELDS = {PII_FIELDS}")
print()

show(
    "trigger_pii_collection (happy path)",
    trigger_pii_collection("3b8c9a44-7c30-4cd9-a4b1-89c2c8b3a1ee"),
)

show(
    "trigger_pii_collection (invalid session_id)",
    trigger_pii_collection(""),
)

show(
    "acknowledge_pii_completion (completed)",
    acknowledge_pii_completion(
        "3b8c9a44-7c30-4cd9-a4b1-89c2c8b3a1ee",
        outcome="completed",
        note="handed off to staff in person",
    ),
)

show(
    "acknowledge_pii_completion (declined)",
    acknowledge_pii_completion(
        "3b8c9a44-7c30-4cd9-a4b1-89c2c8b3a1ee",
        outcome="declined",
    ),
)

print("--- agent tool registry ---")
print([fn.__name__ for fn in get_triage_tools()])
