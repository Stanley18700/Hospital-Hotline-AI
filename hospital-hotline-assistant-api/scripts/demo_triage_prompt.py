"""Concise preview of the triage system prompt.

Prints:
1. The section headers + first content line of each section.
2. A coverage check: which spec requirements are satisfied by which
   text in the prompt.
3. The full character / line counts and the voice-mode delta.
"""

from __future__ import annotations

import re

from app.agent.prompts.triage_system import (
    TRIAGE_SYSTEM_PROMPT,
    build_triage_system_prompt,
)


def print_section_outline(prompt: str) -> None:
    print("=" * 72)
    print("PROMPT SECTION OUTLINE (text mode)")
    print("=" * 72)
    lines = prompt.splitlines()
    current_header: str | None = None
    first_line_shown = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_header = stripped
            print()
            print(current_header)
            first_line_shown = False
            continue
        if current_header and stripped and not first_line_shown:
            preview = stripped if len(stripped) <= 80 else stripped[:77] + "..."
            print(f"  {preview}")
            first_line_shown = True


def coverage_check(prompt: str) -> None:
    print()
    print("=" * 72)
    print("REQUIREMENT COVERAGE")
    print("=" * 72)
    checks: list[tuple[str, str]] = [
        ("introduce itself", r"\bintroduce yourself\b"),
        ("ask for symptoms", r"\bask for symptoms\b"),
        ("ask one follow-up at a time", r"\bone follow-up question at a time\b"),
        ("classify with 5-level JSON", r"\bclassify the case\b.+\bfive-level ladder\b"),
        ("interim advice + department", r"\binterim advice and department guidance\b"),
        ("trigger PII collection level 1 only", r"\btrigger secure pii collection only for level 1\b"),
        ("never ask for name/phone/address", r"\bname, phone number, address\b"),
        ("five-level block present", r"\bFIVE-LEVEL TRIAGE LADDER\b"),
        ("level + color rendered", r"Level\s+1\s+\(Red\)"),
        ("key_question rendered", r"Key question\s*:"),
        ("examples rendered", r"Examples\s*:"),
        ("response time rendered", r"Response time:"),
        ("placement rendered", r"Placement\s*:"),
        ("do not diagnose", r"\bdo not diagnose\b"),
        ("higher severity if uncertain", r"err on\s*\n?\s*the side of higher severity"),
        ("respond in patient language", r"\bin the language the patient is using\b"),
    ]
    for label, pattern in checks:
        ok = re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL) is not None
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}")


def summary(prompt_text: str, prompt_voice: str) -> None:
    print()
    print("=" * 72)
    print("SIZE / VARIANT SUMMARY")
    print("=" * 72)
    print(f"  text-mode length  : {len(prompt_text)} chars / {len(prompt_text.splitlines())} lines")
    print(f"  voice-mode length : {len(prompt_voice)} chars / {len(prompt_voice.splitlines())} lines")
    delta = len(prompt_voice) - len(prompt_text)
    print(f"  voice-mode delta  : +{delta} chars (= voice addendum)")


def main() -> None:
    print_section_outline(TRIAGE_SYSTEM_PROMPT)
    coverage_check(TRIAGE_SYSTEM_PROMPT)
    summary(TRIAGE_SYSTEM_PROMPT, build_triage_system_prompt(input_mode="voice"))


if __name__ == "__main__":
    main()
