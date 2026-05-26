"""Loader for the five-level ER triage source of truth.

This module reads ``app/data/er_triage_five_level_system.json`` once at first
access, validates the basic structure, and exposes small helpers for the rest
of the backend (the ADK agent, prompt builders, severity mapping, etc.).

The JSON is the authoritative triage schema approved by the medical team:
five levels (Red, Orange, Yellow, Green, Blue) with key questions, examples,
response time, and placement guidance. **Do not edit it from code.** If the
schema needs to evolve, update the JSON file and bump any downstream callers.

Design notes:

* Uses only :mod:`pathlib`, :mod:`json`, :mod:`copy`, and :mod:`functools`
  from the standard library — no third-party deps.
* Caches the parsed JSON via ``functools.lru_cache`` so repeated calls are
  cheap, but every public helper returns a deep copy so callers cannot
  mutate the cached config.
* Raises :class:`TriageConfigError` with a clear message when the file is
  missing or structurally invalid, so a misconfigured deploy fails fast at
  startup instead of producing silently wrong triage decisions.
"""

from __future__ import annotations

import copy
import json
import pathlib
from functools import lru_cache
from typing import Any

DATA_PATH: pathlib.Path = (
    pathlib.Path(__file__).resolve().parent.parent
    / "data"
    / "er_triage_five_level_system.json"
)

EXPECTED_LEVELS: frozenset[int] = frozenset({1, 2, 3, 4, 5})


class TriageConfigError(RuntimeError):
    """Raised when the triage JSON is missing, unreadable, or malformed."""


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, Any]:
    """Read, parse, and structurally validate the triage JSON.

    Returns the raw ``dict`` from disk. Callers must not mutate the result —
    use the public helpers in this module, which deep-copy before returning.
    """

    if not DATA_PATH.exists():
        raise TriageConfigError(
            f"Triage configuration file not found at {DATA_PATH}. "
            "Ensure er_triage_five_level_system.json is present in app/data/."
        )

    try:
        text = DATA_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise TriageConfigError(
            f"Triage configuration file at {DATA_PATH} could not be read: {exc}"
        ) from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TriageConfigError(
            f"Triage configuration file at {DATA_PATH} is malformed JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc

    if not isinstance(data, dict):
        raise TriageConfigError(
            "Triage configuration root must be a JSON object."
        )

    five = data.get("five_level_triage")
    if not isinstance(five, dict):
        raise TriageConfigError(
            "Triage configuration missing 'five_level_triage' object."
        )

    levels = five.get("triage_levels")
    if not isinstance(levels, list):
        raise TriageConfigError(
            "Triage configuration 'triage_levels' must be a list."
        )

    seen_levels: set[int] = set()
    required_fields = ("level", "color", "key_question", "examples", "response_time", "placement")
    for index, item in enumerate(levels):
        if not isinstance(item, dict):
            raise TriageConfigError(
                f"triage_levels[{index}] must be an object."
            )
        missing = [field for field in required_fields if field not in item]
        if missing:
            raise TriageConfigError(
                f"triage_levels[{index}] is missing required fields: {missing}."
            )
        level_value = item.get("level")
        if not isinstance(level_value, int):
            raise TriageConfigError(
                f"triage_levels[{index}].level must be an integer, got {type(level_value).__name__}."
            )
        seen_levels.add(level_value)

    if seen_levels != EXPECTED_LEVELS:
        raise TriageConfigError(
            f"Triage levels must be exactly {sorted(EXPECTED_LEVELS)}, got {sorted(seen_levels)}."
        )

    return data


def get_triage_config() -> dict[str, Any]:
    """Return a deep copy of the full triage configuration dictionary."""

    return copy.deepcopy(_load_raw())


def get_triage_levels() -> list[dict[str, Any]]:
    """Return the five triage level definitions, sorted 1 -> 5.

    Each item is a deep copy with keys: ``level``, ``color``, ``key_question``,
    ``examples``, ``response_time``, ``placement``.
    """

    levels = _load_raw()["five_level_triage"]["triage_levels"]
    ordered = sorted(levels, key=lambda item: int(item["level"]))
    return copy.deepcopy(ordered)


def get_triage_level(level: int) -> dict[str, Any]:
    """Return the definition for a single triage level (1..5).

    :raises ValueError: when ``level`` is not an integer in 1..5.
    :raises TriageConfigError: when the JSON omits the requested level
        (should not happen because :func:`_load_raw` validates this).
    """

    if not isinstance(level, int) or isinstance(level, bool) or level not in EXPECTED_LEVELS:
        raise ValueError(f"Triage level must be one of {sorted(EXPECTED_LEVELS)}, got {level!r}.")

    for item in get_triage_levels():
        if int(item["level"]) == level:
            return item
    raise TriageConfigError(f"Triage level {level} not found in configuration.")


def get_examples_for_level(level: int) -> list[str]:
    """Return the list of example presentations for a given level."""

    item = get_triage_level(level)
    examples = item.get("examples") or []
    if not isinstance(examples, list):
        raise TriageConfigError(
            f"'examples' for triage level {level} must be a list, got {type(examples).__name__}."
        )
    return [str(example) for example in examples]


def get_team_composition() -> list[dict[str, Any]]:
    """Return the ER triage team composition definition (deep-copied).

    Returns an empty list if the configuration does not declare a team
    composition section.
    """

    composition = _load_raw()["five_level_triage"].get("er_triage_team_composition")
    if composition is None:
        return []
    if not isinstance(composition, list):
        raise TriageConfigError(
            "'er_triage_team_composition' must be a list when present."
        )
    return copy.deepcopy(composition)
