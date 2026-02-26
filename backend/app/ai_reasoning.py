from __future__ import annotations

import json
import os
from typing import Any
from urllib import request


def build_reasoning_prompt(
    city: str,
    overall_score: float,
    overall_level: str,
    systems: dict[str, dict[str, Any]],
    cascading_failure_risks: list[str],
    confidence_score: int,
    missing_fields: list[str],
) -> str:
    system_blob = json.dumps(systems, separators=(",", ":"))
    cascade_blob = json.dumps(cascading_failure_risks, separators=(",", ":"))
    missing_blob = json.dumps(missing_fields, separators=(",", ":"))
    return (
        "Generate an executive city infrastructure risk briefing.\n"
        f"city={city}\n"
        f"overall_score={overall_score:.1f}\n"
        f"overall_level={overall_level}\n"
        f"systems={system_blob}\n"
        f"cascades={cascade_blob}\n"
        f"confidence_score={confidence_score}\n"
        f"missing_fields={missing_blob}\n"
        "Required style: concise, technical, actionable, no disclaimers."
    )


def _external_reasoning(prompt: str) -> str | None:
    mode = os.getenv("GRIDMIND_REASONING_MODE", "template").strip().lower()
    if mode != "external":
        return None

    endpoint = os.getenv("GRIDMIND_REASONING_URL")
    if not endpoint:
        return None

    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=6) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body.get("summary")
        if isinstance(content, str) and content.strip():
            return content.strip()
    except Exception:
        return None
    return None


def generate_executive_summary(
    city: str,
    overall_score: float,
    overall_level: str,
    systems: dict[str, dict[str, Any]],
    cascading_failure_risks: list[str],
    confidence_score: int,
    missing_fields: list[str],
) -> str:
    prompt = build_reasoning_prompt(
        city=city,
        overall_score=overall_score,
        overall_level=overall_level,
        systems=systems,
        cascading_failure_risks=cascading_failure_risks,
        confidence_score=confidence_score,
        missing_fields=missing_fields,
    )
    llm_summary = _external_reasoning(prompt)
    if llm_summary:
        return llm_summary

    ranked = sorted(
        systems.items(),
        key=lambda x: x[1]["risk_score"],
        reverse=True,
    )
    top_system, top_data = ranked[0]
    second_system, second_data = ranked[1]

    uncertainty_clause = ""
    if confidence_score < 70:
        uncertainty_clause = (
            f" Assessment confidence is reduced ({confidence_score}/100) due to incomplete "
            f"or low-signal inputs: {', '.join(missing_fields) if missing_fields else 'trend sparsity'}."
        )

    cascade_clause = (
        " Cascading pathways currently include "
        + "; ".join(cascading_failure_risks[:2])
        + "."
        if cascading_failure_risks
        else " Cascading pathways are currently limited based on available indicators."
    )

    return (
        f"{city} is operating at {overall_level} infrastructure risk ({overall_score:.1f}/100). "
        f"Primary pressure is in {top_system.replace('_', ' ')} ({top_data['risk_score']:.1f}) "
        f"followed by {second_system.replace('_', ' ')} ({second_data['risk_score']:.1f})."
        f"{cascade_clause} Prioritize preventive actions in the next operational window for these systems."
        f"{uncertainty_clause}"
    )

