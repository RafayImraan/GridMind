from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from .models import RiskAssessmentRequest, RiskAssessmentResponse


_WRITE_LOCK = Lock()


def _storage_path() -> Path:
    configured = os.getenv("GRIDMIND_STORAGE_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    backend_root = Path(__file__).resolve().parents[1]
    return backend_root / "storage" / "assessments.jsonl"


def save_assessment(
    request_payload: RiskAssessmentRequest,
    response_payload: RiskAssessmentResponse,
) -> str:
    path = _storage_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    record_id = f"asm_{uuid4().hex[:12]}"
    record = {
        "assessment_id": record_id,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "request": request_payload.model_dump(mode="json"),
        "response": response_payload.model_dump(mode="json", by_alias=True),
    }

    line = json.dumps(record, separators=(",", ":"))
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return record_id


def list_recent_assessments(limit: int = 20) -> list[dict]:
    path = _storage_path()
    if not path.exists():
        return []

    with _WRITE_LOCK:
        lines = path.read_text(encoding="utf-8").splitlines()

    records: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return list(reversed(records[-limit:]))

