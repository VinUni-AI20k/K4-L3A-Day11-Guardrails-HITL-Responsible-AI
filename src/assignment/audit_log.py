"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    def record_input(
        self,
        *,
        user_id: str,
        text: str,
        request_id: str | None = None,
    ):
        import time

        key = request_id or user_id

        self._open[key] = time.time()

        self._pending[key] = {
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        status: str | None = None,
        error: str | None = None,
        redacted: bool = False,
    ):
        import time

        key = request_id or user_id

        started = self._open.pop(key, None)

        latency_ms = (
            (time.time() - started) * 1000
            if started is not None
            else None
        )

        input_data = self._pending.pop(
            key,
            {
                "request_id": request_id,
                "user_id": user_id,
                "input": None,
                "started_at": None,
            },
        )

        log_entry = {
            **input_data,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "finished_at": utc_now_iso(),
            "status": status or ("blocked" if blocked else "ok"),
            "error": error,
            "redacted": redacted,
            "latency_ms": latency_ms,
        }

        self.logs.append(log_entry)

        return log_entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk."""

        path = Path(filepath or default_audit_log_path())

        # Đảm bảo outputs/ tồn tại
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                self.logs,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
