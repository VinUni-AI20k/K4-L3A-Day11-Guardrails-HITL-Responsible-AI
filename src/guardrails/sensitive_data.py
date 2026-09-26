"""Shared deterministic Blue output/egress checks. No LLM calls."""
from __future__ import annotations

import re
import unicodedata

from core.config import DEMO_SECRETS, load_protected_payload


def normalize_text(text: str, *, fold_accents: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    # Remove format controls (zero-width spaces, soft hyphens, bidi controls).
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    if fold_accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")
        text = text.replace("đ", "d").replace("Đ", "D")
    return text


PATTERNS = {
    "phone": r"(?<!\w)(?:\+84|84|0)[ .-]?[35789](?:[ .-]?\d){8}(?!\w)",
    "email": r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b",
    "national_id": r"(?<!\w)(?:\d{9}|\d{12})(?!\w)",
    "api_key": r"\bsk-[a-zA-Z0-9_-]+\b",
    "password": r"\b(?:password|mat\s+khau|mật\s+khẩu)\s*[\"']?\s*(?:is\b|la\b|là\b|[:=])\s*[\"']?[^\s\"',;}]+",
    "db_host": r"\b[\w.-]+\.internal(?::\d+)?\b",
}


def filter_sensitive_data(text: str) -> dict:
    redacted = normalize_text(text)
    issues = []
    # Use protected values as well as aliases; bare or JSON values must be caught.
    values = set(DEMO_SECRETS)
    values.update(str(v) for v in load_protected_payload()["secrets"].values() if v)
    for secret in sorted(values, key=len, reverse=True):
        pattern = re.escape(normalize_text(secret))
        redacted, count = re.subn(pattern, "[REDACTED]", redacted, flags=re.I)
        if count:
            issues.append(f"protected_secret: {count} found")
    for name, pattern in PATTERNS.items():
        redacted, count = re.subn(pattern, "[REDACTED]", redacted, flags=re.I)
        if count:
            issues.append(f"{name}: {count} found")
    return {"safe": not issues, "issues": issues, "redacted": redacted}
