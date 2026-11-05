"""Structured logging configuration for Cross Section Studio."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import UTC, datetime

# Applied in order. Later patterns skip a value already "[redacted]" so a
# rule never re-redacts (and swallows) text an earlier rule left alone.
_NOT_REDACTED = r"(?!\[redacted\])"
_SCHEMES = r"(?:bearer|basic|digest|token)"
# Quoted value: a backslash escape or any other character except the quote.
# The two branches never match the same text, so a missing closing quote
# fails in linear time instead of backtracking exponentially.
_QUOTED = r"""(["'])(?:\\.|(?!\3)[^\\\n])*\3"""
_SECRET_PATTERNS = (
    # Query parameters, with or without the leading "?" (Sentry passes a bare
    # query_string): ?access_token=...&state=ok keeps the other params.
    (
        re.compile(
            r"(?i)(^|[?&#])((?:access_|refresh_|id_)?token|api[_-]?key|key|password|passwd"
            r"|client_secret|secret)=" + _NOT_REDACTED + r"""(?!["'])([^&\s#]+)"""
        ),
        r"\1\2=[redacted]",
    ),
    # JSON / dict-repr fields: {"api_key": "x"} -> {"api_key": "[redacted]"}.
    (
        re.compile(
            r"""(?i)((["'])(?:[\w-]*(?:api[_-]?key|password|passwd|secret|token)|authorization)\2"""
            r"""\s*:\s*)""" + _QUOTED
        ),
        r"\1\3[redacted]\3",
    ),
    # Authorization header, scheme kept: "Authorization: Bearer [redacted]".
    # The lookahead keeps a second pass from eating the scheme word.
    (
        re.compile(
            r"(?i)\b(authorization\s*[=:]\s*)"
            r"(?!(?:" + _SCHEMES + r"\s+)?\[redacted\])"
            r"((?:" + _SCHEMES + r"\s+)?)\S+"
        ),
        r"\1\2[redacted]",
    ),
    # Bare bearer token, including JWT dots and base64 + / = characters.
    (re.compile(r"(?i)\b(bearer\s*:?\s+)" + _NOT_REDACTED + r"[A-Za-z0-9._~+/\-]+=*"), r"\1[redacted]"),
    # Env-style assignments: CROSS_SECTION_AUTH_PASSWORD=..., GROQ_API_KEY=...
    # Data-structure keys (SORT_KEY, cache_key, primary_key, ...) are not secrets.
    (
        re.compile(
            r"(?i)\b(?![A-Za-z0-9_]*(?:sort|cache|primary|foreign|partition|group|lookup|index"
            r"|unique|composite|row|column|hash|dict|map|join|order|dedupe|state|widget)_key\b)"
            r"([A-Za-z0-9_]*_(?:password|passwd|token|secret|key))(\s*=\s*)"
            + _NOT_REDACTED
            + r"""("[^"\n]*"|'[^'\n]*'|\S+)"""
        ),
        r"\1\2[redacted]",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|password|passwd|secret|token)\s*[=:]\s*"
            + _NOT_REDACTED
            + r"""("[^"\n]*"|'[^'\n]*'|\S+)"""
        ),
        r"\1=[redacted]",
    ),
    # Credentials in a URL: https://user:pass@host -> https://user:[redacted]@host.
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s:/@]+:)" + _NOT_REDACTED + r"[^\s@/]+@"), r"\1[redacted]@"),
    # Cookie headers carry session tokens; drop the whole value.
    (re.compile(r"(?i)\b((?:set-)?cookie\s*:(?!\s*\[redacted\])\s*)[^\r\n]+"), r"\1[redacted]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "sk-[redacted]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{10,}"), "AIza[redacted]"),
    (re.compile(r"\bgsk_[A-Za-z0-9]{10,}"), "gsk_[redacted]"),
    (re.compile(r"(?i)(x-goog-api-key\s*[:=]\s*)\S+"), r"\1[redacted]"),
)


def redact_secrets(text: str) -> str:
    """Best-effort scrub of credentials from log / error text."""
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        exc_text = getattr(record, "exc_text", None)
        if isinstance(exc_text, str) and exc_text:
            scrubbed = redact_secrets(exc_text)
            if scrubbed != exc_text:
                record.exc_text = scrubbed
        return True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(record.getMessage()),
        }
        if record.exc_info:
            payload["exc_info"] = redact_secrets(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    level_name = os.environ.get("CROSS_SECTION_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        if not any(isinstance(item, _RedactingFilter) for item in root.filters):
            root.addFilter(_RedactingFilter())
        return
    handler = logging.StreamHandler(sys.stdout)
    if os.environ.get("CROSS_SECTION_LOG_FORMAT", "").lower() == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    handler.addFilter(_RedactingFilter())
    root.addHandler(handler)
    root.addFilter(_RedactingFilter())
    root.setLevel(level)
