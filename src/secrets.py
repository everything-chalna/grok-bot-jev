from __future__ import annotations

import os

_SECRET_ENV_NAMES = ("OPENAI_API_KEY", "TYPESAFE_API_KEY")


def ensure_typesafe_api_key() -> str:
    """Require the public package's only supported TypeSafe credential source."""
    try:
        return os.environ["TYPESAFE_API_KEY"]
    except KeyError as exc:
        raise SystemExit(
            "TYPESAFE_API_KEY is missing; export it in the process environment"
        ) from exc


def ensure_openai_api_key() -> str:
    """Require an OpenAI Platform API key. ChatGPT or Codex logins are not accepted."""
    try:
        return os.environ["OPENAI_API_KEY"]
    except KeyError as exc:
        raise SystemExit(
            "OPENAI_API_KEY is missing; export an OpenAI Platform API key in the process environment"
        ) from exc


def redact_secrets(text: str) -> str:
    """Remove known credential values before logging or sending task text."""
    redacted = text
    for name in _SECRET_ENV_NAMES:
        secret = os.environ.get(name) or ""
        if len(secret) >= 8 and secret in redacted:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted
