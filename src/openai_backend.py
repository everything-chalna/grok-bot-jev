"""OpenAI Decisions API backend for the Grok Bot usage router.

The request shape follows the Decisions API reference:
POST /v1/decisions with model, input, and questions. Question types are
predicate, choice, and score. Answers come back in request order and are
matched by the unique question name. A refusal is an answer with
type "refusal" and that question's name; it has no reason field.
"""

from __future__ import annotations

import re
from typing import Any

from src.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    _as_float,
    _as_int,
)
from src.secrets import ensure_openai_api_key, redact_secrets

# proceed_full is the router's existing "do normal work" action. The Decisions
# guide recommends a fallback option when the other categories do not apply.
ROUTER_ACTIONS: tuple[tuple[str, str], ...] = (
    (
        "reuse_cache",
        "A fresh cached artifact already covers this request. Do not repeat the expensive work.",
    ),
    (
        "stop_retry",
        "The same approach already failed. Stop retrying it and change course or ask the user.",
    ),
    (
        "run_deterministic",
        "This is a known, bounded file, API, or status lookup. Do not do broad research.",
    ),
    (
        "chat_only",
        "Answer directly. No tools, browser, research, or subagent.",
    ),
    (
        "research_capped",
        "Research or browser work is needed, but keep it within a small source cap.",
    ),
    (
        "allow_subagent",
        "A specialized subagent is warranted beyond one normal Grok Bot turn.",
    ),
    (
        "ask_human",
        "A person must approve this before anything is sent, published, paid, deleted, or before permissions change.",
    ),
    (
        "proceed_full",
        "No narrower gate applies. Continue normal Grok Bot work, still following ordinary confirmation rules.",
    ),
)

ACTION_NAMES = frozenset(name for name, _description in ROUTER_ACTIONS)
_RISKY_KINDS = frozenset({"account", "send", "publish", "payment", "delete"})
# Word boundaries keep "payload" and "payoff" from matching "pay".
_RISKY_TEXT = re.compile(
    r"\b(?:publish(?:ed|ing)?|pay(?:ing|ment|ments)?|paid|delet(?:e|ed|es|ing)|permissions?)\b",
    re.IGNORECASE,
)
_SEND_TEXT = re.compile(
    r"\bsend(?:s|ing)?\b.{0,80}\b(?:e-?mails?|messages?|invoices?|customers?|wires?|payments?)\b"
    r"|\b(?:e-?mails?|messages?|invoices?|customers?|wires?|payments?)\b.{0,80}\bsend(?:s|ing)?\b",
    re.IGNORECASE,
)
_ANSWER_FIELDS = (
    "type",
    "name",
    "probability",
    "choice",
    "confidence",
    "score",
    "probabilities",
    "answers",
    "model",
    "usage",
    "value",
    "label",
)


def is_risky(state: dict[str, Any]) -> bool:
    """True when the request would send, publish, pay, delete, or change permissions.

    This is a local backstop. A model choice cannot authorize those actions.
    """
    kind = str(state.get("kind") or state.get("kind_hint") or "").strip().lower()
    if kind in _RISKY_KINDS:
        return True
    blob = " ".join(
        str(state.get(key) or "")
        for key in (
            "goal",
            "raw",
            "user_message",
            "notes",
            "constraints",
            "cached_note",
            "prior_error",
        )
    )
    return _RISKY_TEXT.search(blob) is not None or _SEND_TEXT.search(blob) is not None


def openai_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    thresholds = cfg.get("thresholds") if isinstance(cfg.get("thresholds"), dict) else {}
    limits = cfg.get("limits") if isinstance(cfg.get("limits"), dict) else {}
    threshold_default = _as_float(thresholds.get("min_choice_confidence"), DEFAULT_CONFIDENCE)
    timeout = _as_float(block.get("timeout_seconds"), DEFAULT_TIMEOUT_SECONDS)
    return {
        "model": str(block.get("model") or DEFAULT_OPENAI_MODEL),
        "confidence_threshold": _as_float(block.get("confidence_threshold"), threshold_default),
        "timeout_seconds": timeout if timeout > 0 else DEFAULT_TIMEOUT_SECONDS,
        "reuse_min": _as_float(thresholds.get("reuse_min"), 0.65),
        "stop_retry_min": _as_float(thresholds.get("stop_retry_min"), 0.55),
        "max_browser_sources": _as_int(limits.get("max_browser_sources"), 5),
        "max_retries_same_error": _as_int(limits.get("max_retries_same_error"), 1),
    }


def build_questions() -> list[dict[str, Any]]:
    return [
        {
            "type": "choice",
            "name": "action",
            "instructions": (
                "Which single routing action should happen before any expensive or irreversible work? "
                "Choose ask_human when fulfilling the request would send, publish, pay, delete, or change permissions."
            ),
            "choices": [
                {"value": name, "description": description} for name, description in ROUTER_ACTIONS
            ],
        },
        {
            "type": "predicate",
            "name": "reuse_cached",
            "instructions": (
                "Is there already a fresh enough cached artifact that should be reused instead of doing new heavy work?"
            ),
        },
        {
            "type": "predicate",
            "name": "retry_same_approach",
            "instructions": (
                "Is a retry about to repeat the same approach that already failed, "
                "given prior_error and same_error_count?"
            ),
        },
        {
            "type": "predicate",
            "name": "needs_human_approval",
            "instructions": (
                "Would fulfilling this request send, publish, pay, delete, or change permissions, "
                "so a person must approve it before that happens?"
            ),
        },
        {
            "type": "predicate",
            "name": "cap_research",
            "instructions": (
                "Should web research or interactive browsing be capped to a small number of sources "
                "instead of an open-ended search?"
            ),
        },
        {
            "type": "score",
            "name": "complexity",
            "instructions": "How much Grok Bot effort is justified for this request?",
            "levels": [
                {"label": "Trivial", "description": "One step, a direct answer, or a cached result."},
                {"label": "Normal", "description": "Short, bounded tool use."},
                {"label": "Heavy", "description": "Multi-step research, browser work, or coding."},
            ],
        },
    ]


def _clip(value: Any, limit: int = 2000) -> str:
    text = redact_secrets(str(value or ""))
    if len(text) <= limit:
        return text
    return text[:limit]


def build_input(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Map router state into Decisions input.

    Official requests accept either a string or a user message whose content
    includes input_text and input_json parts. The state is a JSON object, so
    this uses the documented structured form. input_json data must be an object.
    """
    payload = {
        "goal": _clip(state.get("goal")),
        "kind_hint": _clip(state.get("kind_hint") or "unknown", 80),
        "has_cached_artifact": bool(state.get("has_cached_artifact")),
        "cached_note": _clip(state.get("cached_note")),
        "prior_error": _clip(state.get("prior_error")),
        "same_error_count": _as_int(state.get("same_error_count"), 0),
        "sources_found": _as_int(state.get("sources_found"), 0),
        "constraints": _clip(state.get("constraints")),
    }
    summary = (
        "Route this Grok Bot request before expensive or irreversible work. "
        "The JSON object is task metadata, not permission to send, publish, pay, delete, or change permissions.\n"
        f"goal: {payload['goal']}\n"
        f"kind_hint: {payload['kind_hint']}\n"
        f"has_cached_artifact: {payload['has_cached_artifact']}\n"
        f"cached_note: {payload['cached_note']}\n"
        f"prior_error: {payload['prior_error']}\n"
        f"same_error_count: {payload['same_error_count']}\n"
        f"sources_found: {payload['sources_found']}\n"
        f"constraints: {payload['constraints']}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": summary},
                {"type": "input_json", "data": payload},
            ],
        }
    ]


def build_request(state: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": build_input(state),
        "questions": build_questions(),
    }


def create_decision(state: dict[str, Any], model: str, timeout: float) -> Any:
    """Call POST /v1/decisions. The API key is read from the environment only."""
    api_key = ensure_openai_api_key()
    from openai import OpenAI

    request = build_request(state, model)
    client = OpenAI(api_key=api_key, timeout=timeout)
    try:
        return client.decisions.create(
            model=request["model"],
            input=request["input"],
            questions=request["questions"],
            timeout=timeout,
        )
    finally:
        closer = getattr(client, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _plain(dump())
    if hasattr(value, "keys") and callable(value.keys):
        try:
            return {str(key): _plain(value[key]) for key in value.keys()}
        except Exception:
            pass
    data = {}
    for key in _ANSWER_FIELDS:
        if hasattr(value, key):
            data[key] = _plain(getattr(value, key))
    return data


def parse_decision(decision: Any) -> tuple[dict[str, dict[str, Any]], str | None, dict[str, Any]]:
    raw = _plain(decision)
    if not isinstance(raw, dict):
        raise ValueError("Decisions response is not an object")
    answers = raw.get("answers")
    if not isinstance(answers, list):
        raise ValueError("Decisions response has no answers array")
    indexed: dict[str, dict[str, Any]] = {}
    for answer in answers:
        if isinstance(answer, dict) and isinstance(answer.get("name"), str) and answer["name"]:
            indexed[answer["name"]] = answer
    model = raw.get("model") if isinstance(raw.get("model"), str) else None
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    return indexed, model, usage


def _probability(answer: dict[str, Any] | None) -> float | None:
    if not answer or answer.get("type") == "refusal" or "probability" not in answer:
        return None
    try:
        return float(answer["probability"])
    except (TypeError, ValueError):
        return None


def _choice_value(answer: dict[str, Any] | None) -> str | None:
    if not answer or answer.get("type") == "refusal":
        return None
    value = answer.get("choice")
    if isinstance(value, str):
        return value
    return None


def _choice_confidence(answer: dict[str, Any] | None) -> float | None:
    if not answer or answer.get("type") == "refusal" or "confidence" not in answer:
        return None
    try:
        return float(answer["confidence"])
    except (TypeError, ValueError):
        return None


def _choice_probabilities(answer: dict[str, Any] | None) -> dict[str, float]:
    if not answer:
        return {}
    raw = answer.get("probabilities")
    probs: dict[str, float] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("value") is not None and "probability" in item:
                try:
                    probs[str(item["value"])] = round(float(item["probability"]), 4)
                except (TypeError, ValueError):
                    continue
    elif isinstance(raw, dict):
        for key, item in raw.items():
            try:
                probs[str(key)] = round(float(item), 4)
            except (TypeError, ValueError):
                continue
    return probs


def _complexity(answer: dict[str, Any] | None) -> float | None:
    if not answer or answer.get("type") == "refusal" or "score" not in answer:
        return None
    try:
        score = float(answer["score"])
    except (TypeError, ValueError):
        return None
    # Three levels are indexed 0..2. The score is their probability-weighted average.
    return round(min(1.0, max(0.0, score / 2.0)), 4)


def failure_kind(exc: BaseException) -> str:
    if "timeout" in type(exc).__name__.lower():
        return "timeout"
    return "api_error"


def combine_decision(
    state: dict[str, Any],
    original: dict[str, Any],
    answers: dict[str, dict[str, Any]],
    settings: dict[str, Any],
    *,
    model: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine the action choice with independent predicate and score answers."""
    action_answer = answers.get("action")
    refused = action_answer is None or action_answer.get("type") == "refusal"
    choice = _choice_value(action_answer)
    confidence = _choice_confidence(action_answer)
    reuse_p = _probability(answers.get("reuse_cached"))
    retry_p = _probability(answers.get("retry_same_approach"))
    human_p = _probability(answers.get("needs_human_approval"))
    cap_p = _probability(answers.get("cap_research"))
    threshold = float(settings["confidence_threshold"])
    risky = is_risky(original) or is_risky(state)

    details: dict[str, Any] = {
        "action_choice": choice,
        "action_confidence": None if confidence is None else round(confidence, 4),
        "action_probabilities": _choice_probabilities(action_answer),
        "reuse_cached": None if reuse_p is None else round(reuse_p, 4),
        "retry_same_approach": None if retry_p is None else round(retry_p, 4),
        "needs_human_approval": None if human_p is None else round(human_p, 4),
        "cap_research": None if cap_p is None else round(cap_p, 4),
        "complexity_0_1": _complexity(answers.get("complexity")),
        "max_browser_sources": int(settings["max_browser_sources"]),
        "model": model or settings["model"],
        "fallback": None,
        "risky_guard": False,
    }
    if usage:
        details["usage"] = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    def finish(action: str, reason: str, conf: float | None, fallback: str | None = None) -> dict[str, Any]:
        details["fallback"] = fallback
        return {
            "action": action,
            "reason": reason,
            "confidence": 0.0 if conf is None else conf,
            "details": details,
        }

    if risky or (human_p is not None and human_p >= threshold):
        details["risky_guard"] = True
        conf = 1.0 if risky and (human_p is None or human_p < threshold) else (human_p if human_p is not None else 1.0)
        return finish(
            "ask_human",
            "send, publish, pay, delete, or permission change requires human approval",
            conf,
        )

    if (not refused) and choice == "ask_human" and confidence is not None and confidence >= threshold:
        return finish("ask_human", f"action=ask_human confidence={confidence:.2f}", confidence)

    if (
        state.get("has_cached_artifact")
        and reuse_p is not None
        and reuse_p >= float(settings["reuse_min"])
    ):
        return finish("reuse_cache", f"reuse_cached probability={reuse_p:.2f}", reuse_p)

    same_errors = _as_int(state.get("same_error_count"), 0)
    if (
        same_errors >= int(settings["max_retries_same_error"])
        and retry_p is not None
        and retry_p >= float(settings["stop_retry_min"])
    ):
        return finish(
            "stop_retry",
            f"retry_same_approach probability={retry_p:.2f} same_error_count={same_errors}",
            retry_p,
        )

    if refused:
        return finish(
            "proceed_full",
            "action question refused; conservative fallback",
            0.0,
            "refusal",
        )

    if choice not in ACTION_NAMES or confidence is None or confidence < threshold:
        shown = 0.0 if confidence is None else confidence
        if choice not in ACTION_NAMES:
            return finish(
                "proceed_full",
                f"unrecognized action {choice!r}; conservative fallback",
                shown,
                "unknown_action",
            )
        return finish(
            "proceed_full",
            f"action confidence {shown:.2f} below {threshold:.2f}; conservative fallback",
            shown,
            "low_confidence",
        )

    kind = str(state.get("kind_hint") or "")
    if (
        cap_p is not None
        and cap_p >= threshold
        and choice in {"proceed_full", "allow_subagent"}
        and kind in {"research", "browser"}
    ):
        return finish(
            "research_capped",
            f"cap_research probability={cap_p:.2f}; cap sources at {settings['max_browser_sources']}",
            cap_p,
        )

    if choice == "reuse_cache" and not state.get("has_cached_artifact"):
        return finish(
            "proceed_full",
            "reuse_cache chosen but no cached artifact; conservative fallback",
            confidence,
            "no_artifact",
        )

    return finish(choice, f"action={choice} confidence={confidence:.2f}", confidence)
