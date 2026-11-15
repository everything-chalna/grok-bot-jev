from __future__ import annotations

from typing import Any

from src import config as config_mod
from src.config import resolve_log_path
from src.logger import log_run
from src.openai_backend import failure_kind, is_risky

BYPASS_MARKERS = ("bypass jev", "bypass jev:", "bypass router", "no jev")


def _bypassed(state: dict[str, Any]) -> bool:
    raw = " ".join(
        str(state.get(key, ""))
        for key in ("goal", "raw", "user_message", "notes")
    ).lower()
    return any(marker in raw for marker in BYPASS_MARKERS)


def _clip(value: Any, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit]


def normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    same_error = state.get("same_error_count")
    sources = state.get("sources_found")
    try:
        same_error_count = int(same_error or 0)
    except (TypeError, ValueError):
        same_error_count = 0
    try:
        sources_found = int(sources or 0)
    except (TypeError, ValueError):
        sources_found = 0
    return {
        "goal": _clip(state.get("goal") or state.get("raw") or ""),
        "kind_hint": _clip(state.get("kind") or state.get("kind_hint") or "unknown", 80),
        "has_cached_artifact": bool(state.get("cached_artifact") or state.get("has_cached_artifact")),
        "cached_note": _clip(state.get("cached_note") or ""),
        "prior_error": _clip(state.get("prior_error") or ""),
        "same_error_count": same_error_count,
        "sources_found": sources_found,
        "constraints": _clip(state.get("constraints") or ""),
    }


def _emit(
    cfg: dict[str, Any],
    log_path: Any,
    state: dict[str, Any],
    *,
    action: str,
    reason: str,
    confidence: float | None,
    backend: str,
    jev_used: bool,
    decision_used: bool,
    details: dict[str, Any],
    include_policy: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "action": action,
        "reason": reason,
        "confidence": None if confidence is None else round(float(confidence), 4),
        "mode": cfg.get("mode"),
        "backend": backend,
        "jev_used": jev_used,
        "decision_used": decision_used,
        "details": details,
    }
    if include_policy:
        out["policy"] = {
            "honor_in_active_mode": True,
            "shadow_mode_is_advisory": cfg.get("mode") == "shadow",
        }
    log_run(
        log_path,
        {
            "event": "route",
            "goal": str(state.get("goal") or "")[:300],
            "action": action,
            "reason": reason,
            "confidence": out["confidence"],
            "mode": cfg.get("mode"),
            "backend": backend,
            "jev_used": jev_used,
            "decision_used": decision_used,
            "details": details,
        },
    )
    return out


def route_task(state: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify a Grok Bot request before expensive work."""
    cfg = config_mod.normalize_config(cfg) if cfg is not None else config_mod.load_config()
    log_path = resolve_log_path(cfg)
    backend = str(cfg.get("backend") or "typesafe")

    if not cfg.get("enabled", True) or _bypassed(state):
        return _emit(
            cfg,
            log_path,
            state,
            action="proceed_full",
            reason="router disabled or bypass marker",
            confidence=None,
            backend=backend,
            jev_used=False,
            decision_used=False,
            details={},
            include_policy=False,
        )

    if backend == "openai":
        return _route_openai(state, cfg, log_path)
    if backend == "typesafe":
        return _route_typesafe(state, cfg, log_path)
    return _emit(
        cfg,
        log_path,
        state,
        action="ask_human" if is_risky(state) else "proceed_full",
        reason=f"unknown backend {backend!r}; conservative fallback",
        confidence=0.0,
        backend=backend,
        jev_used=False,
        decision_used=False,
        details={"fallback": "unknown_backend"},
        include_policy=True,
    )


def _route_openai(state: dict[str, Any], cfg: dict[str, Any], log_path: Any) -> dict[str, Any]:
    from src import openai_backend

    normalized = normalize_state(state)
    settings = openai_backend.openai_settings(cfg)
    try:
        decision = openai_backend.create_decision(
            normalized,
            settings["model"],
            float(settings["timeout_seconds"]),
        )
        answers, model, usage = openai_backend.parse_decision(decision)
        combined = openai_backend.combine_decision(
            normalized,
            state,
            answers,
            settings,
            model=model,
            usage=usage,
        )
    except Exception as exc:
        # SystemExit (missing key) is a BaseException and is not swallowed.
        kind = failure_kind(exc)
        reason = (
            "OpenAI Decisions timeout; conservative fallback"
            if kind == "timeout"
            else "OpenAI Decisions request failed; conservative fallback"
        )
        if is_risky(state):
            reason = f"{reason}; risky action requires human approval"
        return _emit(
            cfg,
            log_path,
            state,
            action="ask_human" if is_risky(state) else "proceed_full",
            reason=reason,
            confidence=0.0,
            backend="openai",
            jev_used=False,
            decision_used=False,
            details={"fallback": kind, "error_type": type(exc).__name__, "model": settings["model"]},
            include_policy=True,
        )

    return _emit(
        cfg,
        log_path,
        state,
        action=combined["action"],
        reason=combined["reason"],
        confidence=combined["confidence"],
        backend="openai",
        jev_used=False,
        decision_used=True,
        details=combined["details"],
        include_policy=True,
    )


def _route_typesafe(state: dict[str, Any], cfg: dict[str, Any], log_path: Any) -> dict[str, Any]:
    # Imported here so the kill switch and the OpenAI backend stay usable
    # when the TypeSafe package is not installed.
    from typesafe_sdk import Choice, Noul, Score

    from src.jev_client import system_one

    thr = cfg.get("thresholds") or {}
    limits = cfg.get("limits") or {}
    model = cfg.get("model") or "jev-latest"
    jstate = normalize_state(state)

    questions = {
        "intent": Choice(
            instructions="What kind of work does this request mainly need?",
            criteria={
                "chat": "Short answer or conversation; no tools needed",
                "lookup": "Known file/API/status check; mostly deterministic",
                "research": "Need web/search and synthesis",
                "browser": "Need interactive browser clicks",
                "coding": "Edit code / tests / repo work",
                "write": "Draft long prose/email/doc",
                "account": "Send, publish, pay, delete, change permissions",
            },
        ),
        "reuse_cache": Noul(
            instructions="Is there already a fresh enough cached result that should be reused instead of doing new heavy work?"
        ),
        "needs_subagent": Noul(
            instructions="Does this clearly need an extra specialized bot (research/browser/coding) beyond one Grok Bot turn?"
        ),
        "stop_retry": Noul(
            instructions="Given prior_error and same_error_count, should we STOP retrying the same approach?"
        ),
        "complexity": Score(
            instructions="How much Grok Bot effort is justified?",
            criteria=[
                "Trivial — one step or cached",
                "Normal — short tool use",
                "Heavy — multi-step research/browser/coding",
            ],
        ),
    }

    result = system_one(jstate, questions, model=model)
    intent = result.choices["intent"]
    reuse = result.nouls["reuse_cache"]
    sub = result.nouls["needs_subagent"]
    stop = result.nouls["stop_retry"]
    complexity = result.scores["complexity"]

    reuse_n = float(reuse.noul)
    sub_n = float(sub.noul)
    stop_n = float(stop.noul)
    comp = float(complexity.score) / 2.0

    action = "proceed_full"
    reason = "default full Grok Bot work"

    if jstate["has_cached_artifact"] and reuse_n >= float(thr.get("reuse_min", 0.65)):
        action = "reuse_cache"
        reason = f"reuse_cache noul={reuse_n:.2f}"
    elif jstate["same_error_count"] >= int(limits.get("max_retries_same_error", 1)) and stop_n >= 0.55:
        action = "stop_retry"
        reason = f"stop_retry noul={stop_n:.2f} same_error_count={jstate['same_error_count']}"
    elif intent.choice == "lookup" and float(intent.confidence) >= float(thr.get("min_choice_confidence", 0.55)):
        action = "run_deterministic"
        reason = "intent=lookup"
    elif intent.choice == "chat" and float(intent.confidence) >= float(thr.get("min_choice_confidence", 0.55)):
        action = "chat_only"
        reason = "intent=chat"
    elif intent.choice == "account":
        action = "ask_human"
        reason = "account/irreversible class — require approval"
    elif sub_n >= float(thr.get("subagent_min", 0.75)):
        action = "allow_subagent"
        reason = f"needs_subagent noul={sub_n:.2f}"
    elif intent.choice in {"research", "browser"}:
        action = "research_capped"
        reason = f"cap sources at {limits.get('max_browser_sources', 5)}"

    details = {
        "intent": intent.choice,
        "intent_confidence": round(float(intent.confidence), 4),
        "intent_probs": {k: round(float(v), 4) for k, v in (intent.probabilities or {}).items()},
        "reuse_cache": round(reuse_n, 4),
        "needs_subagent": round(sub_n, 4),
        "stop_retry": round(stop_n, 4),
        "complexity_0_1": round(comp, 4),
        "max_browser_sources": int(limits.get("max_browser_sources", 5)),
    }
    return _emit(
        cfg,
        log_path,
        state,
        action=action,
        reason=reason,
        confidence=float(intent.confidence),
        backend="typesafe",
        jev_used=True,
        decision_used=True,
        details=details,
        include_policy=True,
    )
