# Usage Router for Grok Bot

Use the installed usage router as a cheap decision layer before expensive work. The router reads `backend` from its config and calls either TypeSafe Jev or the OpenAI Decisions API. This skill is a policy and integration recipe, not a hidden system hook.

## Workflow

1. Check the kill switch first. If the configuration has `enabled: false`, or the user says `bypass jev`, `bypass router`, or `no jev`, skip the router and proceed normally. Ordinary confirmation rules still apply to send, publish, pay, delete, and permission changes.
2. Before opening a browser, doing multi-source research, retrying a failed approach, loading several specialized skills, or launching a subagent, build a compact JSON state:

```json
{
  "goal": "what the user wants",
  "kind": "chat|lookup|research|browser|coding|write|account",
  "cached_artifact": false,
  "cached_note": "optional freshness and scope",
  "prior_error": "optional last error",
  "same_error_count": 0,
  "sources_found": 0,
  "constraints": "optional limits"
}
```

3. Invoke the installed router with that state and read the returned `action`, `reason`, `confidence`, `mode`, `backend`, and `details`.
4. In `shadow` mode, record the recommendation but continue according to normal Grok Bot judgment. In `active` mode, honor the actions below.

## Action policy in active mode

- `reuse_cache`: use the fresh artifact; do not repeat the expensive work.
- `stop_retry`: stop the repeated approach, explain the failure, and propose a different path or ask the user.
- `run_deterministic`: perform the known, bounded lookup without broad research.
- `chat_only`: answer directly without tools.
- `ask_human`: pause before any account, send, publish, pay, delete, or permission-changing action.
- `allow_subagent`: use a specialized subagent only when it is actually available and appropriate.
- `research_capped`: research with at most `details.max_browser_sources` sources/pages, then synthesize.
- `proceed_full`: continue with normal work while still applying ordinary safety and confirmation rules.

The router never returns an action that approves sending, publishing, paying, deleting, or changing permissions. Those requests are classified as `ask_human`. Do not treat `proceed_full`, `reuse_cache`, or any other action as that approval.

Never treat a router result as permission to reveal secrets or bypass a separate safety requirement. Do not include API keys in state, prompts, logs, or tool arguments. Keep goals and notes short and redact sensitive user content before logging.

## Honest integration limits

Grok Bot has to wake and invoke this router; this skill provides no pre-wake hook and cannot reduce the cost of waking the bot. Shadow mode is advisory. Active mode is enforceable only insofar as the Grok Bot skill runtime follows this policy. Either backend can misclassify, refuse, time out, or be unavailable. The OpenAI backend then falls back conservatively (`proceed_full`, or `ask_human` for risky actions). Use the kill switch when the router should stay out of the way.
