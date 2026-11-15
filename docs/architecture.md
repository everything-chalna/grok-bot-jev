# Architecture

## Architecture A: skill gates around Grok Bot

This package implements an external decision layer rather than changing Grok Bot. A Grok Bot skill prepares a small task state, calls `src.cli` (or the equivalent Python function), and then applies the returned action before expensive work.

```text
user request
    |
    v
Grok Bot wakes and loads the skill
    |
    +--> kill switch / bypass marker? ---- yes ---> normal Grok Bot path
    |
    v
configured backend
    |-- typesafe: TypeSafe Jev system_one(state, questions)
    |-- openai:   POST /v1/decisions (choice + predicate + score)
    |
    v
router policy: cache | stop | deterministic | chat | capped research |
                subagent | human approval | full work
    |
    v
Grok Bot executes the action (active) or treats it as advice (shadow)
```

The Python router normalizes state, asks the configured backend to classify it, applies thresholds and limits, and appends a JSONL decision record. Credentials are read only from the process environment: `TYPESAFE_API_KEY` for TypeSafe, or `OPENAI_API_KEY` for OpenAI. The key is never written to the log. The default backend remains TypeSafe.

## What works

- Cache reuse, bounded research, retry stopping, direct chat/lookup suggestions, subagent suggestions, and an approval gate for account-changing intents.
- A configuration kill switch and explicit `shadow`/`active` modes.
- Auditable decision records without storing the credential.

## What does not

- This is not a Grok Bot middleware or pre-wake interceptor. The bot must wake, load the skill, and make the router call.
- Shadow mode cannot force behavior. Active mode relies on the installed skill honoring `route.action`.
- Neither backend performs the browser, research, coding, payment, or account work, and neither replaces Grok Bot's confirmation or safety policy.
- The OpenAI path cannot treat a model choice as approval to send, publish, pay, delete, or change permissions. Those requests come back as `ask_human`.
- The included A/B results are local proxy measurements, not proof of lower Grok Bot token usage or a universal speedup.

## Failure behavior

If the router is disabled or bypassed, it steps aside and returns `proceed_full`. Grok Bot's own confirmation rules still apply. On the OpenAI backend, a low-confidence choice, a refusal, a timeout, or an API error falls back to `proceed_full` for ordinary work and to `ask_human` when the request would send, publish, pay, delete, or change permissions. The TypeSafe path still raises if Jev itself cannot be called, matching its previous behavior.
