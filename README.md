# Grok Bot Usage Router

Connect a cheap decision layer to Grok Bot before expensive research, browser, retry, or subagent work. The router classifies the request so Grok Bot can reuse a fresh artifact, stop a failing retry, cap research, or ask for approval.

Two backends are supported. The default is unchanged:

- `typesafe`: [TypeSafe Jev](https://docs.typesafe.ai) `system_one`
- `openai`: [OpenAI Decisions](https://developers.openai.com/api/docs/guides/decisions) `POST /v1/decisions` with `gpt-6-luna`

This package is a small open-source reference implementation. It does not change Grok Bot's foundation model and does not route Cursor models.

## How it looks

Visualization of Grok Bot + Jev at work:

- [`media/jev-grok-bot-demo.mp4`](media/jev-grok-bot-demo.mp4)

![Grok Bot + Jev dashboard](media/jev-grok-bot-dashboard.png)

## What

- A Python router with `backend: typesafe | openai` in config. `typesafe` is the default.
- Explicit actions such as `reuse_cache`, `stop_retry`, `run_deterministic`, `chat_only`, `research_capped`, `allow_subagent`, `ask_human`, and `proceed_full`.
- `shadow` mode for measurement and `active` mode for a skill that honors the returned action.
- A kill switch through `enabled: false` or a `bypass jev`, `bypass router`, or `no jev` marker.

## Why

A small classification call can prevent needless tool work. The router is intentionally conservative: it never sends, publishes, pays, deletes, or changes permissions, and it will not return an action that approves those. It logs decisions without logging the API key. Keys are read only from the environment.

## Install

```bash
git clone https://github.com/Bodila51/grok-bot-jev.git
cd grok-bot-jev
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Python 3.10 or newer is required because the pinned OpenAI SDK requires it.

For the default TypeSafe backend:

```bash
export TYPESAFE_API_KEY=...   # use your secret manager; do not commit the key
```

For the OpenAI Decisions backend, set `backend: openai` in `config.yaml` and export a platform key:

```bash
export OPENAI_API_KEY=...     # OpenAI Platform API key; do not commit it
```

`OPENAI_API_KEY` must be an [OpenAI Platform](https://platform.openai.com/api-keys) API key. A ChatGPT or Codex subscription login is not accepted. Eligible [ChatGPT Pro and Business API credits](https://developers.openai.com/api/docs/guides/chatgpt-api-credits) can cover Decisions usage after the subscription is linked to an API organization. The router still authenticates with that organization's API key. Credits do not turn a ChatGPT login into a credential, and they do not raise in-app ChatGPT limits.

Keep `config.yaml`, `.env`, and local logs uncommitted. `src/secrets.py` reads `TYPESAFE_API_KEY` or `OPENAI_API_KEY` from the environment only. It never reads a secrets file, and the logger strips those values if they appear in a record.

## Run dry-run

The script sends the sample states through the configured backend and prints JSON decisions:

```bash
.venv/bin/python scripts/dry_run.py
# or one state:
.venv/bin/python -m src.cli '{"goal":"summarize the latest release notes","kind":"research"}'
```

`scripts/dry_run.py` and `python -m src.cli` both call `route_task`, so either backend works. `config.yaml` selects it (`backend: typesafe` or `backend: openai`). The OpenAI model defaults to `gpt-6-luna` under `openai.model`.

For a no-network smoke check, disable the router first:

```bash
cp config.example.yaml config.yaml
# edit config.yaml: enabled: false
.venv/bin/python scripts/dry_run.py
```

## OpenAI Decisions cost

With `gpt-6-luna`, uncached Decisions input is $0.10 per 1M tokens. Output tokens are not charged. Prompt-cache reads are discounted and cache writes are charged, at the standard `gpt-6-luna` prompt-caching rates. Regional processing and long-context multipliers can apply. See the [Decisions pricing section](https://developers.openai.com/api/docs/guides/decisions).

One route sends a short task state plus a handful of questions, so this is a classification charge. It is not a measurement of Grok Bot token savings. The included A/B notes below are from TypeSafe Jev runs, not from the OpenAI backend.

## Use with Grok Bot skill

Copy [`skill/jev-usage-router.SKILL.md`](skill/jev-usage-router.SKILL.md) into a Grok Bot skill. The skill describes the request state, when to call this router, and how to honor each action. Run the router in the environment where this repository is installed; pass only task metadata and short context, never credentials.

Start in `mode: shadow`: decisions are logged and advisory. Move to `mode: active` only after reviewing logs; the Grok Bot skill must then honor `route.action`. The router is not a hidden interceptor and cannot force a bot that ignores the skill to stop.

## A/B results highlights

The included local A/B summary is in [`examples/ab_results.md`](examples/ab_results.md), with sanitized source data in [`examples/chatgpt_pack.json`](examples/chatgpt_pack.json). Highlights from the recorded run:

- Across five tasks, flight browser opens went from 1 to 0 when a cached artifact was reused; same-approach retries went from 3 to 0; flight-prep skills loaded went from 6 to 3; model-research pages fetched went from 10 to 4.
- The Grok Bot weekly usage meter moved 37% → 38% in the first phase and 39% after the second phase. Exact per-task Grok tokens were unavailable, so this is not a token-savings claim.
- In the separate 24-candidate timing comparison, the uncapped arm took 53.803 seconds and fetched 14 pages; the Jev-ranked top-five arm took 4.125 seconds and fetched 5 pages, a reported 13.0× wall-time ratio. It used two Jev calls and an estimated $0.000405 Jev cost; fewer hits were by design because of the top-five cap.

These are one local run's proxy measurements, not a benchmark or guarantee.

## Limits

- The decision call is an additional network request and can be wrong, refused, unavailable, or slower than the skipped work. On OpenAI, low confidence, refusal, timeout, and API errors fall back to `proceed_full`, or to `ask_human` when the request would send, publish, pay, delete, or change permissions.
- Shadow mode does not enforce anything; active mode depends on the pasted skill honoring the decision.
- There are no pre-wake hooks here. Grok Bot must wake and invoke the router; this package cannot reduce the cost of the wake-up itself.
- The router does not execute browser actions, research, payments, messages, or account changes. Those remain Grok Bot responsibilities with their own confirmation rules.
- The included A/B data reports proxy metrics; it does not expose exact Grok Bot tokens or Grok Bot dollar cost.

## Links

- [TypeSafe documentation](https://docs.typesafe.ai)
- [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions)
- [ChatGPT subscription API credits](https://developers.openai.com/api/docs/guides/chatgpt-api-credits)
- [Grok Bot documentation](https://cursor.com/docs/grok-bot)
- [Architecture notes](docs/architecture.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Bug and feature issue templates live under `.github/ISSUE_TEMPLATE`.

## License

MIT. See [LICENSE](LICENSE).
