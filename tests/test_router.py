"""Router tests. OpenAI is mocked; these tests do not call the network."""

from __future__ import annotations

import io
import json
import os
import runpy
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EXAMPLE_PATH, normalize_config  # noqa: E402
from src.openai_backend import (  # noqa: E402
    ACTION_NAMES,
    build_request,
    create_decision,
    is_risky,
)
from src.router import route_task  # noqa: E402
from src.cli import main as cli_main  # noqa: E402

SECRET = "sk-test-openai-key-do-not-log"


def install_typesafe_stub() -> None:
    try:
        import typesafe_sdk  # noqa: F401
        return
    except ImportError:
        pass
    module = types.ModuleType("typesafe_sdk")

    class _Question:
        def __init__(self, *args, **kwargs):
            pass

    module.Choice = _Question
    module.Noul = _Question
    module.Score = _Question
    module.TypeSafeClient = _Question
    sys.modules["typesafe_sdk"] = module


class Bag:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def decision(
    choice="chat_only",
    confidence=0.91,
    reuse=0.05,
    retry=0.05,
    human=0.04,
    cap=0.05,
    score=0.4,
    refuse_action=False,
):
    if refuse_action:
        action = {"type": "refusal", "name": "action"}
    else:
        action = {
            "type": "choice",
            "name": "action",
            "choice": choice,
            "confidence": confidence,
            "probabilities": [
                {"value": choice, "probability": confidence},
                {"value": "proceed_full", "probability": round(1 - confidence, 4)},
            ],
        }
    return {
        "model": "gpt-6-luna",
        "answers": [
            action,
            {"type": "predicate", "name": "reuse_cached", "probability": reuse},
            {"type": "predicate", "name": "retry_same_approach", "probability": retry},
            {"type": "predicate", "name": "needs_human_approval", "probability": human},
            {"type": "predicate", "name": "cap_research", "probability": cap},
            {
                "type": "score",
                "name": "complexity",
                "score": score,
                "confidence": 0.7,
                "probabilities": [
                    {"value": 0, "label": "Trivial", "probability": 0.8},
                    {"value": 1, "label": "Normal", "probability": 0.2},
                    {"value": 2, "label": "Heavy", "probability": 0.0},
                ],
            },
        ],
        "usage": {
            "input_tokens": 40,
            "output_tokens": 0,
            "total_tokens": 40,
        },
    }


class RouterCase(unittest.TestCase):
    def setUp(self):
        self._saved_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = SECRET
        self.tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmp.name) / "runs.jsonl"

    def tearDown(self):
        if self._saved_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._saved_key
        self.tmp.cleanup()

    def cfg(self, backend="openai", **overrides):
        data = {
            "enabled": True,
            "backend": backend,
            "mode": "shadow",
            "model": "jev-latest",
            "thresholds": {
                "min_choice_confidence": 0.55,
                "reuse_min": 0.65,
                "subagent_min": 0.75,
            },
            "limits": {"max_browser_sources": 5, "max_retries_same_error": 1},
            "logging": {"path": str(self.log_path)},
            "openai": {
                "model": "gpt-6-luna",
                "confidence_threshold": 0.55,
                "timeout_seconds": 20,
            },
        }
        data.update(overrides)
        return data

    def route(self, state, response=None, cfg=None, side_effect=None):
        cfg = self.cfg() if cfg is None else cfg
        with patch("src.openai_backend.create_decision", side_effect=side_effect, return_value=response) as create:
            out = route_task(state, cfg)
        return out, create

    def log_text(self) -> str:
        if not self.log_path.exists():
            return ""
        return self.log_path.read_text(encoding="utf-8")


class OpenAIMappingTests(RouterCase):
    def test_choice_maps_onto_router_actions(self):
        cases = [
            ({"goal": "What is 2+2?", "kind": "chat"}, "chat_only"),
            ({"goal": "Read the release version from the local changelog", "kind": "lookup"}, "run_deterministic"),
            ({"goal": "Compare three written summaries of the notes", "kind": "research"}, "research_capped"),
            ({"goal": "Use a specialist to inspect the repository", "kind": "coding"}, "allow_subagent"),
            ({"goal": "Draft a status note for me to review", "kind": "write"}, "ask_human"),
        ]
        for state, action in cases:
            with self.subTest(action=action):
                out, create = self.route(state, decision(choice=action, confidence=0.91))
                self.assertEqual(out["action"], action)
                self.assertEqual(out["backend"], "openai")
                self.assertFalse(out["jev_used"])
                self.assertTrue(out["decision_used"])
                self.assertGreaterEqual(out["confidence"], 0.55)
                self.assertIsNone(out["details"]["fallback"])
                self.assertEqual(out["details"]["max_browser_sources"], 5)
                self.assertEqual(create.call_count, 1)
                self.assertNotIn(SECRET, json.dumps(out))

    def test_reuse_predicate_overrides_a_heavier_choice(self):
        state = {
            "goal": "Summarize AI agent news for tomorrow briefing",
            "kind": "research",
            "cached_artifact": True,
            "cached_note": "Fresh briefing written 20 minutes ago",
        }
        out, _create = self.route(
            state,
            decision(choice="research_capped", confidence=0.93, reuse=0.91),
        )
        self.assertEqual(out["action"], "reuse_cache")
        self.assertIn("reuse_cached", out["reason"])

    def test_retry_predicate_stops_a_repeated_approach(self):
        state = {
            "goal": "Retry the same failing scrape",
            "kind": "browser",
            "prior_error": "timeout on selector",
            "same_error_count": 2,
        }
        out, _create = self.route(
            state,
            decision(choice="allow_subagent", confidence=0.9, retry=0.88),
        )
        self.assertEqual(out["action"], "stop_retry")

    def test_cap_research_predicate_narrows_uncapped_work(self):
        state = {"goal": "Find recent notes on the release", "kind": "research"}
        out, _create = self.route(
            state,
            decision(choice="proceed_full", confidence=0.86, cap=0.84),
        )
        self.assertEqual(out["action"], "research_capped")
        self.assertEqual(out["details"]["max_browser_sources"], 5)

    def test_model_dump_response_is_parsed(self):
        payload = decision(choice="chat_only", confidence=0.88)

        class DumpDecision:
            def model_dump(self):
                return payload

        out, _create = self.route({"goal": "What is 2+2?", "kind": "chat"}, DumpDecision())
        self.assertEqual(out["action"], "chat_only")
        self.assertEqual(out["details"]["model"], "gpt-6-luna")
        self.assertEqual(out["details"]["usage"]["input_tokens"], 40)

    def test_request_asks_choice_and_independent_questions_together(self):
        state = normalize_public_state()
        request = build_request(state, "gpt-6-luna")
        self.assertEqual(request["model"], "gpt-6-luna")
        questions = request["questions"]
        self.assertEqual([item["type"] for item in questions], [
            "choice",
            "predicate",
            "predicate",
            "predicate",
            "predicate",
            "score",
        ])
        choice = questions[0]
        self.assertEqual(choice["name"], "action")
        self.assertEqual({item["value"] for item in choice["choices"]}, set(ACTION_NAMES))
        names = {item["name"] for item in questions}
        self.assertEqual(
            names,
            {"action", "reuse_cached", "retry_same_approach", "needs_human_approval", "cap_research", "complexity"},
        )
        content = request["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_text")
        self.assertEqual(content[1]["type"], "input_json")
        self.assertIsInstance(content[1]["data"], dict)
        self.assertEqual(content[1]["data"]["goal"], state["goal"])
        self.assertNotIn(SECRET, json.dumps(request))


class OpenAIFallbackTests(RouterCase):
    def test_low_confidence_falls_back_conservatively(self):
        out, create = self.route(
            {"goal": "What is 2+2?", "kind": "chat"},
            decision(choice="chat_only", confidence=0.2),
        )
        self.assertEqual(out["action"], "proceed_full")
        self.assertEqual(out["details"]["fallback"], "low_confidence")
        self.assertTrue(out["decision_used"])
        self.assertEqual(create.call_count, 1)
        self.assertLess(out["confidence"], 0.55)

    def test_higher_threshold_can_be_configured(self):
        cfg = self.cfg()
        cfg["openai"]["confidence_threshold"] = 0.95
        out, _create = self.route(
            {"goal": "What is 2+2?", "kind": "chat"},
            decision(choice="chat_only", confidence=0.9),
            cfg=cfg,
        )
        self.assertEqual(out["action"], "proceed_full")
        self.assertEqual(out["details"]["fallback"], "low_confidence")

    def test_refusal_falls_back(self):
        out, _create = self.route(
            {"goal": "What is 2+2?", "kind": "chat"},
            decision(refuse_action=True),
        )
        self.assertEqual(out["action"], "proceed_full")
        self.assertEqual(out["details"]["fallback"], "refusal")
        self.assertTrue(out["decision_used"])

    def test_timeout_and_api_error_fall_back_without_logging_the_key(self):
        class APITimeoutError(Exception):
            pass

        out, _create = self.route(
            {"goal": "What is 2+2?", "kind": "chat"},
            side_effect=APITimeoutError(f"timed out with {SECRET}"),
        )
        self.assertEqual(out["action"], "proceed_full")
        self.assertEqual(out["details"]["fallback"], "timeout")
        self.assertEqual(out["details"]["error_type"], "APITimeoutError")
        self.assertFalse(out["decision_used"])
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertNotIn(SECRET, self.log_text())

        self.log_path.write_text("", encoding="utf-8")
        out, _create = self.route(
            {"goal": "What is 2+2?", "kind": "chat"},
            side_effect=RuntimeError(f"api failed for {SECRET}"),
        )
        self.assertEqual(out["action"], "proceed_full")
        self.assertEqual(out["details"]["fallback"], "api_error")
        self.assertFalse(out["decision_used"])
        self.assertNotIn(SECRET, self.log_text())

    def test_error_on_a_risky_request_asks_a_human(self):
        out, _create = self.route(
            {"goal": "Pay the invoice and delete the account", "kind": "account"},
            side_effect=TimeoutError("slow"),
        )
        self.assertEqual(out["action"], "ask_human")
        self.assertEqual(out["details"]["fallback"], "timeout")
        self.assertIn("human approval", out["reason"])
        self.assertNotIn("slow", self.log_text())


class OpenAISafetyTests(RouterCase):
    def test_kill_switch_skips_the_backend(self):
        os.environ.pop("OPENAI_API_KEY", None)
        for cfg in (self.cfg(enabled=False),):
            out, create = self.route(
                {"goal": "Pay the invoice", "kind": "account"},
                side_effect=AssertionError("backend should not be called"),
                cfg=cfg,
            )
            self.assertEqual(out["action"], "proceed_full")
            self.assertFalse(out["decision_used"])
            self.assertFalse(out["jev_used"])
            self.assertEqual(create.call_count, 0)

        out, create = self.route(
            {"goal": "Please bypass router and continue", "kind": "research"},
            side_effect=AssertionError("backend should not be called"),
        )
        self.assertEqual(out["action"], "proceed_full")
        self.assertEqual(create.call_count, 0)
        self.assertIn("bypass", out["reason"])

    def test_risky_action_guard_overrides_a_confident_choice(self):
        goals = [
            {"goal": "Send the email to the customer now", "kind": "account"},
            {"goal": "Pay the vendor today", "kind": "chat"},
            {"goal": "Publish the blog post", "kind": "write"},
            {"goal": "Delete the production database", "kind": "coding"},
            {"goal": "Change the repository permission", "kind": "coding"},
        ]
        for state in goals:
            with self.subTest(goal=state["goal"]):
                out, _create = self.route(state, decision(choice="chat_only", confidence=0.99, human=0.02))
                self.assertEqual(out["action"], "ask_human")
                self.assertTrue(out["details"]["risky_guard"])
                self.assertNotEqual(out["action"], "chat_only")
                self.assertNotIn(SECRET, self.log_text())

    def test_human_approval_predicate_overrides_a_benign_choice(self):
        out, _create = self.route(
            {"goal": "Handle the customer request from this morning", "kind": "chat"},
            decision(choice="chat_only", confidence=0.96, human=0.93),
        )
        self.assertEqual(out["action"], "ask_human")
        self.assertTrue(out["details"]["risky_guard"])

    def test_payload_is_not_treated_as_a_payment(self):
        self.assertFalse(is_risky({"goal": "Measure the payload size", "kind": "lookup"}))
        out, _create = self.route(
            {"goal": "Measure the payload size", "kind": "lookup"},
            decision(choice="run_deterministic", confidence=0.9),
        )
        self.assertEqual(out["action"], "run_deterministic")


class TypeSafePathTests(RouterCase):
    def test_typesafe_mapping_still_works_and_does_not_call_openai(self):
        install_typesafe_stub()
        result = Bag(
            choices={"intent": Bag(choice="lookup", confidence=0.84, probabilities={"lookup": 0.84})},
            nouls={
                "reuse_cache": Bag(noul=0.1),
                "needs_subagent": Bag(noul=0.1),
                "stop_retry": Bag(noul=0.1),
            },
            scores={"complexity": Bag(score=1.0)},
        )
        with patch("src.openai_backend.create_decision", side_effect=AssertionError("openai called")), \
             patch("src.jev_client.system_one", return_value=result) as system_one:
            out = route_task({"goal": "Read the local version file", "kind": "lookup"}, self.cfg(backend="typesafe"))
        self.assertEqual(out["action"], "run_deterministic")
        self.assertEqual(out["backend"], "typesafe")
        self.assertTrue(out["jev_used"])
        self.assertTrue(out["decision_used"])
        self.assertEqual(out["confidence"], 0.84)
        self.assertEqual(system_one.call_count, 1)

    def test_typesafe_cache_and_account_gates(self):
        install_typesafe_stub()

        def result(intent, confidence, reuse=0.1):
            return Bag(
                choices={"intent": Bag(choice=intent, confidence=confidence, probabilities={intent: confidence})},
                nouls={
                    "reuse_cache": Bag(noul=reuse),
                    "needs_subagent": Bag(noul=0.1),
                    "stop_retry": Bag(noul=0.2),
                },
                scores={"complexity": Bag(score=0.4)},
            )

        with patch("src.jev_client.system_one", return_value=result("research", 0.4, reuse=0.8)):
            cached = route_task(
                {"goal": "Summarize the notes", "kind": "research", "cached_artifact": True},
                self.cfg(backend="typesafe"),
            )
        self.assertEqual(cached["action"], "reuse_cache")

        with patch("src.jev_client.system_one", return_value=result("account", 0.99)):
            account = route_task(
                {"goal": "Send the email to the customer now", "kind": "account"},
                self.cfg(backend="typesafe"),
            )
        self.assertEqual(account["action"], "ask_human")
        self.assertIn("approval", account["reason"])


class EntrypointTests(RouterCase):
    def test_cli_openai_and_typesafe(self):
        install_typesafe_stub()
        openai_cfg = self.cfg(backend="openai")
        typesafe_cfg = self.cfg(backend="typesafe")
        result = Bag(
            choices={"intent": Bag(choice="chat", confidence=0.8, probabilities={"chat": 0.8})},
            nouls={
                "reuse_cache": Bag(noul=0.0),
                "needs_subagent": Bag(noul=0.0),
                "stop_retry": Bag(noul=0.0),
            },
            scores={"complexity": Bag(score=0.2)},
        )
        with patch("src.config.load_config", return_value=openai_cfg), \
             patch("src.openai_backend.create_decision", return_value=decision()):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli_main(['{"goal":"What is 2+2?","kind":"chat"}'])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["backend"], "openai")
        self.assertEqual(payload["action"], "chat_only")

        with patch("src.config.load_config", return_value=typesafe_cfg), \
             patch("src.jev_client.system_one", return_value=result):
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli_main(['{"goal":"What is 2+2?","kind":"chat"}'])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["backend"], "typesafe")
        self.assertEqual(payload["action"], "chat_only")

    def test_dry_run_runs_either_backend(self):
        install_typesafe_stub()
        script = str(ROOT / "scripts" / "dry_run.py")
        result = Bag(
            choices={"intent": Bag(choice="chat", confidence=0.7, probabilities={"chat": 0.7})},
            nouls={
                "reuse_cache": Bag(noul=0.0),
                "needs_subagent": Bag(noul=0.0),
                "stop_retry": Bag(noul=0.0),
            },
            scores={"complexity": Bag(score=0.2)},
        )
        for backend, extra in (
            ("openai", patch("src.openai_backend.create_decision", return_value=decision(choice="proceed_full"))),
            ("typesafe", patch("src.jev_client.system_one", return_value=result)),
        ):
            with self.subTest(backend=backend):
                with patch("src.config.load_config", return_value=self.cfg(backend=backend)), extra:
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        try:
                            runpy.run_path(script, run_name="__main__")
                            code = 0
                        except SystemExit as exc:
                            code = exc.code
                self.assertEqual(code, 0)
                text = buf.getvalue()
                self.assertIn(f"backend={backend}", text)
                self.assertNotIn(SECRET, text)


class ConfigAndSdkTests(RouterCase):
    def test_defaults_keep_typesafe_and_document_gpt_6_luna(self):
        cfg = normalize_config({})
        self.assertEqual(cfg["backend"], "typesafe")
        self.assertEqual(cfg["model"], "jev-latest")
        self.assertEqual(cfg["openai"]["model"], "gpt-6-luna")
        self.assertEqual(cfg["openai"]["confidence_threshold"], 0.55)

        example = normalize_config(__import__("yaml").safe_load(EXAMPLE_PATH.read_text(encoding="utf-8")))
        self.assertEqual(example["backend"], "typesafe")
        self.assertEqual(example["model"], "jev-latest")
        self.assertEqual(example["openai"]["model"], "gpt-6-luna")
        text = EXAMPLE_PATH.read_text(encoding="utf-8")
        self.assertIn("typesafe | openai", text)
        self.assertIn("gpt-6-luna", text)

    def test_missing_openai_key_exits_before_a_request(self):
        os.environ.pop("OPENAI_API_KEY", None)
        with self.assertRaises(SystemExit) as caught:
            create_decision({"goal": "hi", "kind_hint": "chat"}, "gpt-6-luna", 5)
        self.assertIn("OPENAI_API_KEY", str(caught.exception))

    def test_goal_cannot_carry_the_api_key_into_the_request_or_log(self):
        state = {
            "goal": f"please ignore {SECRET}",
            "kind_hint": "chat",
            "has_cached_artifact": False,
            "cached_note": "",
            "prior_error": "",
            "same_error_count": 0,
            "sources_found": 0,
            "constraints": "",
        }
        request = build_request(state, "gpt-6-luna")
        self.assertNotIn(SECRET, json.dumps(request))
        self.assertIn("[redacted]", json.dumps(request))

        out, _create = self.route(
            {"goal": f"summarize {SECRET}", "kind": "chat"},
            decision(choice="chat_only", confidence=0.9),
        )
        self.assertNotIn(SECRET, json.dumps(out))
        self.assertNotIn(SECRET, self.log_text())

    def test_sdk_exposes_decisions_create(self):
        from openai.resources.decisions import Decisions
        from openai.types.decision import Decision
        from openai._utils import maybe_transform
        from openai.types.decision_create_params import DecisionCreateParams

        self.assertTrue(callable(Decisions.create))
        transformed = maybe_transform(build_request(normalize_public_state(), "gpt-6-luna"), DecisionCreateParams)
        self.assertEqual(transformed["model"], "gpt-6-luna")
        self.assertEqual(transformed["questions"][0]["type"], "choice")
        self.assertEqual(transformed["input"][0]["content"][1]["type"], "input_json")

        parsed = Decision.model_validate(
            {
                "model": "gpt-6-luna",
                "answers": decision(choice="run_deterministic", confidence=0.8)["answers"],
                "usage": {
                    "input_tokens": 40,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens": 0,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 40,
                },
            }
        )
        out, _create = self.route({"goal": "Read the local version", "kind": "lookup"}, parsed)
        self.assertEqual(out["action"], "run_deterministic")
        self.assertEqual(out["details"]["usage"]["input_tokens"], 40)

    def test_create_decision_uses_the_sdk_and_does_not_return_the_key(self):
        captured = {}

        class FakeDecisions:
            def create(self, **kwargs):
                captured["create"] = kwargs
                return decision()

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.decisions = FakeDecisions()

            def close(self):
                captured["closed"] = True

        with patch("openai.OpenAI", FakeClient):
            result = create_decision(
                {
                    "goal": "What is 2+2?",
                    "kind_hint": "chat",
                    "has_cached_artifact": False,
                    "cached_note": "",
                    "prior_error": "",
                    "same_error_count": 0,
                    "sources_found": 0,
                    "constraints": "",
                },
                "gpt-6-luna",
                12,
            )
        self.assertEqual(captured["client"]["api_key"], SECRET)
        self.assertEqual(captured["client"]["timeout"], 12)
        self.assertEqual(captured["create"]["model"], "gpt-6-luna")
        self.assertEqual(captured["create"]["timeout"], 12)
        self.assertTrue(captured["closed"])
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertEqual(result["answers"][0]["choice"], "chat_only")


def normalize_public_state():
    return {
        "goal": "Summarize the latest release notes",
        "kind_hint": "research",
        "has_cached_artifact": False,
        "cached_note": "",
        "prior_error": "",
        "same_error_count": 0,
        "sources_found": 0,
        "constraints": "",
    }


if __name__ == "__main__":
    unittest.main()
