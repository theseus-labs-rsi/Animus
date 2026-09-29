from __future__ import annotations

import unittest

from agent_harnesses.costing import (
    attach_cost,
    cost_cny,
    extract_usage,
    normalize_usage,
    price_for,
    usage_from_claude_json,
    usage_from_codex_text,
)


class CostingTests(unittest.TestCase):
    def test_normalize_usage_openai_and_anthropic_styles(self):
        openai = normalize_usage(
            {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 40}},
            source="t",
        )
        self.assertEqual(openai["prompt_tokens"], 100)
        self.assertEqual(openai["cached_tokens"], 40)
        self.assertEqual(openai["total_tokens"], 120)
        anthropic = normalize_usage(
            {"input_tokens": 50, "output_tokens": 5, "cache_read_input_tokens": 30},
            source="t",
        )
        self.assertEqual(anthropic["prompt_tokens"], 50)
        self.assertEqual(anthropic["cached_tokens"], 30)
        self.assertIsNone(normalize_usage(None, source="t"))

    def test_cost_cny_known_model_and_unknown_model(self):
        value = cost_cny(
            "deepseek-v4-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000, cached_tokens=0
        )
        self.assertAlmostEqual(value, 3.0)
        self.assertIsNone(cost_cny("some-claude-model", prompt_tokens=1000))
        self.assertIsNone(price_for("some-claude-model"))

    def test_pricing_override(self):
        pricing = {"fake-model": {"input": 10.0, "input_cached": 1.0, "output": 20.0}}
        value = cost_cny("fake-model-v1", prompt_tokens=1_000_000, pricing=pricing)
        self.assertAlmostEqual(value, 10.0)
        self.assertEqual(price_for("fake-model-v1", pricing)["output"], 20.0)

    def test_attach_cost_unknown_model_is_explicit_not_silent(self):
        rec = {"usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cached_tokens": 0}}
        attach_cost(rec, "some-claude-model")
        self.assertIsNone(rec["cost_cny"])
        self.assertIn("单价表", rec["cost_note"])

    def test_attach_cost_codex_unsplit_total_gives_bounds(self):
        rec = {"usage": usage_from_codex_text("tokens used: 5000")}
        attach_cost(rec, "deepseek-v4-flash")
        self.assertIsNone(rec["cost_cny"])
        self.assertAlmostEqual(rec["cost_cny_low"], 5000 * 1.0 / 1_000_000)
        self.assertAlmostEqual(rec["cost_cny_high"], 5000 * 2.0 / 1_000_000)

    def test_usage_from_claude_json_nested_result(self):
        stdout = '{"result": {"usage": {"input_tokens": 10, "output_tokens": 2}}}'
        usage = usage_from_claude_json(stdout)
        self.assertEqual(usage["prompt_tokens"], 10)
        stdout_flat = '{"usage": {"input_tokens": 7, "output_tokens": 1}}'
        self.assertEqual(usage_from_claude_json(stdout_flat)["prompt_tokens"], 7)
        self.assertIsNone(usage_from_claude_json("not json"))

    def test_extract_usage_dispatch(self):
        self.assertIsNotNone(extract_usage("claude", '{"usage": {"input_tokens": 1, "output_tokens": 1}}'))
        self.assertIsNotNone(extract_usage("codex", "tokens used: 42"))
        self.assertIsNone(extract_usage("dsh", "anything"))


if __name__ == "__main__":
    unittest.main()
