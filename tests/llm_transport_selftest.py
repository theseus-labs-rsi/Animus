"""Offline checks for explicit wire configuration without config or SDK imports."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import llm_transport as transport


def configuration():
    return {"version": transport.VERSION, "default_profile": "legacy",
        "model_profiles": {"explicit-reasoner": "high_json"},
        "profiles": {"high_json": {
            "token_limit_parameter": "max_completion_tokens", "reasoning_effort": "high",
            "omit_parameters": ["temperature", "top_p", "logprobs", "top_logprobs"],
            "response_format": {"type": "json_object"},
            "http_timeout_seconds": 300, "deadline_seconds": 300}}}


class TransportTests(unittest.TestCase):
    def build(self, config=None, **kwargs):
        return transport.build_request_parameters(model=kwargs.pop("model", "explicit-reasoner"),
                                                  transport=config, **kwargs)

    def test_legacy_default_matches_current_parameter_construction(self):
        self.assertEqual(self.build(), {"temperature": 0.7, "top_p": 1.0, "max_tokens": 4096})
        self.assertEqual(self.build(temperature=None, top_p=None, max_tokens=10,
                                   min_completion_tokens=30, response_format={"type": "json_object"}),
                         {"temperature": None, "max_tokens": 30, "response_format": {"type": "json_object"}})

    def test_unmapped_model_uses_legacy_with_no_name_heuristic(self):
        config = configuration()
        for model in ("gpt-explicit-reasoner", "EXPLICIT-REASONER", "explicit-reasoner ", "ordinary"):
            with self.subTest(model=model):
                self.assertEqual(transport.resolve_profile(config, model)["name"], "legacy")
                self.assertIn("max_tokens", self.build(config, model=model))

    def test_exact_reasoning_wire_and_no_null_omission(self):
        self.assertEqual(self.build(configuration(), max_tokens=16384, temperature=None, top_p=None,
                                    min_completion_tokens=99999),
            {"reasoning_effort": "high", "max_completion_tokens": 16384,
             "response_format": {"type": "json_object"}})

    def test_explicit_max_tokens_profile_also_uses_exact_budget(self):
        config = configuration()
        config["profiles"]["high_json"] = {"token_limit_parameter": "max_tokens"}
        self.assertEqual(self.build(config, max_tokens=12, min_completion_tokens=900),
                         {"temperature": 0.7, "top_p": 1.0, "max_tokens": 12})

    def test_declared_default_and_explicit_legacy_mapping(self):
        config = configuration()
        config["default_profile"] = "high_json"
        config["model_profiles"]["plain"] = "legacy"
        self.assertEqual(transport.resolve_profile(config, "unlisted")["name"], "high_json")
        self.assertEqual(transport.resolve_profile(config, "plain")["name"], "legacy")

    def test_conflicting_response_format_rejected_without_override(self):
        for value in ({"type": "text"}, {}, {"type": "json_schema", "json_schema": {}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.build(configuration(), response_format=value)
        self.assertEqual(self.build(configuration(), response_format={"type": "json_object"})["response_format"],
                         {"type": "json_object"})

    def test_legacy_preserves_existing_response_format(self):
        value = {"type": "json_schema", "json_schema": {"name": "existing"}}
        result = self.build(response_format=value)
        self.assertEqual(result["response_format"], value)
        result["response_format"]["json_schema"]["name"] = "changed"
        self.assertEqual(value["json_schema"]["name"], "existing")

    def test_forbidden_parameters_cannot_be_smuggled_into_profiles(self):
        for key in ("messages", "model", "max_tokens", "max_completion_tokens", "max_retries",
                    "retries", "api_key", "base_url", "tools", "temperature", "extra_body"):
            config = configuration()
            config["profiles"]["high_json"][key] = "forbidden"
            with self.subTest(key=key), self.assertRaises(ValueError):
                transport.validate_transport(config)

    def test_invalid_omission_lists_and_duplicates_rejected(self):
        for value in (["max_tokens"], ["temperature", "temperature"], "temperature", [None], [{}]):
            config = configuration(); config["profiles"]["high_json"]["omit_parameters"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                transport.validate_transport(config)

    def test_missing_or_unsupported_token_parameter_rejected(self):
        for value in (None, "tokens", "", 10, [], {}):
            config = configuration(); config["profiles"]["high_json"]["token_limit_parameter"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                transport.validate_transport(config)
        config = configuration(); del config["profiles"]["high_json"]["token_limit_parameter"]
        with self.assertRaises(ValueError):
            transport.validate_transport(config)

    def test_reasoning_effort_requires_nonempty_string(self):
        for value in (None, "", "  ", 1, False):
            config = configuration(); config["profiles"]["high_json"]["reasoning_effort"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                transport.validate_transport(config)

    def test_only_json_object_profile_format_supported(self):
        for value in (None, {}, "json_object", {"type": "text"}, {"type": "json_object", "extra": True}):
            config = configuration(); config["profiles"]["high_json"]["response_format"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                transport.validate_transport(config)

    def test_timeouts_are_finite_positive_numbers_and_never_wire_fields(self):
        for key in ("http_timeout_seconds", "deadline_seconds"):
            for value in (0, -1, True, "300", None, float("nan"), float("inf"), 10 ** 500):
                config = configuration(); config["profiles"]["high_json"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    transport.validate_transport(config)
        config = configuration(); config["profiles"]["high_json"]["deadline_seconds"] = 0.25
        self.assertEqual(transport.resolve_profile(config, "explicit-reasoner")["profile"]["deadline_seconds"], 0.25)
        self.assertNotIn("deadline_seconds", self.build(config))
        self.assertNotIn("http_timeout_seconds", self.build(config))
        config["profiles"]["high_json"]["deadline_seconds"] = 2 ** 53 + 1
        self.assertEqual(transport.resolve_profile(config, "explicit-reasoner")["profile"]["deadline_seconds"], 2 ** 53 + 1)

    def test_invalid_outer_config_or_unknown_mapping_rejected(self):
        values = [{}, [], "legacy"]
        config = configuration(); config["version"] = "future"; values.append(config)
        config = configuration(); config["extra"] = 1; values.append(config)
        config = configuration(); config["default_profile"] = "missing"; values.append(config)
        config = configuration(); config["model_profiles"]["x"] = "missing"; values.append(config)
        config = configuration(); config["profiles"]["legacy"] = {"token_limit_parameter": "max_tokens"}; values.append(config)
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                transport.validate_transport(value)

    def test_explicit_profile_rejects_invalid_caller_budget(self):
        for value in (0, -1, True, 1.5, "100"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.build(configuration(), max_tokens=value)

    def test_fingerprints_canonical_but_parameter_sensitive(self):
        first = configuration(); second = deepcopy(first)
        second["profiles"]["high_json"]["omit_parameters"].reverse()
        second["profiles"]["high_json"]["http_timeout_seconds"] = 300.0
        second = dict(reversed(list(second.items())))
        self.assertEqual(transport.transport_fingerprint(first), transport.transport_fingerprint(second))
        second["profiles"]["high_json"]["reasoning_effort"] = "medium"
        self.assertNotEqual(transport.transport_fingerprint(first), transport.transport_fingerprint(second))
        self.assertNotEqual(transport.transport_fingerprint(first), transport.transport_fingerprint(None))

    def test_validation_resolution_and_build_leave_inputs_unchanged(self):
        original = configuration(); before = deepcopy(original)
        normalized = transport.validate_transport(original)
        resolved = transport.resolve_profile(original, "explicit-reasoner")
        request = self.build(original)
        normalized["profiles"]["high_json"]["omit_parameters"].clear()
        resolved["profile"]["response_format"]["type"] = "text"
        request["response_format"]["type"] = "text"
        self.assertEqual(original, before)
        self.assertEqual(self.build(original)["response_format"], {"type": "json_object"})

    def test_import_has_no_config_credentials_or_sdk_side_effect(self):
        code = ("import sys; sys.path.insert(0, " + repr(str(ROOT)) + "); import llm_transport; "
                "assert not ({'config','openai','dotenv'} & set(sys.modules)); "
                "assert llm_transport.validate_transport(None) is None")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
