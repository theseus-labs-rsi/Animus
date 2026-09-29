"""Offline original-JSON quotation and identity checks; no model/config import."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.reference_locations import (VERSION, locate_reference_quote, locate_reference_target,
                                           reference_value_hash, validate_json_value)


class ReferenceLocationTests(unittest.TestCase):
    def test_node_preserves_whole_structured_value_and_bound_identity(self):
        value = {"结论": [{"状态": "未知", "依据": ["甲", "乙"]}], "other": None}
        before = deepcopy(value)
        result = locate_reference_target(value, {"location_scope": "node", "json_pointer": "/结论/0"})
        self.assertEqual(result["location_status"], "unique")
        self.assertEqual(result["source_value"], value["结论"][0])
        self.assertEqual(result["source_value_hash"], reference_value_hash(value["结论"][0]))
        self.assertEqual(result["reference_value_hash"], reference_value_hash(value))
        self.assertNotIn("source_quote", result)
        self.assertNotIn("matching_spans", result)
        result["source_value"]["依据"].append("not original")
        self.assertEqual(value, before)

    def test_node_root_preserves_all_json_types_and_never_decodes_strings(self):
        for value in [None, True, 0, 1.0, [], {}, "", '{"a":null}']:
            with self.subTest(value=value):
                result = locate_reference_target(value, {"location_scope": "node", "json_pointer": ""})
                self.assertEqual(result["source_value"], value)
                self.assertIs(type(result["source_value"]), type(value))
                self.assertEqual(result["source_value_hash"], reference_value_hash(value))
        with self.assertRaises(ValueError):
            locate_reference_target('{"a":null}', {"location_scope": "node", "json_pointer": "/a"})

    def test_node_escaped_keys_empty_keys_and_single_escape_pass(self):
        value = {"a/b~c": [{"": None}], "~1": "literal escape", "01": "object key", "-1": False}
        for pointer, expected in [("/a~1b~0c/0/", None), ("/~01", "literal escape"),
                                  ("/01", "object key"), ("/-1", False)]:
            with self.subTest(pointer=pointer):
                result = locate_reference_target(value, {"location_scope": "node", "json_pointer": pointer})
                self.assertEqual(result["json_pointer"], pointer)
                self.assertEqual(result["source_value"], expected)

    def test_node_rejects_missing_noncanonical_or_invalid_pointers(self):
        value = {"items": ["first", "second"], "~2": "must be escaped", "scalar": 1}
        for pointer in [None, 0, "items", "#/items/0", "/missing", "/items/-1", "/items/01",
                        "/items/+1", "/items/-", "/items/2", "/items/１", "/items/1.0",
                        "/items/", "/~2", "/~", "/scalar/x"]:
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                locate_reference_target(value, {"location_scope": "node", "json_pointer": pointer})
        with self.assertRaises(ValueError):
            locate_reference_target(value, {"location_scope": "node"})

    def test_target_modes_cannot_mix_even_with_null_or_empty_fields(self):
        for target in [None, [], {"location_scope": "unknown", "source_quote": "x"},
                       {"location_scope": None, "source_quote": "x"},
                       {"location_scope": "node", "json_pointer": "", "source_quote": None},
                       {"location_scope": "node", "json_pointer": "", "source_quote": ""},
                       {"location_scope": "node", "json_pointer": "", "source_quote": "x"},
                       {"source_quote": "x", "json_pointer": None},
                       {"location_scope": "quote", "source_quote": "x", "json_pointer": ""}]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                locate_reference_target("x", target)

    def test_target_quote_compatibility_keeps_every_key_value_and_overlap_match(self):
        value = {"aba": "ababa", "items": ["aba", "aba"]}
        expected = locate_reference_quote(value, "aba")
        for target in [{"source_quote": "aba"}, {"location_scope": "quote", "source_quote": "aba"}]:
            self.assertEqual(locate_reference_target(value, target), expected)
        self.assertEqual(len(expected["matching_spans"]), 5)
        self.assertEqual(locate_reference_target(value, {"location_scope": "node", "json_pointer": "/aba"})[
            "source_value"], "ababa")

    def test_string_root_keeps_exact_code_point_offsets(self):
        result = locate_reference_quote("前😀结论，结论", "结论")
        self.assertEqual(result["locator_version"], VERSION)
        self.assertEqual(result["location_status"], "ambiguous")
        self.assertEqual([(x["json_pointer"], x["start"], x["end"]) for x in result["matching_spans"]],
                         [("", 2, 4), ("", 5, 7)])

    def test_mixed_nested_values_and_pointer_escaping(self):
        value = {"a/b~c": [{"": "结果"}, False], "other": 10}
        result = locate_reference_quote(value, "结果")
        self.assertEqual(result["location_status"], "unique")
        self.assertEqual(result["matching_spans"][0]["json_pointer"], "/a~1b~0c/0/")
        self.assertEqual(locate_reference_quote(value, "false")["matching_spans"][0]["json_pointer"], "/a~1b~0c/1")
        self.assertEqual(locate_reference_quote(value, "a/b~c")["matching_spans"][0]["source_kind"], "object_key")

    def test_duplicates_across_keys_leaves_and_overlaps_are_all_retained(self):
        result = locate_reference_quote({"aba": "ababa", "items": ["aba", "aba"]}, "aba")
        self.assertEqual(result["location_status"], "ambiguous")
        self.assertEqual(len(result["matching_spans"]), 5)
        self.assertEqual([(x["source_kind"], x["start"]) for x in result["matching_spans"][:3]],
                         [("object_key", 0), ("string_value", 0), ("string_value", 2)])

    def test_keys_and_values_have_separate_kinds_with_same_pointer(self):
        matches = locate_reference_quote({"same": "same"}, "same")["matching_spans"]
        self.assertEqual([x["json_pointer"] for x in matches], ["/same", "/same"])
        self.assertEqual([x["source_kind"] for x in matches], ["object_key", "string_value"])
        self.assertEqual(matches[0]["source_value_hash"], matches[1]["source_value_hash"])

    def test_string_escapes_are_not_decoded_or_normalized_again(self):
        value = {"text": '甲\n乙\t"引号"\\后'}
        quote = '\n乙\t"引号"\\'
        span = locate_reference_quote(value, quote)["matching_spans"][0]
        self.assertEqual(value["text"][span["start"]:span["end"]], quote)
        self.assertEqual(locate_reference_quote(value, r"甲\n乙")["location_status"], "failed")
        self.assertEqual(locate_reference_quote("e\u0301", "é")["location_status"], "failed")

    def test_numbers_match_only_complete_literal(self):
        value = [10, 1.5, -1, 1, 1.0, "10"]
        matches = locate_reference_quote(value, "1")["matching_spans"]
        self.assertEqual([(x["json_pointer"], x["source_kind"]) for x in matches],
                         [("/3", "number"), ("/5", "string_value")])
        self.assertEqual(locate_reference_quote(value, "1.0")["matching_spans"][0]["json_pointer"], "/4")
        self.assertEqual(locate_reference_quote(-1, "1")["location_status"], "failed")

    def test_null_boolean_string_and_number_remain_distinct(self):
        value = [None, "null", True, "true", 1]
        nulls = locate_reference_quote(value, "null")["matching_spans"]
        self.assertEqual([x["source_kind"] for x in nulls], ["null", "string_value"])
        self.assertNotEqual(nulls[0]["source_value_hash"], nulls[1]["source_value_hash"])
        trues = locate_reference_quote(value, "true")["matching_spans"]
        self.assertEqual([x["source_kind"] for x in trues], ["boolean", "string_value"])
        self.assertEqual([x["json_pointer"] for x in locate_reference_quote(value, "1")["matching_spans"]], ["/4"])

    def test_empty_container_literals_are_present_and_typed(self):
        value = [{}, [], "{}", "[]", ""]
        objects = locate_reference_quote(value, "{}")["matching_spans"]
        arrays = locate_reference_quote(value, "[]")["matching_spans"]
        self.assertEqual([x["source_kind"] for x in objects], ["empty_object", "string_value"])
        self.assertEqual([x["source_kind"] for x in arrays], ["empty_array", "string_value"])
        self.assertNotEqual(objects[0]["source_value_hash"], objects[1]["source_value_hash"])
        self.assertEqual(locate_reference_quote({}, "{")["location_status"], "failed")
        self.assertEqual(locate_reference_quote("", "x")["location_status"], "failed")

    def test_cross_leaf_quote_and_serialized_nonempty_container_do_not_match(self):
        self.assertEqual(locate_reference_quote(["甲", "乙"], "甲乙")["location_status"], "failed")
        self.assertEqual(locate_reference_quote({"a": 1}, '{"a":1}')["location_status"], "failed")

    def test_json_looking_string_is_never_recursively_decoded(self):
        value = '{"a":null}'
        span = locate_reference_quote(value, "null")["matching_spans"][0]
        self.assertEqual((span["json_pointer"], span["source_kind"]), ("", "string_value"))
        self.assertNotEqual(reference_value_hash(value), reference_value_hash({"a": None}))

    def test_value_hash_preserves_types_and_input_is_unchanged(self):
        values = [None, "null", True, 1, 1.0, "1", {}, "{}", [], "[]"]
        self.assertEqual(len({reference_value_hash(v) for v in values}), len(values))
        original = {"b": ["text", {"x": False}], "a": None}
        before, keys = deepcopy(original), list(original)
        report = locate_reference_quote(original, "text")
        report["matching_spans"][0]["json_pointer"] = "/changed"
        self.assertEqual(original, before)
        self.assertEqual(list(original), keys)
        self.assertEqual(locate_reference_quote(original, "text")["matching_spans"][0]["json_pointer"], "/b/0")

    def test_receipt_order_is_stable_for_equivalent_object_order(self):
        self.assertEqual(locate_reference_quote({"b": "x", "a": "x"}, "x"),
                         locate_reference_quote({"a": "x", "b": "x"}, "x"))

    def test_nonfinite_and_non_json_python_values_are_rejected(self):
        for value in [float("nan"), float("inf"), -float("inf"), {"x": float("nan")},
                      {1: "one"}, {True: "one"}, ("a",), {"a"}, object()]:
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(ValueError):
                    locate_reference_quote(value, "x")
        class DerivedInt(int):
            pass
        with self.assertRaises(ValueError):
            reference_value_hash(DerivedInt(1))

    def test_cycles_rejected_but_shared_acyclic_values_allowed(self):
        recursive = []
        recursive.append(recursive)
        with self.assertRaises(ValueError):
            validate_json_value(recursive)
        shared = ["same"]
        matches = locate_reference_quote([shared, shared], "same")["matching_spans"]
        self.assertEqual([x["json_pointer"] for x in matches], ["/0/0", "/1/0"])

    def test_no_empty_or_nontext_quote_and_no_whitespace_rewrite(self):
        for quote in (None, 1, "", [], {}):
            with self.assertRaises(ValueError):
                locate_reference_quote("value", quote)
        span = locate_reference_quote(" leading ", " leading ")["matching_spans"][0]
        self.assertEqual((span["start"], span["end"]), (0, 9))

    def test_import_does_not_load_configuration_or_model_sdk(self):
        script = "import sys; import pipeline.reference_locations; assert 'config' not in sys.modules; assert 'openai' not in sys.modules"
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
