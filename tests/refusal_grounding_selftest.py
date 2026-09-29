"""Regression shapes from fresh insurance Q17; synthetic identifiers only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.lines.L6_refusal import RefusalLine, field_ownership_mentions
from pipeline.world_state import INSUFFICIENT


ENTITY, FIELD, REPORT = "测试客户甲", "分析确认状态", "测试2024年度分析报告"
ORDER = {"line": "L6_refusal", "capability": "L6_refusal", "entity": ENTITY, "field": FIELD,
         "gt": INSUFFICIENT, "aux": {"refusal_type": "T1_adjacent", "lure": {"value": "区域乙"}}}
LURE = {"session": 0, "content": f"{ENTITY}所属市场为区域乙。"}


class RefusalAttributionTests(unittest.TestCase):
    def test_four_real_clause_shapes_clear_other_owner_or_absence(self):
        examples = [
            f"《{REPORT}》的分析客户登记为{ENTITY}；采用来源、采用口径与{FIELD}在本期尚无记录，不得据此标作已采用或已确认。",
            f"对《{REPORT}》进行重新确认，其{FIELD}为已确认；采用来源为《测试财务报告》；客户{ENTITY}档案阶段为审核完毕。",
            f"{ENTITY}所属市场为区域乙，档案阶段为审核完毕；《{REPORT}》以该集团为客户，{FIELD}为已确认。",
            f"经复核操作，{REPORT}的{FIELD}调整为已确认，其关联客户{ENTITY}的档案阶段随之更新为审核完毕。",
        ]
        for example in examples:
            with self.subTest(text=example):
                findings = field_ownership_mentions(ENTITY, FIELD, [example])
                self.assertEqual({item["status"] for item in findings}, {"cleared"})
                self.assertEqual(RefusalLine().ground(ORDER, [LURE, {"session": 0, "content": example}])[0], "grounded")

    def test_explicit_target_assertion_still_rejects_even_with_correct_other_owner(self):
        for text in (f"{ENTITY}的{FIELD}为已确认。", f"{ENTITY}{FIELD}为已确认。",
                     f"《{REPORT}》的{FIELD}为已确认；{ENTITY}的{FIELD}为已确认。"):
            self.assertEqual(RefusalLine().ground(ORDER, [LURE, {"session": 0, "content": text}])[0], "drop")
            self.assertIn("target", {item["status"] for item in field_ownership_mentions(ENTITY, FIELD, [text])})

    def test_ambiguous_owner_stays_dropped_without_claiming_field_exists(self):
        for text in (f"{ENTITY}的档案已到。其{FIELD}为已确认。",
                     f"{ENTITY}在场；客户的{FIELD}为已确认。",
                     f"{ENTITY}已核对；{FIELD}为已确认。",
                     f"{ENTITY}的{FIELD}是什么？"):
            status, reason = RefusalLine().ground(ORDER, [LURE, {"session": 0, "content": text}])
            self.assertEqual(status, "drop")
            self.assertIn("归属歧义待审", reason)
            self.assertNotIn("其实有", reason)

    def test_absence_is_not_a_blanket_exemption_for_later_assertion(self):
        absent = f"{ENTITY}的{FIELD}在本期尚无记录。"
        self.assertEqual(RefusalLine().ground(ORDER, [LURE, {"session": 0, "content": absent}])[0], "grounded")
        later = f"{ENTITY}的{FIELD}为已确认。"
        self.assertEqual(RefusalLine().ground(ORDER, [LURE, {"session": 0, "content": absent + later}])[0], "drop")
        # Whole visible signal text can disprove refusal beyond the chosen pool.
        self.assertEqual(RefusalLine().ground(ORDER, [LURE], later)[0], "drop")
        for contrast in (f"{ENTITY}的{FIELD}未记录，但实际为已确认。",
                         f"不是别人的{FIELD}，而是{ENTITY}的。",
                         f"不是《{REPORT}》的{FIELD}，而是{ENTITY}的。"):
            status, reason = RefusalLine().ground(ORDER, [LURE, {"session": 0, "content": contrast}])
            self.assertEqual(status, "drop")
            self.assertIn("归属歧义待审", reason)

    def test_lure_requirement_is_preserved(self):
        self.assertEqual(RefusalLine().ground(ORDER, [{"session": 0, "content": "只记录无关信息。"}])[0], "drop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
