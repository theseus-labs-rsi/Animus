"""
pipeline.lines.L9_induction —— L9 条件归纳(IF-THEN 条件函数归纳:function-on-unseen)。

【它考什么(与 L8 的本质区别)】
  L8_transition = 状态机【下一合法阶段】:世界已声明一条【单向状态序】,系统回忆当前态 + 归纳生命周期序推进一步。
  L9_induction  = 从散落的【单条情境→动作】执行记录里,归纳一条【确定性阶跃规则 f: trigger 值 → 动作】,
                  然后在一个【从未见过的 trigger 值 x*】上【外推该函数】给出动作。
                  ★禁止在语料/题面写任何【一般化规则句】(如"若 X≥30 则…");只摆【单条情境→动作】的执行实例,
                  逼系统真去【归纳函数】,而非检索一条现成规则句。
  → 这是"函数外推"而非"状态推进":gold 由 apply_rule(rule, x*) 纯查表算,x* ∉ 实例集(真 held-out)。

【死命门守法】
  · gold = world_state.apply_rule(声明规则, x*) —— 纯代码查表/比较,零 LLM。gt() 护城河从 ws 重取规则重算。
  · 答案协议 = 闭选项 MC:把规则声明的【完整 canonical 动作集】(thresholds.action + default)作固定选项随题面给出;
    判分(eval/judge.judge_l9)对选项做 _norm 精确/子串定位命中项、纯代码 EM —— ★绝不回落 judge.py 的 llm_judge
    (自由文本会 default-on 触发 LLM 判分 → 违命门)。

【闸(全 code-falsifiable,well_posed 跑)】
  I0 规则合法      —— op 合法、≥1 threshold、cutoff 可解析、动作集去重后 ≥2 且【互不为 _norm 子串】(MC 定位无歧义)。
  I1 实例自洽      —— 每条实例 canon_action == apply_rule(rule, x)(实例确由该规则产生)。
  I2 全 arm 可观测 —— 每个声明动作臂(含 default)都有 ≥1 条实例(否则该臂在语料里根本没露过、无从归纳)。
  I3 真 held-out   —— x* ∉ 实例 x 集(_to_num 口径),问的是【没见过的输入】。
  I4 is_determined —— ★夹逼:x* 不落在任一 cutoff 的邻域(|x*-cutoff|≥margin)、且每 arm ≥3 点 →
                     使答案与 cutoff 精确值无关(x* 舒适地落在被 ≥3 点见证的臂内部)。
  I5 gold 护城河   —— gold == apply_rule(rule, x*) 且 gt() 重算恒等。
  ★canon 层        —— 每实例对同一 canonical 动作用【不同表面串】渲(surface ≠ canon 逐字);canon 映射只活侧信道,
                     gt/ground 走 canon —— 否则 naive query-RAG 逐字检索 train 文档即免费通关(检索到的表面串 ≠ 选项 canon 串,
                     EM 判错),"测归纳"是假的。well_posed 断言【答案臂】至少一条实例 surface ≠ canon 逐字。
  ★唯一性闸        —— 代码遍历【有限声明规则族】(同 op、同动作次序、cutoff 取实例中点网格),断言恰一【行为等价类】
                     与全实例一致、且它在 x* 上的预测 == gold;否则(实例没把规则钉死 / x* 不被钉定)→ drop。

诚实标注见文末 __main__ 与报告:canon 层 & 唯一性闸【均已落地】(非 stub)。
"""
from __future__ import annotations
import random
import re

from pipeline.lines.base import ProductionLine
from pipeline.world_state import (WorldState, _norm, _to_num, _date_of,
                                   apply_rule, rule_action_set)

_SEED = 20260717
_CUTOFF_MARGIN = 5.0          # I4 x* 距任一 cutoff 的最小间距(夹逼:答案与 cutoff 精确值无关)
_MIN_PER_ARM = 3              # I4 每 arm 最少实例点(≥3 → x* 落在被充分见证的臂内部)


# ── 规则模板(声明式阶跃规则 + canon 动作的表面同义池)────────────────────────────
#   canonical 动作 = 选项本体(随题面给出、判分 EM 目标);surface 池 = 实例渲进语料时用的【不同表面串】(canon 层)。
_RULE_TEMPLATE = {
    "rule_id": "R_dispatch", "trigger_field": "响应时长", "op": ">=", "unit": "分钟",
    "thresholds": [{"cutoff": 30, "action": "升级为紧急工单"},
                   {"cutoff": 60, "action": "上报主管"}],
    "default_action": "常规处理",
    "subject_noun": "工单",          # 实例/探针主体名词
}
# canon → 表面同义池(★每条实例轮流取不同表面串,且【均 ≠ canon 逐字】→ 检索捷径失效)
_SURFACE_POOL = {
    "常规处理":       ["按常规流程处置", "照标准流程办理", "作常规处置"],
    "升级为紧急工单": ["标记为紧急工单", "升为紧急工单处理", "转紧急工单处置"],
    "上报主管":       ["上报直属主管", "呈报主管审批", "上交主管处理"],
}
# 每 arm 的实例 trigger 值(全部【远离 cutoff】≥ margin;每臂 ≥3 点)——确定性,不随机
_ARM_XS = {
    "常规处理":       [8, 15, 22],     # x < 30
    "升级为紧急工单": [38, 45, 52],     # 30 ≤ x < 60
    "上报主管":       [68, 78, 92],     # x ≥ 60
}
_X_STAR = 47                          # held-out 探针值(∈中臂,∉实例集,距 30/60 均 ≥ margin)→ gold=升级为紧急工单


def heldout_leakage(order: dict, texts: list[str]) -> bool:
    """Find explicit held-out condition/action pairs, not unrelated numbers.

    This bounded check covers literal canonical/instance action wording. It
    does not claim to detect all paraphrases or generalized rule disclosure.
    """
    aux = order.get("aux") or {}
    subject = _norm(aux.get("subject_noun", "工单"))
    trigger = _norm(aux.get("trigger_field", ""))
    x_star = _to_num(aux.get("x_star"))
    if x_star is None or not trigger or not subject:
        return False
    actions = {_norm(action) for action in aux.get("options", [])}
    actions.update(_norm(instance.get("surface_action")) for instance in aux.get("instances", []))
    actions.discard("")
    unit = _norm(aux.get("unit", ""))
    for text in texts:
        for clause in re.split(r"[。！？\n]", str(text)):
            compact = _norm(clause)
            if subject not in compact or trigger not in compact or not any(action in compact for action in actions):
                continue
            if any(marker in compact for marker in ("未说明", "无法确定", "没有决定", "不确定", "尚未决定")):
                continue
            for number in re.finditer(r"(?<![\d.])[+-]?\d+(?:\.\d+)?(?![\d.])", compact):
                if _to_num(number.group()) != x_star:
                    continue
                prefix = compact[max(0, number.start() - len(trigger) - 12):number.start()]
                if trigger not in prefix:
                    continue
                if unit and not compact[number.end():].startswith(unit):
                    continue
                return True
    return False


def _candidate_boundaries(xs: list[float]) -> list[float]:
    """有限声明规则族的 cutoff 候选网格 = 相邻不同实例 x 的中点(边界只可能落在两点之间)。"""
    xs = sorted(set(xs))
    return [(xs[i] + xs[i + 1]) / 2.0 for i in range(len(xs) - 1)]


def _consistent_behavior_classes(rule: dict, instances: list[dict], x_star) -> dict:
    """★唯一性闸内核:遍历【同 op、同动作次序、cutoff 取候选网格】的有限规则族,
    收集与【全部实例】一致的候选,按其在 domain=(实例x ∪ {x*}) 上的预测行为签名去重成【行为等价类】。
    返回 {behavior_sig: 该类在 x* 的预测动作(_norm)}。len==1 且预测==gold ⇔ 实例把规则(在 x* 上)钉死。"""
    from itertools import combinations
    thr = sorted((rule.get("thresholds") or []), key=lambda t: _to_num(t.get("cutoff")))
    k = len(thr)
    xs = [_to_num(i.get("x")) for i in instances]
    xs = [x for x in xs if x is not None]
    xsv = _to_num(x_star)
    domain = sorted(set(xs + ([xsv] if xsv is not None else [])))
    # ★候选边界网格从【实例 x ∪ {x*}】取中点(务必含 x*):否则 x* 落在两臂之间的【空档】时,
    #   仅用实例中点会漏采能把 x* 判到另一臂的边界位置 → 假装"已钉死"。含 x* 才能暴露空档歧义。
    grid = _candidate_boundaries(domain)
    classes: dict = {}
    if k == 0 or len(grid) < k:
        return classes
    for cuts in combinations(grid, k):                 # 升序 cutoff 组合(单调阶跃)
        cand = dict(rule)
        cand["thresholds"] = [{"cutoff": c, "action": t.get("action")} for c, t in zip(cuts, thr)]
        if all(_norm(apply_rule(cand, i.get("x"))) == _norm(i.get("canon_action")) for i in instances):
            sig = tuple(_norm(apply_rule(cand, xv)) for xv in domain)
            classes[sig] = _norm(apply_rule(cand, x_star))
    return classes


def inject_rules(ws: WorldState, profile: dict = None, seed: int = _SEED) -> list[dict]:
    """声明一条确定性阶跃规则(→ ws.conditional_rules)+ 生成执行实例(→ ws.rule_instances 侧信道)。
    每实例 = 【单条情境→动作】,canon_action 由 apply_rule 产、surface_action 取【不同表面串】(canon 层)。
    幂等:已注入(ws.rule_instances 非空)则不重注。"""
    blueprint = getattr(ws, "world_blueprint", None) or {}
    if blueprint and not blueprint.get("legacy_adapter"):
        return []                                      # typed blueprint 尚未声明规则基质，禁止跨域注入固定工单生态
    if getattr(ws, "rule_instances", None):
        return ws.rule_instances
    rule = {k: v for k, v in _RULE_TEMPLATE.items() if k != "subject_noun"}
    subject = _RULE_TEMPLATE["subject_noun"]
    n_sess = ws.n_sessions or 1
    rng = random.Random(seed)
    surf_ptr = {a: 0 for a in _SURFACE_POOL}
    instances: list[dict] = []
    idx = 0
    for canon, xs in _ARM_XS.items():
        for x in xs:
            gold_action = apply_rule(rule, x)                      # I1:实例确由规则产
            pool = _SURFACE_POOL.get(gold_action) or [gold_action]
            surface = pool[surf_ptr.get(gold_action, 0) % len(pool)]
            surf_ptr[gold_action] = surf_ptr.get(gold_action, 0) + 1
            sess = idx % n_sess
            instances.append({
                "rule_id": rule["rule_id"], "inst_id": f"{subject}#{seed % 1000}-{idx:02d}",
                "x": x, "canon_action": gold_action, "surface_action": surface,
                "trigger_field": rule["trigger_field"], "unit": rule["unit"],
                "session": sess, "date": ws.date_of_session(sess),
            })
            idx += 1
    ws.conditional_rules = [rule]
    ws.rule_instances = instances
    # canon_map 只活侧信道(surface → canon),供审计/防检索说明;gt/ground 走 canon,不读语料表面串。
    ws.rule_instances_canon_map = {i["surface_action"]: i["canon_action"] for i in instances}
    return instances


class InductionLine(ProductionLine):
    id = "L9_induction"
    title = "条件归纳"
    memory = "从单条情境→动作实例归纳 IF-THEN 阶跃规则,并在未见 trigger 值上外推该函数(function-on-unseen)"
    gt_substrate = "声明式阶跃规则 ws.conditional_rules + apply_rule 纯查表(canon 层防检索;held-out x* 闭选项 MC)"
    implemented = True
    requires: list[str] = []          # 近根线:prepare 自带声明规则 + 实例注入
    auto_activate = True              # ★结构原型:注入规则基质(feasible)即自动激活,不靠议会列入 active_lines
    deterministic_phrasing = True     # ★出题走确定性(render.phrase_questions bypass LLM 润色):闭选项 MC 串须逐字保真,LLM 改写会打乱选项破坏 EM

    def prepare(self, ws, profile: dict):
        """世界基质:声明阶跃规则 + 注入执行实例(侧信道,确定性模板另渲)。幂等。"""
        pre = len(getattr(ws, "rule_instances", None) or [])
        got = inject_rules(ws, profile)
        if len(got) > pre:
            return (f"  ★条件归纳注入:1 条阶跃规则 + {len(got)} 条执行实例(单条情境→动作;canon 层不同表面串渲;"
                    f"gold=apply_rule 纯查表,held-out x* 闭选项 MC)")

    def feasible(self, ws, profile: dict) -> tuple[bool, str]:
        if getattr(ws, "rule_instances", None) and getattr(ws, "conditional_rules", None):
            return (True, f"ws.rule_instances 有 {len(ws.rule_instances)} 条实例 → 可产条件归纳 held-out 探针")
        return (False, "无条件归纳基质(ws.rule_instances/conditional_rules 空)→ 本线不激活")

    def _rule(self, ws, rule_id: str):
        for r in (getattr(ws, "conditional_rules", None) or []):
            if r.get("rule_id") == rule_id:
                return r
        return None

    def _instances(self, ws, rule_id: str) -> list[dict]:
        return [i for i in (getattr(ws, "rule_instances", None) or []) if i.get("rule_id") == rule_id]

    def enumerate(self, ws, target: int = 200, wp=None) -> list[dict]:
        """每条声明规则 → 一道 held-out MC 探针题:给一个 x* ∉ 实例集,问按归纳出的规则应采取哪个动作。"""
        out: list[dict] = []
        for rule in (getattr(ws, "conditional_rules", None) or []):
            if len(out) >= target:
                break
            insts = self._instances(ws, rule["rule_id"])
            if not insts:
                continue
            x_star = _X_STAR
            options = rule_action_set(rule)                        # 闭选项 = 完整 canonical 动作集(含 default)
            gold = apply_rule(rule, x_star)                        # gold 纯查表
            ev = sorted({i["session"] for i in insts})
            out.append({
                "line": self.id, "capability": "L9_induction",
                "entity": f"{_RULE_TEMPLATE['subject_noun']}·待判", "field": rule["trigger_field"],
                "gt": gold, "evidence_sessions": ev,
                "aux": {"rule_id": rule["rule_id"], "x_star": x_star, "options": options,
                        "trigger_field": rule["trigger_field"], "unit": rule.get("unit", ""),
                        "subject_noun": _RULE_TEMPLATE["subject_noun"],
                        "target_action": gold,
                        "instances": [{"inst_id": i["inst_id"], "x": i["x"],
                                       "canon_action": i["canon_action"], "surface_action": i["surface_action"],
                                       "session": i["session"]} for i in insts]},
            })
        return out

    def gt(self, ws, o: dict):
        """护城河:从 ws.conditional_rules 【重取规则】+ apply_rule(rule, x*) 重算(不信 aux.gt)。"""
        aux = o.get("aux") or {}
        rule = self._rule(ws, aux.get("rule_id"))
        if rule is None:
            return o.get("gt")
        return apply_rule(rule, aux.get("x_star"))

    def well_posed(self, order: dict, ws) -> tuple:
        """边 A 闸:I0..I5 + canon 层 + 唯一性闸。纯代码、零 LLM、不碰 corpus。"""
        aux = order.get("aux") or {}
        rule = self._rule(ws, aux.get("rule_id"))
        if rule is None:
            return ("drop", f"order 与世界脱钩:ws.conditional_rules 无 rule_id={aux.get('rule_id')!r}")
        insts = self._instances(ws, rule["rule_id"])
        options = rule_action_set(rule)
        x_star = aux.get("x_star")

        # I0 规则合法
        if rule.get("op") not in (">=", ">", "<=", "<"):
            return ("drop", f"I0 op 非法:{rule.get('op')!r}")
        thr = [t for t in (rule.get("thresholds") or []) if _to_num(t.get("cutoff")) is not None]
        if not thr:
            return ("drop", "I0 无合法 threshold(cutoff 不可解析)")
        if len(options) < 2:
            return ("drop", f"I0 动作集 <2(MC 无区分度):{options}")
        # 动作集互不为 _norm 子串(MC 定位无歧义)
        for a in options:
            for b in options:
                if a is not b and _norm(a) and _norm(a) in _norm(b) and _norm(a) != _norm(b):
                    return ("drop", f"I0 动作互为子串(MC 定位歧义):{a!r}⊂{b!r}")
        # I1 实例自洽
        for i in insts:
            if _norm(i.get("canon_action")) != _norm(apply_rule(rule, i.get("x"))):
                return ("drop", f"I1 实例不自洽:inst {i.get('inst_id')} x={i.get('x')} canon={i.get('canon_action')!r} ≠ apply_rule")
        # I2 全 arm 可观测
        seen_actions = {_norm(i.get("canon_action")) for i in insts}
        for a in options:
            if _norm(a) not in seen_actions:
                return ("drop", f"I2 动作臂无实例(语料没露过该臂,无从归纳):{a!r}")
        # I3 真 held-out
        inst_xs = {_to_num(i.get("x")) for i in insts}
        if _to_num(x_star) in inst_xs:
            return ("drop", f"I3 x*={x_star} 落在实例集内(非 held-out)")
        # I4 is_determined:x* 距每 cutoff ≥ margin + 每 arm ≥ _MIN_PER_ARM 点
        xsv = _to_num(x_star)
        if xsv is None:
            return ("drop", f"I4 x* 不可解析:{x_star!r}")
        for t in thr:
            if abs(xsv - _to_num(t.get("cutoff"))) < _CUTOFF_MARGIN:
                return ("drop", f"I4 x*={x_star} 落在 cutoff={t.get('cutoff')} 邻域(<{_CUTOFF_MARGIN},答案依赖 cutoff 精确值)")
        from collections import Counter
        arm_cnt = Counter(_norm(i.get("canon_action")) for i in insts)
        for a in options:
            if arm_cnt.get(_norm(a), 0) < _MIN_PER_ARM:
                return ("drop", f"I4 动作臂 {a!r} 仅 {arm_cnt.get(_norm(a),0)} 点(<{_MIN_PER_ARM},夹逼不足)")
        # I5 gold 护城河
        gold = apply_rule(rule, x_star)
        if _norm(order.get("gt")) != _norm(gold):
            return ("drop", f"I5 gold {order.get('gt')!r} ≠ apply_rule(rule,x*) {gold!r}")
        if _norm(self.gt(ws, order)) != _norm(gold):
            return ("drop", "I5 护城河漏:gt() 重算 ≠ apply_rule")
        # ★canon 层:答案臂至少一条实例 surface ≠ canon 逐字(否则检索捷径可逐字命中选项)
        ans_surfaces = [i.get("surface_action") for i in insts if _norm(i.get("canon_action")) == _norm(gold)]
        if not any(_norm(s) != _norm(gold) for s in ans_surfaces):
            return ("drop", "canon 层失效:答案臂全部实例 surface==canon 逐字(检索可免费通关)")
        # ★唯一性闸:有限声明规则族里恰一行为等价类与全实例一致,且它在 x* 上预测 == gold
        classes = _consistent_behavior_classes(rule, insts, x_star)
        if len(classes) != 1:
            return ("drop", f"唯一性闸:与全实例一致的行为等价类数={len(classes)}(≠1,实例未把规则钉死)")
        only_pred = next(iter(classes.values()))
        if only_pred != _norm(gold):
            return ("drop", f"唯一性闸:唯一一致类在 x* 预测 {only_pred!r} ≠ gold {_norm(gold)!r}")
        return ("well_posed", "")

    def ground(self, order, evidence_docs, all_signal_text=""):
        """Verify arm witnesses and reject an explicit held-out condition/action pair."""
        from pipeline.grounding import attributed
        aux = order.get("aux") or {}
        insts = aux.get("instances") or []
        docs = [d["content"] for d in evidence_docs]
        x_star = str(aux.get("x_star"))
        options = aux.get("options") or []
        visible = docs + ([all_signal_text] if all_signal_text else [])
        if heldout_leakage(order, visible):
            return ("drop", f"held-out 条件与处置结果已直接出现在正文:x*={x_star}")
        # 每臂至少一条实例 surface 就近其 inst_id
        witnessed = set()
        for i in insts:
            if attributed(i.get("surface_action"), i.get("inst_id"), docs):
                witnessed.add(_norm(i.get("canon_action")))
        for a in options:
            if _norm(a) not in witnessed:
                return ("drop", f"接地:动作臂「{a}」无实例 surface 就近渲进语料 → 无从归纳该臂")
        return ("grounded", f"接地:{len(witnessed)} 臂均有实例 surface 就近渲入(可归纳规则),held-out x*={x_star} 外推")

    def intent(self, o: dict) -> tuple[str, list]:
        """★闭选项 MC:题面给出 held-out x* + 完整 canonical 动作选项集;禁写一般化规则句、禁泄漏历史实例。
        ★该题走【确定性出题】(render.phrase_questions 对本线 bypass LLM 润色)——否则 LLM 改写会打乱选项串、破坏 EM。"""
        aux = o.get("aux") or {}
        subject = aux.get("subject_noun", "工单")
        tf, unit, x_star = aux.get("trigger_field", ""), aux.get("unit", ""), aux.get("x_star")
        options = aux.get("options") or []
        opt_lines = "  ".join(f"{chr(ord('A') + k)}. {a}" for k, a in enumerate(options))
        s = (f"某系统按一条【固定不变的处置规则】处理每一个{subject}(该规则从未直接告知,需你从历史处置记录中【归纳】)。"
             f"现有一个【全新的{subject}】:其「{tf}」为 {x_star}{unit}。"
             f"问:按该系统一贯的处置规则,对这个{subject}应采取以下哪一项动作?\n"
             f"【选项】{opt_lines}\n"
             f"★只从上面选项里回一个【动作名】(照抄动作文字,不要只回字母、不要自造动作、不要复述规则)。")
        # hide=[]:MC 选项本就含 gold(闭选项协议),无须隐藏;历史实例不进题面(它们在语料里)。
        return s, []


# 自检:python -m pipeline.lines.L9_induction
if __name__ == "__main__":
    import sys

    line = InductionLine()
    checks: list[tuple[bool, str]] = []

    def ck(name, cond):
        checks.append((bool(cond), name))

    ws = WorldState({}, n_sessions=6)

    # ── prepare 注入 ──
    note = line.prepare(ws, {})
    ck("prepare 注入规则(conditional_rules 非空)", bool(ws.conditional_rules))
    ck("prepare 注入实例(rule_instances 非空)", bool(ws.rule_instances))
    ck("prepare 幂等(二次不叠加)", (line.prepare(ws, {}) is None) and len(ws.rule_instances) == 9)
    ck("实例 = 3 臂 × 3 点 = 9", len(ws.rule_instances) == 9)
    ck("canon 层:每实例 surface ≠ canon 逐字",
       all(_norm(i["surface_action"]) != _norm(i["canon_action"]) for i in ws.rule_instances))
    ck("同 canon 动作用了 ≥2 种不同表面串",
       len({i["surface_action"] for i in ws.rule_instances if i["canon_action"] == "升级为紧急工单"}) >= 2)

    # ── feasible ──
    ck("feasible: 有规则实例 → True", line.feasible(ws, {})[0])
    ck("feasible: 空世界 → False", not line.feasible(WorldState({}, n_sessions=1), {})[0])

    # ── enumerate:held-out MC 探针 ──
    orders = line.enumerate(ws)
    ck("enumerate 出 1 道探针", len(orders) == 1)
    o = orders[0]
    ck("gold = apply_rule(rule, x*=47) = 升级为紧急工单", o["gt"] == "升级为紧急工单")
    ck("aux.options = 完整动作集(含 default)",
       set(o["aux"]["options"]) == {"升级为紧急工单", "上报主管", "常规处理"})
    ck("x*=47 ∉ 实例 x 集(held-out)", _to_num(o["aux"]["x_star"]) not in {_to_num(i["x"]) for i in ws.rule_instances})

    # ── 护城河:gt() 从世界重取规则重算 == gold ──
    ck("护城河:gt() == gold(apply_rule 重算)", _norm(line.gt(ws, o)) == _norm(o["gt"]))
    # 篡改 aux.gt 不影响护城河(gt() 不信 aux.gt)
    o_tamper = dict(o); o_tamper["gt"] = "上报主管"
    ck("护城河抗篡改:aux.gt 被改仍重算真值", _norm(line.gt(ws, o_tamper)) == "升级为紧急工单")

    # ── well_posed:合法探针过闸 ──
    st, why = line.well_posed(o, ws)
    ck(f"well_posed: 合法探针 → well_posed(why={why})", st == "well_posed")

    # ── well_posed 各闸反例 ──
    # I3 x* 落进实例集
    bad3 = {**o, "aux": {**o["aux"], "x_star": 45}}   # 45 是中臂实例点
    ck("well_posed I3: x* 落实例集 → drop", line.well_posed(bad3, ws)[0] == "drop")
    # I4 x* 落 cutoff 邻域
    bad4 = {**o, "aux": {**o["aux"], "x_star": 31}, "gt": apply_rule(ws.conditional_rules[0], 31)}
    st4, why4 = line.well_posed(bad4, ws)
    ck("well_posed I4: x* 贴 cutoff(31)→ drop", st4 == "drop" and "I4" in why4)
    # I5 gold 篡改
    bad5 = {**o, "gt": "上报主管"}
    ck("well_posed I5: gold 篡改 → drop", line.well_posed(bad5, ws)[0] == "drop")

    # I4/唯一性:每臂 <3 点 或 规则没被钉死 → drop(手构稀疏世界)
    ws_sparse = WorldState({}, n_sessions=4)
    ws_sparse.conditional_rules = [dict(ws.conditional_rules[0])]
    # 只给中臂 1 点、其余臂 1 点 → I2 过但 I4(<3 点)drop
    ws_sparse.rule_instances = [
        {"rule_id": "R_dispatch", "inst_id": "工单#0", "x": 10, "canon_action": "常规处理",
         "surface_action": "按常规流程处置", "session": 0},
        {"rule_id": "R_dispatch", "inst_id": "工单#1", "x": 45, "canon_action": "升级为紧急工单",
         "surface_action": "标记为紧急工单", "session": 1},
        {"rule_id": "R_dispatch", "inst_id": "工单#2", "x": 80, "canon_action": "上报主管",
         "surface_action": "上报直属主管", "session": 2},
    ]
    sparse_orders = line.enumerate(ws_sparse)
    st_sp, why_sp = line.well_posed(sparse_orders[0], ws_sparse)
    ck(f"well_posed I4: 每臂 1 点(<3)→ drop(why={why_sp})", st_sp == "drop" and "I4" in why_sp)

    # ── canon 层反例:答案臂全部 surface==canon 逐字 → drop ──
    ws_leak = WorldState({}, n_sessions=6)
    line.prepare(ws_leak, {})
    for i in ws_leak.rule_instances:                    # 把中臂(答案臂)surface 全改成 canon 逐字
        if i["canon_action"] == "升级为紧急工单":
            i["surface_action"] = "升级为紧急工单"
    leak_orders = line.enumerate(ws_leak)
    st_lk, why_lk = line.well_posed(leak_orders[0], ws_leak)
    ck(f"well_posed canon 层: 答案臂 surface==canon → drop(why={why_lk})",
       st_lk == "drop" and "canon" in why_lk)

    # ── 唯一性闸反例:实例没把 x* 那侧边界钉死(中臂只在 x* 一侧有点)→ 多行为类 → drop ──
    ws_amb = WorldState({}, n_sessions=6)
    ws_amb.conditional_rules = [dict(ws.conditional_rules[0])]
    # 中臂 3 点全在 [30,38](x*=47 右侧空) + 高臂点距远 → 边界可在 (38,68) 间游移跨过 47
    ws_amb.rule_instances = [
        {"rule_id": "R_dispatch", "inst_id": f"工单#L{k}", "x": x, "canon_action": "常规处理",
         "surface_action": "按常规流程处置", "session": 0} for k, x in enumerate([8, 15, 22])
    ] + [
        {"rule_id": "R_dispatch", "inst_id": f"工单#M{k}", "x": x, "canon_action": "升级为紧急工单",
         "surface_action": "标记为紧急工单", "session": 1} for k, x in enumerate([30, 33, 36])
    ] + [
        {"rule_id": "R_dispatch", "inst_id": f"工单#H{k}", "x": x, "canon_action": "上报主管",
         "surface_action": "上报直属主管", "session": 2} for k, x in enumerate([68, 78, 92])
    ]
    amb_orders = line.enumerate(ws_amb)
    st_amb, why_amb = line.well_posed(amb_orders[0], ws_amb)
    ck(f"well_posed 唯一性闸: 中臂在 x* 一侧空(边界游移跨 x*)→ drop(why={why_amb})",
       st_amb == "drop" and "唯一性" in why_amb)

    # ── ground:每臂就近渲入 + held-out ──
    docs = []
    for i in ws.rule_instances:
        docs.append({"content": f"{_date_of(i['session'])} 处置记录:{i['inst_id']} 本次「响应时长」{i['x']}分钟,"
                                 f"值班据此对该{i['inst_id']}作出处置——{i['surface_action']}。归档。",
                     "session": i["session"]})
    st_g, why_g = line.ground(o, docs)
    ck(f"ground: 各臂就近渲入 → grounded(why={why_g})", st_g == "grounded")
    # 缺一臂 → drop
    docs_miss = [d for d in docs if "升级" not in d["content"] and "紧急工单" not in d["content"]]
    ck("ground: 缺答案臂实例 → drop", line.ground(o, docs_miss)[0] == "drop")

    # ── intent:MC 选项随题面 + 不写规则句 + 不泄历史实例 ──
    it, hide = line.intent(o)
    ck("intent: 题面含 held-out x*=47", "47" in it)
    ck("intent: 选项集随题面(三动作都在)",
       all(a in it for a in ["升级为紧急工单", "上报主管", "常规处理"]))
    ck("intent: 不写一般化规则句(无'若…则'/'≥'/'>=' 阈值句)",
       ("若" not in it) and ("≥" not in it) and (">=" not in it) and ("30" not in it) and ("60" not in it))
    ck("intent: 不泄历史实例 inst_id", all(i["inst_id"] not in it for i in ws.rule_instances))
    ck("intent: hide 为空(闭选项含 gold,无须隐藏)", hide == [])

    npass = sum(1 for ok, _ in checks if ok)
    for ok, name in checks:
        if not ok:
            print(f"  ✗ {name}")
    print(f"[L9_induction self-test] {npass}/{len(checks)} PASS")
    sys.exit(0 if npass == len(checks) else 1)
