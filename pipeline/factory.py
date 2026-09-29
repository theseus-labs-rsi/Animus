"""
pipeline.factory —— Benchmark 工厂【薄装配点】+ CLI(原 run_factory_v2,拆分后瘦身)。

数据流 = stage 序列(见 docs/anchors/run_system_design.md §1):
  input → whitepaper → world → disclosure → orders → well_posed → corpus → questions → grounding
各 stage 的重逻辑分散在专门模块(world_gen / render / lines / well_posed / grounding / central_office /
closed_loop);此处只放:场景输入 + stage 薄包装 + STAGES 注册 + CLI。

用法:
  nohup ./venv/bin/python -u -m pipeline.factory --scenario office --target-mtokens 1.0 > /tmp/f.log 2>&1 &
  ./venv/bin/python -m pipeline.factory --scenario office --to world
  ./venv/bin/python -m pipeline.factory --min-questions 80 --per-line L2_relational=15   # 闭环旋钮
  ./venv/bin/python -m pipeline.factory --list-runs
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import argparse, hashlib, json, math, sys, time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.world_state import WorldState
from pipeline.lines import run_lines, prepare_lines as _prepare_lines
from pipeline.central_office import central_office
from pipeline.world_gen import build_world
from pipeline.world_blueprint import WorldBlueprintError, normalize_world_blueprint, relation_capacity
from pipeline.render import render_corpus, phrase_questions, corpus_scale
from pipeline.run import Run, Stage, drive, list_runs, latest_run_for, new_run_id, RUNS_DIR, _run_stage
from pipeline.targetspec import TargetSpec
from pipeline.closed_loop import build_to_target
from tools.diversity_metrics import report as diversity_report
from pipeline.seed_pack import SeedPackError
from pipeline.seed_run import (prepare_seed_input, seed_config, validate_seed_input,
                               validate_seed_identity)
from pipeline.seed_world import validate_seed_world


SCENARIOS = {
    "office": {
        "description": (
            "构造一个真实公司的运营世界：员工受雇并归属部门，部门控制预算，项目跨部门协作并由负责人承担，"
            "里程碑与项目依赖决定推进。世界由入职/离职/调岗/任命生效、预算申请/审批/调整、项目立项/延期/"
            "里程碑验收等事件驱动；人事通知、组织名册、预算审批单、项目周报分别观察这个世界的不同切面。"),
        "few_shot": [
            {"title": "AI工程周报 W17", "content": "本周 P0缺陷率 20%,Oncall 10。负责人:张三。汇报对象:CTO。",
             "doc_type": "周报", "date": "2025-04-17"},
            {"title": "数据平台通报 W17", "content": "本周 SLA 99.1%,值班 8 人。负责人:王五。",
             "doc_type": "通报", "date": "2025-04-17"},
        ],
    },
    "medical": {
        "description": (
            "构造一个慢病诊疗世界：患者经历不规则就诊，诊断、检验指标、处方和用药方案分属不同对象；临床医生"
            "负责诊疗，药物通过处方与患者关联。就诊、检验出结果、确诊/修订诊断、开药/停药、转诊等事件改变病程；"
            "门诊病历、检验报告、处方单和会诊记录存在不同时间戳与观察延迟。"),
        "few_shot": [
            {"title": "门诊记录 0312", "content": "患者本次血压 150/95,空腹血糖 7.8,主治:李医生,会诊上级:王主任。予二甲双胍。本次随访方式:门诊复诊。",
             "doc_type": "门诊记录", "date": "2025-03-12"},
            {"title": "检查报告 0312", "content": "糖化血红蛋白 8.2%,诊断:2型糖尿病。主治:周医生。",
             "doc_type": "检查报告", "date": "2025-03-12"},
        ],
    },
    "legal": {
        "description": (
            "构造一个律所案件世界：案件连接委托人、对手方、承办律师、合伙人、证据与法院程序；案件按立案、"
            "举证、庭审、裁判、执行或和解等生命周期推进。委托建立、律师改派、证据提交、开庭、裁判、和解与"
            "费用入账是领域事件；委托协议、法院文书、证据目录、庭审纪要和工时账单是不同权威层级的证据。"),
        "few_shot": [
            {"title": "案件评审纪要 M0312", "content": "本次评审:争议标的额 320万,累计计费工时 86,风险评分 4。主办律师:陈律,督导合伙人:周合伙人。本期处理策略倾向:庭外和解。",
             "doc_type": "评审纪要", "date": "2025-03-12"},
            {"title": "对手方备忘 M0312", "content": "对手方:鼎兴贸易;争议焦点:供货违约。主办律师:李律。",
             "doc_type": "对手方备忘", "date": "2025-03-12"},
        ],
    },

    # 六个领域输入只描述各自世界，不预埋“必须喂齐哪些能力线”的统一配方。
    "game": {
        "description": (
            "构造一个开放世界 RPG 的进程世界：玩家角色拥有等级与装备，任务指向地区或首领，首领有存活/击败状态"
            "并掉落装备，阵营控制地区且影响任务解锁。探索、接取任务、战斗、击败首领、掉落、拾取、装备、升级、"
            "完成任务与解锁区域组成核心循环；任务日志、战斗记录、战利品清单和角色面板是证据渠道，时间按章节/遭遇推进。"),
        "few_shot": [
            {"title": "第七章 · 剧情日志", "content": "灰袍贤者对玩家好感度升至 70,现效忠『银鹿议会』。当前任务:护送商队(进行中)。本章玩家抉择:谋略。",
             "doc_type": "剧情日志", "date": "2025-05-01"},
            {"title": "第七章 · 角色档案", "content": "黑剑佣兵等级 18,所属阵营『银鹿议会』,效忠领主:贤者。称玩家为『盟友』。",
             "doc_type": "角色档案", "date": "2025-05-01"},
        ],
    },
    "agent": {
        "description": (
            "构造一个长程 Agent 执行世界：一次 run 包含带依赖的任务 DAG，任务由执行体调度并调用工具，工具产生"
            "观测或 artifact，验证器决定产物是否可接受。调度、工具调用、超时/失败、重试、改路由、产物生成、验证"
            "通过与回滚是核心事件；执行 trace、工具日志、artifact 元数据和验证报告分别记录不同层级的事实，时间按执行轮推进。"),
        "few_shot": [
            {"title": "轮次 R12 · 执行轨迹", "content": "子任务『抓取财报』状态:成功。所用工具:web_fetch(成功率 0.82)。执行体:agent_A。本轮修复策略:更换工具。",
             "doc_type": "执行轨迹", "date": "2025-05-02"},
            {"title": "轮次 R12 · 工具调用记录", "content": "工具 sql_query 返回超时,重试 2 次。负责执行体:agent_B,上级编排器:planner_main。",
             "doc_type": "工具调用记录", "date": "2025-05-02"},
        ],
    },
    "cs": {
        "description": (
            "构造一个企业客户成功世界：客户账户包含联系人与合同，服务工单受 SLA 约束并由客服/支持队列承接，"
            "续约风险由未解决问题和承诺履行共同影响。工单创建、分派、升级、响应、解决/重开、责任转移、承诺兑现"
            "与续约是领域事件；CRM 纪要、工单流水、SLA 告警、合同与交接记录具有不同权威性和时效。"),
        "few_shot": [
            {"title": "服务周期 W19 · 工单记录", "content": "客户『云图科技』分层:战略级。满意度 4.6。负责客服:小林。当前工单:计费异常(处理中)。本次接触渠道:电话。",
             "doc_type": "工单记录", "date": "2025-05-09"},
            {"title": "服务周期 W19 · 交接备忘", "content": "客户『云图科技』承诺履约率 92%。负责客服:小林,客服主管:陈经理。",
             "doc_type": "交接备忘", "date": "2025-05-09"},
        ],
    },
    "companion": {
        "description": (
            "构造一个长期陪伴助手的受限记忆世界：用户与现实联系人、持续议题、承诺/纪念日、提醒和隐私边界是不同对象；"
            "一次互动可提出、澄清或撤回事实，议题可出现、缓解或解决，关系称谓与提醒会被更新。对话记录、用户明确更正、"
            "提醒确认和边界设置是不同证据渠道；敏感内容只保存允许的抽象边界，不把具体秘密变成普通字段。"),
        "few_shot": [
            {"title": "互动记录 0510", "content": "本次情绪基线:平稳。压力水平 3/10。主要倾诉对象:大学室友。当前关注议题:转岗准备。本次偏好支持方式:给出建议。",
             "doc_type": "互动记录", "date": "2025-05-10"},
            {"title": "事件备忘 0510", "content": "用户提及 6 月 2 日为其母亲生日(重要纪念日)。主要倾诉对象:大学室友,关系角色:挚友。",
             "doc_type": "事件备忘", "date": "2025-05-10"},
        ],
    },
    "assistant": {
        "description": (
            "构造一个个人生活助手世界：用户、家庭成员、住址/房间、设备、例程、订单、订阅和硬约束分别建模；"
            "设备加入/迁移/离线、例程创建/触发/覆盖、下单/退货、订阅续费/取消、过敏或禁区更新等事件改变世界。"
            "购物凭证、设备事件流、日历、订阅通知和用户明确指令是证据渠道，且账户、家庭成员与设备归属不能混淆。"),
        "few_shot": [
            {"title": "交互记录 0511", "content": "本次下单:厨房用品。月度预算 2000 元。常用账户:家庭主号。本次选购取向:性价比优先。已知过敏原:花生。",
             "doc_type": "交互记录", "date": "2025-05-11"},
            {"title": "设备设置 0511", "content": "智能家居主控设备:客厅音箱,所在房间场景:客厅。推荐命中率 0.74。",
             "doc_type": "设备设置", "date": "2025-05-11"},
        ],
    },
    "kb": {
        "description": (
            "构造一个企业知识治理世界：知识条目由版本构成，版本引用来源并可能依赖其他条目；维护人与领域 owner 承担"
            "审核责任，消费系统引用已发布版本。起草、评审、发布、替代、回滚、废弃、来源失效与依赖断裂是核心事件；"
            "条目正文、版本 diff、评审意见、发布记录、引用图和失效公告共同构成可观察证据，时间按发布批次推进。"),
        "few_shot": [
            {"title": "知识条目 KB-204 · v3", "content": "《退款流程规范》升级至 v3,有效状态:现行。维护人:老周。引用次数 1280,准确率 0.95。本次发布策略:全量发布。",
             "doc_type": "知识条目", "date": "2025-05-12"},
            {"title": "变更记录 KB-204", "content": "v2→v3 修订退款时限。维护人:老周,领域负责人:支付组组长。",
             "doc_type": "变更记录", "date": "2025-05-12"},
        ],
    },
}



ART = {"input": "00_input.json", "whitepaper": "01_whitepaper.json", "world": "02_world.json",
       "orders": "03_orders.json", "questions": "04_questions.json", "disclosure": "02_disclosure_ready.json",
       "corpus": "05_corpus.json",
       "grounding": "06_grounded_questions.json", "quality": "07_release.json",
       "calibration": "08_calibration.json", "selection": "09_selection.json"}
CORPUS_CKPT = "05_corpus.ckpt.json"
CORPUS_WARNING = "05_corpus_warning.json"
QUESTION_WARNING = "04_questions_warning.json"
ORDER_WARNING = "03_orders_warning.json"
WORLD_REPAIR_FAILURE = "02_world_repair_failure.json"
CORPUS_RENDER_CONTRACT_VERSION = 12

# ★作答协议(B类①修复):benchmark 出厂【显式声明】None 的两类语义 + 期望作答,治"None 未定义→理性系统被误判"。
#   契约层一处声明(非逐题补丁),所有 None 题共享;eval 侧据此把 gold 哨兵映射到人类作答。
LEGACY_ANSWER_PROTOCOL = {
    "version": 5,
    "rules": [
        "普通问题:答该项在【题面所指时点】的具体值。",
        "直接给出简短最终答案。状态、类别及闭选项使用资料中的原值或合同明示别名；不要自行合并相近业务状态。",
        "【截至最新一期】:锚点 = 该实体最后一次有效记录(不是全局最后一周),沿用其最近有效值(carry-forward)。"
        "全程在场的实体折到全局最后一期,早退场的实体折到它最后出现那一期——这是【同一条沿用规则】碰上不同寿命,不是两套口径。",
        "【整体趋势题】整体趋势 = 【首末净方向】(看整段、以首期 vs 末期的净变化为准);中途或末期的【局部反跳不改判】。只回『上升』或『下降』。"
        "(末期常有一次反向跳动,是【考你别只看最近一两期】,按首末净方向作答即可。)",
        "【属性归属】每个属性只属于【它本来记录在其上的那类实体】。问某实体它【自身从无记录】的属性(哪怕同名/关联实体有该属性)→ 答『无此项/查无此记录』,"
        "★不得经关系链折算到关联实体的值(如问『某律师的争议焦点』:争议焦点是案件的属性、律师本身没有 → 查无,不要去取他经办案件的争议焦点)。",
        "【从未涉及/不存在】:所问项在本场景根本没有(gold 标记 INSUFFICIENT,产线 ABS)→ 期望答『无此项/查无此记录』。",
        "【曾有但已显式停止统计】:所问项曾被跟踪、现已停更(gold forgotten=True,产线 FORGET)→ 期望答『已停止统计/不再跟踪』。",
        "【所问时间超出资料范围】:不能把已知历史值外推到未记录的未来时点，答『信息不足/不在记录范围内』。这与从未有记录、已停止统计不同。",
        "【停统前的最后值】:若问的是『停止统计前最后一次』(产线 PREEXPIRE)→ 这是另一类问法,照常答停掉那一刻的值(非 None)。",
        "只评分问题明确要求的主答案；排序题比较完整事件及先后。未要求的附带日期和解释不会自动获得事实正确性认证。",
    ],
    "attribute_ownership_no_fold": True,    # ★个人/角色不具案件级属性,问及判查无、不经关系折算(run112358 D:Q50-56 拒答属性归属未声明 → 此处声明)
    "trend_means_net_first_to_last": True,  # ★L7 趋势=首末净方向、末期反跳不改判(run110317 D 点"假二选一":协议没定义非单调如何裁 → 此处定义,二选一即公平、recency 陷阱保住)
    "latest_means_carry_forward": True,     # ★"最新一期"=该实体末次有效记录沿用,统一口径(run160053 D 点的"双标"实为读者侧表面歧义,gold 本就单一真源 latest_valid)
    "two_none_types_distinguished": True,   # ★区分"从未存在"(ABS)vs"曾有已停"(FORGET)是考点
    "scoring_scope": "primary_answer",
    "additional_facts": "not_assessed",
    "gold_sentinel_map": {"INSUFFICIENT": "无此项/查无此记录", "forgotten=true": "已停止统计/不再跟踪",
                          "out_of_scope": "信息不足/不在记录范围内"},
}

from copy import deepcopy
from eval.answer_task_review import POLICY_VERSION

ANSWER_PROTOCOL = deepcopy(LEGACY_ANSWER_PROTOCOL)
ANSWER_PROTOCOL.update(version=7, scoring_policy=POLICY_VERSION,
                       scoring_scope="task_with_supporting_reasons",
                       additional_facts="record_unrelated_separately")
ANSWER_PROTOCOL["rules"][-1] = (
    "评分遵循下方公开政策：题目要求的主要任务及回答直接支撑它的关键理由均在范围内；"
    "允许自然简略和有效的不同证据路径，确实无关的附言另记。")
ANSWER_PROTOCOL["rules"].insert(2,
    "【周期编号】资料的 session/周期编号从0开始：编号0是第1期，编号1是第2期，以此类推。"
    "题面中的第N周/期使用从1开始的自然序号；日期仍按资料的实际日期理解。")
ANSWER_PROTOCOL["rules"].insert(3,
    "【历史时点的记录状态】题面所问时点在资料覆盖的周期内时，采用截至该时点最后一条有效记录的值；"
    "当期没有新的有效状态记录，沿用此前最近有效值。明确停止统计、撤销或替代记录须据其含义处理。"
    "这不表示现实中所有事件均会被记录，也不允许向资料范围外的未来外推。")


def _answer_protocol(wp):
    policy = (wp.get("quality_contract") or {}).get("scoring_policy")
    if policy not in (None, POLICY_VERSION):
        raise ValueError("Unknown whitepaper scoring policy")
    return ANSWER_PROTOCOL if policy else LEGACY_ANSWER_PROTOCOL


def stage_input(run: Run):
    scenario = prepare_seed_input(run)
    if scenario is None:
        if run.scenario not in SCENARIOS:
            raise ValueError(f"未知场景 {run.scenario!r}；使用内置 --scenario 或 --seed-pack")
        scenario = SCENARIOS[run.scenario]
    run.write(ART["input"], scenario)
    about = {"answer_protocol": ANSWER_PROTOCOL}
    if scenario.get("seed"):
        about["seed"] = scenario["seed"]
        about["generation_mode"] = "real_task_seeded_synthetic"
        run.set_algo(seed=scenario["seed"])
    run.write("00_about.json", about)


def _pin_game_primary(wp: dict) -> None:
    """把游戏唯一主角固定为一个实例，并同步白皮书的实体总数。

    这是 game 场景的产品语义，不由通用 closed-loop 猜测；外围角色仍可扩容，
    主角则由 ``exact`` 策略永久锁为 1。
    """
    blueprint = wp.get("world_blueprint") or {}
    types = blueprint.get("entity_types") or []
    primaries = [item for item in types if isinstance(item, dict) and item.get("primary") is True]
    if len(primaries) != 1:
        raise WorldBlueprintError(
            f"game 白皮书必须且只能有一个 primary entity type，当前={len(primaries)}")
    primaries[0]["count"] = 1
    primaries[0]["cardinality_policy"] = "exact"
    # 标量 FK 的容量取决于 owner 数量；主角收缩为 1 后，同步收紧不可实现的关系下限。
    for relation in blueprint.get("relation_types") or []:
        minimum = relation.get("min_count")
        capacity = relation_capacity(blueprint, relation)
        if (isinstance(minimum, int) and not isinstance(minimum, bool)
                and capacity > 0 and minimum > capacity):
            relation["min_count"] = capacity
    normalize_world_blueprint(wp)
    total = sum(int(item.get("count", 0)) for item in types if isinstance(item, dict))
    wp.setdefault("shared_world_spec", {}).setdefault("entities", {})["count"] = total


def stage_whitepaper(run: Run):
    sc = run.read(ART["input"])
    pack = validate_seed_input(run, sc)
    cfg = run.manifest.get("config") or {}
    joint = (cfg.get("production_control") or {}).get("version") == "supply-driven/v6"
    joint_audit_name = "01_joint_design_audit.json"
    joint_identity = None
    if joint:
        from pipeline.capability_contract import digest
        from pipeline.joint_design import (VERSION as JOINT_VERSION, MAX_REVISIONS,
                                           MAX_INITIAL_SCHEMA_ATTEMPTS)
        joint_identity = digest({"input": sc, "config": cfg})
        if run.has(joint_audit_name):
            prior = run.read(joint_audit_name)
            if (prior.get("version") == JOINT_VERSION and prior.get("identity") == joint_identity
                    and prior.get("status") == "accepted"
                    and prior.get("receipt", {}).get("whitepaper_hash") == digest(prior.get("accepted"))):
                wp = prior["accepted"]
                run.write("01_supply_plan.json", wp["supply_plan"])
                run.write("01_instance_plan.json", wp["business_instance_plan"])
                if wp.get("seed_audit"):
                    run.write("01_seed_audit.json", wp["seed_audit"])
                run.write(ART["whitepaper"], wp)
                run.write("01_joint_design_receipt.json", prior["receipt"])
                run.set_algo(active_lines=[l.get("line") for l in wp.get("active_lines", [])],
                             medium=wp.get("output_medium") or wp.get("domain_profile", {}).get("medium"))
                return
            raise ValueError("Saved unpublished joint design needs explicit diagnosis; paid author calls cannot restart")
        run.write(joint_audit_name, {"version": JOINT_VERSION, "identity": joint_identity,
                                    "status": "initial_schema_requested", "max_revisions": MAX_REVISIONS})
    extra = {}
    if cfg.get("delivery_target"):
        from pipeline.supply import author_brief
        extra["delivery_brief"] = author_brief(cfg["delivery_target"], cfg.get("question_budget"),
                                                cfg.get("single_pass_world_limits"), cfg.get("delivery_survival_rates"))
        if cfg.get("production_control"):
            from pipeline.supply_capacity import requirements
            extra["delivery_brief"].update(version=2, requirements=requirements(extra["delivery_brief"]["candidate_allocation"]))
    joint_options = {"max_world_attempts": MAX_INITIAL_SCHEMA_ATTEMPTS,
                     "defer_business_review": True} if joint else {}
    try:
        if pack is None:
            wp = central_office(sc["description"], sc["few_shot"], run.tracer, run.log,
                                **extra, **joint_options)
        else:
            wp = central_office(sc["description"], sc["few_shot"], run.tracer, run.log,
                                seed_pack=pack, **extra, **joint_options)
            validate_seed_identity(run, wp)
            if not joint:
                run.write("01_seed_audit.json", wp["seed_audit"])
    except BaseException as error:
        if joint:
            record = run.read(joint_audit_name)
            record.update(status="initial_schema_failed", error_type=type(error).__name__,
                          error=str(error))
            run.write(joint_audit_name, record)
        raise
    if joint:
        record = run.read(joint_audit_name)
        record.update(status="initial_schema_ready", schema_hash=digest(wp["world_blueprint"]),
                      initial_schema=wp)
        run.write(joint_audit_name, record)
    if run.scenario == "game":
        _pin_game_primary(wp)
    elif pack is not None or cfg.get("production_control"):
        # Freeze the execution strategy with the original whitepaper. Existing
        # saved papers retain their own strategy and review bindings.
        wp["world_generation"] = {"strategy": "agentic", "version": 2, "blueprint_repair_attempts": 1,
                                  "disclosure_strategy": "direct_batches_v1"}
    cfg = run.manifest.get("config") or {}
    if cfg.get("single_pass_world_limits") and not cfg.get("production_control"):
        from pipeline.closed_loop import _scale_world_contract, _ensure_event_role_capacity
        limits = cfg["single_pass_world_limits"]
        bp = wp.get("world_blueprint") or {}
        entities = sum(int(t.get("count", 0)) for t in bp.get("entity_types", []))
        entities = entities or int((wp.get("shared_world_spec", {}).get("entities") or {}).get("count", 8))
        periods = limits.get("time_span_weeks") or (bp.get("temporal_model") or {}).get("n_sessions", 6)
        _scale_world_contract(wp, min(entities, limits["max_world_entities"]), periods,
                              narrative=run.scenario == "game")
        _ensure_event_role_capacity(wp)
        actual = sum(int(t.get("count", 0)) for t in (wp.get("world_blueprint") or {}).get("entity_types", []))
        if actual > limits["max_world_entities"]:
            raise WorldBlueprintError("Seed entity minima exceed this run's max_world_entities")
    wp["generation_contract"] = {"version": "generation-first/v2", "material_first": True}
    if cfg.get("delivery_target"):
        from pipeline.supply import attach_plan
        plan = attach_plan(wp, cfg["delivery_target"], cfg.get("question_budget"),
                           cfg.get("single_pass_world_limits"), cfg.get("delivery_survival_rates"),
                           capacity_driven=bool(cfg.get("production_control")),
                           instance_driven=(cfg.get("production_control") or {}).get("version") in
                           ("supply-driven/v4", "supply-driven/v5", "supply-driven/v6"))
        if not joint:
            run.write("01_supply_plan.json", plan)
    wp["quality_contract"] = {"version": 4, "corpus_review": True,
                              "world_semantic_review": True,
                              "public_disclosure": True,
                              "process_proposals": False,  # Opt in while capability quality is being calibrated.
                              "scoring_policy": POLICY_VERSION,
                              "public_semantic_review": True,
                              "isolated_reference_audit": True,
                              "release_requires": "question_corpus_and_world_contracts"}
    if joint:
        from pipeline.joint_design import accept
        audit = run.read(joint_audit_name)
        audit.update(status="initial_draft_ready", draft=wp,
                     initial_schema_hash=digest(wp["world_blueprint"]))
        run.write(joint_audit_name, audit)
        wp, receipt = accept(run, wp, pack, extra["delivery_brief"], audit,
                             lambda current: run.write(joint_audit_name, current))
        validate_seed_identity(run, wp)
        run.write("01_supply_plan.json", wp["supply_plan"])
        run.write("01_instance_plan.json", wp["business_instance_plan"])
        if wp.get("seed_audit"):
            run.write("01_seed_audit.json", wp["seed_audit"])
        run.write(ART["whitepaper"], wp)
        run.write("01_joint_design_receipt.json", receipt)
    else:
        run.write(ART["whitepaper"], wp)
    run.set_algo(active_lines=[l.get("line") for l in wp.get("active_lines", [])],
                 medium=wp.get("output_medium") or wp.get("domain_profile", {}).get("medium"))



def stage_world(run: Run):
    """Route an author's explicit blueprint conflict to its original architect."""
    from pipeline.blueprint_feasibility import BlueprintReviewRequested, revise_constraints
    from pipeline.production import ensure_instance_plan
    ensure_instance_plan(run)
    original = run.read(ART["whitepaper"])
    policy = original.get("world_generation") or {}
    allowed = (policy.get("strategy") == "agentic" and policy.get("version") == 2
               and policy.get("blueprint_repair_attempts") == 1)
    archive = "02_blueprint_revision_attempts.json"
    attempts = run.read(archive).get("attempts", []) if run.has(archive) else []
    original_seed_audit = run.read("01_seed_audit.json") if run.has("01_seed_audit.json") else None
    changed = False
    try:
        while True:
            try:
                return _stage_world_once(run)
            except BlueprintReviewRequested as request:
                # A published world has downstream consumers. Revising its
                # definition requires a fresh generation, never an in-place edit.
                from pipeline.supply_capacity import CapacityReviewRequested, revise_capacity
                capacity_request = isinstance(request, CapacityReviewRequested)
                control = (run.manifest.get("config", {}).get("production_control") or {}).get("version")
                if control == "supply-driven/v6":
                    if capacity_request:
                        raise
                    raise CapacityReviewRequested({"phase": "world_blueprint_review",
                        "blueprint_review": request.evidence}) from request
                if capacity_request and control:
                    raise
                limit = (run.read(ART["whitepaper"]).get("delivery_target") or {}).get("max_supply_rounds", 1) if capacity_request else 1
                if not allowed or len(attempts) >= limit or run.has(ART["world"]):
                    raise
                run.log("  ↻ 世界作者请求上游复核字段约束；保留旧稿，交原架构师修订并独立复核")
                audit = {}
                try:
                    revise = revise_capacity if capacity_request else revise_constraints
                    revised = revise(run.read(ART["whitepaper"]), run.tracer, request.evidence, audit)
                    validate_seed_identity(run, revised)
                finally:
                    attempts.append(audit)
                    run.write(archive, {"attempts": attempts})
                run.write(ART["whitepaper"], revised)
                if revised.get("supply_plan"):
                    run.write("01_supply_plan.json", revised["supply_plan"])
                changed = True
                if "seed_audit" in revised:
                    run.write("01_seed_audit.json", revised["seed_audit"])
    except BaseException:
        if changed:
            run.write(ART["whitepaper"], original)
            if original.get("supply_plan"):
                run.write("01_supply_plan.json", original["supply_plan"])
            if original_seed_audit is not None:
                run.write("01_seed_audit.json", original_seed_audit)
        raise


def _stage_world_once(run: Run):
    wp = run.read(ART["whitepaper"])
    validate_seed_identity(run, wp)
    existing = None
    # game 的 Story Ledger 必须基于单一 canon；即使 manifest 残留 augment 也始终全量重建。
    if (run.scenario != "game" and run.manifest["config"].get("augment")
            and run.has(ART["world"])):                         # ★增量(§10.1):旧世界上 augment 新实体
        existing = WorldState.from_dict(run.read(ART["world"]))
    from pipeline import world_semantics, disclosure
    review_enabled = world_semantics.enabled(wp, run.manifest.get("config", {}))
    disclosure_enabled = disclosure.enabled(wp)
    if ((wp.get("seed_contract") or {}).get("schema_version") == 2
            and not (review_enabled and disclosure_enabled)):
        raise WorldBlueprintError("Seed v2 requires the original world review and public disclosure context")
    if disclosure_enabled and not review_enabled:
        raise WorldBlueprintError("公开信息安排必须经过原世界业务审阅")
    draft = {}
    options = {"draft_out": draft} if review_enabled else {}
    agent_options = {}
    if (wp.get("world_generation") or {}).get("strategy") == "agentic":
        import hashlib
        identity = json.dumps({"whitepaper": wp,
            "existing": existing.to_dict() if existing is not None else None},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        checkpoint_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        agent_options["checkpoint_path"] = run.dir / f"02_world_agent_{checkpoint_id}.json"
        reused = (run.manifest.get("derived_from") or {}).get("world_construction_checkpoints", [])
        if reused and not run.has(ART["world"]) and agent_options["checkpoint_path"].name not in reused:
            raise WorldBlueprintError("恢复参数与已保存世界不匹配；停止，避免重新生成世界")
        options.update(agent_options)
    try:
        ws = build_world(wp, run.tracer, run.log, existing=existing,
                         narrative=(run.scenario == "game"), **options)
    finally:
        if draft:
            run.write("02_world_draft.json", draft)
    from pipeline.construction import enabled as construction_enabled, resolve as resolve_construction
    if construction_enabled(wp):
        from pipeline.instance_plan import active_mapping
        ws.supply_construction = resolve_construction(wp, active_mapping((draft or {}).get("agent") or {}))
    _prepare_lines(wp, ws, run.log)
    if wp.get("business_instance_plan"):
        from pipeline.instance_plan import fulfillment, publication_obligations
        instance_state = (draft or {}).get("agent") or {}
        receipt = fulfillment(wp,ws,instance_state)
        run.write("02_instance_fulfillment.json",receipt)
        if not receipt["passed"]:
            from pipeline.supply_capacity import CapacityReviewRequested
            raise CapacityReviewRequested({"instance_fulfillment":receipt})
        run.write("02_publication_obligations.json",publication_obligations(wp,instance_state))
    seed_audit = validate_seed_world(wp, ws)
    review = None
    review_warning = None
    repair_failure = None
    if review_enabled:
        task_input = run.read(ART["input"])
        run.write("02_world_draft.json", draft)
        attempts = []
        disclosure_attempts = []
        review_limit = (3 if (run.manifest.get("config", {}).get("production_control") or {}).get("version")
                        == "supply-driven/v6" else 2)
        for attempt in range(review_limit):
            unit_plan = [{"unit_id": unit["unit_id"], "intent": unit["intent"],
                          "entities": [item["name"] for item in unit["raw"]["entities"]],
                          "events": [item["id"] for item in unit["raw"]["events"]]}
                         for unit in (draft.get("agent") or {}).get("units", [])]
            author_context = (draft.get("repair_log") if attempt else None)
            if unit_plan:
                author_context = {"business_work_plan": unit_plan,
                                  "repair_responses": author_context,
                                  "scope": "Fallible author intent; verify actual world facts independently"}
            if wp.get("business_instance_plan"):
                author_context = {**(author_context or {}), "compiled_instance_plan":wp["business_instance_plan"],
                    "publication_obligations":run.read("02_publication_obligations.json")}
            if disclosure_enabled:
                disclosure_feedback = attempts[-1] if attempts else None
                if unit_plan and disclosure_feedback is None:
                    disclosure_feedback = {"business_work_plan": unit_plan,
                        "scope": "Author grouping for navigation; acquisition times still need independent authoring"}
                if wp.get("business_instance_plan"):
                    disclosure_feedback = {"previous":disclosure_feedback,
                        "publication_obligations":run.read("02_publication_obligations.json"),
                        "instruction":"按原业务规则安排所列事实的公开期和渠道；发现计划与事实或seed矛盾时明确返修。"}
                plan_report = disclosure.author_plan(
                    wp, ws, run.tracer, task_input=task_input,
                    feedback=disclosure_feedback,
                    **({"checkpoint_dir": run.dir} if (wp.get("world_generation") or {}).get(
                        "disclosure_strategy") == "direct_batches_v1" else {}))
                disclosure_attempts.append(plan_report)
                run.write("02_disclosure_plan_attempts.json", {"attempts": disclosure_attempts})
                if plan_report.get("status") != "ready" or disclosure.validate_plan(ws):
                    run.write("02_world_candidate.json", ws.to_dict())
                    run.log("  ⚠ 公开信息安排未形成有效候选；保留作者记录，世界候选继续进入审阅与下游诊断")
            run.write("02_world_candidate.json", ws.to_dict())
            review = world_semantics.review_world(
                wp, ws, run.tracer, task_input=task_input,
                previous=attempts[-1] if attempts else None,
                author_responses=author_context)
            attempts.append(review)
            run.write("02_world_review_attempts.json", {"attempts": attempts})
            if review.get("status") == "passed" and not world_semantics.validate_review(
                    review, wp, ws, task_input=task_input):
                break
            targets = review.get("repair_targets") or {}
            truth_repair = bool(targets.get("intrinsic") or targets.get("structure"))
            disclosure_repair = disclosure_enabled and targets.get("disclosure") is True
            if (attempt >= review_limit - 1 or review.get("status") not in ("failed", "unresolved")
                    or not (truth_repair or disclosure_repair)):
                # Preserve the exact candidate and opinion so recovery can
                # refresh this bounded review without rebuilding the world.
                review_warning = world_semantics.generation_warning(
                    review, wp, ws, task_input=task_input)
                run.log("  ⚠ 世界审阅未认证当前候选；候选带待补审警告继续生产")
                break
            if not truth_repair:
                run.log("  ↻ 公开信息安排需要返修；保留世界真值，返回原作者重拟安排并复核")
                continue
            run.log("  ↻ 世界业务审阅发现待核问题，返回原作者进行一次有限返修")
            repaired_draft = {}
            # One call per selected intrinsic author, plus the original bounded
            # structure/validation retries. Keep a finite ceiling for large worlds.
            repair_call_limit = min(12, max(4, len(targets.get("intrinsic", []))
                                             + (3 if targets.get("structure") else 2)))
            try:
                ws = build_world(wp, run.tracer, run.log, existing=existing,
                    narrative=(run.scenario == "game"), draft_out=repaired_draft,
                    repair_input={"draft": draft, "feedback": review,
                                  "targets": {k: targets[k] for k in ("intrinsic", "structure") if k in targets},
                                  "max_calls": repair_call_limit}, **agent_options)
            except Exception as exc:
                # The review already left us a complete candidate and a
                # review opinion. A failed optional repair must not
                # discard those artifacts or terminate the remaining factory.
                # Preserve the exact pre-repair candidate for bounded recovery.
                repair_failure = {
                    "version": "world-repair-failure/v1", "status": "error",
                    "release_eligible": False,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "binding": {
                        "whitepaper_hash": _canonical_hash(wp),
                        "world_hash": _canonical_hash(ws.to_dict()),
                        "review_hash": _canonical_hash(review),
                    },
                }
                review_warning = world_semantics.generation_warning(
                    review, wp, ws, task_input=task_input)
                run.log(f"  ⚠ 世界有限返修未完成:{type(exc).__name__}: {str(exc)[:160]}；"
                        "保留返修前候选和审阅意见，继续后续生产")
                break
            finally:
                if repaired_draft:
                    run.write("02_world_repair_draft.json", repaired_draft)
            draft = repaired_draft
            run.write("02_world_draft.json", draft)
            _prepare_lines(wp, ws, run.log)
            seed_audit = validate_seed_world(wp, ws)
    bundle = {ART["world"]: ws.to_dict()}
    if wp.get("seed_contract"):
        bundle["02_seed_audit.json"] = seed_audit
    metadata = {"entities": len(ws.entities), "sessions": ws.n_sessions}
    if disclosure_enabled:
        metadata["public_disclosure"] = {
            "records": len(ws.disclosure.get("records", [])),
            "undisclosed_references": len(ws.disclosure.get("undisclosed", [])),
            "author_attempts": len(disclosure_attempts),
            "scope": "LLM-reviewed publication schedule; not a question difficulty claim"}
    if review is not None:
        bundle[world_semantics.REVIEW_ARTIFACT] = review
        if review_warning is not None:
            bundle[world_semantics.WARNING_ARTIFACT] = review_warning
        if repair_failure is not None:
            bundle[WORLD_REPAIR_FAILURE] = repair_failure
        metadata["world_semantic_review"] = {
            "status": review["status"], "attempts": len(attempts),
            "generation_disposition": "warning_continue" if review_warning else "certified",
            "release_eligible": review_warning is None}
    if (run.manifest.get("config", {}).get("production_control") or {}).get("version") == "supply-driven/v6":
        from pipeline.capability_contract import digest
        receipt = run.read("02_instance_fulfillment.json")
        if review_warning is not None or review is None:
            run.write("02_source_supply_gate.json", {"status": "business_review_unresolved",
                "world_hash": digest(ws.to_dict()), "review": review,
                "source_supply_passed": receipt["passed"]})
            raise WorldBlueprintError("Source supply has candidates but the original world business review is unresolved")
        bundle["02_source_supply_gate.json"] = {
            "version": "source-supply-ready/v1", "status": "source_supply_ready",
            "whitepaper_hash": digest(wp), "world_hash": digest(ws.to_dict()),
            "seed_world_audit_hash": digest(seed_audit),
            "business_review_hash": digest(review),
            "actual_supply_hash": digest(receipt),
            "per_line": receipt["actual_supply"],
            "native_trend": receipt["native_trend"],
            "scope": "Original bounded enumerators, answer recomputation, well-posed and family keys; public materials and final selection remain unchecked"}
    remove = []
    if review_warning is None:
        remove.append(world_semantics.WARNING_ARTIFACT)
    if repair_failure is None:
        remove.append(WORLD_REPAIR_FAILURE)
    _publish_world_bundle(run, bundle, metadata, remove=remove)


def _business_work_plan_context(run: Run):
    """Recover the original author's bounded navigation hint from its saved draft."""
    if not run.has("02_world_draft.json"):
        return None
    draft = run.read("02_world_draft.json")
    units = (draft.get("agent") or {}).get("units", []) if isinstance(draft, dict) else []
    plan = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        raw = unit.get("raw") or {}
        plan.append({"unit_id": unit.get("unit_id"), "intent": unit.get("intent"),
                     "entities": [item.get("name") for item in (raw.get("entities") or [])
                                  if isinstance(item, dict) and item.get("name")],
                     "events": [item.get("id") for item in (raw.get("events") or [])
                                if isinstance(item, dict) and item.get("id")]})
    if not plan:
        return None
    return {"business_work_plan": plan,
            "scope": "Author grouping for navigation; acquisition times still need independent authoring"}


def _release_current_world_review(run: Run, wp, ws, task) -> bool:
    """Require a current successful world review before per-question review."""
    from pipeline import world_semantics
    if (run.has(world_semantics.WARNING_ARTIFACT)
            or not run.has(world_semantics.REVIEW_ARTIFACT)):
        return False
    try:
        review = run.read(world_semantics.REVIEW_ARTIFACT)
        return (review.get("status") == "passed"
                and not world_semantics.validate_review(review, wp, ws, task_input=task))
    except (ValueError, OSError, TypeError, KeyError, AttributeError):
        return False


def _recoverable_current_world_review(run: Run, wp, ws, task):
    """Return an exactly bound nonpassing opinion for targeted recovery only."""
    from pipeline import world_semantics
    if not (run.has(world_semantics.REVIEW_ARTIFACT)
            and run.has(world_semantics.WARNING_ARTIFACT)):
        return None
    try:
        review = run.read(world_semantics.REVIEW_ARTIFACT)
        warning = run.read(world_semantics.WARNING_ARTIFACT)
        if not world_semantics.validate_generation_warning(
                warning, review, wp, ws, task_input=task):
            return review
    except (ValueError, OSError, TypeError, KeyError, AttributeError):
        pass
    return None


def stage_disclosure(run: Run):
    """Complete or validate public disclosure without regenerating the frozen world.

    Recovery runs may already contain whitepaper, world, orders and questions.
    This stage resumes only the missing disclosure author checkpoint, refreshes
    the bound world opinion once, then lets corpus production continue.  World
    truth is checked before and after so this recovery stage cannot rewrite it.
    """
    from pipeline import disclosure, world_semantics
    wp = run.read(ART["whitepaper"])
    task = run.read(ART["input"])
    ws = WorldState.from_dict(run.read(ART["world"]))
    validate_seed_identity(run, wp)
    validate_seed_world(wp, ws)
    truth_before = disclosure._world(ws)
    old_plan_hash = ((getattr(ws, "disclosure", None) or {}).get("plan_hash"))

    if not disclosure.enabled(wp):
        receipt = {"version": "disclosure-ready/v1", "status": "disabled",
                   "world_hash": _canonical_hash(ws.to_dict())}
        run.write(ART["disclosure"], receipt)
        return

    # A completed historical plan and opinion can be certified with zero calls.
    # This is the common insurance recovery path.
    plan_errors = disclosure.validate_plan(ws)
    review_current = (not plan_errors
                      and _release_current_world_review(run, wp, ws, task))

    disclosure_attempts = (run.read("02_disclosure_plan_attempts.json").get("attempts", [])
                           if run.has("02_disclosure_plan_attempts.json") else [])
    if plan_errors:
        report = disclosure.author_plan(
            wp, ws, run.tracer, task_input=task,
            feedback=_business_work_plan_context(run), checkpoint_dir=run.dir)
        disclosure_attempts.append(report)
        run.write("02_disclosure_plan_attempts.json", {"attempts": disclosure_attempts})
        plan_errors = disclosure.validate_plan(ws)
        if report.get("status") != "ready" or plan_errors:
            raise WorldBlueprintError(
                "公开安排恢复尚未完成；已保留检查点，可从 disclosure 阶段继续: "
                + str(report.get("error") or plan_errors[:1]))

    if disclosure._world(ws) != truth_before:
        raise WorldBlueprintError("公开安排恢复改动了世界真值；拒绝发布恢复候选")

    review = run.read(world_semantics.REVIEW_ARTIFACT) if (
        review_current and run.has(world_semantics.REVIEW_ARTIFACT)) else None
    warning = (run.read(world_semantics.WARNING_ARTIFACT)
               if review_current and run.has(world_semantics.WARNING_ARTIFACT) else None)
    recoverable_review = None if review_current else _recoverable_current_world_review(run, wp, ws, task)
    can_target_repair = (isinstance(recoverable_review, dict)
                         and (recoverable_review.get("repair_targets") or {}).get("disclosure") is True
                         and (ws.disclosure or {}).get("strategy") in
                         {"direct-disclosure/v1", "direct-disclosure/v2"})
    if can_target_repair:
        review = recoverable_review
    elif not review_current:
        review = world_semantics.review_world(
            wp, ws, run.tracer, task_input=task,
            author_responses=_business_work_plan_context(run))
        prior = (run.read("02_world_review_attempts.json").get("attempts", [])
                 if run.has("02_world_review_attempts.json") else [])
        prior.append(review)
        run.write("02_world_review_attempts.json", {"attempts": prior})
        errors = world_semantics.validate_review(review, wp, ws, task_input=task)
        warning = None if review.get("status") == "passed" and not errors else \
            world_semantics.generation_warning(review, wp, ws, task_input=task)

    # A disclosure-only semantic rejection edits only the named public records,
    # then receives one fresh independent review.  It never reauthors the world,
    # the full disclosure plan, questions, or corpus, and never loops.
    semantic_repair = None
    if ((review.get("repair_targets") or {}).get("disclosure") is True
            and (ws.disclosure or {}).get("strategy") in
            {"direct-disclosure/v1", "direct-disclosure/v2"}):
        from pipeline import disclosure_batches
        semantic_repair = disclosure_batches.repair(wp, ws, run.tracer, review)
        run.write("02_disclosure_semantic_repair.json", semantic_repair)
        if semantic_repair.get("status") == "ready":
            if disclosure._world(ws) != truth_before:
                raise WorldBlueprintError("公开安排局部返修改动了世界真值；拒绝发布恢复候选")
            author_context = _business_work_plan_context(run) or {}
            author_context = {**author_context,
                              "disclosure_semantic_repair": {
                                  "status": "ready",
                                  "repaired_record_ids": semantic_repair.get("repaired_record_ids", [])}}
            # Review checkpoint names and input bindings include the complete
            # world and previous opinion. A changed plan cannot replay an old
            # transcript. Keep those transcripts as historical call evidence.
            semantic_repair["retained_review_checkpoints"] = len(
                list(run.dir.glob("02_world_review_*.ckpt.json")))
            run.write("02_disclosure_semantic_repair.json", semantic_repair)
            previous_review = review
            review = world_semantics.review_world(
                wp, ws, run.tracer, task_input=task, previous=previous_review,
                author_responses=author_context)
            prior = (run.read("02_world_review_attempts.json").get("attempts", [])
                     if run.has("02_world_review_attempts.json") else [])
            # The earlier opinion was already recorded when it was obtained.
            # This branch buys exactly one new review and records it once.
            prior.append(review)
            run.write("02_world_review_attempts.json", {"attempts": prior})
            errors = world_semantics.validate_review(review, wp, ws, task_input=task)
            warning = None if review.get("status") == "passed" and not errors else \
                world_semantics.generation_warning(review, wp, ws, task_input=task)

    new_plan_hash = (ws.disclosure or {}).get("plan_hash")
    receipt = {"version": "disclosure-ready/v1", "status": "ready",
               "truth_hash": _canonical_hash(truth_before),
               "plan_hash": new_plan_hash,
               "world_hash": _canonical_hash(ws.to_dict()),
               "review_hash": _canonical_hash(review),
               "review_status": review.get("status"),
               "release_eligible": warning is None}
    bundle = {ART["world"]: ws.to_dict(), ART["disclosure"]: receipt,
              world_semantics.REVIEW_ARTIFACT: review}
    remove = []
    if warning is None:
        remove.append(world_semantics.WARNING_ARTIFACT)
    else:
        bundle[world_semantics.WARNING_ARTIFACT] = warning
    _publish_world_bundle(run, bundle, {
        "public_disclosure": {"records": len(ws.disclosure.get("records", [])),
                              "undisclosed_references": len(ws.disclosure.get("undisclosed", [])),
                              "author_attempts": len(disclosure_attempts),
                              "recovery_stage": True},
        "world_semantic_review": {"status": review.get("status"),
                                  "generation_disposition": "warning_continue" if warning else "certified",
                                  "release_eligible": warning is None,
                                  "targeted_disclosure_repair": (semantic_repair or {}).get("status")}},
        remove=remove, invalidate_corpus=(old_plan_hash != new_plan_hash))
    # Historical large runs can contain a complete frozen corpus plus an
    # incomplete optional L3 proposal.  Preserve the warning and bind its exact
    # impact to L3 question identities; later stages may withhold only that
    # line.  Any mismatch leaves the warning unresolved and release-ineligible.
    from pipeline import order_warning
    resolution_path = run.dir / order_warning.RESOLUTION_ARTIFACT
    # The warning scope belongs to existing question identities.  A material-
    # first recovery can reach this stage before any questions have been made.
    if (run.has(ART["questions"])
            and run.has(order_warning.WARNING_ARTIFACT)
            and run.has(order_warning.PROPOSAL_ARTIFACT)):
        try:
            resolution = order_warning.build_resolution(
                wp, ws.to_dict(), run.read(ART["questions"]),
                run.read(order_warning.WARNING_ARTIFACT),
                run.read(order_warning.PROPOSAL_ARTIFACT))
            run.write(order_warning.RESOLUTION_ARTIFACT, resolution)
        except (ValueError, TypeError, KeyError, AttributeError):
            resolution_path.unlink(missing_ok=True)
    else:
        resolution_path.unlink(missing_ok=True)


def _disclosure_is_current(run: Run) -> bool:
    try:
        from pipeline import disclosure, world_semantics
        wp = run.read(ART["whitepaper"])
        ws = WorldState.from_dict(run.read(ART["world"]))
        receipt = run.read(ART["disclosure"])
        if not disclosure.enabled(wp):
            return receipt == {"version": "disclosure-ready/v1", "status": "disabled",
                               "world_hash": _canonical_hash(ws.to_dict())}
        if disclosure.validate_plan(ws):
            return False
        _require_current_world_review(run, wp, ws)
        review = run.read(world_semantics.REVIEW_ARTIFACT)
        warning = run.read(world_semantics.WARNING_ARTIFACT) if run.has(world_semantics.WARNING_ARTIFACT) else None
        expected = {"version": "disclosure-ready/v1", "status": "ready",
                    "truth_hash": _canonical_hash(disclosure._world(ws)),
                    "plan_hash": ws.disclosure.get("plan_hash"),
                    "world_hash": _canonical_hash(ws.to_dict()),
                    "review_hash": _canonical_hash(review),
                    "review_status": review.get("status"),
                    "release_eligible": warning is None}
        return receipt == expected
    except (ValueError, OSError, TypeError, KeyError, AttributeError):
        return False


def _publish_world_bundle(run: Run, bundle: dict, metadata: dict, remove=(), *, invalidate_corpus=True):
    """Restore the previously published world/opinion if the commit is interrupted.

    The stage lock serializes writers. Readers still verify content bindings,
    so a crash between individual atomic writes cannot certify a mixed bundle.
    """
    names = list(dict.fromkeys([*bundle, *remove, *([CORPUS_CKPT] if invalidate_corpus else [])]))
    previous = {name: (run.dir / name).read_bytes() if (run.dir / name).exists() else None
                for name in names}
    prior_algo = deepcopy(run.manifest.get("algo", {}))
    try:
        for name, value in bundle.items():
            run.write(name, value)
        for name in remove:
            (run.dir / name).unlink(missing_ok=True)
        run.set_algo(**metadata)
        if invalidate_corpus:
            (run.dir / CORPUS_CKPT).unlink(missing_ok=True)
    except BaseException:
        for name, content in previous.items():
            path = run.dir / name
            if content is None:
                path.unlink(missing_ok=True)
            else:
                restore = path.with_name("." + path.name + ".world-restore")
                try:
                    restore.write_bytes(content)
                    restore.replace(path)
                finally:
                    restore.unlink(missing_ok=True)
        run.manifest["algo"] = prior_algo
        if hasattr(run, "_save_manifest"):
            run._save_manifest()
        raise


def _require_current_world_review(run: Run, wp: dict, ws=None):
    """Permit a bound candidate warning while keeping release certification closed."""
    from pipeline import world_semantics, disclosure
    if not world_semantics.enabled(wp, run.manifest.get("config", {})):
        if disclosure.enabled(wp):
            raise WorldBlueprintError("公开信息安排要求原世界业务审阅，不能在续跑时关闭")
        return
    if not run.has(world_semantics.REVIEW_ARTIFACT) or not run.has(ART["input"]):
        raise WorldBlueprintError("缺少当前世界的业务审阅；请先运行原 world 阶段")
    ws = ws if ws is not None else WorldState.from_dict(run.read(ART["world"]))
    review = run.read(world_semantics.REVIEW_ARTIFACT)
    errors = world_semantics.validate_review(review, wp, ws, task_input=run.read(ART["input"]))
    if review.get("status") == "passed" and not errors:
        return
    if run.has(world_semantics.WARNING_ARTIFACT):
        warning_errors = world_semantics.validate_generation_warning(
            run.read(world_semantics.WARNING_ARTIFACT), review, wp, ws,
            task_input=run.read(ART["input"]))
        if not warning_errors:
            return
    raise WorldBlueprintError("世界业务审阅及其候选继续生产凭据缺失或已过期；请先运行原 world 阶段")


def _world_is_current(run: Run) -> bool:
    try:
        _require_current_world_review(run, run.read(ART["whitepaper"]))
        return True
    except (ValueError, OSError, TypeError, KeyError):
        return False


def _require_current_corpus_review(run: Run, wp: dict, corpus_obj=None):
    """Recheck saved corpus receipts before downstream work, without model calls."""
    if not (wp.get("quality_contract") or {}).get("corpus_review"):
        return
    from pipeline.corpus_contract import validate_corpus
    ws = WorldState.from_dict(run.read(ART["world"]))
    corpus_obj = corpus_obj if corpus_obj is not None else run.read(ART["corpus"])
    report = validate_corpus(ws, corpus_obj)
    if report.get("status") != "passed" or report.get("issues"):
        if run.has(CORPUS_WARNING):
            warning = run.read(CORPUS_WARNING)
            binding = warning.get("binding") if isinstance(warning, dict) else None
            candidate_hash = (_canonical_hash(run.read("05_corpus_candidate.json"))
                              if run.has("05_corpus_candidate.json") else None)
            if (warning.get("version") == "corpus-generation-warning/v1"
                    and warning.get("status") == "warning"
                    and warning.get("release_eligible") is False
                    and binding == {"whitepaper_hash": _canonical_hash(wp),
                                    "world_hash": _canonical_hash(ws.to_dict()),
                                    "corpus_hash": _canonical_hash(corpus_obj),
                                    "candidate_hash": candidate_hash}):
                return
        codes = list(dict.fromkeys(issue.get("code", "unknown") for issue in report.get("issues", [])))
        raise ValueError(f"正文审阅缺失、未通过或已过期:{codes}；请先运行原 corpus 阶段")


def _corpus_is_current(run: Run) -> bool:
    try:
        _require_current_corpus_review(run, run.read(ART["whitepaper"]))
        cfg = run.manifest.get("config", {})
        if cfg.get("corpus_token_target") is not None:
            from pipeline.corpus_tokens import CorpusTokenCounter, plan_filler_batch
            report = run.read("05_corpus_token_scale.json")
            name, separator, version = cfg.get("corpus_tokenizer", "cl100k_base").partition("@")
            if cfg.get("delivery_target") and (not separator or not version):
                return False
            counter = CorpusTokenCounter(name, expected_version=version or "0.12.0")
            measurement = report.get("measurement") or {}
            plan = report.get("plan") or {}
            if (report.get("version") != "corpus-token-scale/v1"
                    or report.get("status") != "completed"
                    or report.get("attempt_complete") is not True
                    or report.get("target_tokens") != cfg["corpus_token_target"]
                    or report.get("haystack_ratio") != cfg.get("haystack_ratio")
                    or report.get("tokenizer") != counter.tokenizer
                    or report.get("tokenizer_id") != counter.tokenizer_id
                    or measurement.get("tokenizer") != counter.tokenizer
                    or report.get("binding") != _corpus_scale_binding(run)
                    or report.get("total_tokens") != (measurement.get("total") or {}).get("tokens")
                    or report.get("target_met") is not True
                    or report.get("deficit_tokens") != 0
                    or plan != plan_filler_batch(measurement, cfg["corpus_token_target"],
                                                 cfg.get("haystack_ratio"))):
                return False
        elif cfg.get("haystack_ratio") is not None:
            report = run.read("05_corpus_scale.json")
            if (report.get("target_chars") != cfg.get("target_tokens", 1_000_000)
                    or report.get("haystack_ratio") != cfg["haystack_ratio"]
                    or report.get("binding") != _corpus_scale_binding(run)):
                return False
        return True
    except (ValueError, OSError, TypeError, KeyError, AttributeError):
        return False


def _corpus_scale_binding(run):
    from pipeline.semantic_review import visible_documents
    corpus = run.read(ART["corpus"])
    documents, _ = visible_documents(corpus)
    return {"whitepaper_hash": _canonical_hash(run.read(ART["whitepaper"])),
            "world_hash": _canonical_hash(run.read(ART["world"])),
            "orders_hash": _canonical_hash(run.read(ART["orders"])) if run.has(ART["orders"]) else None,
            "corpus_hash": _canonical_hash(documents),
            "scale_roles": _canonical_hash([
                [doc.get("role"), doc.get("type")]
                for session in sorted(corpus.get("corpus", corpus)["sessions"], key=lambda x: int(x["session_id"]))
                for doc in session["docs"]])}


def _require_current_question_wording(run: Run, wp: dict, questions=None):
    """Replay original wording receipts before downstream work; never call a model."""
    if not (wp.get("quality_contract") or {}).get("scoring_policy"):
        return
    from pipeline.question_wording import validate_wording
    questions = questions if questions is not None else run.read(ART["questions"])
    if not isinstance(questions, list) or any(not isinstance(q, dict) for q in questions):
        raise ValueError("题面产物格式无效；请先运行原 questions 阶段")
    invalid = []
    for index, question in enumerate(questions):
        errors = validate_wording(question)
        if errors:
            invalid.append({"qid": question.get("qid", index),
                            "codes": [error.get("code", "unknown") for error in errors]})
    if invalid:
        raise ValueError(f"题面审阅缺失、未通过或已过期:{invalid}；请先运行原 questions 阶段")


def _questions_is_current(run: Run) -> bool:
    try:
        wp = run.read(ART["whitepaper"])
        _require_current_question_wording(run, wp)
        _require_current_question_sources(run, wp)
        return True
    except (ValueError, OSError, TypeError, KeyError, AttributeError):
        return False


def _question_source_binding(run, wp=None, questions=None):
    """Bind reusable candidate text to its task, not the mutable corpus."""
    from pipeline.benchmark_export import public_view
    _, protocol = public_view(run)
    return {"version": "generation-candidates/v2",
            "whitepaper_hash": _canonical_hash(wp if wp is not None else run.read(ART["whitepaper"])),
            "orders_hash": _canonical_hash(run.read(ART["orders"])),
            "protocol_hash": _canonical_hash(protocol),
            "questions_hash": _canonical_hash(questions if questions is not None else run.read(ART["questions"]))}


def _require_current_question_sources(run, wp, questions=None):
    """Keep original author provenance; a new corpus only expires review."""
    if (wp.get("generation_contract") or {}).get("version") != "generation-first/v2":
        return
    from pipeline.question_wording import validate_authoring, authoring_binding
    from pipeline.benchmark_export import public_view
    questions = run.read(ART["questions"]) if questions is None else questions
    if not run.has("04_wording_report.json"):
        raise ValueError("Missing actual question author report")
    report = run.read("04_wording_report.json")
    if report.get("candidate_binding") != _question_source_binding(run, wp, questions):
        raise ValueError("Candidate task, protocol or saved output has changed; resume questions")
    _, protocol = public_view(run)
    corpus = run.read(ART["corpus"])
    expected = authoring_binding(corpus, protocol)
    invalid = {q.get("qid"): validate_authoring(q, corpus, protocol, expected_binding=expected)
               for q in questions}
    invalid = {qid: errors for qid, errors in invalid.items() if errors}
    if invalid:
        raise ValueError(f"Invalid original author provenance: {invalid}")


def _grounding_source_binding(run):
    from pipeline.semantic_review import visible_documents
    from pipeline.benchmark_export import public_view
    import config
    documents, _ = visible_documents(run.read(ART["corpus"]))
    _, protocol = public_view(run)
    wp = run.read(ART["whitepaper"])
    cfg = run.manifest.get("config") or {}
    return {"version": "generation-grounding-input/v1",
            "public_hash": _canonical_hash({"documents": documents, "protocol": protocol}),
            "questions_hash": _canonical_hash(run.read(ART["questions"])),
            "review_model": config.REVIEWER_MODEL,
            "review_contract": wp.get("quality_contract"),
            "options": {key: cfg.get(key) for key in
                        ("semantic_max_calls", "semantic_max_tokens", "semantic_max_input_chars")},
            "implementation": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                               for name in ("grounding_review.py", "semantic_review.py", "reference_audit.py")}}


def _grounding_is_current(run):
    try:
        report = run.read("06_grounding_report.json")
        from pipeline.grounding_review import enabled, validate_current_review, candidates_with_evidence
        wp = run.read(ART["whitepaper"])
        if report.get("source_binding") is not None:
            if report["source_binding"] != _grounding_source_binding(run):
                return False
        elif (wp.get("generation_contract") or {}).get("version") == "generation-first/v2":
            return False
        if enabled(wp) and run.read(ART["questions"]):
            from eval.provenance import public_protocol
            corpus = run.read(ART["corpus"])
            candidates = candidates_with_evidence(run.read(ART["questions"]), corpus,
                isolated_reference=(wp.get("quality_contract") or {}).get("isolated_reference_audit", False))
            validate_current_review(candidates, corpus,
                                    public_protocol(run.read("00_about.json")),
                                    run.read("06_semantic_review.json"))
        elif not enabled(wp):
            from pipeline.grounding import run_grounding
            kept, _ = run_grounding(run.read(ART["questions"]), run.read(ART["corpus"]))
            if kept != run.read(ART["grounding"]):
                return False
        return True
    except (ValueError, OSError, TypeError, KeyError, AttributeError):
        return False


def _global_generation_failure(error):
    from execution_control import global_failure
    return global_failure(error)


def _require_v6_supply_gate(run, wp, world):
    if (run.manifest.get("config", {}).get("production_control") or {}).get("version") != "supply-driven/v6":
        return
    from pipeline.capability_contract import digest
    if not (run.has("01_joint_design_receipt.json") and run.has("02_source_supply_gate.json")
            and run.has("02_instance_fulfillment.json")):
        raise ValueError("Joint design or original source supply has no committed receipt")
    design = run.read("01_joint_design_receipt.json")
    supply = run.read("02_source_supply_gate.json")
    if (design.get("status") != "ready_for_world_probe"
            or design.get("whitepaper_hash") != digest(wp)
            or design.get("supply_plan_hash") != digest(wp["supply_plan"])
            or design.get("business_instance_plan_hash") != digest(wp["business_instance_plan"])
            or supply.get("status") != "source_supply_ready"
            or supply.get("whitepaper_hash") != digest(wp)
            or supply.get("world_hash") != digest(world)
            or supply.get("actual_supply_hash") != digest(run.read("02_instance_fulfillment.json"))):
        raise ValueError("Joint design/source supply receipt is stale or changed")


def stage_orders(run: Run):
    wp = run.read(ART["whitepaper"]); ws = WorldState.from_dict(run.read(ART["world"]))
    (run.dir / ORDER_WARNING).unlink(missing_ok=True)
    validate_seed_identity(run, wp)
    validate_seed_world(wp, ws)
    _require_current_world_review(run, wp, ws)
    _require_v6_supply_gate(run, wp, run.read(ART["world"]))
    cfg = run.manifest["config"]
    quotas = cfg.get("quotas")
    # Closed-loop floors own their supply plan; ordinary runs use a total budget.
    # Historical runs without this config retain their explicit legacy behavior.
    budget = cfg.get("question_budget") if quotas is None else None
    if cfg.get("delivery_target"):
        if quotas is not None:
            raise ValueError("Delivery target and legacy quotas require separate runs")
        budget = wp["supply_plan"]["candidate_budget"]
    supply = {}
    process_enabled = (cfg.get("process_proposals") is True
                       or (wp.get("quality_contract") or {}).get("process_proposals") is True)
    # Allocate in the original registry before a bounded proposal. The preview
    # is deterministic and read-only; it does not regenerate world or questions.
    orders = run_lines(wp, ws, (lambda *args: None) if process_enabled else run.log,
                       quotas=quotas, question_budget=budget, stats=supply)
    if process_enabled:
        import config
        from pipeline.lines import line_for
        from pipeline.process_proposals import propose_process_orders
        from eval.answer_task_review import POLICY_VERSION
        quality = wp.get("quality_contract") or {}
        active = any(getattr(line_for(item.get("line")), "id", None) == "L3_process"
                     for item in wp.get("active_lines", []))
        target = 0
        if active and budget is not None:
            target = next((row["allocated"] for row in supply.get("lines", [])
                           if row["line"] == "L3_process"), 0)
        elif active:
            target = int((quotas or {}).get("L3_process",
                         wp.get("capability_targets", {}).get("total_q", 40)))
        if active and target > 0 and ws.events:
            if not (quality.get("scoring_policy") == POLICY_VERSION
                    and quality.get("public_semantic_review") is True
                    and quality.get("isolated_reference_audit") is True):
                raise ValueError("Process proposals require original public semantic review and task-support scoring")
            proposal = {}
            proposal_error = None
            try:
                proposal = propose_process_orders(wp, ws, target=target,
                    chat_json=run.tracer.chat_json, model=config.MODEL,
                    checkpoint_path=run.dir / "03_process_proposals.ckpt.json")
            except Exception as exc:
                proposal_error = exc
            finally:
                if proposal:
                    run.write("03_process_proposals.json", proposal)
            if proposal.get("status") == "completed":
                (run.dir / ORDER_WARNING).unlink(missing_ok=True)
                orders = run_lines(wp, ws, run.log, quotas=quotas,
                    question_budget=budget, stats=supply, process_proposals=proposal)
            else:
                run.write(ORDER_WARNING, {"version": "order-generation-warning/v1",
                    "status": "warning", "release_eligible": False,
                    "error_type": type(proposal_error).__name__ if proposal_error else "IncompleteProposal",
                    "error": str(proposal_error) if proposal_error else str(proposal.get("error") or
                        "Original process proposal did not complete"),
                    "binding": {"whitepaper_hash": _canonical_hash(wp),
                                "world_hash": _canonical_hash(ws.to_dict()),
                                "proposal_hash": _canonical_hash(proposal)}})
                run.log("  ⚠ L3 过程订单提案未完成；保留提案记录，其余可用订单继续进入出题和最终质量报告")
        else:
            # Ordinary lines keep exactly their original selection behavior.
            run.log(f"  Process proposal not requested: active={active}, target={target}, typed_events={len(ws.events)}")
    from pipeline.question_contract import attach_question_contract, bind_question_world
    orders = [attach_question_contract(bind_question_world(order, ws), wp)
              for order in orders]
    run.write(ART["orders"], orders)
    if cfg.get("delivery_target"):
        run.write("03_raw_orders.json", orders)
    run.write("03_supply_report.json", supply)
    by_line: dict = {}
    for o in orders:
        by_line[o.get("line", "?")] = by_line.get(o.get("line", "?"), 0) + 1
    run.set_algo(orders=len(orders), orders_by_line=by_line, question_supply=supply)


def stage_well_posed(run: Run):
    """★边 A 闸(§V 良定义):出题【前】逐题验"gold 是题面在世界里的唯一正确解",ill-posed 即弃。
    纯代码、零 LLM、不碰 corpus。过闸 orders 覆写 03_orders(下游出题用),弃因逐条留 03_well_posed_report.json。
    源头修复后(week_label / L3 跳复现 / L5 排己)新鲜 order 应≈全过 → 本闸=兜底+防回归。"""
    from pipeline.well_posed import run_well_posed
    orders = run.read(ART["orders"]); ws = WorldState.from_dict(run.read(ART["world"]))
    if run.has(ART["whitepaper"]):
        wp = run.read(ART["whitepaper"])
        validate_seed_identity(run, wp)
        validate_seed_world(wp, ws)
        _require_current_world_review(run, wp, ws)
    kept, report = run_well_posed(orders, ws)
    process_count = sum(order.get("capability") == "L3_process_trace" for order in kept)
    if process_count:
        report["process_reference_scope"] = {
            "n": process_count, "checked": "frozen_world_witness_identity",
            "natural_reference": "awaiting_public_semantic_review"}
        run.log(f"  业务过程题 {process_count} 道：此处仅核验冻结世界引用；自然答案与关键理由仍待公开材料审阅")
    run.write(ART["orders"], kept)                          # 过闸 orders 覆写(下游 stage_questions 只对良定义题出题)
    run.write("03_well_posed_report.json", report)
    o = report["overall"]
    run.set_algo(well_posed={"overall": o, "by_line": report["by_line"], "n_dropped": report["n_dropped"]})
    run.log(f"  ★边A闸(良定义):{o['well_posed']}/{o['n']} 良定义({(o['pass_rate'] or 0):.0%}),弃 {report['n_dropped']} "
            f"→ 过闸 orders 下游出题")


def stage_questions(run: Run):
    orders = run.read(ART["orders"]); wp = run.read(ART["whitepaper"])
    pack = validate_seed_identity(run, wp)
    _require_current_world_review(run, wp)
    # The compiled question contract and the solver's published instructions
    # move together, including when regenerating questions in an older run.
    about = run.read("00_about.json")
    protocol = _answer_protocol(wp)
    if about.get("answer_protocol") != protocol:
        about["answer_protocol"] = protocol
        run.write("00_about.json", about)
    from pipeline.benchmark_export import public_view
    corpus = run.read(ART["corpus"])
    _require_current_corpus_review(run, wp, corpus)
    _, published_protocol = public_view(run)
    audit = {}
    question_error = None
    try:
        qs = phrase_questions(orders, wp, run.tracer, run.log, audit=audit,
                              checkpoint_path=run.dir / "04_wording.ckpt.json",
                              corpus=corpus, public_protocol=published_protocol)
    except Exception as exc:
        question_error = exc
        qs = deepcopy(getattr(exc, "completed_questions", []))
        if isinstance(getattr(exc, "report", None), dict):
            audit.update(exc.report)
    finally:
        if audit:
            run.write("04_wording_report.json", audit)
    if pack is not None:
        provenance = run.read(ART["input"])["seed"]
        qs = [{**q, "seed": provenance} for q in qs]
    run.write(ART["questions"], qs)
    audit["candidate_binding"] = _question_source_binding(run, wp, qs)
    run.write("04_wording_report.json", audit)
    if question_error is None:
        (run.dir / QUESTION_WARNING).unlink(missing_ok=True)
    else:
        run.write(QUESTION_WARNING, {"version": "question-generation-warning/v1",
            "status": "warning", "release_eligible": False,
            "error_type": type(question_error).__name__, "error": str(question_error),
            "binding": {"whitepaper_hash": _canonical_hash(wp),
                        "orders_hash": _canonical_hash(orders),
                        "questions_hash": _canonical_hash(qs)}})
        run.log(f"  ⚠ 出题阶段没有形成可继续使用的题面:{type(question_error).__name__}: "
                f"{str(question_error)[:160]}；保存完整审计并继续语料、接地和质量报告")
    run.set_algo(questions=len(qs))
    if question_error is not None and _global_generation_failure(question_error):
        raise question_error


def _corpus_checkpoint_identity(wp: dict, world: dict, target: int,
                                delta_mode: bool, only: set | None,
                                pairs: set[tuple[str, int]] | None, orders=None) -> str:
    """计算渲染中断点的稳定身份；输入或渲染范围变化即不可续用。"""
    payload = {
        "version": CORPUS_RENDER_CONTRACT_VERSION,
        "whitepaper": wp,  # style_spec 属于白皮书，随整体一起绑定。
        "world": world,
        "orders": orders,
        "target_tokens": target,
        "delta_scope": {
            "mode": "delta" if delta_mode else "full",
            "entities": sorted(only or []),
            "entity_sessions": [[entity, session] for entity, session in sorted(pairs or set())],
        },
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def stage_corpus(run: Run):
    from pipeline.supply_capacity import require_order_capacity
    require_order_capacity(run)
    wp = run.read(ART["whitepaper"]); frozen_wp = deepcopy(wp); world = run.read(ART["world"])
    prior_published_corpus = deepcopy(run.read(ART["corpus"])) if run.has(ART["corpus"]) else None
    orders = run.read(ART["orders"]) if run.has(ART["orders"]) else []
    ws = WorldState.from_dict(world)
    validate_seed_identity(run, wp)
    validate_seed_world(wp, ws)
    _require_current_world_review(run, wp, ws)
    _require_v6_supply_gate(run, wp, world)
    # Resuming an old world still uses the current corpus contract. This copy
    # does not silently rewrite the frozen whitepaper or the world.
    wp = {**wp, "quality_contract": {**wp.get("quality_contract", {}), "corpus_review": True}}
    if run.scenario == "game" and not ws.narrative:
        raise WorldBlueprintError(
            "game corpus 缺少合法 Story Ledger；请先强制重跑 world，禁止退化为普通语料渲染")
    cfg = run.manifest["config"]
    requested_delta = "render_only" in cfg or "render_only_pairs" in cfg
    # 叙事世界也允许在【已有完整 05】上定向补充实体×章节；render_corpus 会继续
    # 使用 Story Ledger/reviewer，并在全量旧+新语料上复核全部 canonical event。
    # 没有既有语料时禁止把 delta 冒充首次全量渲染。
    if requested_delta and ws.narrative:
        if not run.has(ART["corpus"]):
            raise WorldBlueprintError(
                "game narrative delta 需要已有完整 corpus；首次渲染必须覆盖全部 Story Ledger")
        published = run.read(ART["corpus"])
        published_corpus = published.get("corpus", published) if isinstance(published, dict) else {}
        expected_sessions = set(ws.sessions())
        published_sessions = {
            item.get("session_id") for item in (published_corpus.get("sessions") or [])
            if isinstance(item, dict)
        }
        published_done = set(published.get("done_weeks") or []) if isinstance(published, dict) else set()
        if published_sessions != expected_sessions or published_done != expected_sessions:
            raise WorldBlueprintError(
                "game narrative delta 的既有 corpus 未精确覆盖全部章节，拒绝把局部语料当完整基线")
    delta_mode = requested_delta
    refresh_reasons = []
    if requested_delta and run.has(ART["corpus"]):
        from pipeline.corpus_contract import validate_corpus
        previous_quality = validate_corpus(ws, run.read(ART["corpus"]))
        if previous_quality["status"] != "passed":
            refresh_reasons = sorted({issue["code"] for issue in previous_quality["issues"]})
            delta_mode = False
            run.log(f"  ⓘ 旧正文验收与当前世界不一致，增量请求升级为全量重渲:{refresh_reasons}")
    if delta_mode and ws.narrative:
        run.log("  ⓘ game Story Ledger 定向补渲：只追加指定范围，收口仍复核全部 canonical event")
    run.set_algo(render_strategy={"requested": "delta" if requested_delta else "full",
                                 "effective": "delta" if delta_mode else "full",
                                 "refresh_reasons": refresh_reasons})
    target = int(run.manifest["config"].get("target_tokens", 1_000_000))
    token_target = cfg.get("corpus_token_target")
    token_setting = cfg.get("corpus_tokenizer", "cl100k_base")
    token_encoding, token_sep, token_version = token_setting.partition("@")
    if token_target is not None and cfg.get("delivery_target") and (not token_sep or not token_version):
        raise ValueError("Delivery tokenizer must pin encoding@version")
    token_state = {}
    only = set(cfg.get("render_only") or []) if delta_mode else None
    pairs = ({(item[0], int(item[1])) for item in (cfg.get("render_only_pairs") or [])
              if isinstance(item, (list, tuple)) and len(item) == 2} if delta_mode else None)
    identity = _corpus_checkpoint_identity(wp, world, target, delta_mode, only, pairs, orders)
    material_identity = _corpus_checkpoint_identity(wp, world, 0, delta_mode, only, pairs, orders)
    if token_target is not None:
        identity = _canonical_hash({"legacy_identity": identity, "corpus_token_target": token_target,
                                    "corpus_tokenizer": token_encoding})
        if run.has("05_corpus_token_scale.json"):
            previous_tokens = run.read("05_corpus_token_scale.json")
            if previous_tokens.get("material_identity") == material_identity:
                token_state = deepcopy(previous_tokens)
    ckpt = run.dir / CORPUS_CKPT
    if ckpt.exists():                                       # 中断续渲只读独立 checkpoint
        try:
            st = run.read(CORPUS_CKPT)
            if not isinstance(st, dict) or not isinstance(st.get("corpus"), dict) \
                    or not isinstance(st.get("done_weeks"), list):
                raise ValueError("corpus checkpoint shape is invalid")
        except Exception as exc:
            # A checkpoint is only an optimization.  Its corruption cannot
            # invalidate a previously published corpus or stop a clean rebuild.
            run.log(f"  ⚠ 语料续跑 checkpoint 无法读取:{type(exc).__name__}: {str(exc)[:160]}；"
                    "忽略坏缓存，从正式语料或空候选继续")
            if delta_mode and run.has(ART["corpus"]):
                st = run.read(ART["corpus"])
                corpus, done = st["corpus"], set(st["done_weeks"])
            else:
                corpus, done = {"sessions": []}, set()
        else:
            if st.get("identity") == identity or st.get("material_identity") == material_identity:
                corpus, done = st["corpus"], set(st["done_weeks"])
                if token_target is not None and isinstance(st.get("token_state"), dict):
                    token_state = deepcopy(st["token_state"])
                run.log(f"  ↻ 续渲:已完成 {len(done)} 周")
            elif delta_mode and run.has(ART["corpus"]):
                st = run.read(ART["corpus"])
                corpus, done = st["corpus"], set(st["done_weeks"])
                run.log("  ⓘ 渲染 checkpoint 不匹配，增量改从已发布完整语料开始")
            else:
                corpus, done = {"sessions": []}, set()
                run.log("  ⓘ 渲染 checkpoint 与当前输入不匹配，忽略并从空语料开始")
    elif run.has("05_corpus_candidate.json") and run.has(CORPUS_WARNING):
        # A fail-soft run publishes its partial corpus for diagnostics and may
        # later be copied into a derived recovery run without the live
        # checkpoint.  Reuse that work only when the warning binds the exact
        # frozen whitepaper, world and candidate bytes used by this run.
        try:
            candidate = run.read("05_corpus_candidate.json")
            warning = run.read(CORPUS_WARNING)
            binding = warning.get("binding") if isinstance(warning, dict) else None
            sessions = (candidate.get("corpus") or {}).get("sessions")
            candidate_done = candidate.get("done_weeks")
            session_ids = [item.get("session_id") for item in sessions]
            expected_sessions = set(ws.sessions())
            if (warning.get("version") != "corpus-generation-warning/v1"
                    or not isinstance(binding, dict)
                    or binding.get("whitepaper_hash") != _canonical_hash(frozen_wp)
                    or binding.get("world_hash") != _canonical_hash(world)
                    or binding.get("candidate_hash") != _canonical_hash(candidate)
                    or not isinstance(sessions, list)
                    or not isinstance(candidate_done, list)
                    or any(type(sid) is not int for sid in session_ids)
                    or len(session_ids) != len(set(session_ids))
                    or set(session_ids) != set(candidate_done)
                    or not set(candidate_done) <= expected_sessions):
                raise ValueError("partial corpus recovery binding or coverage is invalid")
            corpus, done = candidate["corpus"], set(candidate_done)
            run.log(f"  ↻ 续渲:复用精确绑定的部分语料 {len(done)}/{len(expected_sessions)} 期")
        except Exception as exc:
            corpus, done = {"sessions": []}, set()
            run.log(f"  ⓘ 部分语料恢复证据无效:{type(exc).__name__}: {str(exc)[:160]}；"
                    "从空候选开始")
    elif delta_mode and run.has(ART["corpus"]):             # 非剧情闭环的增量渲染从上一版成品起步
        st = run.read(ART["corpus"])
        corpus, done = st["corpus"], set(st["done_weeks"])
    else:
        corpus, done = {"sessions": []}, set()

    # A size-only change reuses accepted prose. Validate against this exact
    # world before extending it; normal material changes still regenerate.
    if (not corpus.get("sessions") and not delta_mode and prior_published_corpus
            and (cfg.get("haystack_ratio") is not None or token_target is not None)):
        from pipeline.corpus_contract import validate_corpus
        scale_binding = run.read("05_corpus_scale.json").get("binding") if run.has("05_corpus_scale.json") else None
        current_scale_binding = _corpus_scale_binding(run)
        same_world = (isinstance(scale_binding, dict)
                      and all(scale_binding.get(key) == current_scale_binding[key]
                              for key in ("whitepaper_hash", "world_hash", "orders_hash")))
        if ((same_world
                or (not run.has("05_corpus_scale.json") and not run.has(CORPUS_WARNING)))
                and validate_corpus(ws, prior_published_corpus).get("status") == "passed"):
            corpus = deepcopy(prior_published_corpus["corpus"])
            done = set(prior_published_corpus["done_weeks"])
            run.log("  ↻ 仅补语料规模差额，复用已有正文")

    def save():
        run.write(CORPUS_CKPT, {"identity": identity, "material_identity": material_identity, "corpus": corpus,
                                "done_weeks": sorted(done),
                                **({"token_state": token_state} if token_target is not None else {})})

    def save_token_scale(state):
        run.write("05_corpus_token_scale.json", {**state, "material_identity": material_identity,
                                                "status": "in_progress"})

    corpus_error = None
    try:
        render_corpus(wp, ws, target, run.tracer, corpus, done, save, run.log,
                        only_entities=only, only_entity_sessions=pairs,
                        orders=orders,
                      **({"corpus_token_target": token_target, "corpus_tokenizer": token_encoding,
                           "token_state": token_state, "token_save_cb": save_token_scale}
                         if token_target is not None else {}),
                      **({"haystack_ratio": cfg["haystack_ratio"]} if cfg.get("haystack_ratio") is not None else {}))
    except Exception as exc:
        corpus_error = exc
    # A partial corpus is still a useful candidate for downstream diagnostics.
    # Bind the warning to exact inputs; release remains closed and a later
    # explicit rerun can reuse the checkpoint to fill the missing material.
    candidate_corpus = {"corpus": corpus, "done_weeks": sorted(done)}
    if corpus_error is None:
        run.write(ART["corpus"], candidate_corpus)
        (run.dir / CORPUS_WARNING).unlink(missing_ok=True)
        (run.dir / "05_corpus_candidate.json").unlink(missing_ok=True)
        ckpt.unlink(missing_ok=True)
    else:
        run.write("05_corpus_candidate.json", candidate_corpus)
        published_corpus = prior_published_corpus if prior_published_corpus is not None else candidate_corpus
        if prior_published_corpus is None:
            run.write(ART["corpus"], published_corpus)
        run.write(CORPUS_WARNING, {"version": "corpus-generation-warning/v1",
            "status": "warning", "release_eligible": False,
            "error_type": type(corpus_error).__name__, "error": str(corpus_error),
            "evidence": deepcopy(getattr(corpus_error, "report", None)),
            "binding": {"whitepaper_hash": _canonical_hash(frozen_wp),
                        "world_hash": _canonical_hash(world),
                        "corpus_hash": _canonical_hash(published_corpus),
                        "candidate_hash": _canonical_hash(candidate_corpus)}})
        run.log(f"  ⚠ 正文阶段留下待补材料:{type(corpus_error).__name__}: {str(corpus_error)[:160]}；"
                "已保存局部结果；正文阶段停止，下游保持未完成")
    ch = sum(len(dd.get("content", "")) for x in corpus["sessions"] for dd in x["docs"])
    run.set_algo(docs=sum(len(x["docs"]) for x in corpus["sessions"]), chars=ch)
    if token_target is not None:
        from pipeline.corpus_tokens import CorpusTokenCounter, plan_filler_batch
        published = run.read(ART["corpus"])["corpus"]
        counter = CorpusTokenCounter(token_encoding, expected_version=token_version or "0.12.0")
        measured = counter.measure(published, cache=token_state.get("measurement", {}).get("cache"))
        token_plan = plan_filler_batch(measured, token_target, cfg.get("haystack_ratio"))
        report = {**deepcopy(token_state), "version": "corpus-token-scale/v1",
                  "status": "completed" if corpus_error is None else "incomplete",
                  "material_identity": material_identity, "binding": _corpus_scale_binding(run),
                  "target_tokens": token_target, "haystack_ratio": cfg.get("haystack_ratio"),
                  "tokenizer": counter.tokenizer, "tokenizer_id": counter.tokenizer_id,
                  "measurement": measured, "plan": token_plan,
                  "target_met": token_plan["target_met"],
                  "deficit_tokens": token_plan["remaining_filler_tokens"],
                  "total_tokens": measured["total"]["tokens"],
                  "total_characters": measured["total"]["characters"],
                  "attempt_complete": corpus_error is None,
                  "warning": ("corpus_token_generation_incomplete" if corpus_error is not None else
                              None if token_plan["target_met"] else "corpus_token_target_unmet")}
        if corpus_error is not None and token_state.get("measurement"):
            report["candidate_measurement"] = token_state["measurement"]
        run.write("05_corpus_token_scale.json", report)
        run.set_algo(corpus_tokens={key: report[key] for key in
                     ("target_tokens", "tokenizer", "total_tokens", "total_characters",
                      "deficit_tokens", "target_met", "attempt_complete", "warning")})
        # Keep the legacy character diagnostic separate from token acceptance.
        scale = corpus_scale(published, 0, None)
        scale.update(binding=_corpus_scale_binding(run), warning=None,
                     attempt_complete=corpus_error is None, acceptance_unit="tokenizer_tokens")
        run.write("05_corpus_scale.json", scale)
    elif cfg.get("haystack_ratio") is not None:
        scale = corpus_scale(run.read(ART["corpus"])["corpus"], target, cfg["haystack_ratio"])
        scale.update(binding=_corpus_scale_binding(run),
                     warning=None if scale["target_met"] else "corpus_scale_target_unmet",
                     attempt_complete=corpus_error is None,
                     retry_from_checkpoint="--force --from corpus" if corpus_error is not None else None)
        run.write("05_corpus_scale.json", scale)
        run.set_algo(corpus_scale={k: v for k, v in scale.items() if k != "binding"})

    # ★多样性硬指标(诊断附加项,非主链):测本次 corpus 全部文档正文,写进 manifest.algo.diversity。
    #   失败绝不拖垮整个 run —— 只 log 一句警告 + 跳过。
    try:
        texts = [dd.get("content", "") for x in corpus["sessions"] for dd in x["docs"] if dd.get("content")]
        if len(texts) >= 2:
            div = diversity_report(texts)
            run.set_algo(diversity=div)
            ndg = div.get("NDG_n-gram多样性(越高越好)", {}).get("point")
            ido = div.get("IDO_跨文档重叠(越低越好)", {}).get("point")
            cr = div.get("CR_gzip压缩比(越低越好)", {}).get("point")
            run.log(f"  ◆ 多样性:NDG={ndg} IDO={ido} CR={cr}(n_docs={div.get('n_docs')})")
        else:
            run.log(f"  ⓘ 多样性跳过:文档不足 2 篇({len(texts)})")
    except Exception as e:
        run.log(f"  ⚠ 多样性测量失败(已跳过,不影响 run):{e}")
    if corpus_error is not None:
        raise corpus_error


def stage_grounding(run: Run):
    """Merge original questions and corpus under the whitepaper's review contract.

    New runs use public LLM review; lexical matches remain diagnostics. Legacy
    runs retain their declared grounding path. Keep all review outcomes in the
    report. Parsed per-candidate format failures remain pending; publish only
    the completely certified subset after the shared delivery validation.
    """
    from pipeline.grounding import run_grounding
    wp = {}
    if run.has(ART["whitepaper"]):
        wp = run.read(ART["whitepaper"])
        validate_seed_identity(run, wp)
        _require_current_world_review(run, wp)
        if wp.get("seed_contract"):
            validate_seed_world(wp, WorldState.from_dict(run.read(ART["world"])))
    elif run.manifest.get("config", {}).get("seed_pack_digest"):
        raise SeedPackError("种子运行缺少白皮书，不能发布题库")
    questions = run.read(ART["questions"])
    _require_current_question_wording(run, wp, questions)
    _require_current_question_sources(run, wp, questions)
    corpus_obj = run.read(ART["corpus"])
    _require_current_corpus_review(run, wp, corpus_obj)
    from pipeline.grounding_review import enabled, review_grounding, REVIEW_ARTIFACT
    if enabled(wp):
        import config
        from eval.provenance import public_protocol
        cfg = run.manifest.get("config", {})
        kept, report, semantic = review_grounding(
            questions, corpus_obj, public_protocol(run.read("00_about.json")),
            chat_json=run.tracer.chat_json, model=config.REVIEWER_MODEL,
            max_calls=cfg.get("semantic_max_calls"),
            isolated_reference=(wp.get("quality_contract") or {}).get("isolated_reference_audit", False),
            max_tokens=cfg.get("semantic_max_tokens", 4096),
            max_input_chars=cfg.get("semantic_max_input_chars", 200000),
            checkpoint_dir=run.dir / "06_review_checkpoints",
            cache_namespace=(_grounding_source_binding(run)["public_hash"]
                             if (wp.get("generation_contract") or {}).get("material_first") else None),
            workers=cfg.get("semantic_workers", 1))
        run.write(REVIEW_ARTIFACT, semantic)
        run.write("06_grounding_report.json", report)
        if not report["delivery_safe"]:
            # Keep the full trace and the questions that already have complete
            # per-item certification. Unfinished candidates stay pending; an
            # execution fault in a later item must not erase earlier decisions.
            report["provisional_before_delivery_validation"] = {
                "overall": deepcopy(report.get("overall", {})),
                "by_line": deepcopy(report.get("by_line", {})),
                "by_capability": deepcopy(report.get("by_capability", {})),
                "candidate_count": len(kept)}
            run.log("  ⚠ 逐题审阅发生执行故障；保留已完成认证的题，其余题保持待审")
    else:
        kept, report = run_grounding(questions, corpus_obj)
    from pipeline import order_warning
    if (run.has(order_warning.WARNING_ARTIFACT)
            and run.has(order_warning.PROPOSAL_ARTIFACT)
            and run.has(order_warning.RESOLUTION_ARTIFACT)):
        receipt = run.read(order_warning.RESOLUTION_ARTIFACT)
        warning = run.read(order_warning.WARNING_ARTIFACT)
        proposal = run.read(order_warning.PROPOSAL_ARTIFACT)
        world = run.read(ART["world"])
        if not order_warning.validate_resolution(receipt, wp, world, questions, warning, proposal):
            kept, report = order_warning.apply_resolution(kept, report, receipt)
            run.log(f"  ⓘ 订单警告按产线收口：隔离 {report['n_scoped_excluded']} 道 L3 入选题，其余产线继续")
    run.write(ART["grounding"], kept)
    report["source_binding"] = _grounding_source_binding(run)
    run.write("06_grounding_report.json", report)
    o = report["overall"]
    algo_update = {"grounding": {"overall": o, "by_line": report["by_line"],
                                  "by_capability": report["by_capability"],
                                  "n_dropped": report["n_dropped"],
                                  "n_pending": report.get("n_pending", 0),
                                  **({"execution_complete": report["execution_complete"],
                                      "delivery_safe": report["delivery_safe"]}
                                     if "delivery_safe" in report else {})}}
    # grounding 是产物汇合点，必须据当前 06 重算闭环账本。否则一次早期失败后
    # 从中游恢复，即使最终题量已达标，manifest 仍会永久携带陈旧 UNMET。
    target = (run.manifest.get("algo") or {}).get("targetspec") or {}
    floors = target.get("per_line_min") or {}
    if target and isinstance(floors, dict):
        per_line_final = {
            line_id: int((report["by_line"].get(line_id) or {}).get("grounded", 0) or 0)
            for line_id in floors
        }
        min_questions = int(target.get("min_questions", 0) or 0)
        met = (int(o.get("grounded", 0) or 0) >= min_questions
               and all(per_line_final[line_id] >= int(floor)
                       for line_id, floor in floors.items()))
        algo_update.update({
            "per_line_final": per_line_final,
            "met_status": "MET" if met else "UNMET_GROUNDING",
        })
    run.set_algo(**algo_update)
    run.log(f"  ★接地闸:{o['grounded']}/{o['n']} 接地({(o['survival'] or 0):.0%}),弃 {report['n_dropped']}，待审 {report.get('n_pending', 0)} "
            f"→ 出厂题库 {ART['grounding']}")



def stage_quality(run: Run):
    """Summarize and bind the per-question decisions produced by grounding."""
    from pipeline.quality import evaluate_release
    if (run.read(ART["whitepaper"]).get("seed_contract") or {}).get("schema_version") == 2:
        from pipeline.seed_lineage import ARTIFACT as lineage_artifact, seed_lineage_report
        run.write(lineage_artifact, seed_lineage_report(run.dir))
    report = evaluate_release(run.dir)
    run.write(ART["quality"], report)
    if run.manifest.get("config", {}).get("delivery_target"):
        from pipeline.supply import write_delivery_report
        write_delivery_report(run)
    run.set_algo(quality={"version": report["version"], "status": report["status"],
                          "eligible": report["eligible"], "scope": report["scope"],
                          "issue_count": len(report["issues"])})
    counts = ((report.get("checks") or {}).get("partition") or {}).get("counts") or {}
    if not report["eligible"]:
        codes = list(dict.fromkeys(issue["code"] for issue in report["issues"]))
        if codes:
            run.log(f"  ⚠ 逐题结果汇总失败:{codes}；全部候选和审阅记录仍已落盘")
        else:
            run.log("  ⓘ 逐题结果已收口，可用题为 0；淘汰和待审记录均已保留")
        return
    run.log("  ✓ 逐题结果已收口:"
            f"可用 {counts.get('released', 0)}，淘汰 {counts.get('rejected', 0)}，"
            f"待审 {counts.get('pending_review', 0)}，范围排除 {counts.get('scoped_excluded', 0)}")


def _quality_is_current(run: Run) -> bool:
    from pipeline.quality import quality_snapshot
    snapshot = quality_snapshot(run.dir)
    codes = {issue.get("code") for issue in snapshot.get("issues", []) if isinstance(issue, dict)}
    return (run.has(ART["quality"]) and snapshot.get("status") in ("passed", "failed")
            and "invalid_release_receipt" not in codes)


STAGES = [
    Stage("input",      [],                       stage_input,      ART["input"]),
    Stage("whitepaper", ["input"],                stage_whitepaper, ART["whitepaper"]),
    Stage("world",      ["whitepaper"],           stage_world,      ART["world"], is_current=_world_is_current),
    Stage("disclosure", ["whitepaper", "world"],  stage_disclosure, ART["disclosure"],
          is_current=_disclosure_is_current, refreshes=("world",), freshness_covers=("world",)),
    Stage("orders",     ["whitepaper", "world", "disclosure"],  stage_orders,     ART["orders"]),
    Stage("well_posed", ["orders", "world"],      stage_well_posed, "03_well_posed_report.json"),  # ★边A闸:出题前剔 ill-posed(覆写 03_orders)
    Stage("corpus",     ["whitepaper", "world", "disclosure", "well_posed"], stage_corpus, ART["corpus"], is_current=_corpus_is_current),
    Stage("questions",  ["orders", "well_posed", "corpus"], stage_questions, ART["questions"], is_current=_questions_is_current),
    Stage("grounding",  ["questions", "corpus"], stage_grounding, ART["grounding"], is_current=_grounding_is_current),
    Stage("quality", ["world", "questions", "corpus", "grounding"], stage_quality, ART["quality"],
          is_current=_quality_is_current),
]

def run_generation_tail(run: Run):
    """Use the registered material -> wording -> review order in supply rounds."""
    names = [stage.name for stage in STAGES]
    for stage in STAGES[names.index("corpus"):names.index("grounding") + 1]:
        _run_stage(run, stage.name, stage.fn, stage.artifact)


def generation_stages(run: Run, *, finalize_only=False):
    """Calibration is a generation tail, executed once after the supply loop."""
    if not run.manifest.get("config", {}).get("calibration"):
        return STAGES
    from pipeline.calibration import (stage_calibration, stage_selection,
                                      calibration_can_skip, selection_is_current)
    # Finalization consumes the saved question partition and material. The
    # existing quality function validates those files directly; old generator
    # receipts must not cause world generation or semantic review to run again.
    prefix = ([Stage("quality", [], stage_quality, ART["quality"], is_current=_quality_is_current)]
              if finalize_only else STAGES)
    return [*prefix,
            Stage("calibration", ["quality"], stage_calibration, ART["calibration"],
                  is_current=calibration_can_skip),
            Stage("selection", ["calibration"], stage_selection, ART["selection"],
                  is_current=selection_is_current, refreshes=("calibration",))]


def finish_generation(run: Run, to_stage=None, force=False):
    """Finish a quantity-driven world without rerunning its production rounds."""
    if run.manifest.get("config", {}).get("calibration") and to_stage in (None, "calibration", "selection"):
        drive(run, generation_stages(run, finalize_only=True), "quality", to_stage, None, force)



def _print_runs():
    rows = list_runs()
    if not rows:
        print("(无 run;output/runs/ 为空)"); return
    print(f"{'run_id':<34} {'status':<8} {'tag':<10} active_lines / Q / docs")
    for m in rows:
        a = m.get("algo", {})
        print(f"{m['run_id']:<34} {m.get('status', '?'):<8} {str(m.get('tag') or '-'):<10} "
              f"{a.get('active_lines', '?')} / {a.get('questions', '?')} / {a.get('docs', '?')}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--process-questions", action="store_true", default=None,
                    help="Enable original L3 business-process proposals for a frozen older run")
    ap.add_argument("--seed-pack", help="策展种子 JSON；增强 input→whitepaper，后续阶段保持同一合同")
    ap.add_argument("--delivery-target", type=Path,
                    help="交付目标 JSON：真实 corpus token、筛后题量、逐线配额及供给补充轮数")
    ap.add_argument("--delivery-survival-rates", type=Path,
                    help="可选：带来源及样本数的逐线候选到筛后留存率 JSON")
    ap.add_argument("--world-semantic-review", action="store_true", default=None,
                    help="为旧白皮书显式启用原 world 阶段的业务语义审阅；新白皮书自动启用")
    ap.add_argument("--target-mchars", "--target-mtokens", dest="target_mtokens", type=float, default=None,
                    help="目标正文百万字符数；新 run 缺省 1.0，续 run 沿用；旧名称 target-mtokens 兼容保留")
    ap.add_argument("--question-budget", type=int, default=None,
                    help="普通流程总订单上限；新 run 缺省 30，续 run 沿用；不承诺最终存活题数")
    ap.add_argument("--semantic-workers", type=int, default=None,
                    help="逐题语义审阅并行数；只并行相互独立的题，最终仍按原顺序统一验收")
    ap.add_argument("--calibration-config", type=Path,
                    help="生成末尾执行运动员试答、判分和简单题筛选；JSON配置在run中冻结")
    ap.add_argument("--release", action="store_true",
                    help="完整发布流程：原有生成→四运动员评测→删除全员答对题；缺省使用examples/release_four.json")
    ap.add_argument("--tag", default=None, help="人类标签(进 manifest,不影响 run_id)")
    ap.add_argument("--from", dest="from_stage", default=None, help=f"从哪个 stage 起跑 {list(ART)}")
    ap.add_argument("--to", dest="to_stage", default=None, help="跑到哪个 stage 止")
    ap.add_argument("--only", default=None, help="只跑某一个 stage")
    ap.add_argument("--force", action="store_true", help="重跑已完成的 stage")
    ap.add_argument("--run", dest="run_id", default=None, help="在已有 run 上继续/重跑(不新开)")
    ap.add_argument("--resume", action="store_true", help="[兼容]续跑该 scenario 最近一次 run")
    ap.add_argument("--list-runs", action="store_true", help="列出所有 run 后退出")
    # ── §S 闭环旋钮:给了 --min-questions 即走 build_to_target(反推世界规模/配额→①供给环→渲→接地→②纠偏) ──
    ap.add_argument("--min-questions", type=int, default=None, help="出厂题库总下限;给了即启用闭环旋钮(否则按原 drive 跑)")
    ap.add_argument("--per-line", action="append", default=None,
                    help="每线 floor,可重复:--per-line L1_timeline=30 --per-line L2_relational=15(缺省按白皮书 weight 派生)")
    ap.add_argument("--haystack-ratio", type=float, default=None,
                    help="草堆字符/其余正文字符的目标下限；新 run 缺省9(约90%%草堆)，旧 run 沿用")
    ap.add_argument("--time-span-weeks", type=int, default=None, help="时间跨度周数(None=反推/默认)")
    ap.add_argument("--max-rounds", type=int, default=2, help="②实测纠偏环最多整轮重渲次数(bounded)")
    ap.add_argument("--total-only", action="store_true", help="总题量为硬下限；白皮书逐线权重用于生产配额，显式 --per-line 仍为硬下限")
    ap.add_argument("--max-world-entities", type=int, default=80, help="闭环每个独立世界的实体上限(8–80)，种子结构最低要求仍须满足")
    a = ap.parse_args()
    if a.release and a.calibration_config is None:
        a.calibration_config = Path(__file__).resolve().parents[1] / "examples/release_four.json"

    if a.list_runs:
        _print_runs(); return
    if a.haystack_ratio is not None and (not math.isfinite(a.haystack_ratio) or a.haystack_ratio < 0):
        ap.error("--haystack-ratio 必须为有限非负数")
    if a.target_mtokens is not None and (not math.isfinite(a.target_mtokens) or a.target_mtokens < 0):
        ap.error("--target-mchars 必须为有限非负数")
    if a.question_budget is not None and a.question_budget < 1:
        ap.error("--question-budget 必须为正整数")
    if a.question_budget is not None and a.min_questions is not None:
        ap.error("--question-budget 与闭环 --min-questions 不能同时指定")
    if a.semantic_workers is not None and not 1 <= a.semantic_workers <= 16:
        ap.error("--semantic-workers 必须在 1 到 16 之间")
    if a.total_only and a.min_questions is None:
        ap.error("--total-only requires --min-questions")
    if not 8 <= a.max_world_entities <= 80 or a.max_rounds < 1 or (a.min_questions is not None and a.min_questions < 1):
        ap.error("Invalid quantity/world limits")
    if a.time_span_weeks is not None and not 6 <= a.time_span_weeks <= 26:
        ap.error("--time-span-weeks must be between 6 and 26")

    if a.seed_pack and a.scenario:
        ap.error("--seed-pack 已定义场景，不能同时指定 --scenario")
    try:
        seed_cfg = seed_config(a.seed_pack) if a.seed_pack else {}
    except (SeedPackError, OSError) as error:
        ap.error(str(error))
    requested_scenario = ("seed_" + seed_cfg["seed_id"] if seed_cfg
                          else a.scenario or "office")

    if a.run_id:                                            # 在已有 run 上继续:scenario 取自其 manifest
        run_id, scenario = a.run_id, requested_scenario
        mf = RUNS_DIR / a.run_id / "manifest.json"
        if mf.exists():
            scenario = json.loads(mf.read_text(encoding="utf-8")).get("scenario", requested_scenario)
    elif a.resume:
        run_id = latest_run_for(requested_scenario) or new_run_id(requested_scenario)
        scenario = requested_scenario
    else:
        run_id, scenario = new_run_id(requested_scenario), requested_scenario

    cfg = {"from": a.from_stage, "to": a.to_stage, "only": a.only}
    if a.delivery_target:
        from pipeline.delivery_target import DeliveryTarget
        from pipeline.corpus_tokens import CorpusTokenCounter
        try:
            delivery_target = DeliveryTarget.from_dict(json.loads(a.delivery_target.read_text(encoding="utf-8")))
            if not delivery_target.requested_lines:
                raise ValueError("Production delivery target must explicitly name requested_lines")
            if a.min_questions is not None or a.per_line:
                raise ValueError("Use a separate run for legacy grounding-stage quantity targets")
            name, separator, version = delivery_target.tokenizer.partition("@")
            if not separator or not version:
                raise ValueError("tokenizer must name encoding@version, for example cl100k_base@0.12.0")
            CorpusTokenCounter(name, expected_version=version)
            from pipeline.supply import author_brief
            survival_rates = (json.loads(a.delivery_survival_rates.read_text(encoding="utf-8"))
                              if a.delivery_survival_rates else None)
            author_brief(delivery_target.to_dict(), a.question_budget, survival_rates=survival_rates)
            previous_path = RUNS_DIR / run_id / "manifest.json"
            if previous_path.exists():
                previous_config = json.loads(previous_path.read_text(encoding="utf-8")).get("config", {})
                previous_target = previous_config.get("delivery_target")
                if previous_target != delivery_target.to_dict():
                    raise ValueError("Delivery target is frozen for an existing run; use a new run for changes")
                if a.delivery_survival_rates and previous_config.get("delivery_survival_rates") != survival_rates:
                    raise ValueError("Delivery survival evidence is frozen for an existing run")
            cfg.update(delivery_target=delivery_target.to_dict(), corpus_token_target=delivery_target.corpus_tokens,
                       corpus_tokenizer=delivery_target.tokenizer)
            if survival_rates is not None:
                cfg["delivery_survival_rates"] = survival_rates
        except (OSError, ValueError, RuntimeError) as exc:
            ap.error(str(exc))
    elif a.delivery_survival_rates:
        ap.error("--delivery-survival-rates requires --delivery-target")
    if a.calibration_config:
        from pipeline.calibration import load_config
        try:
            cfg["calibration"] = load_config(a.calibration_config)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
    if a.semantic_workers is not None:
        cfg["semantic_workers"] = a.semantic_workers
    if a.world_semantic_review:
        cfg["world_semantic_review"] = True
    manifest_path = RUNS_DIR / run_id / "manifest.json"
    previous_config = json.loads(manifest_path.read_text(encoding="utf-8")).get("config", {}) if manifest_path.exists() else {}
    effective_delivery = cfg.get("delivery_target") or previous_config.get("delivery_target")
    if effective_delivery:
        if a.min_questions is not None or a.per_line or a.target_mtokens is not None:
            ap.error("Delivery mode uses its frozen token target and candidate budget; legacy quantity flags require a separate run")
        if previous_config.get("delivery_target") and a.question_budget is not None:
            from pipeline.supply import author_brief
            expected = author_brief(effective_delivery, previous_config.get("question_budget"),
                                    survival_rates=previous_config.get("delivery_survival_rates"))["candidate_budget"]
            if a.question_budget != expected:
                ap.error("Candidate reserve is frozen with the delivery target; use a new run for changes")
    if a.min_questions is None and not manifest_path.exists():
        cfg["single_pass_world_limits"] = {"max_world_entities": a.max_world_entities,
                                          "time_span_weeks": a.time_span_weeks}
    if a.haystack_ratio is not None:
        cfg["haystack_ratio"] = a.haystack_ratio
    elif not manifest_path.exists():
        cfg["haystack_ratio"] = 9.0
    if a.process_questions:
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if (previous.get("stages", {}).get("orders", {}).get("done")
                and previous.get("config", {}).get("process_proposals") is not True
                and not (a.force and (a.only == "orders" or (a.only is None and
                    a.from_stage in (None, "input", "whitepaper", "world", "orders"))))):
            ap.error("Enabling process questions requires regenerating original orders and downstream stages")
        cfg["process_proposals"] = True
    if a.question_budget is not None:
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if (previous.get("stages", {}).get("orders", {}).get("done")
                and previous.get("config", {}).get("question_budget") != a.question_budget
                and not (a.force and (a.only == "orders" or (a.only is None and
                    a.from_stage in (None, "input", "whitepaper", "world", "orders"))))):
            ap.error("修改已有订单预算需 --force --from orders（或 --force --only orders 后续跑），以失效旧的下游产物")
        cfg["question_budget"] = a.question_budget
    elif not manifest_path.exists() and a.min_questions is None and not cfg.get("delivery_target"):
        cfg["question_budget"] = 30
    if seed_cfg and manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            seed_cfg = seed_config(a.seed_pack, previous.get("config") or {})
        except (SeedPackError, OSError) as error:
            ap.error(str(error))
    cfg.update(seed_cfg)
    if a.target_mtokens is not None:
        cfg["target_tokens"] = int(a.target_mtokens * 1_000_000)
    elif not (RUNS_DIR / run_id / "manifest.json").exists():
        cfg["target_tokens"] = 0 if cfg.get("delivery_target") else 1_000_000
    if effective_delivery and not manifest_path.exists():
        if not effective_delivery["requested_lines"]:
            ap.error("Supply-driven production requires explicit requested_lines in the delivery target")
        cfg["production_control"] = {"version": "supply-driven/v6"}
        cfg["acceptance_scope"] = {"version":1,"count_stage":"generation_quality" if a.to_stage == "quality" else "selection_complete"}
    if previous_config.get("cumulative_recovery") is not None and (
            a.force or a.only is not None or a.from_stage is not None):
        ap.error("Cumulative recovery must enter its dedicated post-corpus world review first")
    from pipeline.cumulative_entry_guard import check_before_run_init
    check_before_run_init(manifest_path)
    run = Run(scenario, run_id, tag=a.tag, config_meta=cfg)
    if (run.manifest.get("config") or {}).get("cumulative_recovery") is not None:
        from pipeline.post_corpus_world_review import run_once
        run_once(run)
        return
    finalize_only = (bool(run.manifest.get("config", {}).get("calibration")) and
                     (a.from_stage in ("quality", "calibration", "selection") or
                      a.only in ("quality", "calibration", "selection")))
    stages = generation_stages(run, finalize_only=finalize_only)
    if any(name in ("calibration", "selection") for name in (a.from_stage, a.to_stage, a.only)) and len(stages) == len(STAGES):
        ap.error("calibration/selection requires --calibration-config or a frozen run configuration")

    calibration = run.manifest["config"].get("calibration")
    reaches_calibration = (a.only == "calibration" or
        (a.only is None and a.from_stage != "selection" and a.to_stage in (None, "calibration", "selection")))
    if calibration and reaches_calibration:
        from pipeline.calibration import preflight_release
        try:
            preflight_release(calibration, log=run.log)
        except ValueError as exc:
            ap.error(str(exc))

    t0 = time.time()
    tgt = run.manifest["config"].get("target_tokens", 1_000_000)
    run.log(f"=== run {run_id}(scenario={scenario},目标 {tgt/1e6:.1f}M 字符)===")

    if a.min_questions is not None and not finalize_only:   # 显式末尾续跑优先，避免复用命令时再次进入供给闭环。
        drive(run, STAGES, None, "whitepaper", None, a.force)
        plm: dict = {}
        for kv in (a.per_line or []):
            k, _, v = kv.partition("=")
            if v.strip().isdigit():
                plm[k.strip()] = int(v)
        spec = TargetSpec(min_questions=a.min_questions, per_line_min=plm,
                          haystack_ratio=run.manifest["config"].get("haystack_ratio", 4.0), time_span_weeks=a.time_span_weeks,
                          total_only=a.total_only, max_world_entities=a.max_world_entities)
        _, status = build_to_target(run, spec, max_rounds=a.max_rounds)
        finish_generation(run, a.to_stage, a.force)
        run.log(f"=== DONE {run_id}:闭环 {status} / {run.tracer.n} 次 LLM / {round((time.time() - t0) / 60, 1)} min / 留痕 {run.dir} ===")
        return

    if (run.manifest.get("config", {}).get("production_control")
            and a.only is None and a.from_stage in (None, "input", "whitepaper", "world")):
        from pipeline.production import produce
        production = produce(run, to_stage=a.to_stage, force=a.force)
        run.log(f"  逐线生产状态: {production['status']} / 第{production['round']}轮 / 证据 11_production.json")
    else:
        drive(run, stages, a.from_stage, a.to_stage, a.only, a.force)
    run.log(f"=== DONE {run_id}:{run.tracer.n} 次 LLM / {round((time.time() - t0) / 60, 1)} min / 留痕 {run.dir} ===")


if __name__ == "__main__":
    main()
