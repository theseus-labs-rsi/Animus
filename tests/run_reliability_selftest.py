"""
Run 基建离线自检：原子 JSON、续跑调用计数、stage 生命周期和 force 下游失效。

跑：./venv/bin/python tests/run_reliability_selftest.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import pipeline.run as run_module
from pipeline.run import Run, Stage, _dependent_stage_names, _run_stage
from pipeline.world_state import WorldState


checks: list[tuple[bool, str]] = []


def ck(name, cond):
    checks.append((bool(cond), name))


with tempfile.TemporaryDirectory() as atomic_temp:
    atomic_path = Path(atomic_temp) / "state.json"
    real_replace = run_module.os.replace
    attempts = []
    def briefly_locked(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            raise PermissionError("temporary Windows sharing violation")
        return real_replace(source, target)
    with patch.object(run_module.os, "replace", side_effect=briefly_locked), \
         patch.object(run_module.time, "sleep") as waited:
        run_module._atomic_write_json(atomic_path, {"complete": True})
    ck("Windows 临时文件占用有限重试后仍原子发布完整 JSON",
       json.loads(atomic_path.read_text(encoding="utf-8")) == {"complete": True}
       and len(attempts) == 3 and waited.call_count == 2)


empty_response = SimpleNamespace(
    choices=[SimpleNamespace(
        finish_reason="length", message=SimpleNamespace(content=""))],
    usage=SimpleNamespace(
        completion_tokens=8192,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=8192)))
try:
    config._completion_text(empty_response)
except ValueError as exc:
    empty_detail = str(exc)
else:
    empty_detail = ""
ck("空正文错误保留 finish_reason 与 reasoning token 证据",
   "finish_reason=length" in empty_detail and "reasoning_tokens=8192" in empty_detail)


with tempfile.TemporaryDirectory() as td:
    run_module.RUNS_DIR = Path(td)
    run_dir = Path(td) / "game__test"
    run_dir.mkdir(parents=True)
    (run_dir / "prompts.jsonl").write_text(
        '{"i": 1}\n{"i": 2}\n', encoding="utf-8")

    run = Run("game", "game__test")
    ck("续跑从 prompts.jsonl 恢复累计 llm_calls", run.tracer.n == 2 and run.manifest["llm_calls"] == 2)

    run.manifest["config"].update({"from": None, "to": None, "only": "world"})
    run._save_manifest()
    resumed = Run("game", "game__test",
                  config_meta={"from": "corpus", "to": "grounding", "only": None})
    ck("新调用的空 only 会清除 manifest 中上次残留的单步选择",
       resumed.manifest["config"] == {
           "from": "corpus", "to": "grounding", "only": None,
       })
    run = resumed

    run.write("artifact.json", {"中文": [1, 2]})
    ck("Run.write 写出完整合法 JSON", run.read("artifact.json") == {"中文": [1, 2]})
    ck("原子写不遗留临时文件", not list(run_dir.glob(".*.tmp")))

    second_lock_rejected = False
    with run.stage_write_lock("holder"):
        try:
            with run.stage_write_lock("contender"):
                pass
        except RuntimeError:
            second_lock_rejected = True
    ck("同一 Run 的第二个进程级写锁被非阻塞拒绝", second_lock_rejected)

    init_lock_rejected = False
    with run.stage_write_lock("holder"):
        try:
            Run("game", "game__test")
        except RuntimeError:
            init_lock_rejected = True
    ck("第二个 Run 实例初始化 manifest 前也必须取得进程锁", init_lock_rejected)

    # 两个对象可先后指向同一 Run；后执行者必须在取得锁后刷新 manifest，不能
    # 用构造时的旧快照抹掉前一个对象刚写入的 stage 状态或配置。
    run.manifest["config"]["shared_option"] = "base"
    run._save_manifest()
    peer = Run("game", "game__test")

    def ok_stage(r):
        r.write("a.json", {"ok": True})

    def counted_stage(r):
        ok_stage(r)
        r.tracer.n += 1
        r.tracer._log(r.tracer.n, "offline-test", [], {}, {})

    run.manifest["config"]["shared_option"] = "fresh"
    _run_stage(run, "a", counted_stage, "a.json")
    a = run.manifest["stages"]["a"]
    ck("stage 成功显式记录 succeeded", a["done"] is True and a["status"] == "succeeded" and a["attempt"] == 1)

    peer.manifest["config"]["peer_option"] = "kept"
    _run_stage(peer, "peer", lambda r: r.write("peer.json", {"ok": True}), "peer.json")
    peer_disk = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ck("独立 Run 实例取得锁后保留磁盘上的新 stage 状态",
       peer_disk["stages"]["a"]["status"] == "succeeded"
       and peer_disk["stages"]["peer"]["status"] == "succeeded")
    ck("manifest 刷新合并调用方尚未落盘的 config",
       peer_disk["config"].get("peer_option") == "kept"
       and peer_disk["config"].get("shared_option") == "fresh")
    ck("旧 Run 实例进 stage 前从 prompts.jsonl 追平调用计数",
       peer.tracer.n == 3 and peer_disk["llm_calls"] == 3)

    # 后续继续使用第一个旧对象时也必须再次刷新，不能反向抹掉 peer。
    _run_stage(run, "after_peer", lambda r: r.write("after_peer.json", {"ok": True}),
               "after_peer.json")
    refreshed_disk = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ck("两个旧 Run 实例交替执行不互相覆盖 stage",
       all(refreshed_disk["stages"][name]["status"] == "succeeded"
           for name in ("a", "peer", "after_peer")))
    run_module._update_run_metadata(peer, algo={"driver_marker": "kept"})
    metadata_disk = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ck("driver 级元数据更新会刷新旧快照，不覆盖其他实例的新 stage",
       metadata_disk["algo"]["driver_marker"] == "kept"
       and metadata_disk["stages"]["after_peer"]["status"] == "succeeded")

    def bad_stage(_run):
        raise RuntimeError("boom")

    try:
        _run_stage(run, "a", bad_stage, "a.json")
    except RuntimeError:
        pass
    a = run.manifest["stages"]["a"]
    ck("重跑开始清除旧完成态，失败显式记录 failed", a["done"] is False and a["status"] == "failed")
    ck("stage 重跑递增 attempt", a["attempt"] == 2)
    ck("失败同步写回磁盘 manifest", json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["stages"]["a"]["status"] == "failed")

    blocked_on_failed_dependency = False
    try:
        run_module.drive(run, [
            Stage("a", [], ok_stage, "a.json"),
            Stage("b", ["a"], lambda r: r.write("b.json", {"ok": True}), "b.json"),
        ], only="b")
    except SystemExit:
        blocked_on_failed_dependency = True
    ck("失败 stage 即使留有旧产物也不得启动下游", blocked_on_failed_dependency)

    stages = [
        Stage("a", [], ok_stage, "a.json"),
        Stage("b", ["a"], lambda r: r.write("b.json", {"ok": True}), "b.json"),
        Stage("c", ["b"], lambda r: r.write("c.json", {"ok": True}), "c.json"),
    ]
    drive = run_module.drive
    drive(run, stages)
    stale_before_force = Run("game", "game__test")
    drive(run, stages, only="a", force=True)
    ck("force 上游自动失效全部后续 stage",
       all(run.manifest["stages"][name]["status"] == "invalidated" and
           run.manifest["stages"][name]["done"] is False for name in ("b", "c")))
    ck("存在失效下游时 Run 不误报 done", run.manifest["status"] == "invalidated")
    ck("force 重跑自身成功并递增 attempt",
       run.manifest["stages"]["a"]["status"] == "succeeded" and
       run.manifest["stages"]["a"]["attempt"] == 4)

    branched = [
        Stage("world", [], ok_stage, "world.json"),
        Stage("questions", ["world"], ok_stage, "questions.json"),
        Stage("corpus", ["world"], ok_stage, "corpus.json"),
        Stage("grounding", ["questions", "corpus"], ok_stage, "grounding.json"),
    ]
    ck("force 失效传播按 DAG：重出题不误伤独立 corpus 分支",
       _dependent_stage_names("questions", branched) == ["questions", "grounding"])

    invalidated_dependency_rejected = False
    try:
        drive(run, stages, only="c")
    except SystemExit:
        invalidated_dependency_rejected = True
    ck("CLI 单跑不能绕过显式 invalidated 依赖", invalidated_dependency_rejected)

    drive(stale_before_force, stages, from_stage="b")
    recovered = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ck("旧 Run 实例会刷新失效状态并补跑下游，不会用旧快照误报 done",
       recovered["status"] == "done"
       and all(recovered["stages"][name]["status"] == "succeeded"
               for name in ("a", "b", "c"))
       and recovered["stages"]["b"]["attempt"] == 2
       and recovered["stages"]["c"]["attempt"] == 2)

    try:
        _run_stage(run, "c", bad_stage, "c.json")
    except RuntimeError:
        pass
    drive(run, stages, only="a")
    failed_run = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    ck("无关单步完成后不能把仍有 failed stage 的 Run 误报为 done",
       failed_run["status"] == "failed"
       and failed_run["stages"]["c"]["status"] == "failed")

    # corpus 中断点与最终产物分离：失败不覆盖旧成品，续跑成功后再原子发布。
    import pipeline.factory as factory

    corpus_run = Run("office", "office__corpus-test", config_meta={"target_tokens": 10})
    corpus_run.write(factory.ART["whitepaper"], {"domain_profile": {}})
    corpus_run.write(factory.ART["world"], WorldState(n_sessions=1).to_dict())
    old_final = {"corpus": {"sessions": [{"session_id": 99, "date": "old", "docs": []}]},
                 "done_weeks": [99]}
    corpus_run.write(factory.ART["corpus"], old_final)

    def interrupted_render(_wp, _ws, _target, _tracer, corpus, done, save, _log=None, **_kwargs):
        corpus["sessions"].append({"session_id": 0, "date": "new", "docs": []})
        done.add(0)
        save()
        raise RuntimeError("interrupted")

    original_render_corpus = factory.render_corpus
    factory.render_corpus = interrupted_render
    try:
        factory.stage_corpus(corpus_run)
    except RuntimeError:
        pass
    ck("corpus 中断只更新独立 checkpoint，不覆盖旧成品",
       corpus_run.has(factory.CORPUS_CKPT)
       and corpus_run.read(factory.ART["corpus"]) == old_final)

    def resumed_render(_wp, _ws, _target, _tracer, corpus, done, save, _log=None, **_kwargs):
        ck("续渲从中断点恢复", 0 in done and corpus["sessions"][0]["session_id"] == 0)
        save()

    factory.render_corpus = resumed_render
    factory.stage_corpus(corpus_run)
    ck("corpus 成功后发布成品并清理 checkpoint",
       corpus_run.read(factory.ART["corpus"])["done_weeks"] == [0]
       and not corpus_run.has(factory.CORPUS_CKPT))

    # checkpoint 必须绑定所有会改变渲染结果的输入；旧格式或任一输入变化都不得续用。
    wp_identity = {"style_spec": {"tone": "简洁"}, "domain_profile": {}}
    world_identity = WorldState(n_sessions=1).to_dict()
    identity = factory._corpus_checkpoint_identity(
        wp_identity, world_identity, 10, True, {"e1"}, {("e1", 0)})
    reordered_identity = factory._corpus_checkpoint_identity(
        {"domain_profile": {}, "style_spec": {"tone": "简洁"}},
        dict(reversed(list(world_identity.items()))), 10, True, {"e1"}, {("e1", 0)})
    ck("corpus checkpoint identity 是稳定的 64 位 SHA256",
       reordered_identity == identity and len(identity) == 64)
    variants = [
        factory._corpus_checkpoint_identity(
            {"style_spec": {"tone": "叙事"}, "domain_profile": {}},
            world_identity, 10, True, {"e1"}, {("e1", 0)}),
        factory._corpus_checkpoint_identity(
            wp_identity, WorldState(n_sessions=2).to_dict(),
            10, True, {"e1"}, {("e1", 0)}),
        factory._corpus_checkpoint_identity(
            wp_identity, world_identity, 11, True, {"e1"}, {("e1", 0)}),
        factory._corpus_checkpoint_identity(
            wp_identity, world_identity, 10, True, {"e2"}, {("e2", 0)}),
    ]
    ck("corpus checkpoint identity 覆盖 style/world/target/delta scope",
       all(value != identity for value in variants))

    corpus_run.write(factory.CORPUS_CKPT, {
        "identity": "stale", "corpus": {"sessions": [{"session_id": 7, "docs": []}]},
        "done_weeks": [7],
    })

    def stale_checkpoint_render(_wp, _ws, _target, _tracer, corpus, done, save,
                                _log=None, **_kwargs):
        ck("identity 不匹配的 checkpoint 从空语料重启",
           corpus == {"sessions": []} and done == set())
        save()

    factory.render_corpus = stale_checkpoint_render
    factory.stage_corpus(corpus_run)

    game_run = Run("game", "game__stale-delta-test",
                   config_meta={"target_tokens": 10, "render_only": ["old-entity"]})
    game_run.write(factory.ART["whitepaper"], {"domain_profile": {}})
    game_run.write(factory.ART["world"], WorldState(n_sessions=1, narrative={"version": 1}).to_dict())
    game_final = {"corpus": {"sessions": [{"session_id": 0,
                                         "date": WorldState(n_sessions=1).date_of_session(0), "docs": []}]},
                  "done_weeks": [0]}
    game_run.write(factory.ART["corpus"], game_final)

    def narrative_render(_wp, _ws, _target, _tracer, corpus, done, save, _log=None, **kwargs):
        ck("narrative delta 只在已有完整语料上追加，并保留定向范围",
           kwargs.get("only_entities") == {"old-entity"}
           and kwargs.get("only_entity_sessions") == set()
           and corpus == game_final["corpus"] and done == set(game_final["done_weeks"]))
        save()

    factory.render_corpus = narrative_render
    factory.stage_corpus(game_run)

    empty_narrative_delta = Run(
        "game", "game__empty-narrative-delta-test",
        config_meta={"target_tokens": 10, "render_only_pairs": [["old-entity", 0]]})
    empty_narrative_delta.write(factory.ART["whitepaper"], {"domain_profile": {}})
    empty_narrative_delta.write(
        factory.ART["world"], WorldState(n_sessions=1, narrative={"version": 1}).to_dict())
    try:
        factory.stage_corpus(empty_narrative_delta)
    except factory.WorldBlueprintError:
        empty_narrative_delta_rejected = True
    else:
        empty_narrative_delta_rejected = False
    ck("narrative delta 没有既有完整 corpus 时 fail-closed",
       empty_narrative_delta_rejected)

    partial_narrative_delta = Run(
        "game", "game__partial-narrative-delta-test",
        config_meta={"target_tokens": 10, "render_only_pairs": [["old-entity", 0]]})
    partial_narrative_delta.write(factory.ART["whitepaper"], {"domain_profile": {}})
    partial_narrative_delta.write(
        factory.ART["world"], WorldState(n_sessions=1, narrative={"version": 1}).to_dict())
    partial_narrative_delta.write(factory.ART["corpus"], old_final)
    try:
        factory.stage_corpus(partial_narrative_delta)
    except factory.WorldBlueprintError:
        partial_narrative_delta_rejected = True
    else:
        partial_narrative_delta_rejected = False
    ck("narrative delta 的既有 corpus 缺章或多章时 fail-closed",
       partial_narrative_delta_rejected)

    missing_story_run = Run("game", "game__missing-story-test",
                            config_meta={"target_tokens": 10})
    missing_story_run.write(factory.ART["whitepaper"], {"domain_profile": {}})
    missing_story_run.write(factory.ART["world"], WorldState(n_sessions=1).to_dict())
    missing_story_rendered = []
    factory.render_corpus = lambda *_args, **_kwargs: missing_story_rendered.append(True)
    try:
        factory.stage_corpus(missing_story_run)
    except factory.WorldBlueprintError:
        missing_story_rejected = True
    else:
        missing_story_rejected = False
    ck("game corpus 缺 Story Ledger 时 fail-closed，不退化为普通渲染",
       missing_story_rejected and not missing_story_rendered)

    # game 即使 manifest 遗留 augment，也不能把旧世界作为 Story Ledger 的 canon 输入。
    game_world_run = Run("game", "game__stale-augment-test", config_meta={"augment": True})
    game_world_run.write(factory.ART["whitepaper"], {})
    game_world_run.write(factory.ART["world"], WorldState(n_sessions=1).to_dict())
    game_world_run.write(factory.CORPUS_CKPT, {"identity": "old"})
    captured = {}
    original_build_world = factory.build_world
    original_prepare_lines = factory._prepare_lines

    def capture_build(_wp, _tracer, _log, existing=None, narrative=False):
        captured.update(existing=existing, narrative=narrative)
        return WorldState(n_sessions=1)

    factory.build_world = capture_build
    factory._prepare_lines = lambda *_args, **_kwargs: None
    factory.stage_world(game_world_run)
    ck("game stage_world 无条件忽略 stale augment",
       captured == {"existing": None, "narrative": True})
    ck("新 world 成功后清理旧 corpus checkpoint",
       not game_world_run.has(factory.CORPUS_CKPT))

    # 主角 pin 会改变标量 FK 容量：沿用现有容量计算收紧 min_count，再硬校验蓝图。
    game_wp = {
        "shared_world_spec": {"entities": {"count": 3}},
        "world_blueprint": {
            "version": 1,
            "entity_types": [
                {"id": "player", "noun": "玩家", "count": 2, "primary": True,
                 "fields": [{"name": "所属阵营", "kind": "reference"},
                            {"name": "等级", "kind": "numeric", "range": [1, 99]}]},
                {"id": "faction", "noun": "阵营", "count": 1, "primary": False,
                 "fields": [{"name": "声望", "kind": "numeric", "range": [0, 100]}]},
            ],
            "relation_types": [
                {"id": "belongs", "label": "归属", "from_type": "player",
                 "to_type": "faction", "field": "所属阵营", "temporal": False,
                 "min_count": 2},
            ],
            "event_types": [
                {"id": "level_up", "label": "升级", "roles": {"player": "player"},
                 "effect_fields": [{"role": "player", "field": "等级"}], "min_count": 1},
            ],
            "causal_rules": [],
            "temporal_model": {"unit": "章", "cadence": "每章", "n_sessions": 2,
                               "step_days": 1},
            "evidence_channels": ["剧情日志"],
        },
    }
    factory._pin_game_primary(game_wp)
    ck("pin 唯一主角后 relation min_count 不超过实际容量",
       game_wp["world_blueprint"]["entity_types"][0]["count"] == 1
       and game_wp["world_blueprint"]["relation_types"][0]["min_count"] == 1
       and game_wp["shared_world_spec"]["entities"]["count"] == 2)

    # 从中游恢复时，grounding 必须以当前 06 覆盖历史 UNMET 状态。
    import pipeline.grounding as grounding_module
    grounding_run = Run("game", "game__grounding-ledger-test")
    grounding_run.manifest.setdefault("algo", {}).update({
        "met_status": "UNMET_ORDER_SUPPLY: stale",
        "targetspec": {"min_questions": 3, "per_line_min": {"L1_timeline": 2}},
    })
    grounding_run._save_manifest()
    grounding_run.write(factory.ART["questions"], [{"qid": str(i)} for i in range(3)])
    grounding_run.write(factory.ART["corpus"], {"corpus": {"sessions": []}})
    original_run_grounding = grounding_module.run_grounding
    grounding_module.run_grounding = lambda *_args: (
        [{"qid": str(i)} for i in range(3)],
        {"overall": {"n": 3, "grounded": 3, "survival": 1.0},
         "by_line": {"L1_timeline": {"n": 3, "grounded": 3, "survival": 1.0}},
         "by_capability": {}, "n_dropped": 0, "drops": []},
    )
    factory.stage_grounding(grounding_run)
    ck("grounding 汇合点按当前产物把陈旧 UNMET 重算为 MET",
       grounding_run.manifest["algo"]["met_status"] == "MET"
       and grounding_run.manifest["algo"]["per_line_final"] == {"L1_timeline": 3})
    grounding_module.run_grounding = original_run_grounding

    factory.build_world = original_build_world
    factory._prepare_lines = original_prepare_lines
    factory.render_corpus = original_render_corpus


npass = sum(1 for ok, _ in checks if ok)
for ok, name in checks:
    if not ok:
        print(f"  ✗ {name}")
print(f"[run reliability self-test] {npass}/{len(checks)} PASS")
sys.exit(0 if npass == len(checks) else 1)
