"""
tools.monitor —— 图形化 Run 监控窗口(Tkinter,原生窗口,零依赖)。

完全不碰 pipeline:只轮询 output/runs/<run_id>/ 已有的实时产物——
  manifest.json(stage 完成/算法指标/状态)· prompts.jsonl(每行=1 次 LLM 调用)·
  05_corpus.json(逐周 checkpoint=语料实时进度)· run.log(日志尾)。
所以能监控任何 run:前台、nohup 后台、甚至历史 run。可【先开窗口再启动 pipeline】,自动latch 上最新 run。

用法:
  ./venv/bin/python tools/monitor.py                       # 跟随最新 run
  ./venv/bin/python tools/monitor.py --run office__YYYYMMDD-HHMMSS   # 钉住某个 run
  ./venv/bin/python tools/monitor.py --interval 0.5        # 刷新更快
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "output" / "runs"

# 镜像 factory.STAGES 的顺序(此处硬编码,避免 import 整条 pipeline + OpenAI 客户端)
STAGE_ORDER = ["input", "whitepaper", "world", "orders", "well_posed", "questions", "corpus", "grounding"]

# LLM step code → 人话(让"现在在干啥"可读)
STEP_HUMAN = {
    "world.batch": "生成世界实体", "world.repair": "修复世界缺陷",
    "council.observe": "议会·观测", "council.skeptic": "议会·怀疑",
    "council.map": "议会·映射", "council.medium": "议会·媒介", "council.style": "议会·文风",
    "council.traps": "议会·陷阱", "council.critique": "议会·批判", "render.signal": "渲染信号文档",
    "render.filler": "渲染草堆文档", "render.conflict": "渲染矛盾文档", "phrase": "出题润色",
}

# ── 纯数据层(可无头测试,与 Tk 视图解耦)──────────────────────────────────


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail(p: Path, n: int) -> list:
    try:
        return p.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except Exception:
        return []


def _scan_prompts(p: Path) -> dict:
    """单遍扫 prompts.jsonl(每行=1 次 LLM 调用):总数/失败数/按 step 直方图/末条 ts+step/最大延迟。
    兼容 A 之前的旧记录(无 ts/ok 时用 out_preview 的 __error__ 兜底)。"""
    n = errors = max_latency = 0
    steps: dict = {}
    ebs: dict = {}
    last_ts, last_step = None, ""
    try:
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            n += 1
            st = r.get("step", "")
            steps[st] = steps.get(st, 0) + 1
            bad = (r.get("ok") is False) or (isinstance(r.get("out_preview"), str)
                                             and r["out_preview"].startswith('{"__error__"'))
            if bad:
                errors += 1
                ebs[st] = ebs.get(st, 0) + 1                # ★失败按 step 分桶:卡在哪一步一眼看出
            if r.get("ts"):
                last_ts = r["ts"]
            last_step = st or last_step
            max_latency = max(max_latency, r.get("latency_ms") or 0)
    except Exception:
        pass
    return {"n": n, "errors": int(errors), "steps": steps, "errors_by_step": ebs,
            "last_ts": last_ts, "last_step": last_step, "max_latency_ms": max_latency}


def _last_stage_in_log(p: Path) -> str:
    for ln in reversed(_tail(p, 200)):
        i = ln.find("→ stage: ")
        if i >= 0:
            return ln[i + len("→ stage: "):].strip()
    return ""


def _closed_loop_state(run_dir: Path) -> dict:
    """从 run.log 尾提取闭环 build_to_target 的最近状态(第几轮 / 谁赤字 / 在长世界重建)——只为可观测。
    闭环会倒回重跑 world,这几行让"为什么又在跑 world"可见,不靠猜。非闭环 run 全空。"""
    rnd = deficit = grow = ""
    for ln in _tail(run_dir / "run.log", 250):
        if "═ 第" in ln and "轮" in ln:
            rnd = "第" + ln.split("第", 1)[1].split("轮", 1)[0].strip() + "轮"
        elif "赤字" in ln and "{" in ln:
            deficit = ln.split("赤字", 1)[1].strip()
        elif "长世界" in ln and "重建" in ln:
            grow = ln.split("→", 1)[-1].strip()
    return {"round": rnd, "deficit": deficit, "grow": grow}


def resolve_run(pin: str | None) -> Path | None:
    if pin:
        d = RUNS_DIR / pin
        return d if (d / "manifest.json").exists() else None
    if not RUNS_DIR.exists():
        return None
    cands = [d for d in RUNS_DIR.iterdir() if (d / "manifest.json").exists()]
    return max(cands, key=lambda d: (d / "manifest.json").stat().st_mtime) if cands else None


# 镜像 pipeline/lines 注册表(built=已建产线 / 否则规划中)。硬编码以保持监控零 import pipeline。
LINES_ALL = [
    ("L1_timeline", "时间线", True), ("L2_relational", "关系多跳", True),
    ("L3_process", "过程序列", True), ("L4_preference", "偏好隐式", True),
    ("L5_conflict", "冲突可信", True), ("L6_refusal", "拒答边界", True),
    ("L7_consolidation", "巩固摘要", True),
]


def log_slice(run_dir: Path, key: str, n: int = 50) -> list:
    """某节点的日志切片:stage → 从最后一次「→ stage: X」到下一个 stage banner;line:<id> → 含该线 id 的行。"""
    try:
        lines = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    if key.startswith("line:"):
        lid = key[5:]
        short = lid.split("_")[0]
        hits = [ln for ln in lines if lid in ln or f"{short}_" in ln or f"{short}:" in ln]
        return hits[-n:] or ["(该线暂无独立日志;详见 orders / grounding 节点)"]
    idxs = [i for i, ln in enumerate(lines) if f"→ stage: {key}" in ln]
    if not idxs:                                            # 无 banner(如闭环外的 input)→ 退化为含 key 的行
        return [ln for ln in lines if key in ln][-n:]
    out = [lines[idxs[-1]]]
    for ln in lines[idxs[-1] + 1:]:
        if "→ stage: " in ln:
            break
        out.append(ln)
    return out[-n:]


def snapshot(run_dir: Path) -> dict:
    """读全部实时产物 → 一份给视图渲染的快照(纯函数,无 Tk 依赖)。"""
    now = time.time()
    m = _read_json(run_dir / "manifest.json") or {}
    sd = m.get("stages", {})
    status = m.get("status", "?")
    # 当前 stage:status=running 时【信任】manifest.current_stage(E 探针);非运行态无 current。
    # ★闭环 build_to_target 会【重跑已 done 的 stage】(world→orders→well_posed→[赤字]→world…),
    #   此刻 current_stage 指向的 stage 其 done 仍是上一轮留下的 True——绝不能据此清空,那恰恰是【正在重跑】。
    #   (旧逻辑"current 指向 done 就清空"会把重跑中的 world 误判成已完成 → 没有节点高亮,像卡死。)
    current = (m.get("current_stage", "") or "") if status == "running" else ""
    if status == "running" and not current:                    # 仅 current_stage 缺失才兜底推断
        current = _last_stage_in_log(run_dir / "run.log") or \
            next((s for s in STAGE_ORDER if not sd.get(s, {}).get("done")), "")
    rerun = bool(current and sd.get(current, {}).get("done"))   # 当前 stage 已 done 过却又是 current = 闭环倒带重跑

    stages = []
    for s in STAGE_ORDER:
        info = sd.get(s, {})
        if s == current:                                       # ★当前(含【重跑已 done】的)优先标 running,实时计时
            stages.append((s, "running", (now - info["started_ts"]) if info.get("started_ts") else None))
        elif info.get("done"):
            stages.append((s, "done", info.get("elapsed_s")))
        else:
            stages.append((s, "pending", None))

    corpus = _read_json(run_dir / "05_corpus.json") or {}
    docs = corpus.get("corpus", {}).get("sessions", [])
    chars = sum(len(d.get("content", "")) for sess in docs for d in sess.get("docs", []))
    n_docs = sum(len(sess.get("docs", [])) for sess in docs)
    target = int(m.get("config", {}).get("target_tokens", 0) or 0)

    created = m.get("created", "")                              # 用时:running→实时墙钟;否则→各 stage 之和
    if status == "running" and created:
        try:
            elapsed = (datetime.now() - datetime.fromisoformat(created)).total_seconds()
        except Exception:
            elapsed = 0
    else:
        elapsed = sum(v.get("elapsed_s", 0) for v in sd.values())

    pr = _scan_prompts(run_dir / "prompts.jsonl")
    stall = int(now - pr["last_ts"]) if (status == "running" and pr["last_ts"]) else None  # A:距上次 LLM 调用

    # 各产线状态(orders"一分为七":built 4 条 + 规划 3 条;激活/产单/接地存活)
    algo = m.get("algo", {})
    active_set = {str(x) for x in (algo.get("active_lines") or [])}
    obl = algo.get("orders_by_line") or {}
    gbl = ((algo.get("grounding") or {}).get("by_line")) or {}

    def _line_active(lid):
        if lid in active_set or obl.get(lid):
            return True
        short = lid.split("_")[0]
        return any(str(x).startswith(short) for x in active_set)
    lines_v = []
    for lid, title, built in LINES_ALL:
        g = gbl.get(lid) or {}
        lines_v.append({"id": lid, "title": title, "built": built,
                        "active": bool(built and _line_active(lid)),
                        "orders": (obl.get(lid, 0) if built else None),
                        "grounded": g.get("grounded"), "survival": g.get("survival")})

    return {
        "run_id": m.get("run_id", run_dir.name), "scenario": m.get("scenario", "?"),
        "tag": m.get("tag"), "status": status, "stages": stages, "current": current,
        "lines": lines_v,
        "llm": pr["n"], "errors": pr["errors"], "steps": pr["steps"], "errors_by_step": pr["errors_by_step"],
        "max_latency_ms": pr["max_latency_ms"], "stall_s": stall,
        "step_now": STEP_HUMAN.get(pr["last_step"], pr["last_step"]),
        "elapsed_s": int(elapsed), "chars": chars, "n_docs": n_docs, "target": target,
        "algo": algo,
        "env": m.get("env", {}),                               # 工程配置快照(模型/并发/端点)
        "cfg": m.get("config", {}),                            # per-run 配置(target_tokens/quotas/from-to-only)
        "rerun": rerun,                                        # ★当前 stage 是否闭环倒带重跑(让"看似卡死"现形)
        "loop": _closed_loop_state(run_dir) if status == "running" else {},  # 闭环轮次/赤字/重建(可观测)
        "log_tail": _tail(run_dir / "run.log", 14),
    }


def preview(run_dir: Path, stage: str) -> dict:
    """点击某节点按需读产物 → 人读预览。stage=阶段名 或 line:<id>(单产线详情)。返回 {title, kv?, lines?}。"""
    def J(name):
        return _read_json(run_dir / name) or {}

    if stage.startswith("line:"):                          # 单条产线详情(orders 一分为七的一支)
        lid = stage[5:]
        title = next((t for i, t, _ in LINES_ALL if i == lid), lid)
        built = next((b for i, _, b in LINES_ALL if i == lid), False)
        if not built:
            return {"title": f"{lid} · {title}", "lines": ["(规划中,尚未建成 —— 议会能激活、但无代码 gt,不产题)"]}
        orders = [o for o in (_read_json(run_dir / "03_orders.json") or []) if o.get("line") == lid]
        qs = [q for q in (_read_json(run_dir / "04_questions.json") or []) if q.get("line") == lid]
        rep = J("06_grounding_report.json")
        gl = (rep.get("by_line") or {}).get(lid, {})
        drops = [f"✗ {str(d.get('reason', ''))[:60]}" for d in (rep.get("drops") or []) if d.get("line") == lid][:5]
        samp = [f"· {q.get('question', '')}" for q in qs[:5]]
        kv = {"订单": len(orders), "题": len(qs)}
        if gl:
            kv["接地"] = f"{gl.get('grounded', '?')}/{gl.get('n', '?')} ({(gl.get('survival') or 0) * 100:.0f}%)"
        return {"title": f"{lid} · {title}", "kv": kv,
                "lines": (["真题预览:"] + samp if samp else ["(尚无题面)"]) + (["弃因:"] + drops if drops else [])}

    if stage == "input":
        sc = J("00_input.json")
        return {"title": "场景输入", "lines": [sc.get("description", "")[:400],
                f"few-shot:{len(sc.get('few_shot', []))} 篇"]}
    if stage == "whitepaper":
        wp = J("01_whitepaper.json")
        dp = wp.get("domain_profile", {})
        sw = wp.get("shared_world_spec", {})
        al = [f"· {l.get('line')}  w={l.get('weight')}  — {str(l.get('why', ''))[:46]}" for l in wp.get("active_lines", [])]
        traps = [f"· {t.get('trap', '')} → 激 {t.get('boost_line', '')}" for t in (wp.get("traps") or [])][:5]
        return {"title": "白皮书 · 中央议会决策",
                "kv": {"主体": dp.get("entity_noun"),
                       "字段": "、".join(f.get("name") for f in dp.get("field_schema", [])),
                       "媒介": (wp.get("medium") or {}).get("type"), "停用语": dp.get("stopped_phrase"),
                       "议会定规模": f"{(sw.get('entities') or {}).get('count', '?')}实体×{(sw.get('timeline') or {}).get('n_sessions', '?')}周",
                       "目标题量": (wp.get("capability_targets") or {}).get("total_q")},
                "lines": ["激活产线(权重·理由):"] + al + (["陷阱布局:"] + traps if traps else [])
                         + ["(6 视角成功/失败 + 批判状态:点 whitepaper 节点看下方日志)"]}
    if stage == "world":
        ws = J("02_world.json")
        ents = list(ws.get("entities", {}))
        return {"title": "共享世界",
                "kv": {"实体": len(ents), "周": ws.get("n_sessions"), "级联": len(ws.get("cascades", [])),
                       "absent": len(ws.get("absent_fields", [])), "矛盾(L5)": len(ws.get("conflicts", []))},
                "lines": ["实体:" + "、".join(ents[:24]) + (" …" if len(ents) > 24 else "")]}
    if stage == "orders":
        orders = _read_json(run_dir / "03_orders.json") or []
        bycap: dict = {}
        for o in orders:
            bycap[o.get("capability", "?")] = bycap.get(o.get("capability", "?"), 0) + 1
        samp = [f"· [{o.get('line', '?')}/{o.get('capability', '?')}] {o.get('entity', '')}·{o.get('field', '')}"
                f" → gt={str(o.get('gt'))[:36]}" for o in orders[:5]]
        return {"title": f"订单 {len(orders)}", "kv": {"按能力": bycap}, "lines": ["样例:"] + samp}
    if stage == "well_posed":
        rep = J("03_well_posed_report.json")
        ov = rep.get("overall", {})
        drops = [f"✗ [{d.get('line', '?')}] {str(d.get('reason', ''))[:60]}" for d in (rep.get("drops") or [])[:7]]
        return {"title": "边A闸 · 良定义(出题前剔 ill-posed,纯代码零 LLM)",
                "kv": {"良定义": f"{ov.get('well_posed', '?')}/{ov.get('n', '?')} ({(ov.get('pass_rate') or 0) * 100:.0f}%)",
                       "弃": rep.get("n_dropped"), "按线": rep.get("by_line")},
                "lines": (["被剔题逐条原因:"] + drops) if drops else ["(全部良定义,无剔)"]}
    if stage == "questions":
        qs = _read_json(run_dir / "04_questions.json") or []
        samp = [f"· [{q.get('line', '?')}] {q.get('question', '')}" for q in qs[:7]]
        return {"title": f"题面 {len(qs)} · 真问题预览", "lines": samp or ["(空)"]}
    if stage == "corpus":
        cp = J("05_corpus.json").get("corpus", {})
        sess = cp.get("sessions", [])
        types: dict = {}
        sig = fil = conf = 0
        sampdoc = ""
        for s in sess:
            for d in s.get("docs", []):
                types[d.get("type", "?")] = types.get(d.get("type", "?"), 0) + 1
                if d.get("is_filler"):
                    fil += 1
                elif d.get("is_conflict"):
                    conf += 1
                else:
                    sig += 1
                    if not sampdoc and d.get("content"):
                        sampdoc = d["content"][:260]
        return {"title": f"语料 {sig + fil + conf} 篇",
                "kv": {"信号": sig, "草堆": fil, "矛盾": conf, "周": len(sess), "类型": types},
                "lines": ["样例信号文档:", (sampdoc + " …") if sampdoc else "(尚无)"]}
    if stage == "grounding":
        rep = J("06_grounding_report.json")
        ov = rep.get("overall", {})
        algo = J("manifest.json").get("algo", {})
        ts = algo.get("targetspec") or {}
        kv = {"存活": f"{ov.get('grounded', '?')}/{ov.get('n', '?')} ({(ov.get('survival') or 0) * 100:.0f}%)",
              "弃": rep.get("n_dropped"), "按线": rep.get("by_line"), "按能力": rep.get("by_capability")}
        if ts:                                              # 闭环旋钮判据(read-only:manifest.algo)
            kv["旋钮"] = f"min_q={ts.get('min_questions')}  per_line_min={ts.get('per_line_min')}"
        if algo.get("met_status"):
            kv["闭环达标"] = algo["met_status"]
        if algo.get("per_line_final"):
            kv["逐线最终接地"] = algo["per_line_final"]
        drops = [f"✗ [{d.get('line', '?')}/{d.get('capability', '?')}] {str(d.get('reason', ''))[:60]}"
                 for d in (rep.get("drops") or [])[:7]]
        return {"title": "接地报告(B闸)", "kv": kv,
                "lines": (["被弃题逐条原因:"] + drops) if drops else ["(全部接地,无弃题)"]}
    return {"title": stage, "lines": ["(无预览)"]}


def fmt_dur(s: int) -> str:
    s = int(s)
    return f"{s//3600}h{(s%3600)//60:02d}m" if s >= 3600 else f"{s//60}m{s%60:02d}s"


# ── Tk 视图层 ──────────────────────────────────────────────────────────────
BG, CARD, FG, MUTE = "#1e1e2e", "#313244", "#cdd6f4", "#6c7086"
OK, RUN, ERR, ACC = "#a6e3a1", "#f9e2af", "#f38ba8", "#89b4fa"
STATUS_COLOR = {"running": RUN, "done": OK, "failed": ERR}


def launch(pin: str | None, interval: float):
    import tkinter as tk
    from tkinter import font as tkfont

    root = tk.Tk()
    root.title("Benchmark 工厂 · Run 监控")
    root.configure(bg=BG)
    root.geometry("780x640")
    f_big = tkfont.Font(family="Menlo", size=30, weight="bold")
    f_h = tkfont.Font(family="Helvetica", size=15, weight="bold")
    f_n = tkfont.Font(family="Helvetica", size=12)
    f_mono = tkfont.Font(family="Menlo", size=11)
    f_pill = tkfont.Font(family="Helvetica", size=11, weight="bold")

    def card(parent):
        return tk.Frame(parent, bg=CARD, padx=14, pady=10)

    header = tk.Label(root, text="等待 run…", bg=BG, fg=FG, font=f_h, anchor="w")
    header.pack(fill="x", padx=14, pady=(12, 2))
    sub = tk.Label(root, text="先开窗口、再启动 pipeline 也行——自动 latch 最新 run", bg=BG, fg=MUTE, font=f_n, anchor="w")
    sub.pack(fill="x", padx=14)

    # stage 流水线(7 个 pill)
    pill_row = tk.Frame(root, bg=BG)
    pill_row.pack(fill="x", padx=10, pady=10)
    pills = {}
    for s in STAGE_ORDER:
        lb = tk.Label(pill_row, text=s, bg=CARD, fg=MUTE, font=f_pill, padx=9, pady=6)
        lb.pack(side="left", padx=3)
        pills[s] = lb

    # 大指标行
    metrics = tk.Frame(root, bg=BG)
    metrics.pack(fill="x", padx=10)

    def metric_box(title):
        c = card(metrics)
        c.pack(side="left", expand=True, fill="both", padx=4)
        tk.Label(c, text=title, bg=CARD, fg=MUTE, font=f_n).pack(anchor="w")
        v = tk.Label(c, text="—", bg=CARD, fg=FG, font=f_big)
        v.pack(anchor="w")
        return v

    m_llm = metric_box("LLM 调用")
    m_elapsed = metric_box("用时")
    m_corpus = metric_box("语料")

    now_lb = tk.Label(root, text="", bg=BG, fg=ACC, font=f_h, anchor="w")
    now_lb.pack(fill="x", padx=18, pady=(8, 2))

    bar_c = tk.Canvas(root, height=14, bg=CARD, highlightthickness=0)
    bar_c.pack(fill="x", padx=14, pady=2)

    algo_lb = tk.Label(root, text="", bg=BG, fg=FG, font=f_n, justify="left", anchor="w")
    algo_lb.pack(fill="x", padx=16, pady=6)

    log_txt = tk.Text(root, height=14, bg="#11111b", fg=MUTE, font=f_mono,
                      relief="flat", padx=10, pady=8, highlightthickness=0)
    log_txt.pack(fill="both", expand=True, padx=12, pady=(4, 12))
    log_txt.configure(state="disabled")

    state = {"pin": pin}

    def refresh():
        run_dir = resolve_run(state["pin"])
        if run_dir is None:
            header.config(text="等待 run…(output/runs/ 为空)")
            root.after(int(interval * 1000), refresh)
            return
        s = snapshot(run_dir)
        tag = f"  ·  tag: {s['tag']}" if s.get("tag") else ""
        header.config(text=f"{s['run_id']}{tag}")
        col = STATUS_COLOR.get(s["status"], MUTE)
        sub.config(text=f"scenario: {s['scenario']}   ●  {s['status'].upper()}", fg=col)

        for name, lb in pills.items():
            st = next((x for x in s["stages"] if x[0] == name), (name, "pending", None))
            _, stt, el = st
            if stt == "done":
                lb.config(bg=CARD, fg=OK, text=f"{name} ✓" + (f" {int(el)}s" if el else ""))
            elif stt == "running":
                lb.config(bg=RUN, fg="#1e1e2e", text=f"{name} ⟳")
            else:
                lb.config(bg=CARD, fg=MUTE, text=name)

        m_llm.config(text=str(s["llm"]))
        m_elapsed.config(text=fmt_dur(s["elapsed_s"]))
        if s["target"]:
            mc = s["chars"] / 1e6
            m_corpus.config(text=f"{mc:.2f}M")
        else:
            m_corpus.config(text=f"{s['n_docs']}篇")

        now = s["step_now"]
        nt = (f"⟳ 正在:{now}" if now and s["status"] == "running" else ("✓ 完成" if s["status"] == "done" else ""))
        if s.get("rerun") and s.get("loop"):                   # ★闭环倒带:别让重跑看着像卡死
            lp = s["loop"]
            nt += f"  ·  闭环{lp.get('round', '')}倒回重跑 {s['current']}" + (f"(赤字 {lp['deficit']})" if lp.get("deficit") else "")
        now_lb.config(text=nt)

        # 语料进度条(语料阶段才有真总量)
        bar_c.delete("all")
        w = bar_c.winfo_width() or 740
        pct = min(1.0, s["chars"] / s["target"]) if s["target"] else 0
        bar_c.create_rectangle(0, 0, w * pct, 14, fill=ACC, width=0)
        if s["target"]:
            bar_c.create_text(w - 6, 7, text=f"{s['chars']/1e6:.2f}M / {s['target']/1e6:.1f}M  ({pct:.0%})",
                              anchor="e", fill=FG, font=f_mono)

        a = s["algo"]
        parts = []
        if a.get("entities") is not None:
            parts.append(f"实体 {a['entities']} · 周 {a.get('sessions','?')}")
        if a.get("active_lines"):
            parts.append("线 " + "/".join(x.split("_")[0] for x in a["active_lines"]))
        if a.get("orders") is not None:
            parts.append(f"订单 {a['orders']} {a.get('orders_by_line','')}")
        if a.get("questions") is not None:
            parts.append(f"题 {a['questions']}")
        d = a.get("diversity") or {}
        if d:
            ndg = (d.get("NDG_n-gram多样性(越高越好)") or {}).get("point")
            ido = (d.get("IDO_跨文档重叠(越低越好)") or {}).get("point")
            if ndg is not None:
                parts.append(f"多样性 NDG={ndg} IDO={ido}")
        g = a.get("grounding") or {}
        if g.get("overall"):
            parts.append(f"接地 {(g['overall'].get('survival') or 0):.0%}")
        algo_lb.config(text="   ".join(parts))

        log_txt.configure(state="normal")
        log_txt.delete("1.0", "end")
        log_txt.insert("end", "\n".join(s["log_tail"]))
        log_txt.see("end")
        log_txt.configure(state="disabled")

        root.after(int(interval * 1000), refresh)

    refresh()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="图形化 Run 监控窗口")
    ap.add_argument("--run", default=None, help="钉住某个 run_id(默认:跟随最新 run)")
    ap.add_argument("--interval", type=float, default=1.0, help="刷新间隔秒(默认 1.0)")
    a = ap.parse_args()
    launch(a.run, a.interval)


if __name__ == "__main__":
    main()
