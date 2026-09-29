"""
tools/eval_monitor — eval.multi_system 轻量级 web 看板。
纯只读:扫描 output/eval/eval_*/_progress.json(自动找最新 run),不碰 eval 进程。

启动:
  python tools/eval_monitor.py [--port 8071]              # 自动追踪最新 run
  python tools/eval_monitor.py --eval-id eval_20260616-161342  # 钉死看某个 run
"""
from __future__ import annotations
import json, argparse, time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "output" / "eval"

_pinned_id = None  # --eval-id 钉死的 run;None = 自动追踪最新


def _find_latest():
    if _pinned_id:
        p = EVAL_DIR / _pinned_id / "_progress.json"
        return p if p.exists() else None
    best, best_mt = None, 0
    if not EVAL_DIR.exists():
        return None
    for d in EVAL_DIR.iterdir():
        try:
            if not d.is_dir() or not d.name.startswith("eval_"):
                continue
            p = d / "_progress.json"
            if p.exists():
                mt = p.stat().st_mtime
                if mt > best_mt:
                    best, best_mt = p, mt
        except OSError:
            continue
    return best


def _list_runs():
    runs = []
    if not EVAL_DIR.exists():
        return runs
    for d in EVAL_DIR.iterdir():
        try:
            if not d.is_dir() or not d.name.startswith("eval_"):
                continue
            p = d / "_progress.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                runs.append({
                    "eval_id": d.name,
                    "status": data.get("status", "?"),
                    "model": data.get("model", "?"),
                    "ts": data.get("ts", 0),
                    "elapsed_s": data.get("elapsed_s"),
                })
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda r: r["ts"], reverse=True)
    return runs


def _read():
    p = _find_latest()
    if not p:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ── HTML(内嵌 CSS + JS,零外部依赖)────────────────────────────────────────
_CSS = r"""
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1a1a2e;color:#e0e0e0;font-family:'SF Mono','Menlo',monospace;font-size:13px;padding:16px;max-width:860px;margin:0 auto}
.card{background:#16213e;border-radius:8px;padding:12px 16px;margin-bottom:10px}
.hdr{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.tag{padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold}
.running{background:#1a5276;color:#5dade2}
.done{background:#1e8449;color:#82e0aa}
.pending{background:#333;color:#888}
.answering{background:#1a5276;color:#5dade2}
.judging{background:#7d3c98;color:#d2b4de}
.sr{display:flex;align-items:center;gap:10px;margin:5px 0}
.sn{width:170px;font-weight:bold;font-size:12px;white-space:nowrap;overflow:hidden}
.bar{flex:1;height:18px;background:#0a0a1a;border-radius:4px;position:relative;overflow:hidden}
.fill{height:100%;position:absolute;left:0;top:0;transition:width .4s;border-radius:4px}
.fans{background:#1a5276;z-index:1}
.fjdg{background:#2ecc71;z-index:2}
.ss{width:260px;font-size:11px;color:#aaa;text-align:right}
table{border-collapse:collapse;width:100%;font-size:12px}
th{color:#888;font-size:11px;padding:4px 6px;text-align:center;border-bottom:1px solid #0f3460}
td{padding:4px 6px;text-align:center;border-bottom:1px solid #0a0a1a}
.hi{color:#2ecc71}.mid{color:#f39c12}.lo{color:#e74c3c}
.feed{max-height:280px;overflow-y:auto;font-size:11px;margin-top:6px}
.fi{padding:1px 0}
.ok{color:#2ecc71}.fail{color:#e74c3c}.skip{color:#555}
h3{font-size:13px;color:#888;margin-bottom:6px}
.sub{font-size:11px;color:#555;margin-top:2px;word-break:break-all}
.stale{color:#e74c3c;font-size:11px;margin-top:4px}
"""

_JS = r"""
const LB={A:"SimpleMem",B:"FullContext",C:"Iterative",MEM0:"mem0",ZEP:"zep",MEMOS:"memOS"};
function fmt(s){if(!s||s<0)return"—";let m=Math.floor(s/60),sc=Math.floor(s%60);return m?m+"m"+String(sc).padStart(2,"0")+"s":sc+"s"}
function accH(a){if(a==null)return'<span class="skip">—</span>';let p=(a*100).toFixed(0);return`<span class="${a>=.75?"hi":a>=.4?"mid":"lo"}">${p}%</span>`}
function esc(s){let d=document.createElement("div");d.textContent=s||"";return d.innerHTML}

function render(d){
 let el=document.getElementById("app");
 if(!d||!d.status){el.innerHTML='<div class="card" style="color:#888">等待 eval 启动 ...</div>';return}
 let now=Date.now()/1e3;
 let elapsed=d.status==="done"?(d.elapsed_s||0):(now-(d.started_ts||now));
 let stale=d.status==="running"&&d.ts&&(now-d.ts>30);
 let h="";

 // header
 h+=`<div class="card hdr"><span class="tag ${d.status}">${d.status}</span>`
   +`<span>model: <b>${esc(d.model||"?")}</b></span>`
   +`<span>协议: ${d.protocol?"是":"否"}</span>`
   +`<span>选取: ${d.n_selected??d.n_judgeable??0}/${d.n_total||0}</span>`
   +`<span>输入预筛选: ${d.n_eligible??d.n_judgeable??0}</span>`
   +`<span>⏱ ${fmt(elapsed)}</span></div>`;
 if(d.eval_id)h+=`<div class="sub" style="color:#5dade2;margin-bottom:2px">${esc(d.eval_id)}</div>`;
 h+=`<div class="sub">${esc(d.bench||"")}</div>`;
 if(stale)h+='<div class="stale">⚠ 超过30秒无更新(eval可能已崩溃/结束)</div>';

 // embed
 let em=d.embed||{};
 if(em.status){
  h+='<div class="card"><b>Embed</b> ';
  if(em.status==="done")h+=`✓ ${em.n_chunks} chunks (${fmt(em.elapsed_s)})`;
  else{let et=em.ts0?(now-em.ts0):0;let ed=em.done||0,etl=em.total||0;
   let pct=etl?Math.round(ed/etl*100):0;
   h+=`⟳ ${ed}/${etl} docs (${pct}%) ${fmt(et)}`;
   if(etl)h+=`<div class="bar" style="margin-top:4px"><div class="fill fans" style="width:${pct}%"></div></div>`}
  h+="</div>";
 }

 // systems
 h+='<div class="card">';
 for(let s of(d.systems||[])){
  let sd=d.sys[s]||{};
  let pa=sd.total?(sd.done/sd.total*100):0;
  let pj=sd.total?(sd.judged/sd.total*100):0;
  let et=sd.status==="done"?sd.elapsed_s:(sd.ts0?(now-sd.ts0):0);
  let liveAcc=sd.j_real>0?accH(sd.correct/sd.j_real):"—";
  let stat="";
  if(sd.status==="pending")stat='<span class="skip">pending</span>';
  else if(sd.status==="done")stat=`✓ acc=${accH(sd.acc)} 已计分 ${sd.scored??sd.j_real??0}/${sd.total||0} <span style="color:#666">${fmt(et)}</span>`;
  else if(sd.status==="judging")stat=`✅${sd.done} → jdg ${sd.judged}/${sd.total} ~${liveAcc} <span style="color:#666">${fmt(et)}</span>`;
  else stat=`ans ${sd.done}/${sd.total} <span style="color:#666">${fmt(et)}</span>`;
  if(sd.incomplete)stat+=` <span class="skip">待复判 ${sd.incomplete}</span>`;
  h+=`<div class="sr"><span class="sn"><span class="tag ${sd.status||"pending"}" style="margin-right:4px">${sd.status||"pending"}</span>${s}=${LB[s]||s}</span>`
    +`<div class="bar"><div class="fill fans" style="width:${pa}%"></div><div class="fill fjdg" style="width:${pj}%"></div></div>`
    +`<span class="ss">${stat}</span></div>`;
 }
 h+="</div>";

 // accuracy table
 let dones=(d.systems||[]).filter(s=>(d.sys[s]||{}).by_line);
 if(dones.length){
  let lns=new Set();dones.forEach(s=>Object.keys(d.sys[s].by_line||{}).forEach(l=>lns.add(l)));
  let lines=[...lns].sort();
  h+='<div class="card"><h3>逐线 accuracy</h3><table><tr><th>系统</th>';
  lines.forEach(l=>h+=`<th>${l.split("_")[0]}</th>`);
  h+="<th>overall</th></tr>";
  dones.forEach(s=>{
   let sd=d.sys[s];h+=`<tr><td><b>${s}</b></td>`;
   lines.forEach(l=>h+=`<td>${accH((sd.by_line||{})[l])}</td>`);
   h+=`<td>${accH(sd.acc)}</td></tr>`;
  });
  h+="</table></div>";
 }

 // discrimination
 if(d.disc){
  let ds=d.disc;
  h+='<div class="card"><h3>跨系统分数比较</h3>';
  if(ds.status==="incomplete"){
   h+='<div>判分未完成；暂不判断排名、区分度或题库难度。</div></div>';
  }else if(ds.status==="insufficient_systems"){
   h+='<div>至少需要两个不同系统；本次不评估跨系统区分。</div></div>';
  }else{
  if(ds.ranking)h+=`<div>排名: ${ds.ranking.map(s=>s+"="+(ds.overall[s]*100).toFixed(0)+"%").join(" > ")}</div>`;
  h+=`<div>总分离差: ${(ds.ov_spread*100).toFixed(0)}% | 余量: ${(ds.headroom*100).toFixed(0)}%</div>`;
  h+=`<div>${ds.discriminates?"本批分数差达到描述性阈值":"本批分数差未达到描述性阈值"}；不单独证明题库难度。</div></div>`;
  }
 }

 // feed
 let ff=(d.feed||[]).slice().reverse().slice(0,25);
 if(ff.length){
  h+='<div class="card"><h3>最近判分</h3><div class="feed">';
  ff.forEach(f=>{
   let m=f.ok===null?"∅":f.ok?"✓":"✗";
   let c=f.ok===null?"skip":f.ok?"ok":"fail";
   h+=`<div class="fi"><span class="${c}">${m}</span> [${f.s}] ${f.ln}/${esc(f.cap)} ${esc(f.verdict||"")} <span style="color:#666">pred=</span>${esc(f.pred)} <span style="color:#666">gold=</span>${esc(f.gold)}</div>`;
  });
  h+="</div></div>";
 }
 el.innerHTML=h;
}

function tick(){fetch("/api/progress").then(r=>r.json()).then(render).catch(()=>render(null)).finally(()=>setTimeout(tick,2000))}
tick();
"""

_HTML = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Eval Monitor</title>
<style>{_CSS}</style></head>
<body><div id="app"><div class="card" style="color:#888">等待 eval 启动 ...</div></div>
<script>{_JS}</script></body></html>"""


class _H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/progress":
            d = _read()
            body = json.dumps(d or {}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        elif self.path == "/api/runs":
            body = json.dumps(_list_runs(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        else:
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def main():
    global _pinned_id
    ap = argparse.ArgumentParser(description="eval.multi_system 进度看板(只读)")
    ap.add_argument("--port", type=int, default=8071)
    ap.add_argument("--eval-id", default=None,
                    help="钉死看某个 eval run(如 eval_20260616-161342);不传则自动追踪最新")
    a = ap.parse_args()

    _pinned_id = a.eval_id
    srv = HTTPServer(("127.0.0.1", a.port), _H)
    print(f"[eval_monitor] http://127.0.0.1:{a.port}")
    if _pinned_id:
        print(f"[eval_monitor] 钉死 run: {_pinned_id}")
    else:
        print(f"[eval_monitor] 自动追踪最新 eval_*/ run")
    print(f"[eval_monitor] 扫描 {EVAL_DIR}/eval_*/_progress.json")

    p = _find_latest()
    if p:
        print(f"[eval_monitor] 当前: {p.parent.name}")
    else:
        print(f"[eval_monitor] 暂无 eval run,等待中...")
    srv.serve_forever()


if __name__ == "__main__":
    main()
