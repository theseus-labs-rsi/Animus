"""
tools.monitor_web —— 图形化 Run 监控【网页面板】(Python stdlib http.server,零依赖)。

与 tools.monitor 共用同一个已测的 snapshot() 数据层(只读 manifest/prompts/corpus/run.log,
完全不碰 pipeline)。起一个本地服务 + 自动弹浏览器;页面每 interval 秒 fetch /api/snapshot 重绘。

用法:
  ./venv/bin/python tools/monitor_web.py                    # 跟随最新 run,自动开浏览器
  ./venv/bin/python tools/monitor_web.py --run office__YYYYMMDD-HHMMSS   # 钉住某个 run
  ./venv/bin/python tools/monitor_web.py --port 8899 --interval 0.5
  # mac 上想要"独立窗口"感:open -na "Google Chrome" --args --app=http://localhost:8765
"""
from __future__ import annotations
import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo 根上 path:支持 `python tools/monitor_web.py` 当文件直接跑

from tools.monitor import snapshot, resolve_run, preview, log_slice


def web_payload(run_dir) -> dict:
    """snapshot() + 把 algo 里嵌套的多样性/接地拍平成前端好用的简单字段。"""
    s = snapshot(run_dir)
    a = s.get("algo", {})
    d = a.get("diversity") or {}
    g = (a.get("grounding") or {}).get("overall") or {}
    s["metrics"] = {
        "entities": a.get("entities"), "sessions": a.get("sessions"),
        "active_lines": a.get("active_lines"), "orders": a.get("orders"),
        "orders_by_line": a.get("orders_by_line"), "questions": a.get("questions"),
        "ndg": (d.get("NDG_n-gram多样性(越高越好)") or {}).get("point"),
        "ido": (d.get("IDO_跨文档重叠(越低越好)") or {}).get("point"),
        "grounding": g.get("survival"),
    }
    return s


HTML = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Benchmark 工厂 · Run 监控</title>
<style>
:root{--bg:#1e1e2e;--card:#313244;--fg:#cdd6f4;--mute:#6c7086;--ok:#a6e3a1;--run:#f9e2af;--err:#f38ba8;--acc:#89b4fa;--track:#45475a}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,"PingFang SC","Helvetica Neue",sans-serif;padding:24px;max-width:980px;margin:0 auto}
.row{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
#run{font-size:22px;font-weight:700;font-family:Menlo,monospace}
.chip{background:var(--card);color:var(--mute);font-size:12px;padding:3px 9px;border-radius:999px}
#status{font-size:13px;font-weight:700;display:flex;align-items:center;gap:6px;margin-top:6px;flex-wrap:wrap}
#dot{width:9px;height:9px;border-radius:50%;display:inline-block}
#stats{margin-left:8px;display:flex;gap:8px;font-weight:600}
#stats b{background:var(--card);color:var(--mute);padding:2px 9px;border-radius:7px;font-weight:600}
#stats b.bad{background:var(--err);color:#1e1e2e}
#stats b.warn{background:var(--run);color:#1e1e2e}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
/* ── DAG 流程图 ── */
#graph{position:relative;height:520px;margin:16px 0 4px}
#gedges{position:absolute;left:0;top:0;pointer-events:none;z-index:1}
.gnode{position:absolute;transform:translateX(-50%);width:130px;height:50px;display:flex;flex-direction:column;justify-content:center;
  text-align:center;background:var(--card);border:2px solid var(--track);border-radius:10px;cursor:pointer;font-size:13px;font-weight:700;transition:.15s;z-index:2}
.gnode b{display:block;font-weight:700}
.gnode i{font-style:normal;font-size:11px;color:var(--mute);font-weight:600}
.gnode:hover{border-color:var(--acc)}
.gnode.done{border-color:var(--ok);color:var(--ok)}
.gnode.run{border-color:var(--run);background:var(--run);color:#1e1e2e;animation:pulse 1.2s infinite}
.gnode.run i{color:#1e1e2e}
.gnode.err{border-color:var(--err);color:var(--err)}
.gnode.sel{outline:3px solid var(--acc);outline-offset:1px}
/* ── orders 一分为七:产线条 ── */
#lines{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:2px 2px 10px}
.lhdr{color:var(--mute);font-size:12px;margin-right:4px}
.lchip{background:var(--card);color:var(--mute);font-size:12px;font-weight:600;padding:5px 9px;border-radius:7px;cursor:pointer;border:1px solid transparent}
.lchip i{font-style:normal;color:var(--mute);margin-left:5px;font-size:11px}
.lchip:hover{outline:2px solid var(--acc)}
.lchip.fired{color:var(--ok);border-color:var(--ok)}.lchip.fired i{color:var(--ok)}
.lchip.active{color:#1e1e2e;background:var(--run)}.lchip.active i{color:#1e1e2e}
.lchip.planned{opacity:.45;border:1px dashed var(--mute)}
.lchip.sel{outline:2px solid var(--acc)}
#hint{color:var(--mute);font-size:11px;margin:0 2px 12px}
#cards{display:flex;gap:12px;margin:6px 0 4px}
.card{background:var(--card);border-radius:12px;padding:14px 18px;flex:1}
.card .t{font-size:12px;color:var(--mute)}
.card .v{font-size:34px;font-weight:700;font-family:Menlo,monospace;margin-top:2px}
#now{color:var(--acc);font-size:15px;font-weight:600;min-height:22px;margin:10px 2px}
#track{background:var(--track);height:18px;border-radius:9px;overflow:hidden;position:relative}
#fill{background:linear-gradient(90deg,#89b4fa,#a6e3a1);height:100%;width:0;transition:width .4s}
#barlabel{position:absolute;right:10px;top:0;line-height:18px;font-size:11px;font-family:Menlo,monospace}
#algo{color:var(--fg);font-size:13px;margin:14px 2px;display:flex;gap:8px;flex-wrap:wrap}
#algo span{background:var(--card);padding:4px 10px;border-radius:7px}
#engine{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin:12px 0 2px;padding:10px 12px;background:#181825;border:1px solid var(--track);border-radius:10px}
#engine .ehdr{color:var(--mute);font-size:12px;font-weight:700;margin-right:2px}
#engine .ek{background:var(--card);font-size:12px;padding:4px 10px;border-radius:7px;font-weight:600}
#engine .ek b{color:var(--acc);font-weight:700;margin-right:5px}
#llmbreak{display:flex;gap:6px;flex-wrap:wrap;margin:8px 2px}
#llmbreak .lbhdr{color:var(--mute);font-size:11px;margin-right:2px}
.lb{background:var(--card);color:var(--mute);font-size:11px;padding:3px 8px;border-radius:6px}
.lb.lberr{border:1px solid var(--err);color:var(--err);font-weight:700}
#preview{background:#181825;border:1px solid var(--track);border-radius:12px;padding:14px 16px;margin:10px 0;display:none}
#preview h3{font-size:14px;margin-bottom:8px;color:var(--acc)}
#pvkv{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:8px}
#pvkv span{background:var(--card);padding:3px 9px;border-radius:7px;font-size:12px}
#pvlines{font-family:Menlo,monospace;font-size:12px;color:var(--fg);white-space:pre-wrap;line-height:1.55}
#pvlog{font-family:Menlo,monospace;font-size:11px;color:var(--mute);white-space:pre-wrap;line-height:1.5;
  margin-top:10px;border-top:1px solid var(--track);padding-top:8px;max-height:170px;overflow-y:auto}
#log{background:#11111b;color:var(--mute);font-family:Menlo,monospace;font-size:12px;border-radius:12px;
     padding:12px 14px;margin-top:12px;height:200px;overflow-y:auto;white-space:pre-wrap;line-height:1.5}
#foot{color:var(--mute);font-size:11px;margin-top:10px;text-align:right}
</style></head><body>
<div class="row"><span id="run">等待 run…</span><span class="chip" id="tag" style="display:none"></span></div>
<div id="status"><span id="dot"></span><span id="statustext">先开窗口、再启动 pipeline 也行——自动 latch 最新 run</span><span id="stats"></span></div>
<div id="engine"></div>
<div id="graph"><svg id="gedges"></svg><div id="gnodes"></div></div>
<div id="lines"></div>
<div id="hint">▸ 点任意【节点】或【产线】看详情 + 该节点日志切片。运行中节点高亮;orders 一分为七 = 各产线。</div>
<div id="cards">
  <div class="card"><div class="t">LLM 调用</div><div class="v" id="llm">—</div></div>
  <div class="card"><div class="t">用时</div><div class="v" id="elapsed">—</div></div>
  <div class="card"><div class="t">语料</div><div class="v" id="corpus">—</div></div>
</div>
<div id="now"></div>
<div id="llmbreak"></div>
<div id="track"><div id="fill"></div><span id="barlabel"></span></div>
<div id="algo"></div>
<div id="preview"><h3 id="pvtitle"></h3><div id="pvkv"></div><div id="pvlines"></div><div id="pvlog"></div></div>
<pre id="log"></pre>
<div id="foot"></div>
<script>
const INTERVAL=__INTERVAL__;
const C={done:'#a6e3a1',running:'#f9e2af',failed:'#f38ba8','?':'#6c7086'};
const $=id=>document.getElementById(id);
let prev={llm:0,t:0}, sel=null;
// DAG:节点坐标(x=%,y=px,顶部)+ 边(按 STAGES.needs)+ 标签
const NODES={input:[50,6],whitepaper:[50,80],world:[50,154],orders:[26,234],corpus:[74,234],well_posed:[26,308],questions:[26,382],grounding:[50,456]};
const EDGES=[['input','whitepaper'],['whitepaper','world'],['world','orders'],['world','corpus'],['orders','well_posed'],['well_posed','questions'],['questions','grounding'],['corpus','grounding']];
const NLABEL={input:'input',whitepaper:'whitepaper · 议会',world:'world · 共享世界',orders:'orders · 一分为七',well_posed:'well_posed · 边A闸',corpus:'corpus · 渲染',questions:'questions · 题面',grounding:'grounding · 接地闸'};
const NTIP={world:'命门1:唯一真相源(各线 prepare 往里加基质)',orders:'一分为七:各线 enumerate + 代码 gt',well_posed:'边A闸:出题前剔 ill-posed(纯代码零LLM)',corpus:'只认世界,不认 order',grounding:'B闸:逐题过各线 ground()'};
const NH=50;
function dur(s){s=Math.floor(s||0);return s>=3600?`${Math.floor(s/3600)}h${String(Math.floor(s%3600/60)).padStart(2,'0')}m`:`${Math.floor(s/60)}m${String(s%60).padStart(2,'0')}s`}
function esc(x){return String(x).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function drawGraph(s){
  const st={}; (s.stages||[]).forEach(([n,state,el])=>st[n]=[state,el]);
  const g=$('graph'),W=g.clientWidth||720,H=g.clientHeight||520;
  $('gnodes').innerHTML=Object.entries(NODES).map(([id,[x,y]])=>{
    const [state,el]=st[id]||['pending',null];
    const cls=state==='done'?'gnode done':state==='running'?'gnode run':state==='failed'?'gnode err':'gnode';
    const rr=s.rerun&&id===s.current;   // ★闭环倒带重跑该 stage
    const sub=state==='done'?(el?Math.round(el)+'s':'✓'):state==='running'?((el?Math.round(el)+'s ':'')+(rr?'⟳重跑':'⟳')):'待';
    return `<div class="${cls}${sel===id?' sel':''}" style="left:${x}%;top:${y}px" title="${NTIP[id]||''}" onclick="showDetail('${id}')"><b>${NLABEL[id]}</b><i>${sub}</i></div>`;
  }).join('');
  const svg=$('gedges'); svg.setAttribute('width',W); svg.setAttribute('height',H);   // ★显式尺寸,否则 SVG 画布为 0、箭头不显
  svg.innerHTML='<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="#7f849c"/></marker></defs>'+
    EDGES.map(([a,b])=>{const [xa,ya]=NODES[a],[xb,yb]=NODES[b],x1=xa/100*W,y1=ya+NH,x2=xb/100*W,y2=yb-7;
      return `<path d="M${x1} ${y1} C${x1} ${y1+30} ${x2} ${y2-30} ${x2} ${y2}" fill="none" stroke="#7f849c" stroke-width="2" marker-end="url(#ah)"/>`;}).join('');
}
function drawLines(s){
  const ls=s.lines||[];
  $('lines').innerHTML='<span class="lhdr">orders 一分为七 →</span>'+ls.map(l=>{
    const cls=!l.built?'lchip planned':(l.orders>0?'lchip fired':(l.active?'lchip active':'lchip'));
    const cnt=l.built?`<i>${l.orders||0}单${l.survival!=null?' · '+Math.round(l.survival*100)+'%':''}</i>`:'<i>规划</i>';
    return `<div class="${cls}${sel==='line:'+l.id?' sel':''}" onclick="showDetail('line:${l.id}')">${l.id.split('_')[0]} ${l.title}${cnt}</div>`;
  }).join('');
}
async function showDetail(key){
  sel=key;
  try{const r=await fetch('/api/preview?stage='+encodeURIComponent(key));const p=await r.json();
    $('preview').style.display='block';
    $('pvtitle').textContent=p.title||key;
    $('pvkv').innerHTML=Object.entries(p.kv||{}).map(([k,v])=>`<span><b>${esc(k)}</b> ${esc(typeof v==='object'?JSON.stringify(v):v)}</span>`).join('');
    $('pvlines').textContent=(p.lines||[]).join('\n');
    $('pvlog').textContent='— 该节点日志 —\n'+((p.log||[]).join('\n')||'(无)');
  }catch(e){}
}
function render(s){
  if(s.waiting){$('run').textContent='等待 run…';$('statustext').textContent='output/runs/ 为空';$('dot').style.background='#6c7086';$('stats').innerHTML='';return;}
  $('run').textContent=s.run_id;
  const tag=$('tag'); if(s.tag){tag.style.display='';tag.textContent='tag: '+s.tag}else tag.style.display='none';
  const col=C[s.status]||C['?'];
  $('dot').style.background=col; $('statustext').style.color=col;
  $('statustext').textContent=`scenario: ${s.scenario}   ●  ${(s.status||'?').toUpperCase()}`;
  let rate=null; if(prev.t&&s.llm>=prev.llm){const dt=(Date.now()-prev.t)/60000; if(dt>0)rate=Math.round((s.llm-prev.llm)/dt);} prev={llm:s.llm,t:Date.now()};
  const stat=[];
  if(s.errors>0)stat.push(`<b class="bad">⚠ ${s.errors} 失败</b>`);
  if(rate!=null&&s.status==='running')stat.push(`<b>~${rate}/min</b>`);
  if(s.stall_s!=null&&s.stall_s>45)stat.push(`<b class="warn">⏸ ${s.stall_s}s 无 LLM 活动</b>`);
  if(s.max_latency_ms>30000)stat.push(`<b class="warn">最慢 ${(s.max_latency_ms/1000).toFixed(0)}s</b>`);
  $('stats').innerHTML=stat.join('');
  // ⚙ 运行配置(工程信息集中:模型/并发/端点/目标/区间/配额)——这一轮【真实跑的】配置,非读 config 当前 env
  const ev=s.env||{}, cf=s.cfg||{}, eng=[['模型',ev.model||'?'],['并发',ev.llm_concurrency??'?']];
  if(ev.base_url)eng.push(['端点',String(ev.base_url).replace(/^https?:\/\//,'').split('/')[0]]);
  if(cf.target_tokens)eng.push(['目标',(cf.target_tokens/1e6).toFixed(cf.target_tokens>=1e6?1:2)+'M tok']);
  const seg=(cf.from||cf.to)?`${cf.from||'起'}→${cf.to||'终'}`:(cf.only?'only '+cf.only:'');
  if(seg)eng.push(['区间',seg]);
  if(cf.quotas&&Object.keys(cf.quotas).length)eng.push(['配额',Object.entries(cf.quotas).map(([k,v])=>k.split('_')[0]+':'+v).join(' ')]);
  $('engine').innerHTML='<span class="ehdr">⚙ 运行配置</span>'+eng.map(([k,v])=>`<span class="ek"><b>${esc(k)}</b> ${esc(v)}</span>`).join('');
  drawGraph(s); drawLines(s);
  // LLM 花在哪 + 失败卡在哪个 step
  const HS={'world.batch':'世界','world.repair':'修复','council.observe':'议会观测','council.skeptic':'议会怀疑','council.map':'议会映射','council.medium':'议会媒介','council.style':'议会文风','council.traps':'议会陷阱','council.critique':'议会批判','render.signal':'信号','render.filler':'草堆','render.conflict':'矛盾','phrase':'出题'};
  const ebs=s.errors_by_step||{};
  $('llmbreak').innerHTML='<span class="lbhdr">LLM 花在哪 →</span>'+Object.entries(s.steps||{}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>{const e=ebs[k]||0;
    return `<span class="lb${e?' lberr':''}">${HS[k]||k} ${v}${e?' ⚠'+e:''}</span>`;}).join('');
  $('llm').textContent=s.llm; $('elapsed').textContent=dur(s.elapsed_s);
  $('corpus').textContent=s.target?(s.chars/1e6).toFixed(2)+'M':(s.n_docs+'篇');
  let nt=(s.step_now&&s.status==='running')?('⟳ 正在:'+s.step_now):(s.status==='done'?'✓ 完成':'');
  if(s.rerun&&s.loop){const lp=s.loop; nt+=`　·　闭环${lp.round||''}倒回重跑 ${s.current}`+(lp.deficit?`(赤字 ${lp.deficit})`:'');}
  $('now').textContent=nt;
  const pct=s.target?Math.min(1,s.chars/s.target):0;
  $('fill').style.width=(pct*100)+'%';
  $('barlabel').textContent=s.target?`${(s.chars/1e6).toFixed(2)}M / ${(s.target/1e6).toFixed(1)}M (${Math.round(pct*100)}%)`:'';
  const m=s.metrics||{},p=[];
  if(m.entities!=null)p.push(`实体 ${m.entities} · 周 ${m.sessions??'?'}`);
  if(m.questions!=null)p.push(`题 ${m.questions}`);
  if(m.ndg!=null)p.push(`多样性 NDG=${m.ndg} IDO=${m.ido}`);
  if(m.grounding!=null)p.push(`接地 ${Math.round(m.grounding*100)}%`);
  $('algo').innerHTML=p.map(x=>`<span>${x}</span>`).join('');
  if(sel)showDetail(sel);
  $('log').textContent=(s.log_tail||[]).join('\n'); $('log').scrollTop=$('log').scrollHeight;
}
async function tick(){
  try{const r=await fetch('/api/snapshot');render(await r.json());
      $('foot').textContent='刷新于 '+new Date().toLocaleTimeString();}
  catch(e){$('foot').textContent='⚠ 监控服务已断开';}
  setTimeout(tick,INTERVAL);
}
tick();
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    pin = None

    def log_message(self, *a):           # 静音 access log
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path.startswith("/api/snapshot"):
            run_dir = resolve_run(self.pin)
            self._json({"waiting": True} if run_dir is None else web_payload(run_dir))
        elif self.path.startswith("/api/preview"):
            from urllib.parse import urlparse, parse_qs
            key = (parse_qs(urlparse(self.path).query).get("stage") or [""])[0]
            run_dir = resolve_run(self.pin)
            if run_dir is None:
                self._json({"title": "—", "lines": ["(无 run)"], "log": []})
            else:
                p = preview(run_dir, key); p["log"] = log_slice(run_dir, key)   # 详情 + 该节点日志切片
                self._json(p)
        else:
            self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")


def serve(pin, interval, port):
    _Handler.pin = pin
    html = HTML.replace("__INTERVAL__", str(int(interval * 1000))).replace(
        "__STAGES__", json.dumps(["input", "whitepaper", "world", "orders", "questions", "corpus", "grounding"]))
    globals()["HTML"] = html
    for p in range(port, port + 20):                 # 端口占用就顺延
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), _Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit("✗ 找不到可用端口")
    url = f"http://localhost:{p}"
    print(f"◆ Run 监控面板:{url}   (Ctrl-C 退出)")
    print(f"  想要独立窗口:open -na \"Google Chrome\" --args --app={url}")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出监控。")


def main():
    ap = argparse.ArgumentParser(description="图形化 Run 监控网页面板")
    ap.add_argument("--run", default=None, help="钉住某个 run_id(默认:跟随最新 run)")
    ap.add_argument("--interval", type=float, default=1.0, help="刷新间隔秒(默认 1.0)")
    ap.add_argument("--port", type=int, default=8765, help="本地端口(默认 8765,占用则顺延)")
    a = ap.parse_args()
    serve(a.run, a.interval, a.port)


if __name__ == "__main__":
    main()
