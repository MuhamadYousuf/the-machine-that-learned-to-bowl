"""Read-only live dashboard for neuroevolution training runs."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aibowler.dashboard_data import read_csv_tolerant, read_latest_jsonl_tolerant
from aibowler import brain_explorer
from aibowler.env import ARCHETYPE_NAMES, OverEnv
from aibowler.policy import MLPPolicy
from aibowler.replay import load_replay
try:
    from .brain_page import BRAIN_PAGE
except ImportError:
    from brain_page import BRAIN_PAGE
try:
    from .theater_page import THEATER_PAGE
except ImportError:
    from theater_page import THEATER_PAGE


RUN_OPTIONS = ("ga1500", "ga2", "ga200")


HTML_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Bowler · live training</title>
<style>
:root{color-scheme:dark;--bg:#080d18;--panel:#101827;--panel2:#131e30;--line:#26344a;--ink:#edf4ff;--muted:#9aacbf;--cyan:#42d9e8;--violet:#a78bfa;--pink:#fb7185;--green:#5ee0a0;--amber:#fbbf24}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 20% -20%,#172a45 0,transparent 38%),var(--bg);color:var(--ink);font:14px/1.45 Inter,Segoe UI,Arial,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:24px 26px 48px}.top{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:22px}.eyebrow{text-transform:uppercase;letter-spacing:.16em;color:var(--cyan);font-size:11px;font-weight:700}.title{font-size:30px;line-height:1.1;margin:5px 0}.sub{color:var(--muted)}.live{display:flex;align-items:center;gap:8px;border:1px solid #225145;border-radius:999px;padding:8px 12px;background:#0c201e;color:#a8f0cf;font-size:12px;white-space:nowrap}.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green)}
.cards{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:12px;margin-bottom:18px}.card,.panel{background:linear-gradient(145deg,rgba(20,31,48,.96),rgba(13,21,34,.96));border:1px solid var(--line);border-radius:15px;box-shadow:0 12px 32px #0003}.card{padding:14px 16px}.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em}.value{font-size:23px;font-weight:700;margin-top:4px}.small{font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.panel{padding:16px;min-width:0}.panel h2{font-size:15px;margin:0 0 4px}.panel p{font-size:12px;color:var(--muted);margin:0 0 10px}.chart{width:100%;height:245px;display:block}.section{margin:24px 0 11px;display:flex;align-items:baseline;justify-content:space-between}.section h2{margin:0;font-size:18px}.section span{color:var(--muted);font-size:12px}
.champions{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.miniMap{width:100%;height:185px;display:block;background:#0a111e;border:1px solid var(--line);border-radius:10px}.pattern{font-size:12px;color:#d7e6f5;line-height:1.55;margin-top:8px;min-height:36px}.stats{color:var(--muted);font-size:12px;margin-top:8px}
.controls{display:flex;flex-wrap:wrap;gap:9px;align-items:center;margin:10px 0 14px}.controls label{font-size:12px;color:var(--muted)}select,button{font:inherit;color:var(--ink);background:#0b1422;border:1px solid #34465f;border-radius:8px;padding:8px 10px}button{cursor:pointer;background:#123043;border-color:#1e6876}button:hover{background:#174253}.compareGrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.compareMap{height:240px}.foot{color:#74869b;font-size:11px;text-align:center;margin-top:24px}.empty{display:flex;min-height:175px;align-items:center;justify-content:center;color:var(--muted);text-align:center;border:1px dashed var(--line);border-radius:10px;padding:14px}
@media(max-width:950px){.cards{grid-template-columns:repeat(3,1fr)}.champions{grid-template-columns:1fr}.grid{grid-template-columns:1fr}}@media(max-width:600px){.wrap{padding:16px 12px 30px}.cards{grid-template-columns:repeat(2,1fr)}.top{align-items:center}.title{font-size:24px}.compareGrid{grid-template-columns:1fr}.card{padding:12px}.value{font-size:20px}}
</style></head><body><main class="wrap">
<header class="top"><div><div class="eyebrow">AI Bowler · Natural Selection Lab</div><h1 class="title" id="runTitle">Loading run…</h1><div class="sub">Balanced, normalized fitness · common random numbers · held-out champion validation</div></div><div style="display:flex;gap:10px;align-items:center"><label class="small">Run <select id="runSelector"></select></label><a href="/theater" style="color:var(--cyan);font-size:12px">Evolution Theater</a><a href="/brain" style="color:var(--cyan);font-size:12px">Brain Explorer</a><div class="live"><i class="dot"></i><span id="liveText">Connecting</span></div></div></header>
<section class="cards">
 <div class="card"><div class="label">Generation</div><div class="value" id="generation">—</div><div class="small" id="progressText">waiting for log</div></div>
 <div class="card"><div class="label">Seconds / generation</div><div class="value" id="seconds">—</div><div class="small">rolling mean · last 20</div></div>
 <div class="card"><div class="label">Estimated time left</div><div class="value" id="eta">—</div><div class="small">based on recent pace</div></div>
 <div class="card"><div class="label">Champion validation</div><div class="value" id="champion">—</div><div class="small" id="championGen">held-out normalized score</div></div>
 <div class="card"><div class="label">Current best</div><div class="value" id="best">—</div><div class="small">normalized · 0 random / 1 specialist</div></div>
</section>
<section class="grid">
 <article class="panel"><h2>Population fitness</h2><p>Best and mean normalized fitness by generation</p><canvas class="chart" id="fitnessChart"></canvas></article>
 <article class="panel"><h2>Archetype scores</h2><p>Each batter matters equally · dashed line marks the specialist baseline</p><canvas class="chart" id="scoresChart"></canvas></article>
 <article class="panel"><h2>Release speed</h2><p>Generation-best mean release speed</p><canvas class="chart" id="speedChart"></canvas></article>
 <article class="panel"><h2>Length mix</h2><p>Generation-best delivery categories · stacked share</p><canvas class="chart" id="lengthChart"></canvas></article>
</section>
<div class="section"><h2>Latest champion</h2><span id="validationStamp">Waiting for first held-out validation</span></div>
<section class="champions" id="championPanels"></section>
<div class="section"><h2>Checkpoint comparison</h2><span>Replays are sampled on common style-neutral seeds</span></div>
<section class="panel"><div class="controls">
 <label>First checkpoint <select id="checkpointA"></select></label><label>Second checkpoint <select id="checkpointB"></select></label>
 <label>Against <select id="compareArchetype"><option>slogger</option><option>technician</option><option>tailender</option></select></label><button id="compareButton">Compare checkpoints</button><span id="compareStatus" class="small"></span>
 </div><div class="compareGrid" id="comparePanels"></div></section>
<div class="foot">Local read-only dashboard · refreshes every 5 seconds · trajectory rendering remains a separate milestone</div>
</main>
<script>
const RUN_NAME=__RUN_NAME__, RUNS=__RUNS__;
let ACTIVE_RUN=RUN_NAME;
const CATS=['yorker','full','good','short','bouncer','full_toss'];
const ARCHS=['slogger','technician','tailender'];
const COLORS={best:'#42d9e8',mean:'#a78bfa',slogger:'#fb7185',technician:'#5ee0a0',tailender:'#fbbf24',yorker:'#f472b6',full:'#fb923c',good:'#38bdf8',short:'#a3e635',bouncer:'#c084fc',full_toss:'#facc15'};
const n=(v)=>{const x=Number(v);return Number.isFinite(x)?x:null};
function lineChart(canvas,series,{minY=null,maxY=null,threshold=null,stacked=false,marker=null}={}){
 const ctx=canvas.getContext('2d'), dpr=window.devicePixelRatio||1, rect=canvas.getBoundingClientRect(); canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;ctx.scale(dpr,dpr);
 const w=rect.width,h=rect.height,p={l:42,r:12,t:12,b:28};ctx.clearRect(0,0,w,h);ctx.font='10px Segoe UI';ctx.fillStyle='#8396ad';ctx.strokeStyle='#26344a';ctx.lineWidth=1;
 let values=series.flatMap(s=>s.data.filter(Number.isFinite));if(threshold!==null)values.push(threshold);if(!values.length){ctx.fillText('Waiting for training data…',p.l,h/2);return}
 let lo=minY===null?Math.min(...values):minY,hi=maxY===null?Math.max(...values):maxY;if(lo===hi){lo-=1;hi+=1}const pad=(hi-lo)*.08;lo=minY===null?lo-pad:lo;hi=maxY===null?hi+pad:hi;
 const x0=p.l,y0=h-p.b,cw=w-p.l-p.r,ch=h-p.t-p.b;for(let i=0;i<=4;i++){let y=y0-ch*i/4;ctx.beginPath();ctx.moveTo(x0,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.fillText((lo+(hi-lo)*i/4).toFixed(1),3,y+3)}
 const count=Math.max(...series.map(s=>s.data.length),2);const X=i=>x0+cw*i/(count-1),Y=v=>y0-(v-lo)/(hi-lo)*ch;
 if(threshold!==null){ctx.save();ctx.setLineDash([5,5]);ctx.strokeStyle='#fbbf24';ctx.beginPath();ctx.moveTo(x0,Y(threshold));ctx.lineTo(w-p.r,Y(threshold));ctx.stroke();ctx.restore()}
 if(stacked){let lower=Array(count).fill(0);for(const s of series){let upper=lower.map((v,i)=>v+(s.data[i]||0));ctx.beginPath();ctx.moveTo(X(0),Y(upper[0]));for(let i=1;i<count;i++)ctx.lineTo(X(i),Y(upper[i]));for(let i=count-1;i>=0;i--)ctx.lineTo(X(i),Y(lower[i]));ctx.closePath();ctx.globalAlpha=.7;ctx.fillStyle=s.color;ctx.fill();ctx.globalAlpha=1;lower=upper} }
 else for(const s of series){ctx.beginPath();let started=false;for(let i=0;i<s.data.length;i++){const v=s.data[i];if(!Number.isFinite(v)){started=false;continue}if(!started){ctx.moveTo(X(i),Y(v));started=true}else ctx.lineTo(X(i),Y(v))}ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.stroke()}
 if(marker&&marker.index>=0){const mx=X(Math.min(count-1,marker.index));ctx.save();ctx.setLineDash([4,4]);ctx.strokeStyle='#fb7185';ctx.beginPath();ctx.moveTo(mx,p.t);ctx.lineTo(mx,y0);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#fb7185';ctx.fillText(marker.label||'power cut',Math.min(mx+4,w-88),p.t+13);ctx.restore()}
 ctx.fillStyle='#8396ad';ctx.fillText('gen 1',x0,y0+18);ctx.fillText('gen '+(count>1?count:1),Math.max(x0,w-58),y0+18);
 let lx=x0;for(const s of series){ctx.fillStyle=s.color;ctx.fillRect(lx,p.t,9,3);ctx.fillStyle='#a9b8c9';ctx.fillText(s.name,lx+13,p.t+5);lx+=Math.min(125,30+s.name.length*7)}
}
function pitchMap(canvas,points){const ctx=canvas.getContext('2d'),dpr=devicePixelRatio||1,r=canvas.getBoundingClientRect();canvas.width=r.width*dpr;canvas.height=r.height*dpr;ctx.scale(dpr,dpr);const w=r.width,h=r.height,p={l:38,r:12,t:10,b:27};ctx.fillStyle='#0a111e';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#29384e';ctx.fillStyle='#91a2b6';ctx.font='10px Segoe UI';for(let i=0;i<=4;i++){const y=p.t+(h-p.t-p.b)*i/4;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke()}const X=v=>p.l+Math.max(0,Math.min(14,v||0))/14*(w-p.l-p.r),Y=v=>h-p.b-(Math.max(-.8,Math.min(1,v||0))+.8)/1.8*(h-p.t-p.b);ctx.strokeStyle='#dbeafe';ctx.beginPath();ctx.moveTo(p.l,Y(0));ctx.lineTo(w-p.r,Y(0));ctx.stroke();const resultColor={runs:'#5ee0a0',dot:'#8396ad',wide:'#fbbf24',no_ball:'#fb923c',bowled:'#fb7185',lbw:'#f43f5e',caught_behind:'#c084fc',caught_field:'#ef4444'};for(const q of points||[]){ctx.globalAlpha=.75;ctx.fillStyle=resultColor[q.result]||'#42d9e8';ctx.beginPath();ctx.arc(X(q.length_m),Y(q.line_m),2.7,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;ctx.fillStyle='#8396ad';ctx.fillText('bounce distance (m)',p.l,h-6);ctx.fillText('off stump line',4,Y(0)-4)}
function legendText(row){let c=[];for(const cat of CATS){let v=n(row['length_'+cat+'_share']);if(v!==null&&v>.02)c.push(cat+': '+Math.round(v*100)+'%')}return c.join(' · ')}
function renderChampion(latest){const root=document.getElementById('championPanels');root.innerHTML='';if(!latest||!latest.per_archetype){root.innerHTML='<div class="panel empty">No validation record yet. The 200-generation ga200 run can show its saved replay below.</div>';return}for(const name of ARCHS){const s=latest.per_archetype[name]||{},el=document.createElement('article');el.className='panel';el.innerHTML=`<h2>${name[0].toUpperCase()+name.slice(1)}</h2><canvas class="miniMap" id="champMap_${name}"></canvas><div class="pattern">${s.pattern_text||'No ball pattern yet'}</div><div class="stats">Norm ${n(s.normalized_score)?.toFixed(2)??'—'} · W/O ${n(s.wickets_per_over)?.toFixed(2)??'—'} · R/O ${n(s.economy)?.toFixed(2)??'—'}</div>`;root.appendChild(el);pitchMap(el.querySelector('canvas'),s.points||[])} }
function updateCharts(data){const rows=data.rows||[],numeric=(r,k)=>n(r[k]),marker=data.power_cut_generation?{index:rows.findIndex(r=>Number(r.gen)>=data.power_cut_generation),label:`power cut · gen ${data.power_cut_generation}`}:null;
 lineChart(document.getElementById('fitnessChart'),[{name:'best',color:COLORS.best,data:rows.map(r=>numeric(r,'best_fitness'))},{name:'mean',color:COLORS.mean,data:rows.map(r=>numeric(r,'mean_fitness'))}],{marker});
 lineChart(document.getElementById('scoresChart'),ARCHS.map(a=>({name:a,color:COLORS[a],data:rows.map(r=>numeric(r,a+'_normalized_score'))})),{minY:-.3,threshold:1,marker});
 lineChart(document.getElementById('speedChart'),[{name:'km/h',color:COLORS.best,data:rows.map(r=>numeric(r,'mean_speed_kmh'))}],{minY:90,maxY:155,marker});
 lineChart(document.getElementById('lengthChart'),CATS.map(c=>({name:c,color:COLORS[c],data:rows.map(r=>numeric(r,'length_'+c+'_share')||0)})),{minY:0,maxY:1,stacked:true,marker});
}
function prettyTime(seconds){if(!Number.isFinite(seconds)||seconds<0)return'—';let h=Math.floor(seconds/3600),m=Math.floor(seconds%3600/60);return h?`${h}h ${m}m`:`${m}m`}
function setupSelectors(checkpoints){const a=document.getElementById('checkpointA'),b=document.getElementById('checkpointB');const oldA=a.value,oldB=b.value;const opts=checkpoints.length?checkpoints:['No checkpoints'];a.innerHTML='';b.innerHTML='';for(const cp of opts){for(const sel of [a,b]){let op=document.createElement('option');op.value=cp;op.textContent=cp;sel.appendChild(op)}}if(opts.includes(oldA))a.value=oldA;if(opts.includes(oldB))b.value=oldB;else if(opts.length>1)b.selectedIndex=opts.length-1}
async function compare(){const status=document.getElementById('compareStatus'),root=document.getElementById('comparePanels');status.textContent='Evaluating…';root.innerHTML='';const arch=document.getElementById('compareArchetype').value;for(const cp of [document.getElementById('checkpointA').value,document.getElementById('checkpointB').value]){try{const r=await fetch(`/api/compare?run=${encodeURIComponent(ACTIVE_RUN)}&checkpoint=${encodeURIComponent(cp)}&archetype=${arch}`);const d=await r.json();const el=document.createElement('article');el.className='panel';el.innerHTML=`<h2>${cp}</h2><canvas class="miniMap compareMap"></canvas><div class="stats">W/O ${d.stats.wickets_per_over.toFixed(2)} · R/O ${d.stats.runs_per_over.toFixed(2)} · X/O ${d.stats.extras_per_over.toFixed(2)} · speed ${d.stats.mean_speed_kmh.toFixed(1)} km/h</div><div class="stats">${legendText(d.length_distribution)}</div>`;root.appendChild(el);pitchMap(el.querySelector('canvas'),d.points)}catch(e){status.textContent='Could not load checkpoint';return}}status.textContent='30-over comparison per checkpoint'}
async function refresh(){try{const response=await fetch(`/api/data?run=${encodeURIComponent(ACTIVE_RUN)}`,{cache:'no-store'});const d=await response.json();document.getElementById('runTitle').textContent=ACTIVE_RUN;document.getElementById('liveText').textContent=d.running?'Training active · refreshes every 5 s':(d.status==='stopped_at_gate'?'Gate check · run paused':'Latest saved state');const current=d.current_generation||0,total=d.total_generations||current;document.getElementById('generation').textContent=`${current} / ${total}`;document.getElementById('progressText').textContent=d.status==='stopped_at_gate'?'paused for gate review':current>=total&&total?'run complete':`${total?Math.min(100,100*current/total).toFixed(1):0}% complete`;document.getElementById('seconds').textContent=d.seconds_per_generation?d.seconds_per_generation.toFixed(1)+' s':'—';document.getElementById('eta').textContent=prettyTime(Math.max(0,total-current)*(d.seconds_per_generation||0));document.getElementById('best').textContent=d.latest_best===null?'—':d.latest_best.toFixed(3);document.getElementById('champion').textContent=d.champion_score===null?'—':d.champion_score.toFixed(3);document.getElementById('championGen').textContent=d.champion_generation?'best at generation '+d.champion_generation:'held-out normalized score';const championGen=d.champion_validation_generation||d.latest_validation?.generation;const latestGen=d.latest_validation?.generation;document.getElementById('validationStamp').textContent=championGen?'champion validated at generation '+championGen+(latestGen&&latestGen!==championGen?' · latest check '+latestGen:''):(d.fallback_replay?'showing latest saved replay':'Waiting for first held-out validation');updateCharts(d);renderChampion(d.champion_panel);setupSelectors(d.checkpoints||[])}catch(e){document.getElementById('liveText').textContent='Waiting for run files'}}
const runSelector=document.getElementById('runSelector');for(const run of RUNS){const option=document.createElement('option');option.value=run;option.textContent=run;if(run===RUN_NAME)option.selected=true;runSelector.appendChild(option)}runSelector.addEventListener('change',()=>{ACTIVE_RUN=runSelector.value;refresh()});
document.getElementById('compareButton').addEventListener('click',compare);window.addEventListener('resize',()=>refresh());refresh();setInterval(refresh,5000);
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a read-only live GA dashboard")
    parser.add_argument("--run", required=True, help="run directory under runs/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    return parser.parse_args()


def _read_config(run_dir: Path) -> dict:
    path = run_dir / "run_config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_checkpoint(run_dir: Path, checkpoint: str) -> tuple[np.ndarray, dict]:
    safe = Path(checkpoint).name
    filename = safe if safe.endswith(".npz") else f"{safe}.npz"
    path = run_dir / "checkpoints" / filename
    if not path.exists():
        path = run_dir / "best_genomes" / filename
    if not path.exists():
        raise FileNotFoundError(path.name)
    with np.load(path, allow_pickle=False) as data:
        return data["genome"].astype(float), json.loads(str(data["metadata_json"].item()))


def _fallback_panel(run_dir: Path) -> dict | None:
    replay_dir = run_dir / "replays"
    by_arch = {}
    for name in ARCHETYPE_NAMES:
        files = sorted(replay_dir.glob(f"gen_*_{name}.json*"))
        if not files:
            continue
        try:
            doc = load_replay(files[-1])
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        points = []
        patterns = []
        for over in doc.get("overs_data", []):
            sequence = []
            for delivery in over.get("deliveries", []):
                features = delivery.get("features", {})
                outcome = delivery.get("outcome", {})
                release = delivery.get("release", {})
                length = features.get("length_m")
                points.append(
                    {
                        "length_m": 0.0 if length is None else length,
                        "line_m": features.get("line_m", 0),
                        "result": outcome.get("result", "dot"),
                    }
                )
                sequence.append(features.get("length_category", "n/a"))
            patterns.append(sequence)
        summary = doc.get("summary", {})
        by_arch[name] = {
            "normalized_score": None,
            "wickets_per_over": summary.get("wickets_per_over"),
            "economy": summary.get("runs_per_over"),
            "points": points[:: max(1, len(points) // 300)],
            "pattern_text": "Saved replay · modal lengths: " + " → ".join(
                Counter(column[ball] for column in patterns if len(column) > ball).most_common(1)[0][0]
                if any(len(column) > ball for column in patterns) else "n/a"
                for ball in range(6)
            ),
        }
    return {"per_archetype": by_arch, "generation": None} if by_arch else None


def _validation_panel(latest: dict | None) -> dict | None:
    if not latest:
        return None
    per_archetype = {}
    for name, stats in latest.get("per_archetype", {}).items():
        points = stats.get("pitch_points", [])
        patterns = stats.get("ball_patterns", [])
        categories = []
        for ball_index in range(6):
            choices = [sequence[ball_index] for sequence in patterns if len(sequence) > ball_index]
            category = Counter(choices).most_common(1)[0][0] if choices else "n/a"
            speed_values = [
                float(point["speed_kmh"])
                for point in points
                if point.get("ball_pattern_index") == ball_index and "speed_kmh" in point
            ]
            if speed_values:
                med = float(np.median(speed_values))
                pace = "fast" if med >= 123 else "slower" if med <= 108 else "medium"
                categories.append(f"{pace} {category}")
            else:
                categories.append(category)
        per_archetype[name] = {
            "normalized_score": stats.get("normalized_score"),
            "wickets_per_over": stats.get("wickets_per_over"),
            "economy": stats.get("economy"),
            "points": points[:: max(1, len(points) // 300)],
            "pattern_text": "Typical ball 1–6: " + " → ".join(categories),
        }
    return {"generation": latest.get("generation"), "per_archetype": per_archetype}


def _compare_checkpoint(run_dir: Path, name: str, checkpoint: str, archetype: str) -> dict:
    genome, metadata = _load_checkpoint(run_dir, checkpoint)
    policy = MLPPolicy(genome)
    generation = int(metadata.get("generation", 0))
    seed_base = int(metadata.get("seed", 1)) + generation * 100_003 + ARCHETYPE_NAMES.index(archetype) * 1_000_003
    noise_scale = float(metadata.get("noise_scale", 0.05))
    overs = 30
    wickets = runs = extras = 0
    speeds = []
    points = []
    lengths = Counter()
    for over_index in range(overs):
        rng = np.random.default_rng(seed_base + over_index * 101)
        env = OverEnv(noise_scale=noise_scale)
        observation = env.reset(archetype, rng)
        while True:
            observation, _, done, info = env.step(policy.act(observation))
            features = info["features"]
            outcome = info["outcome"]
            category = features.length_category
            lengths[category] += 1
            speeds.append(info["release"].speed_kmh)
            points.append(
                {
                    "length_m": 0.0 if features.length_m is None else features.length_m,
                    "line_m": features.line_m,
                    "result": outcome.result,
                }
            )
            if done:
                break
        summary = env.summary()
        wickets += int(summary["wickets"])
        runs += int(summary["runs_conceded"])
        extras += int(summary["extras"])
    return {
        "checkpoint": checkpoint,
        "generation": generation,
        "stats": {
            "wickets_per_over": wickets / overs,
            "runs_per_over": runs / overs,
            "extras_per_over": extras / overs,
            "mean_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
        },
        "length_distribution": {key: value / max(1, sum(lengths.values())) for key, value in lengths.items()},
        "points": points[:: max(1, len(points) // 400)],
    }


def _make_single_run_handler(run_dir: Path, run_name: str):
    cache: dict[tuple[str, str], dict] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.replace("__RUN_NAME__", json.dumps(run_name)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/data":
                rows = read_csv_tolerant(run_dir / "log.csv")
                config = _read_config(run_dir)
                validation = read_latest_jsonl_tolerant(run_dir / "validation.jsonl")
                champion_validation = None
                champion_validation_path = run_dir / "champion_validation.json"
                if champion_validation_path.exists():
                    try:
                        champion_validation = json.loads(
                            champion_validation_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        champion_validation = None
                checkpoint_files = sorted((run_dir / "checkpoints").glob("*.npz"))
                checkpoints = [path.stem for path in checkpoint_files]
                current = int(rows[-1]["gen"]) if rows else 0
                total = int(config.get("generations", 0)) or (current or 200)
                seconds = [float(row["seconds_per_generation"]) for row in rows[-20:] if row.get("seconds_per_generation")]
                champion_path = run_dir / "checkpoints" / "champion.npz"
                champion_score = None
                champion_generation = None
                if champion_path.exists():
                    try:
                        with np.load(champion_path, allow_pickle=False) as data:
                            meta = json.loads(str(data["metadata_json"].item()))
                        champion_score = meta.get("validation_score")
                        champion_generation = meta.get("validation_generation")
                    except Exception:
                        pass
                latest_panel = _validation_panel(champion_validation or validation)
                fallback = False
                if latest_panel is None:
                    latest_panel = _fallback_panel(run_dir)
                    fallback = latest_panel is not None
                payload = {
                    "rows": rows,
                    "current_generation": current,
                    "total_generations": total,
                    "seconds_per_generation": float(np.mean(seconds)) if seconds else None,
                    "latest_best": float(rows[-1]["best_fitness"]) if rows and rows[-1].get("best_fitness") else None,
                    "champion_score": champion_score,
                    "champion_generation": champion_generation,
                    "latest_validation": {"generation": validation.get("generation")} if validation else None,
                    "champion_validation_generation": (
                        champion_validation.get("generation")
                        if champion_validation
                        else validation.get("generation") if validation else None
                    ),
                    "champion_panel": latest_panel,
                    "fallback_replay": fallback,
                    "checkpoints": checkpoints,
                    "running": bool(total and current < total),
                }
                self._send_json(payload)
                return
            if parsed.path == "/api/compare":
                query = parse_qs(parsed.query)
                checkpoint = query.get("checkpoint", [""])[0]
                archetype = query.get("archetype", ["slogger"])[0]
                if archetype not in ARCHETYPE_NAMES:
                    self._send_json({"error": "unknown archetype"}, 400)
                    return
                key = (checkpoint, archetype)
                if key not in cache:
                    try:
                        cache[key] = _compare_checkpoint(run_dir, run_name, checkpoint, archetype)
                    except (FileNotFoundError, ValueError, OSError) as error:
                        self._send_json({"error": str(error)}, 404)
                        return
                    if len(cache) > 32:
                        cache.pop(next(iter(cache)))
                self._send_json(cache[key])
                return
            self._send_json({"error": "not found"}, 404)

    return Handler


def _resolve_run_dir(runs_root: Path, name: str) -> Path:
    if name not in RUN_OPTIONS:
        raise ValueError(f"unknown run: {name}")
    return runs_root / name


def _available_checkpoints(run_dir: Path) -> tuple[list[str], list[int]]:
    files = []
    for folder in ("checkpoints", "best_genomes"):
        if (run_dir / folder).exists():
            files.extend((run_dir / folder).glob("*.npz"))
    numeric = []
    for path in files:
        if path.stem.startswith("gen_"):
            try:
                numeric.append(int(path.stem.split("_", 1)[1]))
            except ValueError:
                continue
    names = [f"gen_{generation:04d}" for generation in sorted(set(numeric))]
    if any(path.stem == "champion" for path in files):
        names.append("champion")
    return names, sorted(set(numeric))


def make_handler(runs_root: Path, default_run: str):
    compare_cache: dict[tuple[str, str, str], dict] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status)

        def _requested_run(self, query: dict[str, list[str]]) -> tuple[str, Path]:
            name = query.get("run", [default_run])[0]
            return name, _resolve_run_dir(runs_root, name)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                body = HTML_PAGE.replace("__RUN_NAME__", json.dumps(default_run)).replace(
                    "__RUNS__", json.dumps(RUN_OPTIONS)
                ).encode("utf-8")
                self._send(body, "text/html; charset=utf-8")
                return
            if parsed.path == "/brain":
                run_options = "".join(
                    f'<option value="{name}"{" selected" if name == default_run else ""}>{name}</option>'
                    for name in RUN_OPTIONS
                )
                body = BRAIN_PAGE.replace("__RUN_OPTIONS__", run_options).replace(
                    "__RUNS__", json.dumps(RUN_OPTIONS)
                ).replace("__DEFAULT_RUN__", json.dumps(default_run)).encode("utf-8")
                self._send(body, "text/html; charset=utf-8")
                return
            if parsed.path == "/brain_panels.js":
                js_path = ROOT / "src" / "aibowler" / "viz" / "brain_panels.js"
                self._send(js_path.read_bytes(), "text/javascript; charset=utf-8")
                return
            if parsed.path == "/theater":
                body = THEATER_PAGE.replace("__RUNS__", json.dumps(RUN_OPTIONS)).replace(
                    "__DEFAULT_RUN__", json.dumps(default_run)
                ).encode("utf-8")
                self._send(body, "text/html; charset=utf-8")
                return

            try:
                run_name, run_dir = self._requested_run(query)
            except ValueError as error:
                self._send_json({"error": str(error)}, 400)
                return

            if parsed.path == "/api/data":
                rows = read_csv_tolerant(run_dir / "log.csv")
                config = _read_config(run_dir)
                validation = read_latest_jsonl_tolerant(run_dir / "validation.jsonl")
                champion_validation = None
                champion_validation_path = run_dir / "champion_validation.json"
                if champion_validation_path.exists():
                    try:
                        champion_validation = json.loads(champion_validation_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        champion_validation = None
                checkpoints, checkpoint_generations = _available_checkpoints(run_dir)
                current = int(rows[-1]["gen"]) if rows and rows[-1].get("gen", "").isdigit() else 0
                total = int(config.get("generations", 0)) or (current or 200)
                seconds = [
                    float(row["seconds_per_generation"])
                    for row in rows[-20:]
                    if row.get("seconds_per_generation")
                ]
                champion_path = run_dir / "checkpoints" / "champion.npz"
                champion_score = champion_generation = None
                if champion_path.exists():
                    try:
                        with np.load(champion_path, allow_pickle=False) as data:
                            meta = json.loads(str(data["metadata_json"].item()))
                        champion_score = meta.get("validation_score")
                        champion_generation = meta.get("validation_generation")
                    except Exception:
                        pass
                latest_panel = _validation_panel(champion_validation or validation)
                fallback = False
                if latest_panel is None:
                    latest_panel = _fallback_panel(run_dir)
                    fallback = latest_panel is not None
                events = list(config.get("events", []))
                events.extend(
                    {"type": "power_cut", "generation": event.get("interrupted_after_generation", 0), "note": "historical recovery"}
                    for event in config.get("recovery_history", [])
                )
                power_cut = next(
                    (int(event.get("generation", 0)) for event in events if event.get("type") == "power_cut"),
                    None,
                )
                status = config.get("status", "complete" if current >= total and total else "training")
                running = status == "training" if "status" in config else bool(total and current < total)
                self._send_json({
                    "run": run_name,
                    "rows": rows,
                    "current_generation": current,
                    "total_generations": total,
                    "seconds_per_generation": float(np.mean(seconds)) if seconds else None,
                    "latest_best": float(rows[-1]["best_fitness"]) if rows and rows[-1].get("best_fitness") else None,
                    "champion_score": champion_score,
                    "champion_generation": champion_generation,
                    "latest_validation": {"generation": validation.get("generation")} if validation else None,
                    "champion_validation_generation": champion_validation.get("generation") if champion_validation else (validation.get("generation") if validation else None),
                    "champion_panel": latest_panel,
                    "fallback_replay": fallback,
                    "checkpoints": checkpoints,
                    "checkpoint_generations": checkpoint_generations,
                    "events": events,
                    "power_cut_generation": power_cut,
                    "status": status,
                    "running": running,
                })
                return

            if parsed.path == "/api/theater":
                try:
                    requested_generation = max(1, min(1500, int(query.get("generation", [1])[0])))
                    archetype = query.get("archetype", ["slogger"])[0]
                    seed = int(query.get("seed", [10421])[0])
                    if archetype not in ARCHETYPE_NAMES:
                        raise ValueError("unknown archetype")
                    checkpoint, saved_generation = brain_explorer.nearest_checkpoint_name(
                        run_dir, requested_generation
                    )
                    model = brain_explorer.network_payload(run_dir, checkpoint)
                    cam = brain_explorer.brain_cam(run_dir, checkpoint, archetype, seed)
                    population = brain_explorer.population_cloud(
                        run_dir, runs_root, requested_generation
                    )
                    all_runs = {}
                    for candidate in ("ga1500", "ga2"):
                        candidate_dir = runs_root / candidate
                        candidate_config = _read_config(candidate_dir)
                        candidate_events = list(candidate_config.get("events", []))
                        candidate_events.extend(
                            {"type": "power_cut", "generation": event.get("interrupted_after_generation", 0),
                             "note": "historical recovery"}
                            for event in candidate_config.get("recovery_history", [])
                        )
                        all_runs[candidate] = {
                            "rows": read_csv_tolerant(candidate_dir / "log.csv"),
                            "events": candidate_events,
                            "status": candidate_config.get("status", "unknown"),
                            "generations": candidate_config.get("generations", 1500),
                        }
                    lineage = brain_explorer.family_tree(run_dir, "champion")
                    ancestry_ids = []
                    if population.get("genome_ids") and lineage.get("nodes"):
                        lineage_generations = {
                            node["generation"]: int(node["id"])
                            for node in lineage["nodes"] if not node.get("donor")
                        }
                        ancestor_id = lineage_generations.get(population.get("snapshot_generation"))
                        if ancestor_id is not None:
                            ancestry_ids = [ancestor_id]
                    self._send_json({
                        "run": run_name,
                        "requested_generation": requested_generation,
                        "saved_generation": saved_generation,
                        "checkpoint": checkpoint,
                        "archetype": archetype,
                        "seed": seed,
                        "network": model,
                        "cam": cam,
                        "population": population,
                        "ancestry_ids": ancestry_ids,
                        "lineage": lineage,
                        "runs": all_runs,
                    })
                except (FileNotFoundError, ValueError, OSError, KeyError, IndexError) as error:
                    self._send_json({"error": str(error)}, 404)
                return

            if parsed.path == "/api/compare":
                checkpoint = query.get("checkpoint", [""])[0]
                archetype = query.get("archetype", ["slogger"])[0]
                if archetype not in ARCHETYPE_NAMES:
                    self._send_json({"error": "unknown archetype"}, 400)
                    return
                key = (run_name, checkpoint, archetype)
                if key not in compare_cache:
                    try:
                        compare_cache[key] = _compare_checkpoint(run_dir, run_name, checkpoint, archetype)
                    except (FileNotFoundError, ValueError, OSError) as error:
                        self._send_json({"error": str(error)}, 404)
                        return
                    if len(compare_cache) > 48:
                        compare_cache.pop(next(iter(compare_cache)))
                self._send_json(compare_cache[key])
                return

            if parsed.path.startswith("/api/brain/"):
                checkpoint = query.get("checkpoint", ["champion"])[0]
                archetype = query.get("archetype", ["slogger"])[0]
                if archetype not in ARCHETYPE_NAMES and parsed.path not in {"/api/brain/overview", "/api/brain/population", "/api/brain/lineage"}:
                    self._send_json({"error": "unknown archetype"}, 400)
                    return
                try:
                    if parsed.path == "/api/brain/replays":
                        archetype = query.get("archetype", ["slogger"])[0]
                        if archetype not in ARCHETYPE_NAMES:
                            self._send_json({"error": "unknown archetype"}, 400)
                            return
                        replay_dir = run_dir / "replays"
                        replays = sorted(
                            path.name for path in replay_dir.glob(f"*_{archetype}.json*")
                        ) if replay_dir.exists() else []
                        self._send_json({"run": run_name, "archetype": archetype, "replays": replays})
                        return
                    if parsed.path == "/api/brain/overview":
                        rows = read_csv_tolerant(run_dir / "log.csv")
                        config = _read_config(run_dir)
                        checkpoints, generations = _available_checkpoints(run_dir)
                        current = int(rows[-1]["gen"]) if rows and rows[-1].get("gen", "").isdigit() else 0
                        total = int(config.get("generations", 0)) or current
                        champion = (run_dir / "checkpoints" / "champion.npz").exists()
                        champion_generation = None
                        if champion:
                            try:
                                with np.load(run_dir / "checkpoints" / "champion.npz", allow_pickle=False) as data:
                                    champion_meta = json.loads(str(data["metadata_json"].item()))
                                champion_generation = int(champion_meta.get("validation_generation", champion_meta.get("generation", 0)))
                            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                                champion_generation = None
                        self._send_json({
                            "run": run_name, "status": config.get("status", "complete" if current >= total else "training"),
                            "running": config.get("status") == "training" if "status" in config else current < total,
                            "current_generation": current, "total_generations": total,
                            "checkpoints": checkpoints, "checkpoint_generations": generations,
                            "champion_available": champion,
                            "champion_generation": champion_generation,
                        })
                        return
                    if parsed.path == "/api/brain/network":
                        result = brain_explorer.network_payload(run_dir, checkpoint)
                    elif parsed.path == "/api/brain/cam":
                        replay_name = query.get("replay", [""])[0]
                        if replay_name:
                            replay_path = run_dir / "replays" / Path(replay_name).name
                            if not replay_path.exists() or replay_path.name != replay_name:
                                raise FileNotFoundError("saved replay is not available")
                            replay = load_replay(replay_path)
                            if replay.get("initial_archetype") != archetype:
                                raise ValueError("saved replay uses a different batter archetype")
                            first_over = (replay.get("overs_data") or [{}])[0]
                            replay_seed = int(first_over.get("seed", query.get("seed", [10421])[0]))
                            generation = int(replay.get("generation", 1))
                            replay_checkpoint = "champion" if replay.get("label") == "champion" else f"gen_{generation:04d}"
                            result = dict(brain_explorer.brain_cam(
                                run_dir, replay_checkpoint, archetype, replay_seed
                            ))
                            result["replay_source"] = replay_name
                            result["checkpoint_name"] = replay_checkpoint
                        else:
                            result = brain_explorer.brain_cam(
                                run_dir, checkpoint, archetype, int(query.get("seed", [10421])[0])
                            )
                    elif parsed.path == "/api/brain/decision":
                        result = brain_explorer.decision_map(
                            run_dir, checkpoint, archetype,
                            int(query.get("seed", [10421])[0]),
                            int(query.get("ball", [0])[0]),
                            int(query.get("x", [5])[0]), int(query.get("y", [7])[0]),
                            query.get("output", ["speed"])[0],
                        )
                    elif parsed.path == "/api/brain/sensitivity":
                        result = brain_explorer.permutation_importance(run_dir, checkpoint, archetype)
                    elif parsed.path == "/api/brain/neuron":
                        result = brain_explorer.neuron_diagnostics(
                            run_dir, checkpoint, archetype,
                            int(query.get("layer", [0])[0]), int(query.get("neuron", [0])[0]),
                        )
                    elif parsed.path == "/api/brain/population":
                        result = brain_explorer.population_cloud(
                            run_dir, runs_root, int(query.get("generation", [1])[0])
                        )
                    elif parsed.path == "/api/brain/weights":
                        result = brain_explorer.weight_statistics(run_dir, checkpoint, archetype)
                    elif parsed.path == "/api/brain/lineage":
                        result = brain_explorer.family_tree(run_dir, checkpoint)
                    else:
                        self._send_json({"error": "not found"}, 404)
                        return
                    self._send_json(result)
                    return
                except (FileNotFoundError, ValueError, OSError, KeyError, IndexError) as error:
                    self._send_json({"error": str(error)}, 404)
                    return
            self._send_json({"error": "not found"}, 404)

    return Handler


def main() -> None:
    args = parse_args()
    if Path(args.run).name != args.run or args.run in {".", ".."}:
        raise SystemExit("--run must be a single directory name under runs/")
    run_dir = ROOT / "runs" / args.run
    if not run_dir.exists():
        raise SystemExit(f"run directory not found: {run_dir}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(ROOT / "runs", args.run))
    print(f"Dashboard for {args.run}: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
