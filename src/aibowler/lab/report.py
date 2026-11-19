"""Scouting report: one self-contained HTML file an analyst can read, check and forward.

Every number comes from the simulator or the data. Recommendations are generated from those
numbers by templates (no LLM), and every claim about the batter shows its evidence type, so a
coach can see which parts are measured and which are assumed.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .audit import LENGTH_ORDER  # noqa: E402
from .cards import BatterCard, BowlerCard, RewardSpec  # noqa: E402

INK, MUTED, BG, PANEL = "#e8f1fb", "#93a7bd", "#0b1220", "#111a2b"
TEAL, ORANGE, PINK, AMBER = "#49debe", "#fb923c", "#fb7185", "#fbbf24"
LENGTH_LABEL = {"full_toss": "Full toss", "yorker": "Yorker", "full": "Full", "good": "Good", "short": "Short", "bouncer": "Bouncer"}


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#24324a")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def pitch_map(balls: list[dict], hand: str) -> str:
    """Top view of the batter's half: where the plan pitches, coloured by outcome."""
    fig, ax = plt.subplots(figsize=(4.2, 6.0), facecolor=BG)
    _style(ax)
    ax.add_patch(plt.Rectangle((-1.525, -0.3), 3.05, 14.3, color="#a8936a", alpha=0.85, zorder=0))
    ax.add_patch(plt.Rectangle((-1.525, 6.0), 3.05, 2.0, color=TEAL, alpha=0.18, zorder=1))
    ax.plot([-1.83, 1.83], [1.22, 1.22], color="#f4f1e8", lw=1.5, zorder=2)
    ax.plot([-0.114, 0.114], [0, 0], color="#f4f1e8", lw=4, zorder=2)
    flip = 1 if hand == "right" else -1
    xs, ys, cs = [], [], []
    for b in balls:
        if b["length_m"] is None:
            continue
        xs.append(-flip * b["line_m"])  # bowler's-eye view: off side on the left for a right-hander
        ys.append(b["length_m"])
        cs.append(PINK if b["wicket"] else ORANGE if b["runs"] >= 4 else "#5ee0a0" if b["runs"] > 0 else "#c8d3e0")
    ax.scatter(xs, ys, c=cs, s=14, alpha=0.75, edgecolors="none", zorder=3)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(14, -0.6)
    ax.set_xlabel("off side ←   line (m)   → leg side" if hand == "right" else "leg side ←   line (m)   → off side")
    ax.set_ylabel("length from batter's stumps (m)")
    ax.text(1.75, 7.0, "good\nlength", color=TEAL, fontsize=8, ha="right", va="center")
    return _fig_to_data_uri(fig)


def length_bars(plan_mix: dict, base_mix: dict) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 2.8), facecolor=BG)
    _style(ax)
    x = np.arange(len(LENGTH_ORDER))
    ax.bar(x - 0.2, [base_mix.get(k, 0) for k in LENGTH_ORDER], 0.4, color="#4b5d78", label="normal bowling")
    ax.bar(x + 0.2, [plan_mix.get(k, 0) for k in LENGTH_ORDER], 0.4, color=TEAL, label="trained plan")
    ax.set_xticks(x, [LENGTH_LABEL[k] for k in LENGTH_ORDER])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    return _fig_to_data_uri(fig)


def robustness_hist(gaps: list[float]) -> str:
    fig, ax = plt.subplots(figsize=(8.0, 2.6), facecolor=BG)
    _style(ax)
    g = np.asarray(gaps)
    ax.hist(g, bins=16, color=TEAL, alpha=0.85)
    ax.axvline(0, color=PINK, lw=1.5)
    ax.set_xlabel("plan minus normal bowling (reward per over); right of the red line = plan wins", fontsize=9)
    ax.set_ylabel("versions of the batter", fontsize=9)
    return _fig_to_data_uri(fig)


def recommendations(card: BatterCard, a: dict) -> list[str]:
    s, plan, base = a["summary"], a["plan"], a["baseline"]
    recs = []
    top_len = max(s["length_mix"], key=s["length_mix"].get)
    top_line = next(iter(s["line_mix"]), "stumps")
    recs.append(f"Stock ball: <b>{LENGTH_LABEL[top_len].lower()} length</b> ({s['length_mix'][top_len]:.0%} of balls), "
                f"mostly <b>{top_line.replace('_', ' ')}</b>, around {s['speed']['mean']:.0f} km/h.")
    for z in s["hot_zones"][:2]:
        recs.append(f"Wicket ball: {LENGTH_LABEL[z['length']].lower()} length, {z['line'].replace('_', ' ')} — "
                    f"{z['wicket_prob']:.1%} chance of a wicket per ball there ({z['share']:.0%} of the plan).")
    over = s["by_ball"]
    if over:
        seq = ", ".join(f"{b['ball']}: {LENGTH_LABEL[b['length']].lower()}" for b in over)
        recs.append(f"Typical over: {seq}.")
    d_w = plan["wickets_per_over"] - base["wickets_per_over"]
    d_r = plan["economy"] - base["economy"]
    recs.append(f"Against normal bowling this plan changes wickets per over by {d_w:+.2f} and runs per over by {d_r:+.2f}.")
    avoid = [w for w in card.strengths]
    if avoid:
        recs.append("Avoid: " + "; ".join(w.zone.describe() for w in avoid) + " (his scoring zones on the card).")
    return recs


def _badge(kind: str) -> str:
    col = {"data": TEAL, "source": AMBER, "assumption": PINK}[kind]
    return f'<span class="badge" style="border-color:{col};color:{col}">{kind}</span>'


def _evidence(items) -> str:
    if not items:
        return f'<div class="ev">{_badge("assumption")} no evidence given</div>'
    rows = []
    for e in items:
        src = ""
        if e.source:
            s = html.escape(e.source)
            src = f' — <a href="{s}">{s}</a>' if s.startswith("http") else f" — {s}"
        rows.append(f'<div class="ev">{_badge(e.kind)} {html.escape(e.claim)}{src}</div>')
    return "".join(rows)


def _status(status: str) -> str:
    col = {"ok": TEAL, "passed": TEAL, "warn": AMBER, "uncalibrated": AMBER, "fail": PINK, "failed": PINK}[status]
    return f'<span class="status" style="background:{col}">{status.upper()}</span>'


def render(card: BatterCard, bowler: BowlerCard, reward: RewardSpec, calibration: dict, audit_result: dict,
           train_info: dict) -> str:
    a = audit_result
    s = a["summary"]
    plan, base = a["plan"], a["baseline"]
    worst = "fail" if any(c["status"] == "fail" for c in a["checks"]) else "warn" if any(c["status"] == "warn" for c in a["checks"]) else "ok"
    cal_status = calibration["status"]
    verdict = ("Do not use: the plan failed an exploit or robustness check." if worst == "fail"
               else "Use with care: see the warnings and calibration status." if worst == "warn" or cal_status != "passed"
               else "Plan passed every check against a calibrated batter.")
    rows_w = "".join(f"<tr><td>{html.escape(w.zone.describe())}</td><td>{w.dismissal.replace('_', ' ')}</td><td>{w.severity}</td><td>{_evidence(w.evidence)}</td></tr>" for w in card.weaknesses)
    rows_s = "".join(f"<tr><td>{html.escape(w.zone.describe())}</td><td>{w.severity}</td><td>{_evidence(w.evidence)}</td></tr>" for w in card.strengths)
    cal_rows = ""
    if calibration.get("real_fit"):
        real = calibration["real_fit"]["rates"]
        simp = calibration.get("sim_population") or {}
        for k, label in (("strike_rate", "Strike rate"), ("dot_pct", "Dot balls %"), ("boundary_pct", "Boundary %"),
                         ("balls_per_dismissal", "Balls per dismissal"), ("caught_share", "Share of outs caught")):
            sv = simp.get(k, {})
            cal_rows += f"<tr><td>{label}</td><td>{real[k]:.2f}</td><td>{sv.get('mean', float('nan')):.2f} ± {sv.get('sd', float('nan')):.2f}</td></tr>"
        rf = calibration["real_fit"]
        cal_rows = (f"<p class='muted'>Fit on {rf['balls']} balls in {rf['matches']} matches ({rf['first_date']} → {rf['last_date']}).</p>"
                    f"<table><tr><th>Statistic</th><th>Real</th><th>Simulated population</th></tr>{cal_rows}</table>")
        if calibration.get("holdout_check"):
            hc = "".join(f"<tr><td>{k}</td><td>{v['real']:.2f}</td><td>{v['sim']:.2f}</td><td>±{v['allowed']:.2f}</td><td>{'✓' if v['ok'] else '✗'}</td></tr>"
                         for k, v in calibration["holdout_check"].items())
            cal_rows += f"<h4>Held-out recent matches</h4><table><tr><th>Statistic</th><th>Real</th><th>Sim</th><th>Allowed</th><th></th></tr>{hc}</table>"
    checks = "".join(f"<tr><td>{_status(c['status'])}</td><td>{c['name']}</td><td>{html.escape(c['detail'])}</td></tr>" for c in a["checks"])
    over = "".join(f"<tr><td>{b['ball']}</td><td>{LENGTH_LABEL[b['length']]} ({b['length_share']:.0%})</td><td>{b['line'].replace('_', ' ')}</td>"
                   f"<td>{b['speed_kmh']:.0f}</td><td>{b['wicket_prob']:.1%}</td><td>{b['confidence']:.2f}</td></tr>" for b in s["by_ball"])
    recs = "".join(f"<li>{r}</li>" for r in recommendations(card, a))
    rw = reward.resolved()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scouting report: {html.escape(card.name)}</title>
<style>
body{{background:{BG};color:{INK};font:15px/1.55 system-ui,Segoe UI,sans-serif;margin:0}}
main{{max-width:980px;margin:0 auto;padding:28px 18px 60px}}
h1{{font-size:30px;margin:0 0 4px}} h2{{margin-top:36px;border-bottom:1px solid #24324a;padding-bottom:6px;font-size:20px}} h4{{margin:18px 0 6px}}
.muted{{color:{MUTED}}} table{{border-collapse:collapse;width:100%;margin:8px 0}} td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid #1c2a3d;vertical-align:top}}
th{{color:{MUTED};font-weight:600}} .badge{{border:1px solid;border-radius:6px;padding:0 6px;font-size:11px;margin-right:6px}}
.ev{{margin:2px 0}} .status{{color:#06101a;border-radius:6px;padding:1px 8px;font-size:12px;font-weight:700}}
.verdict{{background:{PANEL};border-left:4px solid {TEAL if worst == 'ok' and cal_status == 'passed' else AMBER if worst != 'fail' else PINK};padding:12px 16px;margin:18px 0;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.kpi{{background:{PANEL};border-radius:10px;padding:10px 14px}} .kpi b{{font-size:24px;display:block}}
img{{max-width:100%;border-radius:8px}} .two{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.4fr);gap:18px}}
@media(max-width:720px){{.two{{grid-template-columns:1fr}}}} a{{color:{TEAL}}} li{{margin:6px 0}}
</style></head><body><main>
<p class="muted">Scouting report · AI Bowler Lab · {date.today().isoformat()}</p>
<h1>{html.escape(card.name)}</h1>
<p class="muted">{html.escape(card.description)}</p>
<div class="verdict"><b>{verdict}</b><br><span class="muted">Calibration: {_status(cal_status)} {html.escape(calibration['reason'])}</span></div>
<div class="grid">
<div class="kpi"><span class="muted">Wickets / over</span><b>{plan['wickets_per_over']:.2f}</b><span class="muted">normal bowling {base['wickets_per_over']:.2f}</span></div>
<div class="kpi"><span class="muted">Runs / over</span><b>{plan['economy']:.2f}</b><span class="muted">normal bowling {base['economy']:.2f}</span></div>
<div class="kpi"><span class="muted">Robustness</span><b>{a['robustness']['share']:.0%}</b><span class="muted">of plausible versions beaten</span></div>
<div class="kpi"><span class="muted">Evidence assumed</span><b>{card.assumption_share():.0%}</b><span class="muted">of claims on the card</span></div>
</div>
<h2>The plan for {html.escape(bowler.name)}</h2>
<ul>{recs}</ul>
<div class="two"><div><img src="{pitch_map(a['balls'], card.hand)}" alt="Pitch map of the trained plan"></div>
<div><img src="{length_bars(s['length_mix'], a['baseline_summary']['length_mix'])}" alt="Length mix"><h4>Ball by ball</h4>
<table><tr><th>Ball</th><th>Length</th><th>Line</th><th>km/h</th><th>Wicket chance</th><th>Batter confidence</th></tr>{over}</table></div></div>
<h2>Checks</h2><table>{checks}</table>
<img src="{robustness_hist(a['robustness']['gaps'])}" alt="Robustness histogram">
<h2>The batter card</h2>
<p>{card.hand}-handed · {card.level} · {card.style.replace('_', ' ')} · {card.starter.replace('_', ' ')} · defends straight: {card.defends_straight} · leaves outside off: {card.leaves_outside_off}</p>
{_evidence(card.evidence)}
<h4>Weaknesses</h4><table><tr><th>Zone</th><th>Gets out</th><th>Severity</th><th>Evidence</th></tr>{rows_w}</table>
<h4>Strengths</h4><table><tr><th>Zone</th><th>Severity</th><th>Evidence</th></tr>{rows_s}</table>
<h2>Calibration against real data</h2>{cal_rows or '<p class="muted">No ball-by-ball data supplied. The batter is the card only.</p>'}
<h2>Setup</h2>
<p class="muted">Bowler: {bowler.min_kmh:.0f}–{bowler.max_kmh:.0f} km/h, {bowler.accuracy} accuracy, swing: {bowler.swing}, bouncer: {'yes' if bowler.bouncer else 'no'}.
Reward ({reward.format}): wicket = {rw['wicket']:.0f} runs, run cost {rw['run']}, extra cost {rw['extra']}, max {rw['max_bouncers']} bouncers per over.
Training: {train_info.get('generations')} generations × {train_info.get('population_size')} brains, {train_info.get('overs_per_eval')} overs each, {train_info.get('seconds')} s.
Simulated: {len(a['balls'])} deliveries. This is a simulation of a model of the batter; treat it as a hypothesis to test in the nets, not a fact.</p>
</main></body></html>"""
