"""Scientific plots and a self-contained, offline replay of executed data."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_run(path):
    path = Path(path)
    a = np.load(path)
    m = json.loads(path.with_suffix(".json").read_text())
    dt = m["config"]["dt"]
    H = m["config"]["stage_steps"]
    t = np.arange(len(a["x"])) * dt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axis = ax[0, 0]
    for ox, oy, r in a["obstacles"]:
        axis.add_patch(plt.Circle((ox, oy), r, color="#576270", alpha=0.8))
    sc = axis.scatter(*a["cells"].T, c=a["q"][-1], vmin=0, vmax=1, cmap="viridis", s=65)
    fig.colorbar(sc, ax=axis, label="Final clarity")
    for i, p in enumerate(a["nominal_paths"]):
        axis.plot(
            *p[:, :2].T,
            ":",
            color="#d2872c",
            alpha=0.6,
            label="Nominal (belief mean)" if i == 0 else None,
        )
    axis.plot(*a["x"][:, :2].T, color="#005f95", label="Executed", lw=2)
    axis.scatter(*a["x"][::H, :2].T, color="black", s=20, label="Belief update boundary")
    axis.scatter(*a["probe"], marker="x", s=70, color="#c73341", label="Probe region centre")
    for name in ("roi_a", "roi_b"):
        axis.annotate(name.replace("_", " ").upper(), a[name] + np.array([0, 0.25]))
    axis.set(
        xlim=a["bounds"][:, 0],
        ylim=a["bounds"][:, 1],
        aspect="equal",
        xlabel="x [m]",
        ylabel="y [m]",
    )
    axis.legend(fontsize=7, loc="upper left")
    deficit = np.maximum(a["targets"][None] - a["q"], 0).mean(axis=1)
    ax[0, 1].plot(t, deficit, color="#005f95")
    ax[0, 1].set(xlabel="Mission time [s]", ylabel="Mean clarity deficit", ylim=(0, 0.8))
    ax[1, 0].plot(t[1:], a["correction"], label="Control correction")
    ax[1, 0].plot(t[1:], a["clearance"], label="Minimum clearance [m]")
    ax[1, 0].axhline(0, color="black", lw=0.6)
    ax[1, 0].scatter(
        t[1:][~a["feasible"]],
        a["correction"][~a["feasible"]],
        marker="x",
        color="red",
        label="QP infeasible",
    )
    ax[1, 0].set(xlabel="Mission time [s]")
    ax[1, 0].legend(fontsize=8)
    bt = np.arange(1, len(a["branches"]) + 1) * H * dt
    ax[1, 1].plot(
        bt, np.sqrt(a["belief_variance"]).mean(axis=1), marker="o", label="Mean posterior wind std"
    )
    ax[1, 1].step(bt, a["branches"], where="post", label="Continuation branch", alpha=0.6)
    ax[1, 1].set(xlabel="Belief update time [s]")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle(
        f"{m['scenario']} | {m['method']} | seed {m['seed']} | cost {m['integrated_deficit']:.3f}\n"
        f"{m['backend']} · collision={m['collision']} · QP failure={m['qp_infeasible_fraction']:.1%}",
        fontsize=12,
    )
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)
    data = {
        k: a[k].tolist()
        for k in (
            "x",
            "q",
            "cells",
            "targets",
            "obstacles",
            "bounds",
            "nominal_paths",
            "belief_mean",
            "belief_variance",
            "branches",
            "correction",
            "feasible",
            "probe",
            "observations",
        )
    }
    data.update(dt=dt, H=H, title=f"{m['scenario']} / {m['method']} / seed {m['seed']}")
    template = """<!doctype html><html><head><meta charset="utf-8"><title>VoI IPP replay</title>
<style>body{font:16px system-ui;background:#f5f7fa;color:#153044;margin:25px;max-width:1000px}canvas{background:white;border:1px solid #ccc}input{width:70%}button{padding:8px}#status{font-variant-numeric:tabular-nums;margin:12px 0}p{max-width:850px}</style></head>
<body><h2 id="title"></h2><button id="play">Play / pause</button> <input id="time" type="range" min="0" value="0"><div id="status"></div>
<canvas id="map" width="960" height="570"></canvas>
<p>Blue: executed path. Orange dotted: nominal prediction from the current horizon start, using belief-mean wind.
Circles: monitoring clarity (dark = low, bright = high). Purple arrows and rings: posterior wind mean and uncertainty at monitoring cells.
Black crosses: executed wind measurement locations. Red cross: probe-region centre. Belief changes only at horizon boundaries.
The branch shown is the continuation selected at the most recent completed boundary; subsequent MPC optimization may change it.</p>
<script>const D=__DATA__;const c=document.querySelector('canvas'),g=c.getContext('2d'),slider=document.querySelector('#time');
slider.max=D.x.length-1;document.querySelector('#title').textContent=D.title;
const xy=p=>[60+(p[0]-D.bounds[0][0])/(D.bounds[1][0]-D.bounds[0][0])*840,520-(p[1]-D.bounds[0][1])/(D.bounds[1][1]-D.bounds[0][1])*470];
function line(points,color,dash=[]){g.strokeStyle=color;g.setLineDash(dash);g.beginPath();points.forEach((p,j)=>{let a=xy(p);j?g.lineTo(...a):g.moveTo(...a)});g.stroke();g.setLineDash([])}
function draw(){let i=+slider.value,b=Math.floor(i/D.H)-1,h=Math.min(Math.floor(i/D.H),D.nominal_paths.length-1);g.clearRect(0,0,960,570);
D.obstacles.forEach(o=>{let a=xy(o);g.fillStyle='#576270';g.beginPath();g.ellipse(a[0],a[1],o[2]*840/(D.bounds[1][0]-D.bounds[0][0]),o[2]*470/(D.bounds[1][1]-D.bounds[0][1]),0,0,7);g.fill()});
D.cells.forEach((p,j)=>{let a=xy(p);g.fillStyle=`hsl(${200-150*D.q[i][j]},75%,${25+40*D.q[i][j]}%)`;g.beginPath();g.arc(...a,7,0,7);g.fill();if(b>=0){let v=D.belief_mean[b][j],s=Math.sqrt(D.belief_variance[b][j]);line([p,[p[0]+v[0]*.3,p[1]+v[1]*.3]],'#8439a2');g.strokeStyle='#8439a255';g.beginPath();g.arc(...a,s*40,0,7);g.stroke()}});
line(D.nominal_paths[h].map(x=>x.slice(0,2)),'#d2872c',[4,4]);g.lineWidth=2;line(D.x.slice(0,i+1).map(x=>x.slice(0,2)),'#005f95');g.lineWidth=1;
D.x.slice(1,i+1).forEach(x=>{let a=xy(x);g.fillStyle='#0008';g.fillRect(a[0]-1,a[1]-1,3,3)});
let a=xy(D.x[i]);g.fillStyle='#005f95';g.beginPath();g.arc(...a,7,0,7);g.fill();a=xy(D.probe);g.strokeStyle='red';g.beginPath();g.moveTo(a[0]-6,a[1]-6);g.lineTo(a[0]+6,a[1]+6);g.moveTo(a[0]+6,a[1]-6);g.lineTo(a[0]-6,a[1]+6);g.stroke();
let deficit=D.q[i].reduce((s,q,j)=>s+Math.max(0,D.targets[j]-q),0)/D.q[i].length;
document.querySelector('#status').textContent=`t=${(i*D.dt).toFixed(2)} s | deficit=${deficit.toFixed(3)} | branch=${b<0?'prior':D.branches[b]} | QP=${i===0?'—':D.feasible[i-1]?'feasible':'INFEASIBLE'}`;}
slider.oninput=draw;let playing=false;document.querySelector('#play').onclick=()=>playing=!playing;
setInterval(()=>{if(playing){slider.value=(+slider.value+1)%D.x.length;draw()}},100);draw();</script></body></html>"""
    path.with_suffix(".html").write_text(template.replace("__DATA__", json.dumps(data)))


def summary(directory):
    directory = Path(directory)
    rows = []
    for p in sorted(directory.glob("*.json")):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and "integrated_deficit" in r:
            rows.append(r)
    groups = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["method"]), []).append(r)
    table = []
    for (scenario, method), rs in groups.items():
        vals = np.array([r["integrated_deficit"] for r in rs])
        table.append(
            {
                "scenario": scenario,
                "method": method,
                "n": len(rs),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "collisions": sum(r["collision"] for r in rs),
                "access": sum(r["roi_a_access"] for r in rs),
                "qp_failure": float(np.mean([r["qp_infeasible_fraction"] for r in rs])),
                "optimizer_failures": sum(r.get("optimizer_failure_count", 0) for r in rs),
                "plan_seconds": float(np.mean([r["mean_plan_seconds"] for r in rs])),
                "probe_seconds": float(np.mean([r["probe_seconds"] for r in rs])),
            }
        )
    (directory / "aggregate.json").write_text(json.dumps(table, indent=2))
    scenarios = sorted({r["scenario"] for r in table})
    if not scenarios:
        return
    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(6 * len(scenarios), 5), squeeze=False, layout="constrained"
    )
    for ax, scenario in zip(axes[0], scenarios):
        rr = [r for r in table if r["scenario"] == scenario]
        xx = np.arange(len(rr))
        ax.bar(xx, [r["mean"] for r in rr], yerr=[r["std"] for r in rr], color="#397c9d", capsize=4)
        for i, r in enumerate(rr):
            ax.scatter(
                np.full(r["n"], i),
                [v["integrated_deficit"] for v in groups[(scenario, r["method"])]],
                color="black",
                s=13,
                zorder=3,
            )
        ax.set_xticks(xx, [r["method"] for r in rr], rotation=55, ha="right")
        ax.set(title=scenario, ylabel="Integrated actual clarity deficit")
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)
    text = [
        "# Measured experiment summary",
        "",
        "Dots are paired held-out seeds; error bars are sample standard deviations, not confidence intervals.",
        "",
        "| Scenario | Method | n | Deficit mean ± SD | ROI access | Collisions | QP failure | Optimizer fallbacks | Plan seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in table:
        text.append(
            f"| {r['scenario']} | {r['method']} | {r['n']} | {r['mean']:.4f} ± {r['std']:.4f} | {r['access']}/{r['n']} | {r['collisions']}/{r['n']} | {r['qp_failure']:.2%} | {r['optimizer_failures']} | {r['plan_seconds']:.3f} |"
        )
    (directory / "summary.md").write_text("\n".join(text) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    a = p.parse_args()
    path = Path(a.path)
    if path.is_dir():
        summary(path)
        for file in path.glob("*.npz"):
            render_run(file)
    else:
        render_run(path)


if __name__ == "__main__":
    main()
