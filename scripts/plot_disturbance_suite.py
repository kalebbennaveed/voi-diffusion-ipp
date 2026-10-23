"""Scientific atlas of the declared tasks and physical disturbance fields."""
from pathlib import Path
import json
import jax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from voi_diffusion_ipp.quarter.benchmark_environments import SCENARIOS,make_environment,flow,flow_parameters,describe

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/disturbance_suite';OUT.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,5,figsize=(23,10),constrained_layout=True)
field=jax.jit(flow);records=[]
for ax,s in zip(axes.ravel(),SCENARIOS):
    e=make_environment(s.name);lo,hi=np.asarray(e.bounds)
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],90),np.linspace(lo[1],hi[1],90));p=np.c_[xx.ravel(),yy.ravel()]
    v=np.asarray(field(p,flow_parameters(s))).reshape(90,90,2)
    mag=np.linalg.norm(v,axis=-1)
    im=ax.pcolormesh(xx,yy,mag,cmap='Blues',vmin=0,vmax=4.2,shading='auto',rasterized=True)
    for r in s.rectangles:ax.add_patch(Rectangle(r[:2],r[2]-r[0],r[3]-r[1],fc='#77808a',ec='#394758',lw=.8,zorder=4))
    cells=np.asarray(e.cells);ax.scatter(*cells.T,s=4,c='#d9820f',alpha=.9,zorder=5)
    sample=np.s_[4::10,4::10];vx,vy=v[:,:,0][sample],v[:,:,1][sample]
    valid=np.hypot(vx,vy)>1
    ax.quiver(xx[sample][valid],yy[sample][valid],vx[valid],vy[valid],color='#155b83',scale=32,width=.006,alpha=.85,zorder=3)
    start=(np.asarray(s.patches[0][:2])+np.asarray(s.patches[0][2:]))/2;ax.scatter(*start,s=60,c='black',marker='*',zorder=6)
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',title='Four-branch inspection network' if s.name=='interchange' else s.title)
    ax.set_xlabel(f'{len(s.rectangles)} obstacles · {len(s.patches)} target areas',fontsize=10)
    ax.tick_params(labelsize=8);records.append(describe(s.name))
fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='Physical air speed (m/s)')
fig.suptitle('Ten monitoring tasks: physical air flow, obstacle geometry and monitored cells\nOrange: monitored cells. Black star: initial quadrotor position. Synthetic fields; full nonlinear quadrotor execution.',fontsize=16)
fig.savefig(OUT/'environment_atlas.png',dpi=160);fig.savefig(OUT/'environment_atlas.pdf')
(OUT/'environment_catalog.json').write_text(json.dumps(records,indent=2))
print(OUT/'environment_atlas.png')

diagnostic=OUT/'stall_diagnostic'
if (diagnostic/'same_decision.npz').exists():
    with np.load(diagnostic/'same_decision.npz') as a:
        actual=a['actual'];coarse=a['earlier'][:21,:2];fine=a['fine_predicted'][:21,:2]
        sampled=actual[::25,:2];clock=np.arange(21)*.25
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        s=SCENARIOS[0];ax=axes[0]
        for r in s.rectangles:ax.add_patch(Rectangle(r[:2],r[2]-r[0],r[3]-r[1],fc='#a4adb5',ec='#5a6d7d'))
        ax.plot(*coarse.T,c='#c47216',lw=2.5,label='Earlier prediction')
        ax.plot(*fine.T,c='#1467a3',lw=3,label='Filter prediction, 0.05 s step')
        ax.plot(*actual[:,:2].T,c='#087c68',lw=2,ls='--',label='Nonlinear execution, 100 Hz')
        ax.scatter(*actual[0,:2],s=60,c='black',zorder=5,label='Same reached state')
        ax.set(xlim=(-3.2,5),ylim=(-.8,4.8),aspect='equal',xlabel='x (m)',ylabel='y (m)',title='Same decision and posterior, next five seconds')
        ax.legend(fontsize=9,loc='lower right')
        ax=axes[1]
        ax.plot(clock,np.linalg.norm(coarse-sampled,axis=1),c='#c47216',lw=2.5,label='Earlier prediction')
        ax.plot(clock,np.linalg.norm(fine-sampled,axis=1),c='#1467a3',lw=2.5,label='Filter prediction, 0.05 s step')
        ax.set(xlabel='Time after planning (s)',ylabel='Position prediction error (m)',title='Error against the recorded nonlinear motion')
        ax.legend(fontsize=9);ax.grid(alpha=.2)
        fig.suptitle('Prediction diagnostic: the earlier planning model missed the execution-filter stall',fontsize=13)
        fig.savefig(OUT/'prediction_diagnostic.png',dpi=180)
        fig.savefig(OUT/'prediction_diagnostic.pdf')
