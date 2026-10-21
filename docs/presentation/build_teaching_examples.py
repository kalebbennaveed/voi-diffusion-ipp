"""Deterministic teaching examples. These are not monitoring-mission results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/presentation/wind_aware_mbd/assets'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','text.color':'#00274c','axes.labelcolor':'#00274c'})
N,B,G,O,P='#00274c','#2474bf','#008653','#d77717','#9b4682'
# One-dimensional MBD optimization, with a geometric path model.
s=np.linspace(0,1,201)
def route(a):return np.column_stack((-2+4*s,4*a*s*(1-s)))
def evaluate(a):
 r=route(a); length=np.linalg.norm(np.diff(r,axis=0),axis=1).sum()
 valid=abs(a)<=2 and np.linalg.norm(r,axis=1).min()>=.6
 return float(length), bool(valid)
rng=np.random.default_rng(19); abar=np.geomspace(.98,.16,8); alpha=abar/np.r_[1,abar[:-1]]
y=.35; history=[]; best=None
for i in range(7,-1,-1):
 samples=rng.normal(y/np.sqrt(abar[i]),np.sqrt((1-abar[i])/abar[i]),64)
 costs=np.array([evaluate(a)[0] for a in samples]); valid=np.array([evaluate(a)[1] for a in samples])
 weights=np.zeros(64);weights[valid]=np.exp(-(costs[valid]-costs[valid].min())/.12);weights/=weights.sum()
 mean=float(weights@samples);score=(-y+np.sqrt(abar[i])*mean)/(1-abar[i]);ynext=(y+(1-abar[i])*score)/np.sqrt(alpha[i])
 j=np.where(valid)[0][np.argmin(costs[valid])]
 if best is None or costs[j]<best[1]:best=(float(samples[j]),float(costs[j]))
 history.append(dict(level=i+1,noisy=float(y),proposal_mean=float(y/np.sqrt(abar[i])),proposal_std=float(np.sqrt((1-abar[i])/abar[i])),samples=samples.tolist(),costs=costs.tolist(),valid=valid.tolist(),weights=weights.tolist(),weighted_mean=mean,score=float(score),next_noisy=float(ynext),best=list(best)))
 y=ynext
(OUT/'mbd_toy_data.json').write_text(json.dumps({'description':'Scalar path-height MBD; deterministic seed 19; valid domain [-2,2], clearance >= 0.6; cost path length; temperature .12; 8 reverse levels, 64 samples each. Illustration only.','alpha_bar':abar.tolist(),'history':history},indent=2))

def environment(ax):
 ax.add_patch(Circle((0,0),.5,color='#9aa9b6'));ax.add_patch(Circle((0,0),.6,fill=False,ls=':',ec=N,lw=1.5))
 ax.plot([-2,2],[0,0],'o',c=N,ms=9);ax.text(-2,-.25,'Start',ha='center');ax.text(2,-.25,'Goal',ha='center')
 ax.set(xlim=(-2.4,2.4),ylim=(-2.1,2.1),aspect='equal',xlabel='Horizontal position',ylabel='Vertical position')
 ax.grid(alpha=.12)
fig,ax=plt.subplots(figsize=(9.4,4.4),layout='constrained');environment(ax)
for a in [.0,.9,1.5]:
 r=route(a);ax.plot(*r.T,lw=2.8,label=f'Path height a = {a:.1f}')
ax.legend(loc='lower center',ncol=3,fontsize=12);ax.set_title('One parameter changes the path around the obstacle',fontweight='bold')
fig.savefig(OUT/'mbd_toy_setup.svg');plt.close(fig)

frames=[]
for h in history:
 for phase in range(3):
  fig=plt.figure(figsize=(15,5.4),dpi=100,facecolor='white')
  grid=fig.add_gridspec(1,2,width_ratios=[1.2,1],left=.055,right=.98,bottom=.15,top=.80,wspace=.23)
  ax=fig.add_subplot(grid[0]);environment(ax)
  vals=np.array(h['samples']);weights=np.array(h['weights']);valid=np.array(h['valid'])
  for j,a in enumerate(vals):
   if abs(a)>2.1:continue
   r=route(a)
   c=B if valid[j] else '#bac0c5'
   ax.plot(*r.T,c=c,alpha=.22 if phase==0 else .10+min(1,weights[j]*8)*.65,lw=1 if phase==0 else 1+weights[j]*45)
  if phase==2:
   r=route(h['weighted_mean']);ax.plot(*r.T,c=O,lw=3,ls='--',label='Weighted parameter mean')
   r=route(h['best'][0]);ax.plot(*r.T,c=G,lw=3,label='Best evaluated feasible path')
   ax.legend(loc='lower center',fontsize=11)
  ax.set_title('Model evaluation: paths from sampled heights',fontsize=15,fontweight='bold')
  bx=fig.add_subplot(grid[1]); aa=np.linspace(-2,2,401); cc=np.array([evaluate(a)[0] for a in aa]); vv=np.array([evaluate(a)[1] for a in aa])
  bx.axvspan(-.6,.6,color='#eceff2',label='Insufficient clearance')
  bx.plot(aa[vv & (aa<0)],cc[vv & (aa<0)],c=N,lw=2);bx.plot(aa[vv & (aa>0)],cc[vv & (aa>0)],c=N,lw=2,label='Feasible path length')
  bx.scatter(vals[valid],np.array(h['costs'])[valid],s=35 if phase==0 else 30+weights[valid]*1800,c=B,alpha=.7,label='Evaluated samples')
  if phase>=1:bx.text(.02,.94,'Marker area increases with MBD weight',transform=bx.transAxes,va='top',fontsize=12)
  if phase==2:bx.axvline(h['weighted_mean'],color=O,ls='--',lw=2,label='Weighted parameter mean')
  bx.set(xlim=(-2.1,2.1),ylim=(3.8,6.1),xlabel='Path-height parameter a',ylabel='Path length (cost)');bx.grid(alpha=.15)
  bx.set_title('Parameter-space weighting',fontsize=15,fontweight='bold')
  phase_text=['1. Draw 64 path-height samples from the Gaussian proposal','2. Evaluate length and clearance; assign larger weights to lower cost','3. Form the weighted mean; use the score to update the next level'][phase]
  fig.text(.055,.94,f'MBD iteration {9-h["level"]} of 8  |  diffusion level i = {h["level"]}',fontsize=19,fontweight='bold',color=N)
  fig.text(.055,.86,phase_text,fontsize=16,color=B)
  fig.text(.055,.025,'Illustrative optimization only  •  Grey paths receive zero weight  •  The weighted mean is not guaranteed feasible',fontsize=12,color='#596c7b')
  fig.canvas.draw(); im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert('RGB');frames.append(im)
  if h['level']==1 and phase==2:im.save(OUT/'mbd_toy_poster.png')
  plt.close(fig)
frames[0].save(OUT/'mbd_toy.gif',save_all=True,append_images=frames[1:],duration=[1250]*23+[3500],loop=0,optimize=False)
# Braking example: same position, different velocity; explicitly a teaching model.
fig,axes=plt.subplots(2,1,figsize=(12,4.8),layout='constrained')
for ax,v,c in zip(axes,[1,2],[G,O]):
 stop=-1+v*v/2
 ax.axvspan(0,1.2,color='#f9e5e0');ax.axvline(0,c=N,lw=2)
 ax.plot([-1,stop],[0,0],c=c,lw=6,solid_capstyle='round');ax.plot(-1,0,'o',c=N,ms=12);ax.plot(stop,0,'s',c=c,ms=11)
 ax.annotate('',xy=(-1+.25*v,.17),xytext=(-1,.17),arrowprops=dict(arrowstyle='->',color=N,lw=2))
 ax.text(-1,.29,f'Speed v = {v} m/s',ha='center',fontsize=15)
 ax.text(stop,-.25,f'Stop at p = {stop:g} m',ha='center',c=c,fontweight='bold',fontsize=15)
 ax.text(.65,.30,'Forbidden side',ha='center',fontsize=13)
 ax.text(-1.75,-.02,f'Case {"A" if v==1 else "B"}',fontweight='bold',fontsize=16)
 ax.set(xlim=(-2,1.3),ylim=(-.45,.55));ax.axis('off')
fig.savefig(OUT/'design_policy_example.svg');plt.close(fig)
print(json.dumps({'toy_frames':len(frames),'toy_best':best,'braking_values':[-.5,1.0]}))
