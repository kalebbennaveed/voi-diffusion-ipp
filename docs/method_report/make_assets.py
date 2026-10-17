"""Rebuild report figures from retained recordings; no simulation or JAX required."""
from pathlib import Path
import json, hashlib, os
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"tmp/pdfs/method_report/mplcache"))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'tmp/pdfs/method_report'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','axes.grid':False,'legend.frameon':False})
COLORS=['#087e8b','#d28328','#7c619a']
NAMES=['Anticipated learning','Current belief','Wind-blind nominal + safety']
FOLDERS=['persistent_two_region_2s_01','persistent_two_region_2s_01','persistent_two_region_safety_only_01']
METHODS=['proposed','current_belief','wind_blind_planner']
paths=[ROOT/f'results/budgeted_missions/{f}/{m}.npz' for f,m in zip(FOLDERS,METHODS)]
data=[dict(np.load(p)) for p in paths]
meta=[json.loads(p.with_suffix('.json').read_text()) for p in paths]
audit=[json.loads(p.with_suffix('.audit.json').read_text()) for p in paths]
D=data[0]; cells=D['cells']; shape=(33,41); extent=(-2.7,2.7,-2.1,2.1)
A=np.all(np.abs(cells-[1.5,.9]) <= np.array([1.4,1.2])/2,axis=1)
B=np.all(np.abs(cells-[-1.5,-.9]) <= np.array([1.4,1.2])/2,axis=1)

def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight',pad_inches=.07);plt.close(fig)

def base(ax,labels=True):
 for x,y,r in D['obstacles']:
  ax.add_patch(Circle((x,y),r,color='#506372',zorder=4))
  ax.add_patch(Circle((x,y),r+.06,fill=False,color='#506372',ls=':',lw=.8,zorder=4))
 for (x,y),label in [((1.5,.9),'A'),((-1.5,-.9),'B')]:
  ax.add_patch(Rectangle((x-.7,y-.6),1.4,1.2,fill=False,ec='#273f50',lw=1.1,zorder=5))
  if labels:ax.text(x,y+.69,label,ha='center',weight='bold',color='#183447',zorder=6)
 ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal',xlabel='x (m)',ylabel='y (m)')

# Actual geometry and hidden physical field (diagnostic display only).
fig,ax=plt.subplots(figsize=(7.1,4.0))
mag=np.linalg.norm(D['physical_wind'],axis=1)
im=ax.imshow(mag.reshape(shape),origin='lower',extent=extent,cmap='Blues',vmin=0,vmax=1.85,alpha=.85)
idx=np.arange(len(cells)).reshape(shape)[::3,::3].ravel()
ax.quiver(*cells[idx].T,*D['physical_wind'][idx].T,color='#385e7e',scale=17,width=.0028)
base(ax)
ax.plot(*D['initial_observation_positions'].T,'o',ms=2.7,color='#ad7548',label='Common historical survey')
ax.scatter(1.5,.9,s=60,marker='*',color='#c04439',zorder=7,label='Initial position')
ax.legend(loc='lower right',fontsize=8)
fig.colorbar(im,ax=ax,pad=.025,label='Physical disturbance magnitude (m/s²)',shrink=.8)
save(fig,'geometry')

# Analytic clarity loss and sensing equilibrium.
t=np.linspace(0,20,300);fig,axs=plt.subplots(1,2,figsize=(7.1,2.6))
for Q,c in zip([.04,.08,.16],['#8097a2','#3d95a5','#087e8b']):axs[0].plot(t,.74/(1+Q*.74*t),label=f'Q = {Q:.2f}',color=c)
axs[0].axhline(.7,color='#bb5c40',ls='--',lw=1,label='Target 0.70');axs[0].set(xlabel='Time without sensing (s)',ylabel='Clarity',ylim=(0,1));axs[0].legend(fontsize=7)
r=np.linspace(0,1.5,300);H2=np.exp(-r*r/.5**2);s=H2/.1;qeq=np.sqrt(s)/(np.sqrt(s)+np.sqrt(.16))
axs[1].plot(r,qeq,color=COLORS[0],label='Stationary clarity equilibrium');axs[1].plot(r,H2,color='#ba4d47',ls='--',label='Relative information rate H²')
axs[1].set(xlabel='Distance from sensor (m)',ylabel='Normalized value',ylim=(0,1));axs[1].legend(fontsize=7);fig.tight_layout(w_pad=2);save(fig,'decay')

# Three full recorded executions and task histories.
fig,axs=plt.subplots(3,2,figsize=(7.1,6.0),gridspec_kw={'width_ratios':[1,1.5]})
for i,(d,n,c) in enumerate(zip(data,NAMES,COLORS)):
 ax=axs[i,0];base(ax,False);ax.plot(*d['x'][:,:2].T,color=c,lw=.65,alpha=.8);ax.scatter(*d['x'][0,:2],marker='*',color='#192f43',s=40,zorder=6);ax.set_title(n,loc='left',fontsize=9)
 ax=axs[i,1];t=np.arange(len(d['q']))*.05
 ax.plot(t,d['q'][:,A].mean(1),color=c,lw=1.2,label='Region A');ax.plot(t,d['q'][:,B].mean(1),color=c,lw=1.2,ls='--',label='Region B');ax.axhline(.7,color='#aa5042',lw=.7,ls=':');ax.set(xlim=(0,120),ylim=(.1,.85),ylabel='Mean regional clarity');ax.legend(loc='upper right',ncol=2,fontsize=7)
 if i<2:ax.set_xticklabels([])
 else:ax.set_xlabel('Mission time (s)')
fig.tight_layout(h_pad=1.6);save(fig,'missions')

fig,axs=plt.subplots(1,2,figsize=(7.1,2.8))
for d,n,c in zip(data,NAMES,COLORS):
 cost=np.maximum(d['targets']-d['q'],0).mean(1);t=np.arange(len(cost))*.05
 J=np.r_[0,np.cumsum((cost[1:]+cost[:-1])*.025)]
 axs[0].plot(t,J,label=n,color=c,lw=1.4)
 axs[1].plot(t,cost,color=c,lw=.9,alpha=.85)
axs[0].set(xlabel='Mission time (s)',ylabel='Integrated whole-grid deficit (s)');axs[1].set(xlabel='Mission time (s)',ylabel='Instantaneous whole-grid deficit')
axs[0].legend(fontsize=7,loc='upper left');fig.tight_layout(w_pad=2);save(fig,'cost')

# Whole-grid q snapshots, same scale, predetermined times.
fig,axs=plt.subplots(2,3,figsize=(7.1,4.4),layout='constrained')
for ax,sec in zip(axs.ravel(),[0,10,20,40,80,120]):
 k=round(sec/.05);im=ax.imshow(D['q'][k].reshape(shape),origin='lower',extent=extent,vmin=0,vmax=1,cmap='viridis');base(ax,False);ax.plot(*D['x'][max(0,k-160):k+1,:2].T,color='white',lw=1);ax.scatter(*D['x'][k,:2],s=14,color='#ef665b',zorder=8);ax.set_title(f't = {sec} s',loc='left');ax.tick_params(labelsize=7)
fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.8,pad=.02,label='Clarity q');save(fig,'clarity_maps')

# Wind prediction at initial and final snapshots, common magnitude and sigma ranges.
fig,axs=plt.subplots(2,3,figsize=(7.1,4.5),layout='constrained')
windsets=[D['physical_wind'],D['mean'][0],D['mean'][-1]]
for ax,field,title in zip(axs[0],windsets,['Hidden physical field','POAM mean at 0 s','POAM mean at 120 s']):
 im=ax.imshow(np.linalg.norm(field,axis=1).reshape(shape),origin='lower',extent=extent,vmin=0,vmax=1.85,cmap='Blues');ax.quiver(*cells[idx].T,*field[idx].T,scale=20,color='#234661',width=.004);base(ax,False);ax.set_title(title,fontsize=9,loc='left');ax.tick_params(labelsize=7)
for ax,val,title in zip(axs[1],[D['std'][0],D['std'][-1],np.sqrt(np.mean((D['mean'][-1]-D['physical_wind'])**2,axis=1))],['Component std at 0 s','Component std at 120 s','Component RMSE at 120 s']):
 im2=ax.imshow(val.reshape(shape),origin='lower',extent=extent,vmin=0,vmax=.8,cmap='magma');base(ax,False);ax.set_title(title,fontsize=9,loc='left');ax.tick_params(labelsize=7)
fig.colorbar(im,ax=axs[0].tolist(),shrink=.8,pad=.02,label='Magnitude (m/s²)');fig.colorbar(im2,ax=axs[1].tolist(),shrink=.8,pad=.02,label='m/s²');save(fig,'wind')

# Genuine saved MBD iteration at t=20 s, fixed illustrative index, not selected for gain.
r=json.loads((ROOT/'results/demo/live/budgeted_persistent_two_region_2s_01_proposed_8101/replay.json').read_text())['planning'][10]
fig,axs=plt.subplots(2,2,figsize=(7.1,4.8))
for k,ax in enumerate(axs.ravel()):
 base(ax,False)
 for path,w in zip(r['candidate_paths'][k],r['weights'][k]):
  path=np.asarray(path,dtype=float);ax.plot(*path.T,color='#6c9ba4',alpha=.3+.5*float(w),lw=.7)
 for key,c,ls,label in [('incumbent_paths','#087e8b','-','Best evaluated'),('mean_paths','#ce7b29','--','Denoised mean')]:
  path=np.asarray(r[key][k],dtype=float);ax.plot(*path.T,color=c,ls=ls,lw=1.5,label=label)
 ax.set_title(f'Reverse level {4-k}',loc='left')
 if k==0:ax.legend(fontsize=6.7,loc='lower left')
fig.tight_layout();save(fig,'denoising')
trace=np.asarray(r['trace']);print('t20 MBD trace',trace.tolist())

fig,axs=plt.subplots(1,2,figsize=(7.1,2.75))
for m,n,c in zip(meta,NAMES,COLORS):
 vals=np.array([x['planning_cycle_s'] for x in m['periods']]);axs[0].plot(np.arange(2,60)*2,vals[2:],color=c,lw=1.0,label=n)
axs[0].axhline(2,color='#bb5347',ls='--',lw=1);axs[0].set(xlabel='Mission time (s); first 2 cycles omitted',ylabel='Inclusive planning cycle (s)',ylim=(0,2.25));axs[0].legend(fontsize=6.8,loc='center right')
components=['cache_s','plan_s','verify_s','poam_update_s','snapshot_s'];labels=['Cache','MBD','Uncached check','POAM update','Snapshot']
values=[np.median([x[k] for x in meta[0]['periods'][2:]]) for k in components]
axs[1].barh(labels,values,color=['#8ba9b6','#087e8b','#d28328','#64778d','#b4b5b1']);axs[1].invert_yaxis();axs[1].set(xlabel='Median component time (s)',title='Proposed; illustrative component medians')
fig.tight_layout(w_pad=2);save(fig,'timing')

EQS={
'elbo':r'\mathcal{L}(\theta)=\sum_{n,c}\mathbb{E}_{q(f_{nc})}[\log p(y_{nc}\mid f_{nc})]-\sum_{c=1}^{2}\mathrm{KL}[q(u_c)\Vert p_\theta(u_c)]',
'branches':r'Y=(U_L,U_{F,1},\ldots,U_{F,B}),\qquad b_s=g(x_L^{(s)},q_L^{(s)},\mathcal{B}_L^{(s)})',
'dynamics':r'\dot{p}=v,\qquad \dot{v}=u-\gamma v+d(p),\qquad u\in[-u_{\max},u_{\max}]^2',
'observation':r'y^d_j=d(p_j)+\epsilon_j,\qquad \epsilon_j\sim\mathcal{N}(0,\sigma_d^2 I_2)',
'riccati':r'\dot{P}_i=Q_i-\frac{H_i(p)^2}{R}P_i^2,\qquad q_i=\frac{1}{1+P_i}',
'clarity':r'\dot{q}_i=\frac{H_i(p)^2}{R}(1-q_i)^2-Q_iq_i^2',
'footprint':r'H_i(p)=\exp\left(-\frac{\|p-r_i\|^2}{2\sigma_s^2}\right),\qquad \frac{H_i^2}{R}\geq 0',
'objective':r'c(q)=\frac{1}{N}\sum_{i=1}^{N}\max(\bar{q}_i-q_i,0),\qquad J_T=\int_0^T c(q(t))\,dt',
'decay':r'q_i(t)=\frac{q_i(0)}{1+Q_iq_i(0)t},\qquad q_i^*=\frac{\sqrt{H_i^2/R}}{\sqrt{H_i^2/R}+\sqrt{Q_i}}',
'kernel':r'k_\theta(x,x^\prime)=a\,[w(x)^\top w(x^\prime)]\sum_{\ell=1}^{L}w_\ell(x)w_\ell(x^\prime)\exp\left(-\frac{\|x-x^\prime\|^2}{2\ell_\ell^2}\right)',
'predictmean':r'\mu_X=K_{XZ}K_{ZZ}^{-1}m,\qquad A_X=K_{XZ}K_{ZZ}^{-1}',
'predictcov':r'\Sigma_X=K_{XX}-K_{XZ}K_{ZZ}^{-1}K_{ZX}+A_X S A_X^\top',
'statistics':r'a_{\mathrm{new}}=P^\top a_{\mathrm{old}}+K_{ZX}Y,\qquad B_{\mathrm{new}}=P^\top B_{\mathrm{old}}P+K_{ZX}K_{XZ}',
'posterior':r'C=K_{ZZ}+\sigma_d^{-2}B,\qquad m=\sigma_d^{-2}K_{ZZ}C^{-1}a,\qquad S=K_{ZZ}C^{-1}K_{ZZ}',
'root':r'B=LL^\top,\qquad F=[P^\top L_{\mathrm{old}}\quad K_{ZX}],\qquad F^\top=QR,\qquad L_{\mathrm{new}}=R^\top',
'sample':r'd(x)=g(x)+K_{xZ}K_{ZZ}^{-1}[u-g(Z)],\qquad u\sim\mathcal{N}(m,S)',
'cbf':r'B_j(x;\mathcal{B})=\max_{\ell=1,\ldots,L_R}\ \max_{\tau\in[0,T_B]}\,\widetilde{h}_{j,\ell}(\tau;x)',
'backup':r'\pi_b(x;\mathcal{B})=\mathrm{clip}\left((\gamma-k_b)v-\mu_{\mathcal{B}}(p),-u_{\max},u_{\max}\right)',
'constraint':r'\nabla_v B_j^\top u\leq-\nabla_p B_j^\top v+\gamma\nabla_v B_j^\top v-\max_\ell\nabla_v B_j^\top d_\ell(p)-\alpha B_j',
'qp':r'\min_{u\in\mathcal{U}}\frac{1}{2}\|u-u_{\mathrm{nom}}\|^2\quad\mathrm{subject\ to}\quad A u\leq b',
'slack':r'\delta^*=\min\{\delta\geq0:\ \exists u\in\mathcal{U},\ A u\leq b+\delta\mathbf{1}\}',
'saa':r'\widehat{J}(Y)=\frac{1}{S_o}\sum_{s=1}^{S_o}\left[\int_0^{T_L}c(q^{(s)})dt+\int_{T_L}^{T_L+T_F}c(q^{(s)}_Y)dt\right]',
'voi':r'\Delta_{\mathrm{info}}=\widehat{J}_{\mathrm{frozen}}(x_L,q_L,\mathcal{B}_0)-\widehat{J}_{\mathrm{updated}}(x_L,q_L,\mathcal{B}_L)',
'features':r'z(\mathcal{B})=\frac{1}{\sigma_{\mathrm{prior}}}[\mu_x(0),\mu_y(0),\sigma(0)]^\top,\qquad s(\mathcal{B})=0.6\tanh(Gz)',
'knots':r'r_j=c+(h-0.12)\odot\tanh(\eta_j)+s(\mathcal{B})\sin(j\pi/4),\quad j=1,\ldots,4',
'feedback':r'u_{\mathrm{nom}}=\mathrm{clip}\left(\ddot{r}+1.2(r-p)+1.8(\dot{r}-v)+\gamma v-\mu_{\mathcal{B}}(p),\ \pm u_{\max}\right)',
'forward':r'\alpha_i=1-\beta_i,\quad\bar{\alpha}_i=\prod_{j=1}^{i}\alpha_j,\quad y_D=\sqrt{\bar{\alpha}_D}y_{\mathrm{init}}+\sqrt{1-\bar{\alpha}_D}\epsilon',
'proposal':r'Y_k=\frac{y_i}{\sqrt{\bar{\alpha}_i}}+\sqrt{\frac{1-\bar{\alpha}_i}{\bar{\alpha}_i}}\epsilon_k,\quad\epsilon_k\sim\mathcal{N}(0,I)',
'weights':r'w_k=\frac{\exp[-(\widehat{J}(Y_k)-J_{\min})/\lambda]}{\sum_{r\in\mathcal{V}}\exp[-(\widehat{J}(Y_r)-J_{\min})/\lambda]},\quad k\in\mathcal{V}',
'score':r'\widehat{s}_i=\frac{-y_i+\sqrt{\bar{\alpha}_i}\sum_k w_kY_k}{1-\bar{\alpha}_i},\qquad y_{i-1}=\frac{y_i+(1-\bar{\alpha}_i)\widehat{s}_i}{\sqrt{\alpha_i}}',
'physical':r'd(p)=1.8\exp\left[-\frac{1}{2}\left(\left(\frac{p^\top t}{1.1}\right)^4+\left(\frac{p^\top n}{0.35}\right)^2\right)\right](t+0.2n)',
}
for name,eq in EQS.items():
 fig=plt.figure(figsize=(8,.5));fig.text(.01,.5,'$'+eq+'$',fontsize=13,va='center',color='#183447');fig.savefig(OUT/f'eq_{name}.png',dpi=360,bbox_inches='tight',pad_inches=.025,transparent=True);plt.close(fig)
summary={'audits':audit,'denoising_period':10,'denoising_trace':trace.tolist(),'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'equations':list(EQS)}
(OUT/'figure_manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Assets:',OUT)
