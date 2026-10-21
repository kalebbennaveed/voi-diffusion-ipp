"""Render archived monitoring evidence for Quarto; never runs or alters a planner."""
from pathlib import Path
import os,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/presentation/wind_aware_mbd/assets';OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/slides/mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle,FancyArrowPatch
from PIL import Image
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#00274c','text.color':'#00274c','axes.titleweight':'bold','svg.fonttype':'none','savefig.facecolor':'white'})
C={'proposed':'#008653','current_belief':'#2474bf','wind_blind_planner':'#d77717'}
N={'proposed':'Proposed','current_belief':'Wind-aware','wind_blind_planner':'Wind-blind'}
S={'original':{'proposed':'persistent_two_region_2s_01','current_belief':'persistent_two_region_2s_01','wind_blind_planner':'persistent_two_region_safety_only_01'},'reversed':dict.fromkeys(C,'reversed_wind_01'),'open':dict.fromkeys(['proposed','current_belief'],'open_passage_wind_01')}
DATA={};manifest={'source':'Archived development missions; seed 8101','methods':N,'scenarios':{}}
for case,methods in S.items():
 DATA[case]={};manifest['scenarios'][case]={}
 for m,folder in methods.items():
  p=ROOT/'results/budgeted_missions'/folder/(m+'.npz');z=dict(np.load(p));meta=json.loads(p.with_suffix('.json').read_text());audit=json.loads(p.with_suffix('.audit.json').read_text());dt=meta['metadata']['execution_dt']
  instantaneous=np.maximum(0,z['targets']-z['q']).mean(axis=1)
  cost=np.r_[0,np.cumsum((instantaneous[1:]+instantaneous[:-1])*.5*dt)]
  assert abs(cost[-1]-audit['integrated_whole_grid_deficit'])<3e-5
  z.update(meta=meta,audit=audit,t=np.arange(len(z['x']))*dt,cost=cost);DATA[case][m]=z
  manifest['scenarios'][case][m]={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cost':audit['integrated_whole_grid_deficit'],'contacts':audit['contact_intervals'],'minimum_clearance_m':audit['minimum_dense_clearance'],'transfers':sum(q['count'] for q in audit['directional_routes'].values())}
 for z in DATA[case].values():
  ref=DATA[case]['proposed']
  for k in ['cells','targets','obstacles','bounds','physical_wind','initial_observations','initial_observation_positions']:
   assert np.array_equal(z[k],ref[k]),(case,k)
assert np.allclose(DATA['original']['proposed']['physical_wind'],-DATA['reversed']['proposed']['physical_wind'])
(OUT.parent/'results_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def shape(z,field):
 xx=np.unique(z['cells'][:,0]);yy=np.unique(z['cells'][:,1]);return xx,yy,np.asarray(field).reshape(len(yy),len(xx))

def base(ax,z,wind=True,labels=True):
 ax.set_xlim(-2.7,2.7);ax.set_ylim(-2.1,2.1);ax.set_aspect('equal');ax.set_xticks([-2,0,2]);ax.set_yticks([-2,0,2]);ax.tick_params(labelsize=11);ax.set_facecolor('#f5f7f9')
 for cx,cy,name in [(1.5,.9,'A'),(-1.5,-.9,'B')]:
  ax.add_patch(Rectangle((cx-.7,cy-.6),1.4,1.2,fill=False,ec='#00274c',lw=1.3,zorder=7))
  if labels:ax.text(cx,cy+.73,name,fontsize=18,fontweight='bold',ha='center',zorder=15)
 for x,y,r in z['obstacles']:ax.add_patch(Circle((x,y),r,color='#687885',zorder=9))
 if wind:
  xy=z['cells'][::3];v=z['physical_wind'][::3];keep=np.linalg.norm(v,axis=1)>.07
  xy=xy[keep];v=v[keep]
  ax.quiver(xy[:,0],xy[:,1],v[:,0],v[:,1],color='#586d80',angles='xy',scale_units='xy',scale=4.5,width=.004,alpha=.62,zorder=5)

def save(fig,name):
 for ext in ['svg','png']:fig.savefig(OUT/(name+'.'+ext),dpi=170,bbox_inches='tight')
 plt.close(fig)

# Two real physical fields, exactly negated.
fig,axs=plt.subplots(1,2,figsize=(12.4,4.7))
for ax,case,title in zip(axs,['original','reversed'],['Original passage wind','Exactly reversed wind']):
 z=DATA[case]['proposed'];base(ax,z);ax.set_title(title,fontsize=19,pad=16);ax.plot(*z['x'][0,:2],marker='o',ms=9,color=C['proposed'],zorder=15);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
fig.tight_layout();save(fig,'scenario_fields')

# A monitoring snapshot: use actual q at t=30 s and show both uncertainties separately.
fig,axs=plt.subplots(1,2,figsize=(12,4.8));z=DATA['original']['proposed'];idx=600
for ax,field,title,cmap,vmax in [(axs[0],np.maximum(0,z['targets']-z['q'][idx]),'Monitoring clarity deficit','YlOrRd',.7),(axs[1],z['std'][15],'Wind uncertainty (POAM standard deviation)','Blues',.8)]:
 base(ax,z,wind=False);xx,yy,arr=shape(z,field);im=ax.imshow(arr,origin='lower',extent=[-2.7,2.7,-2.1,2.1],cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest',zorder=1);ax.plot(z['x'][:idx+1,0],z['x'][:idx+1,1],color=C['proposed'],lw=2);ax.plot(*z['x'][idx,:2],marker='o',color='#00274c',ms=6);ax.set_title(title,fontsize=17);fig.colorbar(im,ax=ax,shrink=.7,pad=.025)
fig.tight_layout();save(fig,'two_uncertainties')

# POAM uncertainty snapshots at actual observation boundaries.
fig,axs=plt.subplots(1,3,figsize=(14,4.2));z=DATA['original']['proposed']
for ax,t in zip(axs,[0,20,60]):
 base(ax,z,wind=False);_,_,f=shape(z,z['std'][t//2]);im=ax.imshow(f,origin='lower',extent=[-2.7,2.7,-2.1,2.1],cmap='Blues',vmin=0,vmax=.8,zorder=1);pts=z['observation_positions'][:t*5];ax.scatter(pts[:,0],pts[:,1],s=2,c=C['proposed']);ax.scatter(z['initial_observation_positions'][:,0],z['initial_observation_positions'][:,1],s=5,c='#9f7636',alpha=.7);ax.set_title(f't = {t} s',fontsize=19)
fig.subplots_adjust(wspace=.22,right=.91);cax=fig.add_axes([.935,.22,.014,.58]);fig.colorbar(im,cax=cax,label='Wind standard deviation (m/s²)');save(fig,'poam_evolution')

# Audited cost curves. Trapezoid over the recorded full-grid clarity.
for case in S:
 fig,ax=plt.subplots(figsize=(10,5.2))
 for m,z in DATA[case].items():ax.plot(z['t'],z['cost'],lw=3,color=C[m],label=f"{N[m]}  {z['audit']['integrated_whole_grid_deficit']:.3f}")
 ax.set_xlim(0,120);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.set_xlabel('Mission time (s)');ax.set_ylabel('Cumulative clarity deficit (s)');ax.legend(loc='upper left',frameon=False,fontsize=16);fig.tight_layout();save(fig,case+'_cost')

# Recorded control intervention: the correction norm is already archived.
fig,axs=plt.subplots(1,2,figsize=(12,4.6));z=DATA['original']['proposed'];a,b=1600,1800
base(axs[0],z);axs[0].plot(z['x'][a:b+1,0],z['x'][a:b+1,1],color=C['proposed'],lw=3);axs[0].plot(*z['x'][a,:2],marker='o',color=C['proposed']);axs[0].set_title('Executed motion, 80–90 s',fontsize=18)
axs[1].plot(z['t'][a:b],z['correction'][a:b],color='#a93f80',lw=2.5);axs[1].set_xlabel('Mission time (s)');axs[1].set_ylabel('Control correction norm (m/s²)');axs[1].grid(alpha=.2);axs[1].set_title('RPCBF intervention',fontsize=18);fig.tight_layout();save(fig,'rpcbf_intervention')

# Animated synchronized evidence: 120 physical seconds in 30 playback seconds.
for case in S:
 methods=list(DATA[case]);fig,axs=plt.subplots(1,len(methods),figsize=(15.0,5.1),squeeze=False);axs=axs[0]
 artists=[]
 for ax,m in zip(axs,methods):
  z=DATA[case][m];base(ax,z);xx,yy,arr=shape(z,np.maximum(0,z['targets']-z['q'][0]));im=ax.imshow(arr,origin='lower',extent=[-2.7,2.7,-2.1,2.1],cmap='YlOrRd',vmin=0,vmax=.7,zorder=1)
  full,=ax.plot([],[],color=C[m],lw=1,alpha=.17,zorder=10);tail,=ax.plot([],[],color=C[m],lw=2.7,zorder=11);dot,=ax.plot([],[],marker='o',ms=8,color=C[m],mec='white',mew=1.5,zorder=14)
  foot=Circle((0,0),.5*np.sqrt(2*np.log(2)),fc=(1,.1,.1,.12),ec=(.8,.05,.05,.6),lw=1.5,zorder=12);ax.add_patch(foot);ax.set_title(N[m],color=C[m],fontsize=23,pad=14)
  txt=ax.text(.5,-.12,'',transform=ax.transAxes,ha='center',fontsize=15);artists.append((im,full,tail,dot,foot,txt))
 fig.subplots_adjust(left=.035,right=.965,top=.85,bottom=.18,wspace=.18)
 clock=fig.text(.5,.99,'',ha='center',va='top',fontsize=19,fontweight='bold')
 fig.text(.5,.025,'Yellow → red: clarity deficit    •    Red disk: 50% sensing footprint    •    Arrows: true wind',ha='center',fontsize=13,color='#465c6f')
 frames=[]
 for idx in range(0,2401,10):
  clock.set_text(f't = {idx*.05:5.1f} s / 120 s     |     4× playback')
  for ax,m,arts in zip(axs,methods,artists):
   z=DATA[case][m];im,full,tail,dot,foot,txt=arts;_,_,v=shape(z,np.maximum(0,z['targets']-z['q'][idx]));im.set_data(v);full.set_data(z['x'][:idx+1,0],z['x'][:idx+1,1]);tail.set_data(z['x'][max(0,idx-240):idx+1,0],z['x'][max(0,idx-240):idx+1,1]);pos=z['x'][idx,:2];dot.set_data([pos[0]],[pos[1]]);foot.center=pos;txt.set_text(f'Cumulative deficit  {z["cost"][idx]:.3f}')
  fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert('P',palette=Image.Palette.ADAPTIVE,colors=128);frames.append(im)
  if idx==1200:im.convert('RGB').save(OUT/(case+'_poster.png'))
 # pause the last state for two seconds
 frames[0].save(OUT/(case+'.gif'),save_all=True,append_images=frames[1:],duration=[125]*240+[2000],loop=0,optimize=False,disposal=2)
 plt.close(fig);print(case,'gif', (OUT/(case+'.gif')).stat().st_size,flush=True)
print('assets complete')
