"""Teaching figures and synchronized archived replay; no planner is executed."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/presentation/wind_aware_mbd/assets'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/slides/mplconfig'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
from PIL import Image
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#00274c','text.color':'#00274c','svg.fonttype':'none','savefig.facecolor':'white'})
N='#00274c';BLUE='#2474bf';PURPLE='#9b4682';GREEN='#008653';ORANGE='#d77717'
C={'proposed':GREEN,'current_belief':BLUE,'wind_blind_planner':ORANGE};names={'proposed':'Proposed','current_belief':'Wind-aware','wind_blind_planner':'Wind-blind'}
folders={'original':{'proposed':'persistent_two_region_2s_01','current_belief':'persistent_two_region_2s_01','wind_blind_planner':'persistent_two_region_safety_only_01'},'reversed':dict.fromkeys(C,'reversed_wind_01'),'open':dict.fromkeys(['proposed','current_belief'],'open_passage_wind_01')}
D={}
for case,methods in folders.items():
 D[case]={}
 for m,folder in methods.items():
  z=dict(np.load(ROOT/'results/budgeted_missions'/folder/(m+'.npz')));v=np.maximum(0,z['targets']-z['q']).mean(1);z['cost']=np.r_[0,np.cumsum((v[1:]+v[:-1])*.025)];D[case][m]=z

def field(z,f):return np.asarray(f).reshape(len(np.unique(z['cells'][:,1])),len(np.unique(z['cells'][:,0])))
def world(ax,z,wind=True,labels=True):
 ax.set_xlim(-2.7,2.7);ax.set_ylim(-2.1,2.1);ax.set_aspect('equal');ax.set_xticks([-2,0,2]);ax.set_yticks([-2,0,2]);ax.tick_params(labelsize=10);ax.set_facecolor('#f7fafc')
 for x,y,label in [(1.5,.9,'A'),(-1.5,-.9,'B')]:
  ax.add_patch(Rectangle((x-.7,y-.6),1.4,1.2,fill=False,ec=N,lw=1.3,zorder=8))
  if labels:ax.text(x,y+.71,label,ha='center',fontsize=16,fontweight='bold',zorder=16)
 for x,y,r in z['obstacles']:ax.add_patch(Circle((x,y),r,color='#687885',zorder=9))
 if wind:
  xy=z['cells'][::5];v=z['physical_wind'][::5];keep=np.linalg.norm(v,axis=1)>.15;xy=xy[keep];v=v[keep]
  ax.quiver(xy[:,0],xy[:,1],v[:,0],v[:,1],color=N,angles='xy',scale_units='xy',scale=3.0,width=.0055,alpha=.8,zorder=5)

def save(fig,name):
 for ext in ['svg','png']:fig.savefig(OUT/(name+'.'+ext),dpi=160,bbox_inches='tight')
 plt.close(fig)

z=D['original']['proposed']
fig,axs=plt.subplots(1,2,figsize=(11.5,4.4));t=np.linspace(0,1,100)
for ax in axs:world(ax,z);ax.plot([1.5,-1.5],[.9,-.9],color=ORANGE,lw=3,ls='--',zorder=11);ax.plot(1.5,.9,'o',color=N,ms=8,zorder=15)
axs[0].set_title('Nominally informative route',fontsize=18,pad=18);axs[1].set_title('Wind changes delivered sensing',fontsize=18,pad=18)
x=(1-t)**3*1.5+3*(1-t)**2*t*3.0+3*(1-t)*t*t*(-2)+t**3*(-1.5)
y=(1-t)**3*.9+3*(1-t)**2*t*(-2.7)+3*(1-t)*t*t*(-2.7)+t**3*(-.9)
axs[1].plot(x,y,color=BLUE,lw=3.7,zorder=12);axs[1].plot(x[35],y[35],'o',color=BLUE,ms=8,zorder=16);axs[1].add_patch(Circle((x[35],y[35]),.59,color='#ef5555',alpha=.18,zorder=13))
axs[0].annotate('Wind direction',xy=(.2,.1),xytext=(-2.5,1.65),arrowprops={'arrowstyle':'->','color':N},fontsize=14)
fig.subplots_adjust(wspace=.23,bottom=.14);fig.text(.5,.01,'Dashed orange: nominal route     Blue: possible executed motion     Arrows: wind',ha='center',fontsize=13);save(fig,'motivation_wind')

# Clarity is a normalized inverse variance, with the actual target marked.
fig,ax=plt.subplots(figsize=(6.4,3.3));P=np.linspace(0,5,200);ax.plot(P,1/(1+P),color=BLUE,lw=3);ax.axhline(.7,color=ORANGE,ls='--',lw=2);ax.text(3.15,.75,'Target clarity = 0.7',fontsize=14,color=ORANGE);ax.set_xlabel('Normalized estimation variance');ax.set_ylabel('Clarity');ax.set_ylim(0,1.05);ax.set_xlim(0,5);ax.grid(alpha=.15);fig.tight_layout();save(fig,'clarity_definition')

# Teaching example: smoothing a target density. This is not a mission result.
y=np.linspace(-3,3,700);cost=.10*(y*y-2)**2+.15*(y+1.5)**2;target=np.exp(-cost/.35);target/=np.trapezoid(target,y)
fig,axs=plt.subplots(1,2,figsize=(11,3.35));axs[0].plot(y,cost,color=ORANGE,lw=3);axs[0].set_title('Evaluate the objective',fontsize=17);axs[0].set_ylabel('Cost');axs[0].set_xlabel('One decision coordinate')
for ab,label,c in [(1,'Target',N),(.65,'Less noise',BLUE),(.12,'More noise',PURPLE)]:
 if ab==1:p=target
 else:
  kernel=np.exp(-.5*(y[:,None]-np.sqrt(ab)*y[None,:])**2/(1-ab))/np.sqrt(2*np.pi*(1-ab));p=np.trapezoid(kernel*target[None,:],y,axis=1)
 axs[1].plot(y,p,lw=2.8,label=label,color=c)
axs[1].set_title('Noise smooths the search density',fontsize=17);axs[1].set_xlabel('Noisy decision coordinate');axs[1].set_ylabel('Density');axs[1].legend(frameon=False,fontsize=12)
for ax in axs:ax.grid(alpha=.15)
fig.tight_layout();save(fig,'mbd_density')

fig,ax=plt.subplots(figsize=(5.4,3.4));x=np.array([-1.1,-.5,.05,.5,1.1,1.5]);v=np.array([.05,.09,.21,.34,.22,.09]);ax.axhline(0,color='#bbc8d5');ax.scatter(x,np.zeros_like(x),s=40+1700*v,c=v,cmap='Blues',vmin=0,vmax=.4,edgecolor=BLUE,zorder=3);avg=(x*v).sum();ax.axvline(avg,color=PURPLE,lw=2.3);ax.annotate('Weighted decision\nvector mean',xy=(avg,.05),xytext=(avg,.55),ha='center',arrowprops={'arrowstyle':'->','color':PURPLE},color=PURPLE,fontsize=14);ax.set_ylim(-.25,.9);ax.set_xlim(-1.5,1.9);ax.set_yticks([]);ax.set_xlabel('One nominal decision coordinate');ax.spines['left'].set_visible(False);fig.tight_layout();save(fig,'mbd_weights')

# Backup constraint over time, illustrating definition rather than safety evidence.
fig,ax=plt.subplots(figsize=(6.1,3.3));t=np.linspace(0,1.2,200)
for i,c in enumerate([BLUE,PURPLE,'#8497a7']):
 v=-.25+.19*np.sin(np.pi*t/1.2)+(i-1)*.045*t
 ax.plot(t,v,color=c,lw=2.4,label=f'Wind sample {i+1}')
ax.axhline(0,color=ORANGE,ls='--',lw=2);ax.set_ylim(-.3,.09);ax.set_xlabel('Design-policy time (s)');ax.set_ylabel('Constraint violation');ax.legend(frameon=False,fontsize=11,loc='lower right');ax.text(.03,.02,'Positive = constraint violation',fontsize=12,color=ORANGE);fig.tight_layout();save(fig,'backup_value')

# Both quantities have fixed shared scales across time and methods.
for case,methods in D.items():
 ms=list(methods);n=len(ms);fig,axs=plt.subplots(2,n,figsize=(15.8,6.65),squeeze=False)
 fig.subplots_adjust(left=.12,right=.90,bottom=.09,top=.88,wspace=.24,hspace=.25)
 artists=[]
 for col,m in enumerate(ms):
  z=methods[m]
  for row in [0,1]:
   ax=axs[row,col];world(ax,z,wind=row==0);f=np.maximum(0,z['targets']-z['q'][0]) if row==0 else z['std'][0]
   im=ax.imshow(field(z,f),origin='lower',extent=[-2.7,2.7,-2.1,2.1],cmap='YlOrRd' if row==0 else 'Blues',vmin=0,vmax=.7 if row==0 else .8,zorder=1)
   full,=ax.plot([],[],color=C[m],lw=.8,alpha=.25,zorder=10);tail,=ax.plot([],[],color=C[m],lw=2.4,zorder=11);dot,=ax.plot([],[],marker='o',ms=6,color=C[m],mec='white',mew=1,zorder=15)
   pts=ax.scatter([],[],s=4,c=PURPLE,zorder=12) if row else None
   footprint=Circle((0,0),.5*np.sqrt(2*np.log(2)),fc=(1,.1,.1,.15),ec=(.8,.05,.05,.65),lw=1,zorder=13) if not row else None
   if footprint:ax.add_patch(footprint)
   if not row:ax.set_title(names[m],color=C[m],fontsize=19,pad=10)
   artists.append((m,row,im,full,tail,dot,pts,footprint))
  txt=fig.text(.12+(.78/n)*(col+.5),.033,'',ha='center',fontsize=12,color=C[m]);artists.append((m,'cost',txt))
 # Place one fixed color bar per quantity, shared by all method panels.
 for row,lab,cmap,vmax,y0 in [(0,'Clarity deficit','YlOrRd',.7,.555),(1,'Wind std. (m/s²)','Blues',.8,.115)]:
  cbax=fig.add_axes([.925,y0,.014,.27]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,vmax),cmap=cmap),cax=cbax);cb.set_label(lab,fontsize=13);cb.ax.tick_params(labelsize=11)
 fig.text(.018,.69,'MONITORING\nDeficit below\ntarget clarity',fontsize=14,ha='left',va='center',color=N,fontweight='bold')
 fig.text(.018,.26,'WIND BELIEF\nLighter blue =\nless uncertainty',fontsize=14,ha='left',va='center',color=N,fontweight='bold')
 clock=fig.text(.5,.985,'',ha='center',va='top',fontsize=18,fontweight='bold')
 frames=[]
 for idx in range(0,2401,10):
  now=idx*.05;clock.set_text(f't = {now:5.1f} / 120 s     •     4× playback')
  for item in artists:
   m,row,*a=item;z=methods[m]
   if row=='cost':
    snap=min(int(now//2),len(z['std'])-1)
    a[0].set_text(f'Cost: {z["cost"][idx]:.3f}  •  Mean wind σ: {z["std"][snap].mean():.3f}')
    continue
   im,full,tail,dot,pts,foot=a
   snap=min(int(now//2),len(z['std'])-1)
   im.set_data(field(z,np.maximum(0,z['targets']-z['q'][idx]) if row==0 else z['std'][snap]))
   full.set_data(z['x'][:idx+1,0],z['x'][:idx+1,1]);tail.set_data(z['x'][max(0,idx-160):idx+1,0],z['x'][max(0,idx-160):idx+1,1]);p=z['x'][idx,:2];dot.set_data([p[0]],[p[1]])
   if foot:foot.center=p
   if pts is not None:pts.set_offsets(z['observation_positions'][:int(now//2)*10])
  fig.canvas.draw();frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert('P',palette=Image.Palette.ADAPTIVE,colors=192);frames.append(frame)
  if idx==1200:frame.convert('RGB').save(OUT/(case+'_poster.png'))
 frames[0].save(OUT/(case+'.gif'),save_all=True,append_images=frames[1:],duration=[125]*240+[2000],loop=0,optimize=False,disposal=2)
 plt.close(fig);print(case,'replay with fixed scales and belief history',flush=True)
print('Revision figures complete',flush=True)
