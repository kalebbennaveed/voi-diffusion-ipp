from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
t=np.linspace(0,1,201)
curves=[-.42+.48*np.sin(np.pi*t)*np.exp(-.2*t),-.42+.32*np.sin(np.pi*t),-.42+.7*np.sin(np.pi*t)*np.exp(-.2*t)]
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
for j,(h,c) in enumerate(zip(curves,['#2474bf','#008653','#9b4682'])):
 ax.plot(t,h,color=c,lw=3,label=f'Wind field {j+1}')
 m=np.argmax(h);ax.plot(t[m],h[m],'o',c=c,ms=7)
maxh=max(map(max,curves));ax.axhline(0,color='#00274c',lw=1.7)
ax.hlines(maxh,.0,1,color='#9b4682',ls='--',lw=1.5);ax.text(.72,maxh+.035,r'$\widehat V_a$: largest value',fontsize=16,color='#9b4682')
ax.set(xlim=(0,1),ylim=(-.48,.38),xlabel=r'Time along the design-policy trajectory, $\sigma$',ylabel=r'Constraint value $h_a$',xticks=[0,.5,1],xticklabels=['0','H/2','H'],yticks=[0],yticklabels=['0'])
ax.legend(loc='lower center',ncol=3,fontsize=13);ax.grid(alpha=.12);fig.savefig('output/presentation/wind_aware_mbd/assets/policy_value_traces.svg');plt.close(fig)
