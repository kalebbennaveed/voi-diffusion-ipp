"""Recorded phases and performance; failed runs end at their actual stop."""
import gzip,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]/'output/path_mask_20261105'
report=json.loads((ROOT/'safety-aware-results.json').read_text())
scenarios=sorted(set(r['scenario'] for r in report['runs']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(len(scenarios),3,figsize=(15,4.2*len(scenarios)),squeeze=False)
colors={'baseline1':'#a54c15','pathmask':'#096b83'}
names={'baseline1':'Baseline 1','pathmask':'Proposed'}
for row,scenario in enumerate(scenarios):
    selected=sorted([r for r in report['runs'] if r['scenario']==scenario],key=lambda r:r['method']!='baseline1')
    maximum=max(r['planned_seconds'] for r in selected)
    phase,cost,clarity=axes[row]
    phase.set_title(f'Scenario {scenario} · actual execution phases',loc='left',weight='bold')
    cost.set_title('Accumulated monitoring cost',loc='left',weight='bold')
    clarity.set_title('Clarity in monitoring regions',loc='left',weight='bold')
    for i,summary in enumerate(selected):
        with gzip.open(summary['source'],'rt') as f:r=json.load(f)
        a=np.asarray(r['samples']);method=summary['method'];color=colors[method];label=names[method]
        learning=summary['survey_duration_s'];monitoring=summary['monitoring_duration_s'];end=summary['simulated_seconds']
        phase.barh(i,learning,height=.5,color='#e9b071',edgecolor='white',label='Wind learning' if i==0 else None)
        phase.barh(i,monitoring,left=learning,height=.5,color='#699faf',edgecolor='white',label='MBD' if i==0 else None)
        if summary['contact']:
            phase.plot(summary['contact']['time_s'],i,'x',color='#af1d22',ms=9,mew=2)
        if end<summary['planned_seconds']:
            phase.barh(i,summary['planned_seconds']-end,left=end,height=.5,facecolor='none',edgecolor='#c3c9cc',hatch='///')
        status='contact' if summary['contact'] else 'complete' if summary['completed'] else 'partial'
        phase.text(0,i+.38,f'{learning:g} s learning + {monitoring:g} s MBD · {status}',fontsize=8.5,color='#33434b')
        cost.plot(a[:,0],a[:,14],color=color,label=label,lw=1.8)
        if summary['contact']:cost.plot(end,a[-1,14],'x',color='#af1d22',ms=8,mew=2)
        if summary['gate_time_s'] is not None:cost.axvline(summary['gate_time_s'],color=color,ls=':',lw=1,alpha=.7)
        clarity.plot(a[:,0],a[:,15],color=color,lw=1.2,label=label+' · A')
        clarity.plot(a[:,0],a[:,16],color=color,lw=1.1,ls='--',label=label+' · B')
    phase.set_yticks(range(len(selected)),[names[r['method']] for r in selected]);phase.set_ylim(len(selected)-.35,-.5)
    phase.set_xlim(0,maximum);phase.set_xlabel('Mission time (s)')
    cost.set_xlabel('Mission time (s)');cost.set_ylabel('Integrated mean clarity deficit');cost.legend(fontsize=8,frameon=False)
    clarity.axhline(.7,color='#68757b',ls=':',lw=1);clarity.set_ylim(0,1);clarity.set_ylabel('Mean clarity q')
    clarity.set_xlabel('Mission time (s)');clarity.legend(fontsize=7.5,ncol=2,frameon=False,loc='upper right')
    for ax in (cost,clarity):ax.set_xlim(0,maximum);ax.grid(alpha=.16)
fig.suptitle('Global wind learning versus candidate-path learning',x=.07,ha='left',fontsize=17,weight='bold')
fig.legend(handles=[Patch(facecolor='#e9b071',label='Wind learning'),Patch(facecolor='#699faf',label='MBD execution')],
    loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False,fontsize=10)
fig.text(.07,.012,'All recorded center-in-B trials are shown. Red ×: contact; hatched phase: unexecuted time. '
    'Costs stop with the recording; an early failed run is not a lower-cost completed mission. Safety-aware control, no safety guarantee.',fontsize=9,color='#44535b')
fig.tight_layout(rect=(0,.055,1,.94))
for extension in ('png','pdf'):fig.savefig(ROOT/f'safety-aware-comparison.{extension}',dpi=170,bbox_inches='tight')
print(str(ROOT/'safety-aware-comparison.png'))
