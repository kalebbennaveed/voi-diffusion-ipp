"""Export comparable recorded learning curves; never extrapolate stopped runs."""
from pathlib import Path
import json,gzip,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/task_learning_20261105'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--latest-only',action='store_true');parser.add_argument('--final-only',action='store_true');args=parser.parse_args()
    detailed=args.latest_only or args.final_only
    definitions=[
        ('matched','baseline1','Global waypoints','#276b9c'),
        ('matched','mask','Deficit-weighted waypoints','#dd8d22'),
        ('matched','correlated','Correlation-weighted waypoints','#8a64a3'),
        ('matched','routes','Route-weighted waypoints','#bf4a45'),
        ('information_frontier','mask','Filtered sensing trajectories','#79563a'),
        ('reachable','baseline1','Reachable global acquisition','#1b8a8e'),
        ('reachable','mask','Reachable deficit acquisition','#28844d'),
        ('global_goals','baseline1','Persistent global information goals','#2752d2'),
        ('global_goals','mask','Persistent deficit-weighted goals','#c82f9b'),
        ('adaptive_routes','baseline1','Measured-route global learning','#555ca3'),
        ('adaptive_routes','mask','Measured-route deficit learning','#a37940'),
        ('diversified','baseline1','Alternative-route global learning','#125962'),
        ('diversified','mask','Alternative-route deficit learning','#bd1670'),
        ('route_precision','routeprecision','Route precision → MBD','#7441a2'),
        ('model_support','routeprecision','MBD with wind uncertainty checks','#008272'),
            ('adaptive_information','routeprecision','MBD with targeted wind learning','#b38c1d'),
            ('opportunity_information','routeprecision','Lower-cost opportunity learning','#c1392b'),
        ('../full_learning_20261104/learned','baseline1','Measured-map MBD control','#263746')]
    if args.latest_only:
        definitions=[('diversified','baseline1','Baseline 1: learn everywhere','#276b9c'),
            ('diversified','mask','Clarity-weighted learning','#d17d17'),
            ('route_precision','routeprecision','Clarity weighting + route handoff to MBD','#7441a2'),
            ('model_support','routeprecision','MBD with wind uncertainty checks','#008272'),
            ('adaptive_information','routeprecision','MBD with targeted wind learning','#b38c1d'),
            ('opportunity_information','routeprecision','Lower-cost opportunity learning','#c1392b'),
            ('../full_learning_20261104/learned','baseline1','Measured-map MBD (unrestricted initial data)','#56616b')]
    if args.final_only:
        definitions=[('final_baseline','baseline1','Baseline 1: learn everywhere','#276b9c'),
                     ('final_targeted','routeprecision','Reference-weighted relearning + MBD','#888888'),
                     ('final_clarity','routeprecision','Clarity-weighted relearning + MBD','#bd5c28')]
    rows=4 if detailed else 3
    fig,axes=plt.subplots(rows,2,figsize=(12,12 if detailed else 10),sharex=True,layout='constrained')
    used={};records=[];execution=[]
    for folder,method,label,color in definitions:
        for scenario in (1,2):
            p=OUT/folder/f's{scenario}-center-{method}.json.gz'
            if not p.exists():continue
            r=json.load(gzip.open(p));b=r['belief_frames'];s=np.asarray(r['samples']);sm=r['summary']
            t=np.array([f['t'] for f in b]);error=np.array([f['rmse_m_s'] for f in b]);uncert=np.sqrt(2*np.array([f['mean_variance'] for f in b]))
            style='-' if sm['completed'] else '--'
            series=[(0,t,error),(1,s[:,0],s[:,15]),(2,s[:,0],s[:,16]),(3,s[:,0],s[:,14])] if detailed else [(0,t,error),(1,t,uncert),(2,s[:,0],s[:,14])]
            for row,xx,yy in series:
                line,=axes[row,scenario-1].plot(xx,yy,color=color,lw=1.6,ls=style,label=label)
                if sm['contact']:axes[row,scenario-1].plot(xx[-1],yy[-1],marker='x',ms=6,color=color)
                used[label]=line
            records.append(dict(label=label,scenario=scenario,source=str(p.relative_to(ROOT)),
                recorded_s=sm['simulated_seconds'],complete_300=sm['completed'] and sm['simulated_seconds']==300,
                contact=sm['contact'],gate_time_s=sm['gate_time_s']))
            if args.final_only:
                dispatch=list({plan['t']:plan for plan in r['plans']}.values())
                segments=[]
                for i,plan in enumerate(dispatch):
                    end=dispatch[i+1]['t'] if i+1<len(dispatch) else sm['simulated_seconds']
                    phase=('Learning' if plan['phase']=='learning' else
                           'Filtered hold' if 'execution_fallback' in plan or plan.get('validation',{}).get('no_feasible') else 'Accepted MBD')
                    if end>plan['t']:segments.append((plan['t'],end-plan['t'],phase))
                execution.append((scenario,label,segments,sm['gate_time_s']))
    labels=['Whole-map wind RMSE (m/s)','Mean clarity in Region A','Mean clarity in Region B','Accumulated clarity deficit'] if detailed else ['Whole-map wind RMSE (m/s)','Posterior vector RMS uncertainty (m/s)','Accumulated clarity deficit']
    for row in range(rows):
        for col in range(2):
            ax=axes[row,col];ax.set_xlim(0,300);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False)
            if col==0:ax.set_ylabel(labels[row])
            if row==0:ax.set_title(f'Scenario {col+1}',loc='left',fontweight='bold')
            if row==rows-1:ax.set_xlabel('Executed time (s)')
            if detailed and row in (1,2):ax.set_ylim(0,1);ax.axhline(.7,c='#81919d',ls=':',lw=1)
    axes[0,0].axhline(.2,c='#81919d',ls=':',lw=1);axes[0,1].axhline(.2,c='#81919d',ls=':',lw=1)
    fig.suptitle('Recorded wind learning and monitoring\nDashed: incomplete recording · ×: physical contact · no extrapolation',fontsize=13)
    fig.legend(used.values(),used.keys(),loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    name='final-comparison' if args.final_only else 'latest-comparison' if args.latest_only else 'learning-comparison'
    fig.savefig(OUT/f'{name}.png',dpi=170)
    fig.savefig(OUT/f'{name}.pdf')
    (OUT/f'{name}-provenance.json').write_text(json.dumps(records,indent=2))
    if args.final_only:
        from matplotlib.patches import Patch
        colors={'Learning':'#d49a3e','Accepted MBD':'#187c83','Filtered hold':'#83919b'}
        phasefig,ax=plt.subplots(figsize=(12,5),layout='constrained')
        ordered=sorted(execution,key=lambda row:row[0]);labels=[]
        for row,(scenario,label,segments,gate) in enumerate(ordered):
            for phase,color in colors.items():
                ax.broken_barh([(start,length) for start,length,p in segments if p==phase],(row-.32,.64),facecolors=color)
            if gate is not None:ax.plot(gate,row,'|',color='black',ms=15,mew=1)
            labels.append(f'S{scenario} · {label}')
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,300)
        ax.set_xlabel('Executed time (s)');ax.set_title('Which controller actually drove the robot?\nBlack tick: handoff eligibility; it does not imply MBD execution',loc='left')
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right','left']].set_visible(False)
        phasefig.legend(handles=[Patch(facecolor=c,label=p) for p,c in colors.items()],loc='outside lower center',ncol=3,frameon=False)
        phasefig.savefig(OUT/'final-execution-phases.png',dpi=170)
    print(OUT/f'{name}.png')


if __name__=='__main__':main()
