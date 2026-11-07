"""Standalone figures and tables from actual recordings; contact curves are censored."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from package_final_comparison import OUT,DEST

def main():
    m=json.loads((OUT/'comparison-results.json').read_text())
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#a5b1b8','axes.labelcolor':'#20394a','xtick.color':'#617282','ytick.color':'#617282','pdf.fonttype':42})
    fig,axes=plt.subplots(3,1,figsize=(10.5,10.5),layout='constrained')
    rows=[]
    for ax,s in zip(axes,m['scenarios']):
        ax.set_title(f'Scenario {s["id"]}',loc='left',fontweight='bold');ax.set_ylabel('Mean clarity deficit');ax.grid(alpha=.18)
        if not s.get('environment'):
            ax.text(.5,.5,s['status'],ha='center',va='center',transform=ax.transAxes);continue
        for r in s['runs']:
            if 'curve' not in r:continue
            a=np.asarray(r['curve']);ax.plot(a[:,0],a[:,1],color=r['color'],lw=2.3 if r['method']=='proposed' else 1.5,
                ls={'unfiltered':':','filtered':'--','aware':'-.','global':(0,(6,2)),'proposed':'-'}[r['method']],
                label=r['label'].split(' · ')[0]+(' ('+r['status'].lower()+')' if r['status'] not in ('Completed','Contact') else ''))
            if r['learning_completed'] and r['learning_time_s']>0:
                t=r['learning_time_s'];ax.plot(t,np.interp(t,a[:,0],a[:,1]),marker='D',ms=5,color=r['color'])
            if r['contact']:
                ax.plot(*a[-1],marker='x',ms=8,mew=2,color=r['color'])
                ax.annotate(f'contact {r["contact"]["time_s"]:.1f} s',a[-1],xytext=(9,8),textcoords='offset points',fontsize=8,color=r['color'])
            rows.append(dict(scenario=s['id'],method=r['label'],total_recorded_time_s=r['total_time_s'],
                learning_time_s=r['learning_time_s'],learning_completed=r['learning_completed'],
                final_deficit=r['final_clarity_deficit'],common_window_s=s['comparison_window_s'],
                common_window_deficit=r['comparison_deficit'],status=r['status'],
                common_window_mean_deficit=r.get('common_window_mean_deficit'),
                contact_s=r['contact']['time_s'] if r['contact'] else '',source=r['source']))
        ax.set_ylim(bottom=0);ax.set_xlabel('Simulated mission time (s)');ax.legend(loc='upper right',ncol=2,fontsize=8,framealpha=.95)
        if any(r.get('total_time_s',0)>s['comparison_window_s'] for r in s['runs']):ax.axvline(s['comparison_window_s'],color='#a5b1b8',lw=1,ls='--')
    finished=sum(r['status'] in ('Completed','Contact','Learning limit') for s in m['scenarios'] for r in s['runs'])
    title='Monitoring performance across five methods'
    if finished<15:title+=f'\nInterim export: {finished}/15 outcomes finalized'
    fig.suptitle(title,fontsize=17,fontweight='bold',color='#20394a')
    fig.supxlabel('Lower deficit is better. Diamonds: learning ends. Crosses: contact; curves stop.\nGray vertical lines mark the shared comparison windows; longer recordings remain visible.',fontsize=10)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'clarity-deficit.{ext}',dpi=180,bbox_inches='tight');(DEST/f'clarity-deficit.{ext}').write_bytes((OUT/f'clarity-deficit.{ext}').read_bytes())
    plt.close(fig)
    if rows:
        with (OUT/'comparison-table.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        (DEST/'comparison-table.csv').write_bytes((OUT/'comparison-table.csv').read_bytes())
    print(json.dumps(dict(rows=len(rows),figure=str(OUT/'clarity-deficit.pdf'))))
if __name__=='__main__':main()
