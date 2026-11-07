"""Render saved GP-only evidence; no fitting, motion, or safety simulation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/gp_repair_20261104'


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'})
    reports={name:json.loads((OUT/name/'report.json').read_text()) for name in
             ('full_coverage','resolved768','resolved768_seed2')}
    colors={'full_coverage':'#b84e27','resolved768':'#1666b5','resolved768_seed2':'#5396c8'}
    labels={'full_coverage':'Original POAM · 128 inducing points',
            'resolved768':'Repaired POAM · 768 inducing points',
            'resolved768_seed2':'Repaired POAM · independent repetition'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for scenario,ax in enumerate(axes,1):
        initial=np.load(OUT/'full_coverage'/f's{scenario}-inputs.npz')['true_at_measurements']
        initial_rmse=float(np.sqrt(np.mean(np.sum(initial**2,axis=1))))
        for name,r in reports.items():
            run=next(s for s in r['reports'] if s['scenario']==scenario)
            h=run['history'];counts=[0]+[p['observations'] for p in h]
            errors=[initial_rmse]+[p['vector_rmse_m_s'] for p in h]
            ax.plot(counts,errors,color=colors[name],lw=2,marker='o',ms=4,
                    ls='--' if name.endswith('seed2') else '-',label=labels[name])
        ax.axhline(.2,color='#495361',lw=1,ls=':',label='0.20 m/s accuracy threshold')
        ax.set(title=f'Scenario {scenario}',xlabel='Actual wind measurements',ylabel='Whole-map vector RMSE (m/s)',
               xlim=(0,2450),ylim=(0,2.65))
        ax.grid(alpha=.18)
    handles,legends=axes[0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='outside lower center',ncol=2,frameon=False,fontsize=9)
    fig.suptitle('GP-only test: unrestricted measurements across the entire environment',fontsize=15)
    fig.savefig(OUT/'full-coverage-rmse.png',dpi=170);plt.close(fig)

    env=json.loads((ROOT/'results/airflow/rmse_learning_20261104/checked300/environment.json').read_text())
    nx,ny=env['grid'];indices=np.array(env['free_indices']);extent=(0,8,0,6)
    fig,axes=plt.subplots(2,4,figsize=(15,8.4),layout='constrained')
    for row,scenario in enumerate((1,2)):
        data=np.load(OUT/'full_coverage'/f's{scenario}-inputs.npz')
        truth=data['true_at_measurements']
        original=np.load(OUT/'full_coverage'/f's{scenario}-online_original.npz')['mean'][:len(indices)]
        repaired=np.load(OUT/'resolved768'/f's{scenario}-online_original.npz')['mean'][:len(indices)]
        for column,(values,title) in enumerate(((np.linalg.norm(truth,axis=1),'True wind'),
                (np.linalg.norm(original,axis=1),'Original POAM estimate'),
                (np.linalg.norm(repaired,axis=1),'Repaired POAM estimate'),
                (np.linalg.norm(repaired-truth,axis=1),'Repaired vector error'))):
            ax=axes[row,column];array=np.full(nx*ny,np.nan);array[indices]=values
            plot=ax.imshow(array.reshape(ny,nx),extent=extent,origin='lower',interpolation='nearest',
                cmap='magma' if column==3 else 'Blues',vmin=0,vmax=1.25 if column==3 else 18)
            for x,y,r in env['obstacles']:ax.add_patch(Circle((x,y),r,color='#7a848d'))
            for region in env['regions']:
                lo,hi=region['bounds'];ax.plot([lo[0],hi[0],hi[0],lo[0],lo[0]],
                    [lo[1],lo[1],hi[1],hi[1],lo[1]],c='#334155',lw=.7)
            ax.set(xlim=(0,8),ylim=(0,6),title=title if row==0 else None,aspect='equal',xlabel='x (m)')
            if column==0:ax.set_ylabel(f'Scenario {scenario}\ny (m)')
            else:ax.set_yticklabels([])
            if column==0:wind_plot=plot
            if column==3:error_plot=plot
    fig.colorbar(wind_plot,ax=axes[:,:3],orientation='horizontal',shrink=.8,label='Wind speed (m/s) · same scale for truth and both GP estimates')
    fig.colorbar(error_plot,ax=axes[:,3],orientation='horizontal',shrink=.9,label='Vector error (m/s)')
    fig.suptitle('Same fixed wind and noisy measurements · no robot constraints or RPCBF',fontsize=15)
    fig.savefig(OUT/'full-coverage-maps.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
