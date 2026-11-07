"""Post-run diagnostics; true wind is used here only to audit saved beliefs."""
import argparse,json
from pathlib import Path
import numpy as np
import monitoring_mbd as m


def summarize(directory,site):
    envdata=json.loads((directory/'environment.json').read_text());free=np.asarray(envdata['free_indices'])
    xx,yy=np.meshgrid((np.arange(64)+.5)*.125,(np.arange(37)+.5)*.125);points=np.c_[xx.ravel(),yy.ravel()]
    rows=[]
    for scenario in (1,2):
        truth=np.asarray(m.physical.load_wind(scenario,site).at(points))[free]
        for method in ('aware','baseline'):
            ident=f's{scenario}-{method}';record=json.loads((directory/(ident+'.json')).read_text());samples=np.asarray(record['samples'])
            frames=[]
            for frame in (record['belief_frames'][0],record['belief_frames'][-1]):
                error=np.asarray(frame['mean'])[free]-truth;std=np.asarray(frame['std'])[free]
                frames.append(dict(time_s=frame['t'],wind_component_rmse_m_s=float(np.sqrt(np.mean(error**2))),mean_component_std_m_s=float(std.mean()),fraction_free_components_within_1_96_std=float(np.mean(np.abs(error)<=1.96*std[:,None]))))
            visits={}
            for region in envdata['regions']:
                lo,hi=np.asarray(region['bounds']);inside=((samples[:,1:3]>=lo)&(samples[:,1:3]<=hi)).all(axis=1)
                visits[region['label']]=dict(first_recorded_entry_s=float(samples[inside,0][0]) if inside.any() else None,in_region_samples=int(inside.sum()),total_20hz_samples=len(samples))
            improvements=sum(p['expected_cost']<min(v for v in p['seed_costs'] if v is not None)-1e-8 for p in record['plans'] if p['valid'])
            rows.append(dict(id=ident,summary=record['summary'],region_visits=visits,plans_improving_initial_cost=improvements,total_plans=len(record['plans']),wind_belief_checks=frames))
    report=dict(scope='Descriptive post-run diagnostics of the recorded comparison. True wind is used only in this audit and was never supplied to either planning objective. Confidence-interval coverage is descriptive for this one field, not a global calibration guarantee.',runs=rows)
    (directory/'comparison-diagnostics.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--site',type=Path,default=m.physical.SITE)
    a=p.parse_args();print(json.dumps(summarize(a.directory,a.site),indent=2))
