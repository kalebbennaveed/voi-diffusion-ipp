"""Time the complete planning pipeline from recorded initial conditions.

Run after mission jobs finish. This is an isolated CPU timing experiment, not
an extra mission or a claim that every reached state meets a deadline.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from voi_diffusion_ipp.experiments.urban_quad import UrbanEnvironment
from voi_diffusion_ipp.quarter.continuous import ControlSettings
from voi_diffusion_ipp.quarter.environment import initial_belief,planning_sampler
from voi_diffusion_ipp.quarter.modes import demonstrations,select_initial_modes
from voi_diffusion_ipp.quarter.suite_engine import SuiteEngine

ROOT=Path(__file__).resolve().parents[1]
jax.config.update('jax_compilation_cache_dir',str(ROOT/'.cache/jax_suite'))
p=argparse.ArgumentParser();p.add_argument('--records',nargs='+',type=Path,required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=[]
for record in a.records:
    d=json.loads(record.read_text());settings=d['config'].copy()
    settings['initial_wind_observations']=tuple(tuple(row) for row in settings['initial_wind_observations'])
    settings['wind_kernel_lengths']=tuple(settings['wind_kernel_lengths'])
    cfg=ControlSettings(**settings);e=d['environment']
    env=UrbanEnvironment(*(jnp.array(e[k],jnp.float32) for k in ['buildings','bounds','cells','targets','process_noise','regions']))
    with np.load(record.parent/d['plans'][0]['audit']) as z:
        x=jnp.asarray(z['initial_state']);q=jnp.asarray(z['clarity']);previous=jnp.asarray(z['initial_parameters'][-1])
    setup=time.perf_counter();belief=initial_belief(cfg);belief.mean.block_until_ready();posterior_setup=time.perf_counter()-setup
    sampler=planning_sampler(cfg);engine=SuiteEngine(cfg);construction_key=jax.random.fold_in(jax.random.PRNGKey(d['seed']),271828)
    for method in ['aware','blind_filter']:
        planning=engine.planner(env,method);samples=[]
        for repeat in range(a.repeats+1):
            key=jax.random.fold_in(jax.random.PRNGKey(d['seed']),repeat+300001);ks,kp=jax.random.split(key)
            start=time.perf_counter();mean,fields=sampler(belief,ks,construction_key);fields.block_until_ready();field_seconds=time.perf_counter()-start
            start_geometry=time.perf_counter();bank,metadata=demonstrations(x,env,cfg,d['seed']);initials,_=select_initial_modes(bank,metadata,x,env,cfg,q);initials=jnp.asarray(initials);geometry_seconds=time.perf_counter()-start_geometry
            start_search=time.perf_counter();result=planning(x,q,mean,fields[:cfg.construction_worlds],fields[cfg.construction_worlds:cfg.construction_worlds+cfg.worlds],fields[-cfg.validation_worlds:],previous,kp,jnp.array(1.),jnp.array(True),initials);result[0].block_until_ready();search_seconds=time.perf_counter()-start_search
            item=dict(fields=field_seconds,geometry=geometry_seconds,search_selection=search_seconds,total=field_seconds+geometry_seconds+search_seconds)
            if repeat:samples.append(item)
            else:warmup=item
        row=dict(case=d['case'],method=method,source=str(record),horizon=cfg.horizon,decision_dt=cfg.dt,prediction_dt=cfg.dt/cfg.prediction_substeps,commit=cfg.commit,sensor_half_angle_degrees=cfg.sensor_half_angle_degrees,sensor_radius_at_3_2m=float(3.2*np.tan(np.deg2rad(cfg.sensor_half_angle_degrees))),posterior_setup_seconds=posterior_setup,first_pipeline_call=warmup,repeats=samples,median_total=float(np.median([s['total'] for s in samples])),max_total=max(s['total'] for s in samples),median_components={k:float(np.median([s[k] for s in samples])) for k in ['fields','geometry','search_selection']})
        rows.append(row);print(method,d['case'],row['median_total'],row['median_components'],flush=True)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'devices':list(map(str,jax.devices())),'jax':jax.__version__,'scope':'isolated repeated planning from saved initial state; no other mission jobs; excludes process import and reporting','rows':rows},indent=2))
