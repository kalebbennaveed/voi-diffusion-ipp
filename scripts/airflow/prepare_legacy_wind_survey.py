"""Explicit synthetic noisy survey for the labelled pre-measured experiment.

This is an initial-knowledge condition, not an oracle planning interface. Both
methods receive the same saved posterior. Cold-start runs use no such file.
"""
import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as legacy
import monitoring_mbd as m
from voi_diffusion_ipp.wind.poam import POAM

def prepare(output,site):
    with jax.experimental.disable_x64():
        cfg=legacy.Settings();env,_=m.environment(legacy.physical_settings(cfg))
        locations=np.asarray(env.cells)
        for scenario in (1,2):
            truth=m.physical.load_wind(scenario,site)
            measurements=truth.at(jnp.asarray(locations))+np.random.default_rng(9901).normal(size=(len(locations),2))*.1
            model=POAM.from_data(locations,measurements,jax.random.PRNGKey(731),128,train_steps=0,noise_std=.1,prior_std=1.8,spectral_count=32,prior_support=np.asarray(legacy.POINTS)[::3])
            belief=model.planning_belief();output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output/f'legacy_survey_s{scenario}.npz',positions=locations,measurements=np.asarray(measurements),**{f'belief_leaf_{i:03d}':np.asarray(v) for i,v in enumerate(jax.tree_util.tree_leaves(belief))})
            print(f'Scenario {scenario}: {len(locations)} noisy observations, 128 inducing points',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--site',type=Path,default=m.physical.SITE);args=p.parse_args();prepare(args.output,args.site)
