"""Replay every actual-observation update and verify displayed posterior maps."""
import argparse,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import legacy_monitoring as l
import monitoring_mbd as m

def audit(directory,site=m.physical.SITE,survey_directory=None):
    checks=[]
    for scenario in (1,2):
        for method in ('aware','baseline'):
            ident=f's{scenario}-{method}'
            protocol=json.loads((directory/f'{ident}-protocol.json').read_text())
            record=json.loads((directory/f'{ident}.json').read_text())
            assert protocol['execution_belief']['posterior_update_seconds']==.2
            cfg=l.Settings(**protocol['legacy_settings']);env,grid=m.environment(l.physical_settings(cfg))
            b=l.matched_prior(env)
            if 'initial_survey' in protocol:
                fixture=np.load((survey_directory or directory.parent)/protocol['initial_survey']['file'])
                b=jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(b),[jnp.asarray(fixture[k]) for k in sorted(fixture.files) if k.startswith('belief_leaf_')])
            else:
                mean,cov=b.predict(jnp.asarray(env.cells));np.testing.assert_allclose(mean,0,atol=0);np.testing.assert_allclose(np.sqrt(cov[:,0,0]),1.8,rtol=3e-6,atol=1e-6)
            raw=np.load(directory/f'{ident}-execution.npz');rows=raw['observation_rows']
            truth=m.physical.load_wind(scenario,site)
            noise=np.random.default_rng(protocol['seed']).normal(size=(round(record['summary']['planned_seconds']/.2)+1,2))
            expected=np.asarray(truth.at(jnp.asarray(rows[:,1:3])))+.1*noise[np.rint(rows[:,0]/.2).astype(int)]
            np.testing.assert_allclose(rows[:,3:5],expected,atol=1e-12,rtol=0)
            condition=jax.jit(lambda bb,p,y:bb.condition(p,y,relocate=True))
            maximum=0.;cursor=0
            for frame in record['belief_frames']:
                while cursor<len(rows) and rows[cursor,0]<=frame['t']+1e-8:
                    with jax.experimental.disable_x64():
                        b=condition(l.float32(b),jnp.asarray(rows[cursor:cursor+1,1:3],jnp.float32),jnp.asarray(rows[cursor:cursor+1,3:5],jnp.float32))
                    cursor+=1
                mu,cov=b.predict(jnp.asarray(grid['all_cells']));sigma=np.sqrt(np.asarray(cov)[:,0,0])
                maximum=max(maximum,float(np.max(abs(np.asarray(mu)-frame['mean']))),float(np.max(abs(sigma-frame['std']))))
                survey_count=protocol.get('initial_survey',{}).get('measurements',0)
                assert frame['observations']==survey_count+cursor
            assert maximum<=5.01e-6,(ident,maximum)
            changes=sum(not np.array_equal(a['mean'],b['mean']) for a,b in zip(record['belief_frames'],record['belief_frames'][1:]))
            checks.append(dict(id=ident,observations=len(rows),posterior_frames=len(record['belief_frames']),changed_mean_frames=changes,max_map_rounding_error=maximum,initial_survey_measurements=protocol.get('initial_survey',{}).get('measurements',0)))
            print(json.dumps(checks[-1]),flush=True)
    (directory/'belief-audit.json').write_text(json.dumps(dict(checks=checks,scope='Every local measurement verified against the fixed physical field and seeded noise; every online posterior update replayed; every displayed mean and standard-deviation map checked.'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--site',type=Path,default=m.physical.SITE);p.add_argument('--survey-directory',type=Path);a=p.parse_args();audit(a.directory,a.site,a.survey_directory)
