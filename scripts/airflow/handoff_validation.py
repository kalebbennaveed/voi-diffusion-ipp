"""Simulation-only validation of an RMSE-qualified survey posterior.

The optimizer and safety controller receive only frozen learned POAM fields.
True wind is evaluated afterwards by the plant to test selected forecasts.
No generated readings are assimilated. Outputs only authorize a phase change;
validation trajectories never become references for real execution.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import full_learning_monitoring as model
import efficient_learning_monitoring as efficient


def forecast_passes(accepted,alive,infeasible,rms,maximum,relative,allow_slack=False,metric='position-and-cost'):
    """Declared experiment stopping check, separate from safety filtering.

    Monitoring-cost validation tests the quantity MBD actually minimizes.
    Position errors remain in the report and are not safety certificates.
    """
    if metric not in ('position-and-cost','monitoring-cost'):raise ValueError(metric)
    finite=np.isfinite([rms,maximum,relative]).all()
    accuracy=relative<=.1 and (metric=='monitoring-cost' or (rms<=.25 and maximum<=.5))
    return bool(accepted and alive and (allow_slack or infeasible==0) and finite and accuracy)


class HandoffValidator:
    definition=('After the declared learning threshold, validate the original 15s MBD at the actual state and at rest in A and B, using current clarity. '
        'Fixed independent random seed; frozen learned posterior; true plant only AFTER optimization. '
        'All three selected forecasts must have position RMS error <= 0.25m, maximum position error <= 0.50m, '
        'monitoring-cost error <= 10%, no true contact and no infeasible execution command. '
        'At least one must travel >= 0.50m. Holds are reported and cannot alone pass. '
        'A failed check is retried no sooner than 20 simulated seconds. This is a simulation-only stopping diagnostic, not a safety proof or whole-map accuracy certificate.')
    def __init__(self,env,cfg,pcfg,basis,spec,allow_validation_slack=False,metric='position-and-cost'):
        self.cfg=cfg;self.env=env;self.spec=spec;self.first=True
        self.allow_validation_slack=allow_validation_slack;self.metric=metric
        if metric not in ('position-and-cost','monitoring-cost'):raise ValueError(metric)
        if metric=='monitoring-cost':
            self.definition=self.definition.replace('position RMS error <= 0.25m, maximum position error <= 0.50m, ', '')
            self.definition+=' Position errors are reported separately; the stopping criterion evaluates task cost rather than requiring close geometric agreement.'
        if allow_validation_slack:
            self.definition=self.definition.replace('no true contact and no infeasible execution command',
                'no true contact; infeasible execution commands are measured and reported, not an automatic veto')
        with jax.experimental.disable_x64():
            self.planner=efficient.optimizer(model.legacy.float32(env),cfg,model.legacy.float32(basis),True,prepare_validation=True,allow_validation_slack=allow_validation_slack)
        self.execute=model.execution_block(env,cfg,pcfg,basis,seconds=cfg.horizon)

    def check(self,x,q,b,cache,tables,truth,seed,details=False):
        start=time.perf_counter();reports=[]
        states=[('actual state',np.asarray(x))]+[('region '+r['label'],np.r_[np.mean(r['bounds'],axis=0),[0.,0.]]) for r in self.spec['regions']]
        for i,(label,state) in enumerate(states):
            xx=jnp.asarray(state);key=jax.random.PRNGKey(seed+857711+i)
            drawn=cache.planning(b,tables,key);jax.block_until_ready(drawn)
            seeds,previous,_=model.legacy.seed_bank(xx,q,self.env,self.cfg,None)
            with jax.experimental.disable_x64():
                theta,cost,search,forecast,v=self.planner(*model.legacy.float32((xx,q,drawn,seeds,previous)),key)
                jax.block_until_ready(theta)
            refs,aa=model.legacy.old.nominal_reference(theta.astype(jnp.float64),xx,self.cfg)
            end,rows=self.execute(xx,q,refs,aa,jnp.array(0.),drawn,truth,jnp.array(True));jax.block_until_ready(end)
            xs,qs,us,nom,slack,correction,feasible,gaps,costs,alive,hits=map(np.asarray,rows)
            actual=np.vstack((state,xs[4::5]));prediction=np.asarray(forecast.x).mean(axis=0)
            error=np.linalg.norm(actual[:,:2]-prediction[:,:2],axis=1)
            rms=float(np.sqrt(np.mean(error**2)));maximum=float(error.max())
            actual_cost=float(costs.sum());predicted_cost=float(np.asarray(forecast.monitor).mean())
            relative=abs(actual_cost-predicted_cost)/max(actual_cost,.05)
            displacement=float(np.max(np.linalg.norm(xs[:,:2]-state[:2],axis=1)))
            accepted=not bool(v['no_feasible'])
            passed=forecast_passes(accepted,bool(end[-1]),int((~feasible).sum()),rms,maximum,relative,
                self.allow_validation_slack,self.metric)
            reports.append(dict(state=label,accepted=accepted,position_rmse_m=rms,max_position_error_m=maximum,
                relative_monitoring_cost_error=relative,predicted_monitoring_cost=predicted_cost,true_monitoring_cost=actual_cost,
                contact=not bool(end[-1]),infeasible_steps=int((~feasible).sum()),minimum_clearance_m=float(gaps.min()),
                displacement_m=displacement,passed=passed))
            if details:
                reports[-1]['selection']={key:np.asarray(v[key]).tolist() for key in
                    ('leader_costs','costs','feasible','safety_feasible','contacts','slack','selected','no_feasible')}
                reports[-1]['candidate_reference_displacement_m']=[float(np.max(np.linalg.norm(
                    np.asarray(model.legacy.old.nominal_reference(candidate.astype(jnp.float64),xx,self.cfg)[0])[:,:2]-state[:2],axis=1)))
                    for candidate in v['candidates']]
        result=dict(passed=all(r['passed'] for r in reports) and any(r['displacement_m']>=.5 for r in reports),
            cases=reports,wall_seconds=time.perf_counter()-start,compilation_included=self.first,
            allow_validation_slack=self.allow_validation_slack,metric=self.metric,definition=self.definition)
        self.first=False;return result
