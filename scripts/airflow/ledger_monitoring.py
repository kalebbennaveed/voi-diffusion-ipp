"""Task-directed covariance reduction with a predicted monitoring-loss ledger.

This is a constrained two-objective planning policy, not an estimate of exact
long-term information value. The MBD equations and physical models are reused.
Route relevance is computed before geometric survey starts, so initialization
and optimization use exactly the same targets. Actual observations alone
update the executed POAM belief.
"""
from dataclasses import replace
from typing import NamedTuple
import jax
import jax.numpy as jnp
import legacy_monitoring as legacy
import survey_monitoring as previous_method
from exploration_monitoring import forecast as coarse_forecast
from survey_forecast import forecast as refined_forecast
from survey_information import prepare_full_targets, full_target_variance_reduction_components
from voi_diffusion_ipp.quarter.multimode import search
from voi_diffusion_ipp.wind.gp import WorldField


class Relevance(NamedTuple):
    weights:jax.Array
    uncertainty:jax.Array
    force:jax.Array
    benefits:jax.Array
    values:jax.Array


def prepare_relevance(env,cfg):
    """One bounded paired task test, reused by seed construction and MBD."""
    hard_cfg=replace(cfg,safety_margin=.12,construction_worlds=8,prediction_filter='hard')
    @jax.jit
    def evaluate(b,x,q,mean,construction,physical,seeds,positions,weights,route_indices):
        known_construction=jnp.broadcast_to(physical,(8,*physical.shape))
        def probe(theta):
            uncertain=coarse_forecast(theta,x,q,mean,construction[:8],physical,env,hard_cfg)
            known=coarse_forecast(theta,x,q,physical,known_construction,physical,env,hard_cfg)
            def metric(r):
                return jnp.array([r.monitor+r.penalty,
                    jnp.mean(jnp.linalg.norm(r.x[:,:2]-r.nominal[:,:2],axis=1)),
                    r.terminated.astype(x.dtype)])
            return metric(uncertain),metric(known)
        uncertain,known=jax.vmap(probe)(seeds)
        benefits,_=previous_method.task_information_benefits(uncertain,known)
        weights=previous_method.route_task_weights(weights,route_indices,benefits)
        winds=jax.vmap(lambda field:WorldField(legacy.BASIS,field).at(positions))(construction)
        weights,force=previous_method.force_relevance(winds,positions,weights)
        targets=prepare_full_targets(b,positions,weights)
        uncertainty=jnp.sqrt(targets.total_variance/jnp.maximum(2*jnp.sum(weights),1e-12))
        return Relevance(weights,uncertainty,force,benefits,
                         previous_method.sampled_route_values(uncertain,known))
    return evaluate


def select_with_ledger(task,commit,gain,safe,credit,enabled):
    """Maximize attainable learning within the available predicted-loss credit.

The comparator is the minimum full-horizon monitoring cost in the same
independently validated candidate bank. The debit is its exact committed-cost
difference, clipped at zero. Near-equal gains are resolved by monitoring cost.
    """
    safe=safe&jnp.isfinite(task)&jnp.isfinite(commit)&jnp.isfinite(gain)
    monitor_index=jnp.argmin(jnp.where(safe,task,jnp.inf))
    loss=jnp.maximum(commit-commit[monitor_index],0.)
    eligible=safe&(loss<=credit+1e-7)
    best_gain=jnp.max(jnp.where(eligible,gain,-jnp.inf))
    learning=enabled&(best_gain>gain[monitor_index]+1e-6)
    tied=eligible&(gain>=best_gain-1e-6)
    survey_index=jnp.argmin(jnp.where(tied,task,jnp.inf))
    index=jnp.where(learning,survey_index,monitor_index)
    return index,monitor_index,loss[index],learning,eligible


def replenish(credit,debit,seconds,rate=.01,capacity=.6):
    """The explicit ledger bounds sum of predicted positive committed losses."""
    return jnp.minimum(capacity,jnp.maximum(credit-debit,0.)+rate*seconds)


class Validated(NamedTuple):
    values:jax.Array
    states:jax.Array
    min_clearance:jax.Array
    max_slack:jax.Array
    prefix_max_slack:jax.Array
    prefix_infeasible_steps:jax.Array


class Result(NamedTuple):
    theta:jax.Array
    score:jax.Array
    expected:jax.Array
    search:object
    index:jax.Array
    task_costs:jax.Array
    information:jax.Array
    contacts:jax.Array
    budget:jax.Array
    initial_budget:jax.Array
    exploring:jax.Array
    route_uncertainty:jax.Array
    route_relevance:jax.Array
    relevance_benefits:jax.Array
    score_normalization:jax.Array
    route_values:jax.Array
    validation_bonuses:jax.Array
    amortization_factor:jax.Array
    information_enabled:jax.Array
    task_index:jax.Array
    candidate_indices:jax.Array
    candidates:jax.Array
    validation:object
    commit_costs:jax.Array
    ledger_debit:jax.Array
    ledger_credit:jax.Array
    ledger_eligible:jax.Array


def optimizer(env,cfg,mode='survey',cost_allowance=1.0):
    del cost_allowance # The old per-call allowance is replaced by explicit credit.
    if mode not in ('survey','passive'):raise ValueError(mode)
    search_cfg=replace(cfg,safety_margin=.06,construction_worlds=2,prediction_filter='soft')
    hard_cfg=replace(cfg,safety_margin=.12,construction_worlds=8,prediction_filter='hard')
    times=jnp.arange(1,round(cfg.horizon/.2)+1)*.2
    indices=jnp.minimum(jnp.floor((times-1e-6)/cfg.dt).astype(int),cfg.steps-1)
    offsets=times-indices*cfg.dt
    @jax.jit
    def plan(b,x,q,mean,construction,worlds,validation,previous,key,seeds,positions,weights,
             route_indices,remaining_seconds,relevance,credit):
        del route_indices,remaining_seconds
        targets=prepare_full_targets(b,positions,weights)
        enabled=(relevance.uncertainty>.30)&(relevance.force>.10)&(jnp.sum(weights)>1e-8)&(mode=='survey')
        def information(r,samples,committed_cost):
            components=full_target_variance_reduction_components(b,samples,targets)
            gain=jnp.clip(components.sum()/jnp.maximum(targets.total_variance,1e-30),0.,1.)
            gain=jnp.where(r.terminated|~enabled,0.,gain)
            # Keep legacy viewer columns0..5; column6 is exact committed cost.
            return jnp.array([r.monitor+r.penalty,gain,r.terminated.astype(x.dtype),
                              gain,r.monitor,r.penalty,committed_cost])
        def coarse_commit(r):
            qq=r.q[:cfg.commit_steps+1]
            deficits=jax.vmap(lambda z:legacy.m.clarity.deficit(z,env.targets))(qq)
            monitoring=cfg.dt*jnp.sum((deficits[:-1]+deficits[1:])*.5)
            xx=r.x[1:cfg.commit_steps+1]
            penalty=cfg.violation_weight*jax.nn.relu(cfg.planning_clearance-jax.vmap(env.clearance)(xx))**2
            penalty+=cfg.speed_penalty*jax.nn.relu(jnp.linalg.norm(xx[:,2:],axis=1)-cfg.speed_max)**2
            return monitoring+cfg.dt*jnp.sum(penalty)
        def measures(theta,fields):
            def one(field):
                r=legacy.rollout(theta,x,q,mean,construction[:2],field,env,search_cfg)
                world=WorldField(legacy.BASIS,field)
                samples=jax.vmap(lambda z,u,h:legacy.m.physical.rk4(z,u,world,h)[:2])(r.x[indices],r.u[indices],offsets)
                return information(r,samples,coarse_commit(r))
            return jax.vmap(one)(fields)
        initials=jnp.r_[seeds,previous[None]]
        values=jax.lax.map(lambda th:measures(th,worlds),initials,batch_size=cfg.batch_size)
        task=values[:,:,0].mean(axis=1);committed=values[:,:,6].mean(axis=1)
        safe=jnp.all(values[:,:,2]==0,axis=1)
        comparator=jnp.argmin(jnp.where(safe,task,jnp.inf))
        initial_budget=committed[comparator]+credit
        def training_scores(values):
            safe=jnp.all(values[:,2]==0)
            allowed=safe&(~enabled|(values[:,6].mean()<=initial_budget+1e-7))
            return jnp.where(allowed,jnp.where(enabled,-values[:,1],values[:,0]),jnp.inf)
        def evaluate(theta):return training_scores(measures(theta,worlds))
        cached=jax.vmap(training_scores)(values)
        result=search(evaluate,seeds,previous,key,cfg,1.,0.,jnp.array(False),initial_field_values=cached)
        candidate_indices,candidates=previous_method.validation_candidates(result,seeds)
        def validate(theta):
            def one(field):
                fine=refined_forecast(theta,x,q,mean,construction[:8],field,env,hard_cfg,fine_seconds=cfg.horizon)
                r=fine.rollout
                return Validated(information(r,fine.full_observation_positions,fine.commit_cost),r.x,
                    r.min_clearance,r.max_slack,fine.prefix_max_slack,fine.prefix_infeasible_steps)
            return jax.vmap(one)(validation)
        checked=jax.lax.map(validate,candidates,batch_size=cfg.validation_batch_size or None)
        values=checked.values
        task=values[:,:,0].mean(axis=1);gain=values[:,:,1].mean(axis=1);committed=values[:,:,6].mean(axis=1)
        safe=jnp.all(values[:,:,2]==0,axis=1)
        index,task_index,debit,exploring,eligible=select_with_ledger(task,committed,gain,safe,credit,enabled)
        score=jnp.where(safe[index],jnp.where(exploring,-gain[index],task[index]),jnp.inf)
        return Result(candidates[index],score,task[index],result,index,task,gain,values[:,:,2],
            committed[task_index]+credit,initial_budget,exploring,relevance.uncertainty,relevance.force,
            relevance.benefits,jnp.array(1.,x.dtype),relevance.values,gain,jnp.array(0.,x.dtype),
            enabled,task_index,candidate_indices,candidates,checked,committed,debit,credit,eligible)
    return plan

field_sampler=previous_method.field_sampler
