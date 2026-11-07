"""Route-targeted online survey with the original MBD reverse equations.

A finite set of geometric monitoring routes defines the survey targets. The
survey score is full-predictive-GP target covariance reduction, evaluated at
sensor locations on full physical/RPCBF forecasts. This is a declared planning
surrogate, not a claim about exact future relocated POAM covariance. Actual
observations alone update the executed estimator.
"""
from typing import NamedTuple
from dataclasses import replace
import jax
import jax.numpy as jnp
import legacy_monitoring as legacy
import active_sensing_monitoring as active
from exploration_monitoring import forecast as coarse_forecast
from survey_forecast import forecast as refined_forecast
from survey_information import prepare_full_targets, full_target_variance_reduction_components
from voi_diffusion_ipp.quarter.multimode import search
from voi_diffusion_ipp.wind.gp import WorldField
from voi_diffusion_ipp.quadrotor.aerodynamics import acceleration

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


class Validated(NamedTuple):
    # Values columns: total task cost, information fraction, contact,
    # learning bonus, monitoring cost, safety/speed penalty.
    values:jax.Array
    states:jax.Array
    min_clearance:jax.Array
    max_slack:jax.Array
    prefix_max_slack:jax.Array
    prefix_infeasible_steps:jax.Array


def route_tangents(positions, points_per_route=8):
    """Geometric probing directions, with no jumps between distinct routes."""
    paths=positions.reshape((-1,points_per_route,2))
    delta=jnp.diff(paths,axis=1)
    delta=jnp.concatenate((delta,delta[:,-1:]),axis=1).reshape(positions.shape)
    return .7*delta/jnp.maximum(jnp.linalg.norm(delta,axis=1,keepdims=True),1e-6)


def force_relevance(winds,positions,weights,force_fn=acceleration):
    """Sampled aerodynamic response of task routes; not exact task VOI.

    ``force_fn`` defaults to the unchanged physical force law. Its explicit
    argument enables a controlled zero-coupling property test without changing
    any experimental robot configuration.
    """
    tangent=route_tangents(positions)
    forces=jax.vmap(lambda w:force_fn(tangent,w,xp=jnp))(winds)
    reverse=jax.vmap(lambda w:force_fn(-tangent,w,xp=jnp))(winds)
    sensitivity=jnp.maximum(jnp.sum(jnp.var(forces,axis=0),axis=1),jnp.sum(jnp.var(reverse,axis=0),axis=1))
    relevance=jnp.sqrt(jnp.sum(weights*sensitivity)/jnp.maximum(jnp.sum(weights),1e-12))
    return weights*jnp.maximum(sensitivity-.10**2,0.),relevance


def task_information_benefits(uncertain,known):
    """Compare route metrics in the same posterior-sampled physical field.

    Inputs have columns [monitoring+safety cost, mean tracking error, contact].
    Returned columns are [cost reduction, tracking-error reduction, rescued
    contact]. A counterfactual that itself contacts earns no claimed benefit.
    This finite sampled relevance test is not a task value-of-information
    estimate and never changes the cost used to rank final trajectories.
    """
    known_safe=known[:,2]==0
    cost=jnp.where(known_safe,uncertain[:,0]-known[:,0],0.)
    tracking=jnp.where(known_safe,uncertain[:,1]-known[:,1],0.)
    rescued=(uncertain[:,2]>0)&known_safe
    benefits=jnp.stack((cost,tracking,rescued.astype(cost.dtype)),axis=1)
    useful=jnp.any((cost>.01)|(tracking>.10)|rescued)
    return benefits,useful


def route_task_weights(weights,route_indices,benefits,points_per_route=8):
    """Retain information targets only on routes with sampled task benefit."""
    useful=(benefits[:,0]>.01)|(benefits[:,1]>.10)|(benefits[:,2]>0)
    return mapped_route_weights(weights,route_indices,useful,points_per_route)


def mapped_route_weights(weights,route_indices,selected,points_per_route=8):
    groups=weights.reshape((-1,points_per_route))
    matching=jnp.arange(len(groups))[:,None]==route_indices[None,:]
    mask=jnp.any(matching&selected[None,:],axis=1)
    return (groups*mask[:,None]).reshape(weights.shape)


def sampled_route_values(uncertain,known):
    """Cost-unit value relative to the best of the three current task routes.

    A known-field route must be contact-free. Current-uncertainty routes retain
    their full recorded absorbing monitoring/safety cost even if they contact;
    this is a finite paired-policy diagnostic, not an optimal value of knowing
    the field. Tracking improvement alone does not create a cost-unit reward.
    """
    best_current=jnp.min(uncertain[:,0])
    return jnp.where(known[:,2]==0,jnp.maximum(best_current-known[:,0],0.),0.)


def stationary_learning_bonus(route_values,route_fractions,remaining_seconds,horizon):
    """Declared repeated-use linear learning surrogate in monitoring units.

    Stationary wind can improve repeated monitoring. At most twenty remaining
    horizon-length uses are credited; their actual future policy is not solved.
    The maximum prevents adding benefits of mutually alternative passages.
    """
    uses=jnp.clip(remaining_seconds/horizon,1.,20.)
    return uses*jnp.max(route_values*route_fractions,axis=-1)


def validation_candidates(search_result,seeds):
    """Five training-selected controls; -1 identifies the original hold seed."""
    winner=jnp.argmax(jnp.all(search_result.finalists==search_result.final,axis=1))
    indices=jnp.array([winner,-1,1,2,3])
    candidates=jnp.stack((search_result.finalists[winner],seeds[0],*search_result.finalists[1:4]))
    return indices,candidates

def optimizer(env,cfg,mode='survey',cost_allowance=1.0):
    """Select one declared objective per plan, sharing all proposal work.

    Survey mode minimizes monitoring cost minus a finite cost-unit learning
    surrogate, with an explicit monitoring-loss allowance. Pure monitoring wins
    whenever attainable learning does not repay the extra monitoring loss.
    The gate uses exact aerodynamic acceleration and three coarse paired route
    forecasts as an explicitly finite value surrogate. MBD search retains the
    original relaxed RPCBF; independent final validation uses the execution
    hard-filter definition and cadence for the complete 15-second horizon.
    """
    if mode not in ('survey','passive'):
        raise ValueError(mode)
    search_cfg=replace(cfg,safety_margin=.06,construction_worlds=2,prediction_filter='soft')
    hard_cfg=replace(cfg,safety_margin=.12,construction_worlds=8,prediction_filter='hard')
    times=jnp.arange(1,round(cfg.commit/.2)+1)*.2
    indices=jnp.minimum(jnp.floor((times-1e-6)/cfg.dt).astype(int),cfg.commit_steps-1)
    offsets=times-indices*cfg.dt
    @jax.jit
    def plan(b,x,q,mean,construction,worlds,validation,previous,key,seeds,positions,weights,route_indices=None,remaining_seconds=None):
        if route_indices is None:
            raise ValueError('Pass metadata task_route_indices to map three task probes to their geometric target groups')
        if route_indices.shape != (3,):
            raise ValueError('Task route indices must contain the three original geometric path indices')
        remaining=cfg.horizon if remaining_seconds is None else remaining_seconds
        amortization=jnp.clip(remaining/cfg.horizon,1.,20.)
        # Sampled force variation is a task relevance test, not a replacement
        # plant. A zero wind-to-force coupling produces exactly zero relevance.
        # Reuse construction samples for relevance. Fresh validation fields
        # must not determine the objective or generate candidate trajectories.
        winds=jax.vmap(lambda field:WorldField(legacy.BASIS,field).at(positions))(construction)
        # Tangents are geometric probing velocities. Opposite directions both
        # matter because monitoring requires return trips.
        # All three inferred task passages are probed. Both sides use the
        # same sampled physical field and reference; only what the controller
        # knows about that field differs. No true environment field is queried.
        physical_probe=worlds[0]
        known_construction=jnp.broadcast_to(physical_probe,(8,*physical_probe.shape))
        def probe(theta):
            uncertain=coarse_forecast(theta,x,q,mean,construction[:8],physical_probe,env,hard_cfg)
            known=coarse_forecast(theta,x,q,physical_probe,known_construction,physical_probe,env,hard_cfg)
            def metric(r):
                tracking=jnp.mean(jnp.linalg.norm(r.x[:,:2]-r.nominal[:,:2],axis=1))
                return jnp.array([r.monitor+r.penalty,tracking,r.terminated.astype(x.dtype)])
            return metric(uncertain),metric(known)
        uncertain_metrics,known_metrics=jax.vmap(probe)(seeds[1:4])
        relevance_benefits,_=task_information_benefits(uncertain_metrics,known_metrics)
        route_values=sampled_route_values(uncertain_metrics,known_metrics)
        # A benefit on one passage does not justify surveying irrelevant
        # passages. Map each probe back to its own eight geometric targets.
        useful_task_weights=mapped_route_weights(weights,route_indices,route_values>0)
        target_weights,relevance=force_relevance(winds,positions,useful_task_weights)
        # Ignore force variations below 0.10 m/s² on these task routes. This
        # explicit engineering tolerance is not a proof of decision relevance.
        targets=prepare_full_targets(b,positions,target_weights)
        uncertainty=jnp.sqrt(targets.total_variance/jnp.maximum(2*jnp.sum(target_weights),1e-12))
        information_enabled=(uncertainty>.30)&(relevance>.10)&(jnp.sum(target_weights)>1e-8)&jnp.any(route_values>0)&(mode=='survey')
        route_variance=(2*target_weights*targets.variance).reshape((-1,8)).sum(axis=1)
        def information(r,samples):
            components=full_target_variance_reduction_components(b,samples,targets)
            fractions=jnp.clip(components.reshape((-1,8)).sum(axis=1)/jnp.maximum(route_variance,1e-30),0.,1.)
            bonus=stationary_learning_bonus(route_values,fractions[route_indices],remaining,cfg.horizon)
            gain=jnp.clip(components.sum()/jnp.maximum(targets.total_variance,1e-30),0.,1.)
            gain=jnp.where(r.terminated,0.,gain)
            bonus=jnp.where(r.terminated|~information_enabled,0.,bonus)
            return jnp.array([r.monitor+r.penalty,gain,r.terminated.astype(x.dtype),bonus,r.monitor,r.penalty])
        def measures(theta,fields):
            def one(field):
                r=legacy.rollout(theta,x,q,mean,construction[:2],field,env,search_cfg)
                world=WorldField(legacy.BASIS,field)
                samples=jax.vmap(lambda z,u,h:legacy.m.physical.rk4(z,u,world,h)[:2])(r.x[indices],r.u[indices],offsets)
                return information(r,samples)
            return jax.vmap(one)(fields)
        initials=jnp.r_[seeds,previous[None]]
        values=jax.lax.map(lambda th:measures(th,worlds),initials,batch_size=cfg.batch_size)
        costs=values[:,:,0].mean(axis=1);safe=jnp.all(values[:,:,2]==0,axis=1)
        initial_budget=jnp.min(jnp.where(safe,costs,jnp.inf))+cost_allowance
        def evaluate(theta):
            values=measures(theta,worlds);cost=values[:,0]
            safe=jnp.all(values[:,2]==0)
            scores=cost-values[:,3]
            allowed=safe&(~information_enabled|(cost.mean()<=initial_budget))
            return jnp.where(allowed,scores,jnp.inf)
        initial_scores=values[:,:,0]-values[:,:,3]
        initial_allowed=safe&(~information_enabled|(costs<=initial_budget))
        initial_fields=jnp.where(initial_allowed[:,None],initial_scores,jnp.inf)
        result=search(evaluate,seeds,previous,key,cfg,1.,0.,jnp.array(False),initial_field_values=initial_fields)
        # A declared finite validation budget preserves all three task-passage
        # modes, the globally best trained candidate, and the original hold.
        # Each mode leader already includes its final reverse iterate. No
        # validation field is used to choose this five-candidate subset.
        candidate_indices,candidates=validation_candidates(result,seeds)
        def validate(theta):
            def one(field):
                fine=refined_forecast(theta,x,q,mean,construction[:8],field,env,hard_cfg,fine_seconds=cfg.horizon)
                r=fine.rollout
                return Validated(information(r,fine.observation_positions),r.x,
                                 r.min_clearance,r.max_slack,fine.prefix_max_slack,
                                 fine.prefix_infeasible_steps)
            return jax.vmap(one)(validation)
        checked=jax.lax.map(validate,candidates,batch_size=cfg.validation_batch_size or None)
        values=checked.values
        task=values[:,:,0].mean(axis=1);gain=values[:,:,1].mean(axis=1);bonus=values[:,:,3].mean(axis=1)
        safe=jnp.all(values[:,:,2]==0,axis=1)&jnp.isfinite(task)&jnp.isfinite(gain)&jnp.isfinite(bonus)
        budget=jnp.min(jnp.where(safe,task,jnp.inf))+cost_allowance
        scores=jnp.where(safe&(~information_enabled|(task<=budget)),task-bonus,jnp.inf)
        index=jnp.argmin(scores)
        task_index=jnp.argmin(jnp.where(safe,task,jnp.inf))
        exploring=information_enabled&(task[index]>task[task_index]+1e-6)&(bonus[index]>0)
        return Result(candidates[index],scores[index],task[index],result,index,task,gain,values[:,:,2],budget,initial_budget,exploring,uncertainty,relevance,relevance_benefits,jnp.array(1.,x.dtype),route_values,bonus,amortization,information_enabled,task_index,candidate_indices,candidates,checked)
    return plan

field_sampler=active.field_sampler
