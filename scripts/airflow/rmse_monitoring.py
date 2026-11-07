"""Separate learn-then-MBD experiment: full posterior, guarded execution.

The oracle RMSE gate lives in the experiment runner, outside these APIs.
No true field is accepted by this planner or control law.
"""
from implicit_monitoring_v3 import *
import implicit_monitoring_v2 as core
import implicit_monitoring_v3 as old


def field_builder(points,shape,cfg):
    @jax.jit
    def build(b,count,performance_key,safety_key):
        def draw(key,n):
            eps=jax.random.normal(key,(n//2,*b.sample_shape),dtype=b.mean.dtype)
            coeff=b.sample_coefficients(jnp.concatenate((eps,-eps)))
            return jax.vmap(lambda c:b.basis.evaluate(points,c))(coeff).reshape(n,*shape)
        mu,cov=b.predict(points)
        return Tables(mu.reshape(shape),jnp.sqrt(cov[:,0,0]).reshape(*shape[:2],1),
            jnp.ones((*shape[:2],1)),draw(performance_key,cfg.worlds),
            draw(safety_key,cfg.construction_worlds),
            draw(jax.random.fold_in(performance_key,104729),cfg.validation_worlds))
    return build


def boundary_constraints(x,mean,std_table,basis,env,cfg):
    """Robust relative-degree-two boundary CBF, exact aerodynamic bound.

    h = distance to a wall minus margin; h''+2λh'+λ²h >= 0.
    Conditional on initial h,h'+λh >= 0 and a valid disturbance envelope,
    this supplies an independent wall-braking constraint. It does not certify
    finite sampling, posterior calibration, or an infeasible actuator problem.
    """
    normal=jnp.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    h=jnp.array([x[0]-env.bounds[0,0],env.bounds[1,0]-x[0],
                 x[1]-env.bounds[0,1],env.bounds[1,1]-x[1]])-cfg.safety_margin
    sigma=jnp.maximum(basis.evaluate(x[:2],std_table)[0,0],0.)
    radius=getattr(cfg,'safety_confidence',3.)*jnp.sqrt(sigma**2+.1**2)
    A=-normal;rate=1.5
    b=2*rate*(normal@x[2:])+rate**2*h-core.force_support(A,x[2:],mean.mean_at(x[:2])[0],radius)
    return A,b


def control(x,nominal,mean,fields,std_table,basis,env,cfg,hard=False):
    grad,B,bound=core.constraints(x,mean,fields,std_table,basis,env,cfg,kappa=getattr(cfg,'safety_confidence',3.))
    wall_A,wall_b=boundary_constraints(x,mean,std_table,basis,env,cfg)
    A=jnp.concatenate((grad[:,2:],wall_A));b=jnp.concatenate((bound,wall_b))
    if hard:
        sol=solve(nominal,A,b,cfg.umax,design_policy(x,mean,cfg))
        return sol.control,FilterInfo(B,grad,jnp.linalg.norm(sol.control-nominal),
            sol.feasible,sol.slack,sol.residual,jnp.array(0.))
    u,slack=soft_projection(nominal,A,b,cfg.umax,cfg.slack_weight)
    return u,jnp.max(slack)


def execution_block(env,cfg,pcfg,basis,seconds=.2):
    return core.execution_block(env,cfg,pcfg,basis,seconds,control_fn=control)


def optimizer(env,cfg,basis,aware=True,control_fn=control,reuse_training=False,batch_validation=False,wind_error_reserve=None,prepare_validation=False,information_trigger='stalled',allow_validation_slack=False):
    """Original independent MBD chains; validate optimized mode leaders.

    Retain the existing two-finalist compute budget. Unlike the earlier
    baseline, the second candidate is an optimized mode leader, not merely
    an unoptimized geometric seed. Hold is a separately logged fallback.
    The explicit safety-aware option permits nonzero filter relaxation in
    contact-free forecasts. It changes only finalist eligibility, not the
    objective, filter, posterior, disturbance bounds, or actuator limits.
    """
    import time
    from wind_model_trust import trajectory_error_bound,authority_margin,reference_support
    @jax.jit
    def optimize(x,q,tables,seeds,previous,key):
        def evaluate(theta):
            r=jax.vmap(lambda w:old.rollout(theta,x,q,tables,w,basis,env,cfg,aware,control_fn))(tables.planning)
            return jnp.where(r.terminated,jnp.inf,r.monitor+r.penalty)
        result=search(evaluate,seeds,previous,key,cfg,jnp.array(1.),jnp.array(0.),jnp.array(False))
        leaders=result.finalists[:cfg.mode_count]
        leader_cost=(result.finalist_field_costs[:cfg.mode_count] if reuse_training else jax.vmap(evaluate)(leaders)).mean(axis=1)
        return result,leader_cost
    @jax.jit
    def validate(theta,x,q,tables):
        options={'return_dense':True} if wind_error_reserve is not None else {}
        return jax.vmap(lambda w:core.execution_forecast(theta,x,q,tables,w,basis,env,cfg,aware,control_fn,**options))(tables.validation)
    validate_pair=jax.jit(jax.vmap(validate,in_axes=(0,None,None,None)))
    reference_bounds=jax.jit(lambda params,x,tables:jax.vmap(lambda theta:
        reference_support(theta,x,tables,basis,cfg,old.legacy.old.nominal_reference))(params))
    forecast_bounds=jax.jit(lambda states,tables:jnp.max(trajectory_error_bound(states,tables,basis),axis=(-1,-2)))
    forecast_margin=jax.jit(lambda states,controls,tables:jnp.min(authority_margin(
        trajectory_error_bound(states,tables,basis)[...,:-1],controls,cfg.umax),axis=(-1,-2)))
    reference_clearance=jax.jit(lambda params,x:jax.vmap(lambda theta:jnp.min(jax.vmap(env.clearance)(
        old.legacy.old.nominal_reference(theta,x,cfg)[0])))(params)) if information_trigger=='lower-cost' else None
    @jax.jit
    def validation_summary(ff,tables):
        cost=jnp.mean(ff.monitor+ff.penalty,axis=1)
        valid=jnp.all(~ff.terminated,axis=1)&jnp.isfinite(cost)
        if not allow_validation_slack:
            valid&=jnp.all(ff.max_slack<1e-4,axis=1)
        if wind_error_reserve is not None:
            valid&=(forecast_bounds(ff.x,tables)<=wind_error_reserve)&(forecast_margin(ff.x,ff.u,tables)>=-1e-5)
        return cost,valid
    prepared=False
    compiled_validation={};compiled_summary={}
    def plan(x,q,tables,seeds,previous,key):
        nonlocal prepared
        if prepare_validation and not prepared:
            # Compile every finite finalist-count branch in the first request.
            # No extra rollout is executed, sampled or scored here. This moves
            # first-use compilation into the explicitly reported startup call.
            compiled_validation['single']=validate.lower(seeds[0],x,q,tables).compile()
            compiled_validation['pair']=validate_pair.lower(jnp.stack((seeds[0],seeds[0])),x,q,tables).compile()
            abstract=jax.eval_shape(validate,seeds[0],x,q,tables)
            for count in (2,3):
                xs=jax.ShapeDtypeStruct((count,*abstract.x.shape),abstract.x.dtype)
                us=jax.ShapeDtypeStruct((count,*abstract.u.shape),abstract.u.dtype)
                if wind_error_reserve is not None:
                    forecast_bounds.lower(xs,tables).compile()
                    forecast_margin.lower(xs,us,tables).compile()
                ff=jax.tree.map(lambda z:jax.ShapeDtypeStruct((count,*z.shape),z.dtype),abstract)
                compiled_summary[count]=validation_summary.lower(ff,tables).compile()
            prepared=True
        t=time.perf_counter();(result,leader_cost)=optimize(x,q,tables,seeds,previous,key);jax.block_until_ready(leader_cost)
        search_s=time.perf_counter()-t
        # Global winner and the best different mode leader, under training cost.
        params=np.asarray(result.finalists[:cfg.mode_count])
        scores=np.asarray(leader_cost)
        distinct=np.any(params!=np.asarray(result.final)[None],axis=1)
        order=np.argsort(np.where(distinct,scores,np.inf))
        second=params[order[0]] if np.any(distinct) else np.asarray(seeds[0])
        candidates=[result.final,jnp.asarray(second)];origins=['MBD global winner','MBD alternate mode leader']
        support_bounds=None
        if wind_error_reserve is not None:
            pool=jnp.concatenate((result.finalists,result.initial_parameters))
            costs=np.asarray(jnp.concatenate((result.finalist_field_costs,result.initial_field_costs)).mean(axis=1))
            bounds,margins,mean_margins=reference_bounds(pool,x,tables);support_bounds=np.asarray(bounds);support_margins=np.asarray(margins)
            admissible=(support_bounds<=wind_error_reserve)&(support_margins>=-1e-5)
            order=np.argsort(np.where(admissible,costs,np.inf))
            selected=[];tokens=set()
            for idx in order:
                if not np.isfinite(costs[idx]) or not admissible[idx]:continue
                token=np.asarray(pool[idx]).tobytes()
                if token in tokens:continue
                selected.append(int(idx));tokens.add(token)
                if len(selected)==2:break
            candidates=[pool[i] for i in selected]
            origins=['model-supported MBD archive member '+str(i) for i in selected]
            while len(candidates)<2:
                candidates.append(seeds[0]);origins.append('model-support hold fallback')
        forecasts=[];seen={};t=time.perf_counter()
        if batch_validation and not np.array_equal(candidates[0],candidates[1]):
            pair=compiled_validation.get('pair',validate_pair)(jnp.stack(candidates),x,q,tables);jax.block_until_ready(pair)
            for i,theta in enumerate(candidates):
                f=jax.tree.map(lambda z:z[i],pair);seen[np.asarray(theta).tobytes()]=f;forecasts.append(f)
        else:
            for theta in candidates:
                token=np.asarray(theta).tobytes()
                if token not in seen:
                    seen[token]=compiled_validation.get('single',validate)(theta,x,q,tables);jax.block_until_ready(seen[token])
                forecasts.append(seen[token])
        def arrays():
            ff=jax.tree.map(lambda *v:jnp.stack(v),*forecasts)
            cost,valid=compiled_summary.get(len(forecasts),validation_summary)(ff,tables)
            return ff,cost,valid
        ff,cost,valid=arrays()
        if not bool(jnp.any(valid)):
            hold=compiled_validation.get('single',validate)(seeds[0],x,q,tables);jax.block_until_ready(hold)
            candidates.append(seeds[0]);origins.append('hold fallback');forecasts.append(hold)
            ff,cost,valid=arrays()
        choice=int(jnp.argmin(jnp.where(valid,cost,jnp.inf))) if bool(jnp.any(valid)) else len(candidates)-1
        audit=dict(leader_reevaluations=0 if reuse_training else cfg.mode_count,leader_costs=leader_cost,candidates=jnp.stack(candidates),costs=cost,feasible=valid,
            contacts=ff.terminated,slack=ff.max_slack,selected=choice,no_feasible=~jnp.any(valid),
            safety_feasible=jnp.all(~ff.terminated&(ff.max_slack<1e-4),axis=1),allow_validation_slack=allow_validation_slack,
            unique_evaluations=len(seen)+(len(candidates)>2),search_s=search_s,validation_s=time.perf_counter()-t)
        if wind_error_reserve is not None:
            # Compare on the SAME training worlds. A mean-infeasible reference
            # is not a reason to collect more information about an impossible
            # nominal command. Only uncertainty-blocked candidates qualify.
            uncertain=(~admissible)&(np.asarray(mean_margins)>=-1e-5)&np.isfinite(costs)
            if information_trigger=='lower-cost':
                # Information targets must represent free workspace. This is
                # a reference shortlist screen, not an execution certificate.
                geometric=np.asarray(reference_clearance(pool,x))>=cfg.safety_margin
                uncertain&=geometric
            best_blocked=int(np.argmin(np.where(uncertain,costs,np.inf)))
            selected_training_cost=(float(costs[selected[choice]]) if choice<len(selected)
                else float(np.asarray(result.initial_field_costs[0]).mean()))
            improvement=max(0.,selected_training_cost-float(costs[best_blocked])) if uncertain.any() else 0.
            displacement=float(jnp.max(jnp.linalg.norm(ff.x[choice,...,:2]-x[:2],axis=-1)))
            request_learning=bool(uncertain.any() and improvement>1e-4 and
                (information_trigger=='lower-cost' or not bool(jnp.any(valid)) or displacement<cfg.sensor_sigma))
            learning_reference=old.legacy.old.nominal_reference(pool[best_blocked] if uncertain.any() else seeds[0],x,cfg)[0]
            audit.update(wind_error_reserve_m_s2=wind_error_reserve,
                reference_error_bounds=jnp.asarray(support_bounds),forecast_error_bounds=forecast_bounds(ff.x,tables),
                reference_authority_margins=jnp.asarray(support_margins),forecast_authority_margins=forecast_margin(ff.x,ff.u,tables),
                selected_archive_indices=jnp.asarray(selected,dtype=jnp.int32),
                uncertainty_blocked_improvement=improvement,selected_forecast_displacement_m=displacement,
                request_targeted_learning=request_learning,learning_reference=learning_reference)
        return candidates[choice],cost[choice],result,jax.tree.map(lambda z:z[choice],ff),audit
    return plan
