"""Choose geometric sensing paths by actual filtered motion, without MBD.

Seven existing geometric/hold references, a five-second executed prefix,
full double-integrator/aerodynamic/RPCBF prediction, and the exact conditional
GP reduction in weighted integrated variance for the predicted .2s readings.
Two coherent training fields and two held-out validation fields are retained.
No true wind input, synthetic assimilation, extra POAM approximation, or
changes to the downstream 15-second monitoring MBD objective.

This is finite-candidate, finite-horizon Bayesian experimental design. Kernel
parameters are frozen for the acquisition calculation; real updates still
retrain POAM. Task weights are explicitly a proxy for monitoring relevance.
"""
from dataclasses import replace
from time import perf_counter
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import full_learning_monitoring as model
from monitoring_routes import coarse_paths,smooth_reference_seed,reference_seed


def batch_variance_reduction(cross,measurement_covariance,weights,noise_variance):
    """Tr(W C_xP (C_PP + R)^-1 C_Px), with the full GP covariance."""
    C=(measurement_covariance+measurement_covariance.T)/2
    L=jnp.linalg.cholesky(C+(noise_variance+1e-9)*jnp.eye(len(C)))
    whitened=jsp.linalg.solve_triangular(L,cross.T,lower=True)
    return jnp.sum(weights*jnp.sum(whitened**2,axis=0))


def prefix_points(path,spacing=.025,maximum=1024):
    """Bounded geometric sampling for initialization, never plant integration."""
    path=np.asarray(path);parts=[path[:1]]
    for a,z in zip(path[:-1],path[1:]):
        parts.append(np.linspace(a,z,max(2,int(np.ceil(np.linalg.norm(z-a)/spacing))+1))[1:])
    points=np.concatenate(parts)[:maximum]
    return points,np.pad(points,((0,maximum-len(points)),(0,0)),mode='edge')


def usable_prefix(points,holding_limit,authority):
    """Stop a geometric initialization before its first unsupported point.

    This sufficient holding-authority screen does not certify a trajectory.
    The exact hard-filter forecast still decides whether it is admissible.
    """
    bad=np.flatnonzero(np.asarray(holding_limit)[:len(points)]>authority)
    last=max(0,int(bad[0])-1) if len(bad) else len(points)-1
    result=np.asarray(points)[:last+1]
    if len(result)>2:
        segments=np.diff(result,axis=0)
        direction=segments/np.maximum(np.linalg.norm(segments,axis=1)[:,None],1e-12)
        corners=np.linalg.norm(np.diff(direction,axis=0),axis=1)>1e-7
        result=result[np.r_[True,corners,True]]
    return result


@jax.jit
def information_gains(b,nodes,Kmap,correction,weights,positions):
    def one(points):
        Kp=b.kernel(points,b.basis.centers)
        cross=b.kernel(nodes,points)+Kmap@(correction@Kp.T)
        joint=b.kernel(points,points)+Kp@correction@Kp.T
        return batch_variance_reduction(cross,joint,weights,b.noise_std**2)
    return jax.lax.map(one,positions,batch_size=2)


class InformationPathPlanner:
    def __init__(self,env,grid,cfg,basis,prefix_seconds=5.,frontier_prefixes=False,frontier_graph=False):
        self.env=env;self.grid=grid;self.cfg=cfg;self.nodes=jnp.asarray(grid['all_cells'])
        self.prefix=prefix_seconds;self.prefix_cfg=replace(cfg,horizon=prefix_seconds,commit=min(cfg.commit,prefix_seconds))
        self.compiled_batches=set()
        self.frontier_prefixes=frontier_prefixes
        self.frontier_graph=frontier_graph
        @jax.jit
        def holding_limits(points,mean,std):
            mu=basis.evaluate(points,mean);sigma=jnp.maximum(basis.evaluate(points,std)[:,0],0.)
            radius=3*jnp.sqrt(sigma**2+.1**2);norm=jnp.linalg.norm(mu,axis=1);k=.1/.680
            # The same analytic nonlinear-force enclosure as force_support.
            # It is a candidate screen, not a change to wind or robot physics.
            return jnp.max(jnp.abs(k*norm[:,None]*mu),axis=1)+k*(2*norm*radius+radius**2)
        self.holding_limits=holding_limits
        self.dimension=2*self.prefix_cfg.steps
        fenv=model.legacy.float32(env);fbasis=model.legacy.float32(basis)
        def forecast(theta,x,q,tables,world):
            return model.core.execution_forecast(theta[:self.dimension],x,q,tables,world,fbasis,fenv,self.prefix_cfg,
                True,control_fn=model.control,motion_only=True)
        # A dynamic candidate dimension is deliberately restricted to 7, 2 or
        # 1; those three compiled batches cover search, validation and hold.
        self.forecast=jax.jit(jax.vmap(lambda theta,x,q,tables,worlds:
            jax.vmap(lambda world:forecast(theta,x,q,tables,world))(worlds),in_axes=(0,None,None,None,None)))

    def plan(self,x,q,b,cache,tables,weights,key):
        compiled_before=self.compiled_batches.copy()
        timings={};start=perf_counter()
        tables=cache.planning(b,tables,key);jax.block_until_ready(tables)
        timings['fields']=perf_counter()-start;start=perf_counter()
        _,cov=cache.map(b);jax.block_until_ready(cov)
        priority=np.asarray(weights)*np.asarray(cov)[:,0,0]
        admissible=None
        if self.frontier_graph:
            admissible=np.asarray(self.holding_limits(self.nodes,tables.mean,tables.std))<=.75*self.cfg.umax
        paths,metadata=coarse_paths(np.asarray(x),np.asarray(q),self.env,self.cfg,
            value_weights=priority[np.asarray(self.grid['free_indices'])],admissible_nodes=admissible)
        metadata['frontier_graph']=self.frontier_graph
        if admissible is not None:metadata['admissible_graph_nodes']=int(np.sum(admissible))
        full_paths=paths
        if self.frontier_prefixes:
            sampled=[prefix_points(p) for p in paths]
            limits=np.asarray(self.holding_limits(jnp.asarray(np.concatenate([p[1] for p in sampled])),tables.mean,tables.std)).reshape(len(paths),-1)
            paths=[usable_prefix(p[0],limit,.75*self.cfg.umax) for p,limit in zip(sampled,limits)]
            metadata['frontier_prefixes']=True
            metadata['holding_authority_fraction']=.75
            metadata['full_geometric_paths']=[p.tolist() for p in full_paths]
        hold=reference_seed(np.asarray(x[:2])[None],np.asarray(x),self.cfg)
        references=[smooth_reference_seed(p,np.asarray(x),self.env,self.cfg)[0] for p in paths]
        parameters=jnp.asarray(np.stack([hold,*references]));paths=[np.asarray(x[:2])[None],*paths]
        timings['geometry']=perf_counter()-start
        def evaluate(parameters,worlds):
            self.compiled_batches.add((len(parameters),len(worlds)))
            start=perf_counter();args=model.legacy.float32((parameters,x,q,tables,worlds))
            with jax.experimental.disable_x64():result=self.forecast(*args);jax.block_until_ready(result)
            motion_time=perf_counter()-start;start=perf_counter()
            states,controls,slack,clearance,terminated=result
            # The returned exact .05 s states include each actual .2 s sensor
            # time. No path interpolation or invented sensor footprint.
            readings=states[:,:,4::4,:2]
            count,world_count,number,_=readings.shape
            cache.refresh(b)
            gains=information_gains(b,self.nodes,cache.Kmap,cache.posterior[1],jnp.asarray(weights),
                readings.astype(jnp.float64).reshape(count*world_count,number,2)).reshape(count,world_count)
            gains=np.asarray(gains);valid=np.all(~np.asarray(terminated)&(np.asarray(slack)<1e-4),axis=1)&np.isfinite(gains).all(axis=1)
            return result,gains.mean(axis=1),valid,motion_time,perf_counter()-start
        forecast,gains,valid,motion_s,gain_s=evaluate(parameters,tables.planning)
        timings['search_motion']=motion_s;timings['search_information']=gain_s
        order=np.argsort(np.where(valid,-gains,np.inf))
        indices=order[:2]
        checked,vg,vvalid,motion_s,gain_s=evaluate(parameters[indices],tables.validation)
        vvalid&=valid[indices]
        timings['validation_motion']=motion_s;timings['validation_information']=gain_s
        no_feasible=not bool(np.any(vvalid));fallback=False
        if no_feasible:
            checked,vg,vvalid,motion_s,gain_s=evaluate(parameters[:1],tables.validation)
            vvalid&=valid[:1]
            timings['fallback_motion']=motion_s;timings['fallback_information']=gain_s
            indices=np.array([0]);fallback=True;no_feasible=not bool(np.any(vvalid))
        local=int(np.argmax(np.where(vvalid,vg,-np.inf))) if not no_feasible else 0
        selected=int(indices[local]);states=checked[0][local]
        theta=parameters[selected]
        refs,aa=model.legacy.old.nominal_reference(theta.astype(jnp.float64),x,self.cfg)
        record=dict(path=paths[selected].tolist(),candidate_paths=[p.tolist() for p in paths],
            selected_candidate=selected,expected_variance_reduction=float(vg[local]),
            expected_vector_mse_reduction=2*float(vg[local]),
            candidate_information=gains.tolist(),candidate_feasible=valid.tolist(),
            validation_candidates=indices.tolist(),validation_information=vg.tolist(),validation_feasible=vvalid.tolist(),
            no_feasible=no_feasible,hold_fallback=fallback,predicted_states=np.asarray(states).tolist(),
            timing=timings,initialization=metadata,forecast_seconds=self.prefix,
            reason='Filtered-motion information acquisition',selected_parameters=np.asarray(theta).tolist())
        record['compilation_included']=self.compiled_batches!=compiled_before
        return refs,aa,tables,record
