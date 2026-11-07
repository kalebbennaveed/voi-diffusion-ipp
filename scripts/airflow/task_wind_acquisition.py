"""Task-weighted wind learning: bounded acquisition, unchanged POAM posterior.

`mask` is the requested pointwise clarity-deficit times wind-variance rule.
`correlated` uses expected reduction of integrated task-weighted variance.
`routes` additionally weights the existing geometric alternatives by their
monitoring need and local wind-to-acceleration sensitivity. This is an explicit
acquisition heuristic, not a claim of exact long-horizon value of information.

Only current clarity, geometry and the learned GP enter these APIs. Truth is
never accepted. The one-measurement variance-reduction calculation holds kernel
parameters fixed; actual POAM updates still train on real readings only.
"""
from time import perf_counter
import numpy as np
import jax
import jax.numpy as jnp
from monitoring_routes import coarse_paths,clearance


def deficit_weights(q,targets,decay,horizon):
    """Exact no-sensing clarity solution at three existing seed-ranking times."""
    times=np.array([.25,.5,1.])*horizon
    future=np.asarray(q)[None]/(1+times[:,None]*np.asarray(decay)[None]*np.asarray(q)[None])
    return np.maximum(np.asarray(targets)[None]-future,0.).mean(axis=0)


def expected_variance_reduction(cross_covariance,candidate_variance,weights,noise_variance):
    """Exact Gaussian one-reading reduction, including conditional residual."""
    weights=jnp.asarray(weights)
    return jnp.sum(weights[:,None]*cross_covariance**2,axis=0)/jnp.maximum(candidate_variance+noise_variance,1e-12)


@jax.jit
def covariance_gains(b,nodes,Kmap,correction,candidates,weights,variance):
    Kc=Kmap[candidates]
    cross=b.kernel(nodes,nodes[candidates])+Kmap@(correction@Kc.T)
    return expected_variance_reduction(cross,variance[candidates],weights,b.noise_std**2)


def distinct_candidates(priority,nodes,count,spacing=.35):
    priority=np.asarray(priority).copy();chosen=[]
    for _ in range(count):
        idx=int(np.argmax(priority))
        if not np.isfinite(priority[idx]):break
        chosen.append(idx);priority[np.linalg.norm(nodes-nodes[idx],axis=1)<spacing]=-np.inf
    return chosen


def route_weights(nodes,paths,need,cells,mean,variance,speed=.4,drag=.1/.680,width=.25):
    """Route occupancy times expected squared wind-force Jacobian norm.

    For d=k||w-v||(w-v), ||D_w d||_F^2=5k²||w-v||².
    Its expectation is exact under the GP marginal, including both components.
    Occupancy is a declared geometric proxy for performance sensitivity.
    """
    result=np.zeros(len(nodes));normalizer=0.
    for path in paths:
        if len(path)<2:continue
        end=np.asarray(path[-1]);value=float(np.sum(need*np.exp(-np.sum((cells-end)**2,axis=1)/(.5**2))))
        length=np.linalg.norm(np.diff(path,axis=0),axis=1).sum()
        if value<=0 or length<1e-8:continue
        contribution=np.zeros(len(nodes))
        for a,z in zip(path[:-1],path[1:]):
            delta=z-a;distance=np.linalg.norm(delta)
            if distance<1e-8:continue
            fraction=np.clip((nodes-a)@delta/distance**2,0,1)
            lateral=np.linalg.norm(nodes-a-fraction[:,None]*delta,axis=1)
            velocity=speed*delta/distance
            sensitivity=5*drag**2*(np.sum((mean-velocity)**2,axis=1)+2*variance)
            contribution+=distance/length*np.exp(-.5*(lateral/width)**2)*sensitivity
        result+=value*contribution;normalizer+=value
    return result/max(normalizer,1e-12)


class TaskWindAcquisition:
    def __init__(self,env,grid,cfg,kind='mask',candidate_count=32):
        if kind not in ('mask','correlated','routes'):raise ValueError(kind)
        self.env=env;self.grid=grid;self.cfg=cfg;self.kind=kind;self.count=candidate_count
        self.nodes=np.asarray(grid['all_cells']);self.free=np.asarray(grid['free_indices'])
        self.available=np.flatnonzero(clearance(self.nodes,np.asarray(env.obstacles),np.asarray(env.bounds))>=max(.1,cfg.safety_margin+.04))

    def compute(self,x,q,b,cache,mean,variance,persistent_index=None,reachable=None):
        began=perf_counter();need=deficit_weights(q,self.env.targets,self.env.process_noise,self.cfg.horizon)
        weights=np.zeros(len(self.nodes));weights[self.free]=need
        paths=[]
        if self.kind=='routes':
            paths,_=coarse_paths(np.asarray(x),np.asarray(q),self.env,self.cfg)
            weights=route_weights(self.nodes,paths,need,np.asarray(self.env.cells),mean,variance)
            free_mask=np.zeros(len(weights),bool);free_mask[self.free]=True;weights[~free_mask]=0.
        weights/=max(weights.sum(),1e-12)
        raw=weights*variance;priority=raw.copy();candidates=[]
        if self.kind!='mask' and weights.sum()>0:
            available=self.available if reachable is None else self.available[np.asarray(reachable)[self.available]]
            eligible=np.full(len(weights),-np.inf);eligible[available]=raw[available]
            candidates=distinct_candidates(eligible,self.nodes,self.count//2)
            # Include remote candidates: a correlated observation can help the
            # task even if its own cell has zero clarity-deficit weight.
            eligible[available]=variance[available]
            for i in candidates:eligible[np.linalg.norm(self.nodes-self.nodes[i],axis=1)<.35]=-np.inf
            candidates+=distinct_candidates(eligible,self.nodes,self.count-len(candidates))
            candidates=np.asarray(candidates,int)
            if len(candidates)==0:
                return np.zeros_like(weights),weights,dict(acquisition=self.kind,seconds=perf_counter()-began,
                    weighted_vector_std_m_s=float(np.sqrt(2*np.dot(weights,variance))),active_cells=int(np.count_nonzero(weights)),
                    maximum_priority=0.,candidate_indices=[],geometric_paths=[p.tolist() for p in paths])
            candidates=np.pad(candidates,(0,self.count-len(candidates)),mode='edge')
            if persistent_index is not None and persistent_index not in candidates:
                candidates[-1]=persistent_index
            cache.refresh(b)
            gain=covariance_gains(b,jnp.asarray(self.nodes),cache.Kmap,cache.posterior[1],jnp.asarray(candidates),jnp.asarray(weights),jnp.asarray(variance))
            gain=np.asarray(gain);priority=np.zeros(len(weights));priority[candidates]=np.maximum(gain,0.)
        vector_std=float(np.sqrt(2*np.dot(weights,variance)))
        return priority,weights,dict(acquisition=self.kind,seconds=perf_counter()-began,
            weighted_vector_std_m_s=vector_std,active_cells=int(np.count_nonzero(weights>0)),
            maximum_priority=float(priority.max()),candidate_indices=np.asarray(candidates).tolist(),
            geometric_paths=[p.tolist() for p in paths])
