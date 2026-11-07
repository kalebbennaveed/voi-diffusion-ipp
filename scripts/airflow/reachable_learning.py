"""Cheap GP experimental design on the currently supported geometric component.

Only waypoint initialization is discretized at .05 m. The plant, sensor, POAM,
clarity and execution RPCBF are unchanged. The holding screen is sufficient,
not a guarantee of dynamic reachability: actual RPCBF feasibility is logged.
No truth or synthetic observation enters this policy.
"""
from exact_poam_variance import grid_standard_deviation,PosteriorStandardDeviation,route_standard_deviation
from time import perf_counter
import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from guarded_waypoints import GuardedWaypointPolicy
from monitoring_routes import clearance,visible
from task_wind_acquisition import deficit_weights,distinct_candidates,expected_variance_reduction
from information_paths import prefix_points
from full_learning_monitoring import boundary_constraints,obstacle_constraints,TableMean
from voi_diffusion_ipp.safety.qp import project
from wind_model_trust import reference_information_weights


def measured_route_mask(nodes,mean,sigma,measured_positions,umax,radius=.15,confidence=3.):
    """Use the same recovery bound to exclude locally measured route edges.

    Unknown locations keep the existing mean-wind routing assumption. This
    geometric screen never changes the GP, wind samples or safety uncertainty.
    The .15 m neighborhood is an explicit initialization heuristic, matching
    the existing observed-inducing support radius; it is not a sensor footprint.
    """
    observed=cKDTree(np.asarray(measured_positions)).query(nodes)[0]<=radius
    R=confidence*np.sqrt(np.maximum(sigma,0.)**2+.01);norm=np.linalg.norm(mean,axis=1);k=.1/.680
    coarse=np.max(np.abs(k*norm[:,None]*mean),axis=1)+k*(2*norm*R+R*R)
    angles=np.arange(32)*2*np.pi/32
    w=mean[:,None]+R[:,None,None]*np.c_[np.cos(angles),np.sin(angles)][None]
    force=k*np.linalg.norm(w,axis=-1)[...,None]*w
    angular=np.max(np.abs(force),axis=(1,2))+2*k*R*(norm+R)*np.pi/32
    bound=np.minimum(coarse,angular)
    return ~observed|(bound<=.75*umax),bound,observed


def braking_feasible(position,mean,std,basis,env,cfg):
    """Existence of a reserved bounded command satisfying at-rest braking CBFs.

    This only screens reference geometry. It is not the moving-state RPCBF
    and makes no safety certificate for a trajectory or an uncalibrated GP.
    """
    x=jnp.r_[position,jnp.zeros(2)]
    wall_A,wall_b=boundary_constraints(x,mean,std,basis,env,cfg)
    obstacle_A,obstacle_b=obstacle_constraints(x,mean,std,basis,env,cfg)
    box=jnp.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    A=jnp.concatenate((wall_A,obstacle_A,box))
    b=jnp.concatenate((wall_b,obstacle_b,jnp.full(4,.75*cfg.umax)))
    return project(jnp.zeros(2),A,b)[1]


@jax.jit
def observation_gain_and_variance(b,map_nodes,Kmap,correction,weights,queries):
    Kq=b.kernel(queries,b.basis.centers)
    cross=b.kernel(map_nodes,queries)+Kmap@(correction@Kq.T)
    variance=jnp.diag(b.kernel(queries,queries))+jnp.sum((Kq@correction)*Kq,axis=1)
    return expected_variance_reduction(cross,variance,weights,b.noise_std**2),variance


def observation_gain(b,map_nodes,Kmap,correction,weights,queries):
    return observation_gain_and_variance(b,map_nodes,Kmap,correction,weights,queries)[0]


def acquisition_variances(interpolated,candidates,direct):
    """Use direct posterior variance to decide whether queried goals are learned.

    The field interpolation remains unchanged for dynamics/safety and for the
    geometric shortlist. Only the already-queried candidate entries are
    replaced, with no threshold adjustment or posterior modification.
    """
    variance=np.array(interpolated,copy=True)
    variance[np.asarray(candidates,int)]=np.maximum(np.asarray(direct),0.)
    return variance


class ReachableLearning:
    def __init__(self,env,grid,cfg,basis,kind='all',global_goals=False,vector_precision=.2,prefix_screen='holding',learned_route_bounds=False,diversify_stalls=False,route_precision=False,downstream_information=False):
        if kind not in ('all','mask'):raise ValueError(kind)
        self.env=env;self.grid=grid;self.cfg=cfg;self.kind=kind;self.field_basis=basis
        self.global_goals=global_goals
        self.learned_route_bounds=learned_route_bounds
        self.track_route_precision=route_precision;self.vector_precision=vector_precision
        self.task_reference=None
        self.task_weights=None # Optional candidate-path domain; never changes the GP.
        self.downstream_information=downstream_information
        self.policy=GuardedWaypointPolicy(env,umax=cfg.umax,confidence=cfg.safety_confidence,spacing=.05,
            minimum_target_distance=.05,reference_lookahead=.3,
            learned_variance=.5*vector_precision**2 if global_goals else .25,
            require_target_precision=global_goals,diversify_stalls=diversify_stalls)
        nodes=jnp.asarray(self.policy.nodes)
        self.moments=jax.jit(lambda mean,std:(basis.evaluate(nodes,mean),basis.evaluate(nodes,grid_standard_deviation(std))[:,0]))
        self.gap=clearance(self.policy.nodes,np.asarray(env.obstacles),np.asarray(env.bounds))-cfg.safety_margin
        self.coarse_nodes=jnp.asarray(grid['all_cells'])
        self.route_visited=np.zeros(len(self.policy.nodes),bool);self.last_route_position=None
        self.route_handoff_ids=None;self.route_current_ids=None;self.route_goal=None
        nx,ny=grid['grid'];index=np.floor(self.policy.nodes/.125).astype(int)
        self.coarse_index=np.clip(index[:,1],0,ny-1)*nx+np.clip(index[:,0],0,nx-1)
        @jax.jit
        def supported(points,mean,std):
            mu=basis.evaluate(points,mean);sigma=jnp.maximum(basis.evaluate(points,std)[:,0],0.)
            R=cfg.safety_confidence*jnp.sqrt(sigma*sigma+.01);norm=jnp.linalg.norm(mu,axis=1);k=.1/.680
            error=k*(2*norm*R+R*R);drift=k*norm[:,None]*mu
            boundary=jnp.minimum(jnp.min(points-env.bounds[0],axis=1),jnp.min(env.bounds[1]-points,axis=1))
            obstacle=jnp.min(jnp.linalg.norm(points[:,None]-env.obstacles[None,:,:2],axis=2)-env.obstacles[None,:,2],axis=1)
            gap=jnp.minimum(boundary,obstacle)-cfg.safety_margin
            return (jnp.max(jnp.abs(drift),axis=1)+error<=.75*cfg.umax)&(error<=1.5**2*gap)
        self.supported=supported
        if prefix_screen=='braking-feasible':
            self.supported=jax.jit(lambda p,mean,std:jax.vmap(lambda z:
                braking_feasible(z,TableMean(basis,mean),std,basis,env,cfg))(p))
        elif prefix_screen!='holding':raise ValueError(prefix_screen)

    def route_indices(self,positions):
        nx,ny=np.rint((self.policy.bounds[1]-self.policy.bounds[0])/self.policy.spacing).astype(int)
        ij=np.floor((np.asarray(positions)-self.policy.bounds[0])/self.policy.spacing).astype(int)
        ij=np.clip(ij,[0,0],[nx-1,ny-1]);return np.unique(ij[:,1]*nx+ij[:,0])

    def update_route_precision(self,x,sigma,allow_grid_handoff=True):
        if not self.track_route_precision:return
        previous=x[:2] if self.last_route_position is None else self.last_route_position
        segment=np.linspace(previous,x[:2],max(2,int(np.ceil(np.linalg.norm(x[:2]-previous)/self.policy.spacing))+1))
        self.route_visited[self.route_indices(segment)]=True;self.last_route_position=x[:2].copy()
        if self.policy.path is None or self.policy.target is None:return
        points,_=prefix_points(self.policy.path,spacing=self.policy.spacing)
        ids=np.union1d(np.flatnonzero(self.route_visited),self.route_indices(points));self.route_current_ids=ids
        precision=float(np.sqrt(2*np.mean(sigma[ids]**2)))
        if allow_grid_handoff and self.route_handoff_ids is None and precision<=self.vector_precision:
            self.route_handoff_ids=ids;self.route_goal=self.policy.nodes[self.policy.target].tolist()

    def route_metric(self,tables):
        ids=self.route_handoff_ids if self.route_handoff_ids is not None else self.route_current_ids
        if ids is None:return None
        mu,sigma=self.moments(tables.mean,tables.std)
        sigma=(route_standard_deviation(tables.std,self.policy.nodes[ids]) if isinstance(tables.std,PosteriorStandardDeviation)
               else np.maximum(np.asarray(sigma)[ids],0.))
        precision=float(np.sqrt(2*np.mean(sigma**2)))
        if self.route_handoff_ids is None and self.policy.target is not None and precision<=self.vector_precision:
            self.route_handoff_ids=ids.copy();self.route_goal=self.policy.nodes[self.policy.target].tolist()
        return dict(vector_std_m_s=precision,
            positions=self.policy.nodes[ids],mean=np.asarray(mu)[ids],std=sigma,
            handoff_route_selected=self.route_handoff_ids is not None,goal=self.route_goal)

    def plan(self,x,q,b,cache,tables,clock,local_mean,local_std):
        began=perf_counter();mu,sigma=self.moments(tables.mean,tables.std)
        mu=np.asarray(mu);sigma=np.maximum(np.asarray(sigma),0.);variance=sigma*sigma
        self.update_route_precision(x,sigma,allow_grid_handoff=not isinstance(tables.std,PosteriorStandardDeviation))
        R=self.cfg.safety_confidence*np.sqrt(variance+.01);norm=np.linalg.norm(mu,axis=1);k=.1/.680
        error=k*(2*norm*R+R*R);drift=k*norm[:,None]*mu
        # At rest, u=-E[d] meets the robust wall/circle braking inequalities
        # whenever error <= lambda² h. Also retain the shared authority reserve.
        admissible=(np.max(np.abs(drift),axis=1)+error<=.75*self.cfg.umax)&(error<=1.5**2*self.gap)
        policy=self.policy;allowed=policy.free.copy() if self.global_goals else policy.free&admissible
        route_admissible=None;observed=np.zeros(len(allowed),bool)
        if self.global_goals and self.learned_route_bounds:
            actual_positions=[z for z,_ in policy.history]+[x[:2]]
            route_admissible,_,observed=measured_route_mask(policy.nodes,mu,sigma,actual_positions,self.cfg.umax,confidence=self.cfg.safety_confidence)
            allowed&=route_admissible
        near=np.flatnonzero(allowed&(np.linalg.norm(policy.nodes-x[:2],axis=1)<=policy.spacing*np.sqrt(2)+1e-9))
        near=near[visible(x[:2],policy.nodes[near],policy.ob,policy.bounds,0.)]
        reachable=np.zeros(len(allowed),bool)
        if len(near):
            start=near[np.argmin(np.linalg.norm(policy.nodes[near]-x[:2],axis=1))]
            edges=allowed[policy.rows]&allowed[policy.cols]
            if self.global_goals:
                velocity=policy.speed*(policy.nodes[policy.cols]-policy.nodes[policy.rows])/policy.weights[:,None]
                for endpoint in (policy.rows,policy.cols):
                    relative=mu[endpoint]-velocity
                    force=k*np.linalg.norm(relative,axis=1)[:,None]*relative
                    edges&=np.max(np.abs(force),axis=1)<1.2
            graph=csr_matrix((policy.weights[edges],(policy.rows[edges],policy.cols[edges])),shape=policy.graph.shape)
            reachable=np.isfinite(dijkstra(graph,indices=start))&allowed
        weights=np.zeros(len(self.coarse_nodes));free=self.grid['free_indices']
        weights[free]=1. if self.kind=='all' else deficit_weights(q,self.env.targets,self.env.process_noise,self.cfg.horizon)
        if self.task_weights is not None:
            weights=np.asarray(self.task_weights,float).copy()
            if weights.shape!=(len(self.coarse_nodes),) or not np.isfinite(weights).all() or np.any(weights<0):
                raise ValueError('Task-domain weights must be finite, nonnegative and match the map')
        weights/=max(weights.sum(),1e-12)
        gain_nodes=self.coarse_nodes;gain_weights=weights;gain_kernel=None
        if self.task_reference is not None:
            reference=jnp.asarray(self.task_reference)
            # Use the same current field interpolation as execution, but use
            # exact POAM cross-covariance for expected information gain.
            reference_mean=self.field_basis.evaluate(reference[:,:2],tables.mean)
            reference_std=jnp.maximum(self.field_basis.evaluate(reference[:,:2],tables.std)[:,0],0.)
            need=jnp.asarray(deficit_weights(q,self.env.targets,self.env.process_noise,self.cfg.horizon))
            gain_weights=np.asarray(reference_information_weights(reference,reference_mean,reference_std**2,
                need,self.env.cells,self.cfg.sensor_sigma,downstream=self.downstream_information))
            gain_nodes=reference[:,:2]
            distance=np.sum((np.asarray(self.coarse_nodes)[:,None]-np.asarray(gain_nodes)[None])**2,axis=-1)
            shortlist=(np.exp(-distance/(2*.25**2))@gain_weights)
            weights[:]=0.;weights[free]=shortlist[free];weights/=max(weights.sum(),1e-30)
        score=np.where(reachable,variance*weights[self.coarse_index],-np.inf)
        separation=.35 if self.global_goals else .05
        candidates=distinct_candidates(score,policy.nodes,16,spacing=separation)
        score=np.where(reachable,variance,-np.inf)
        for i in candidates:score[np.linalg.norm(policy.nodes-policy.nodes[i],axis=1)<separation]=-np.inf
        candidates+=distinct_candidates(score,policy.nodes,32-len(candidates),spacing=separation)
        priority=np.zeros(len(allowed))
        if candidates:
            candidates=np.pad(np.asarray(candidates,int),(0,32-len(candidates)),mode='edge')
            if policy.target is not None and reachable[policy.target]:candidates[-1]=policy.target
            cache.refresh(b)
            gain_kernel=cache.Kmap if self.task_reference is None else b.kernel(gain_nodes,b.basis.centers)
            gain,direct_variance=observation_gain_and_variance(b,gain_nodes,gain_kernel,cache.posterior[1],
                jnp.asarray(gain_weights),jnp.asarray(policy.nodes[candidates]))
            variance=acquisition_variances(variance,candidates,direct_variance)
            priority[candidates]=np.maximum(np.asarray(gain),0.)
        path,decision=policy.plan(x,variance,clock,mu,local_mean,local_std,priority=priority,
            admissible_nodes=route_admissible if self.global_goals else admissible)
        if self.global_goals and not decision['reason'].startswith('retreat'):
            decision['global_geometric_path']=path.tolist()
            points,padded=prefix_points(path)
            accepted=np.asarray(self.supported(jnp.asarray(padded),tables.mean,tables.std))[:len(points)]
            bad=np.flatnonzero(~accepted)
            if len(bad):points=points[:max(1,int(bad[0]))]
            path=points
            decision['global_information_goal']=True
        _,cov=cache.map(b)
        task_std=float(np.sqrt(2*np.dot(weights,np.asarray(cov)[:,0,0])))
        decision.update(acquisition='reachable integrated variance reduction',acquisition_seconds=perf_counter()-began,
            waypoint_variance='direct full POAM posterior at candidate locations',
            admissible_nodes=int(allowed.sum()),reachable_nodes=int(reachable.sum()),
            candidate_count=int(len(np.unique(candidates))),task_vector_std_m_s=task_std,
            measured_route_exclusions=int(np.sum(observed&~allowed&policy.free)),
            temporary_passage_cuts=policy.cuts[policy.blocked_cuts].tolist(),
            route_precision_cells=0 if self.route_current_ids is None else len(self.route_current_ids),
            information_reference=None if self.task_reference is None else np.asarray(self.task_reference).tolist(),
            geometric_spacing_m=policy.spacing)
        return path,decision,weights,priority
