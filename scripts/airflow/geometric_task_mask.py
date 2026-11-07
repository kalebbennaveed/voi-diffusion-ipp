"""Clarity-valued candidate-path wind learning, with no access to wind truth.

The compact path tube is an acquisition-domain approximation, not a wind or
robot model. Full posterior covariance, including its conditional residual,
determines the value of measurements. The original MBD seed generator is used
unchanged; this module neither optimizes a trajectory nor updates the belief.
"""
import numpy as np
from time import perf_counter
from monitoring_routes import coarse_paths
from task_wind_acquisition import TaskWindAcquisition, deficit_weights


def candidate_path_weights(nodes, paths, cells, need, free_indices,
                           sensor_sigma=.5, tube_radius=.25):
    """Normalized compact tubes valued by the deficits served at their ends.

    A route's transit points inherit its endpoint sensing value; a pointwise
    clarity mask would incorrectly drop those connecting passages. Use the
    maximum radial weight across segments so adding vertices cannot increase
    a route's importance. Identical geometric paths are counted once.
    """
    if tube_radius <= 0 or sensor_sigma <= 0:
        raise ValueError('Positive geometric tube and sensor scales required')
    nodes=np.asarray(nodes);cells=np.asarray(cells);need=np.asarray(need)
    weights=np.zeros(len(nodes));details=[];seen=set()
    for route in paths:
        route=np.asarray(route,float)
        if len(route)<2:continue
        route=route[np.r_[True,np.linalg.norm(np.diff(route,axis=0),axis=1)>1e-10]]
        if len(route)<2:continue
        key=tuple(np.round(route,9).ravel())
        if key in seen:continue
        seen.add(key)
        value=float(np.dot(need,np.exp(-np.sum((cells-route[-1])**2,axis=1)/sensor_sigma**2)))
        if value<=0:continue
        distance=np.full(len(nodes),np.inf)
        for start,end in zip(route[:-1],route[1:]):
            delta=end-start
            fraction=np.clip((nodes-start)@delta/np.dot(delta,delta),0.,1.)
            distance=np.minimum(distance,np.linalg.norm(nodes-start-fraction[:,None]*delta,axis=1))
        r=np.minimum(distance/tube_radius,1.)
        tube=(1-r)**4*(1+4*r) # compact, C2 Wendland taper
        weights+=value*tube
        details.append(dict(path=route.tolist(),endpoint_sensing_value=value,
                            length_m=float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum())))
    allowed=np.zeros(len(nodes),bool);allowed[np.asarray(free_indices,int)]=True
    weights[~allowed]=0.
    if weights.sum()>0:weights/=weights.sum()
    return weights,details


class CandidatePathAcquisition(TaskWindAcquisition):
    """The same one-observation Bayesian acquisition on a restricted domain."""
    def __init__(self,env,grid,cfg,tube_radius=.25):
        super().__init__(env,grid,cfg,kind='correlated')
        self.tube_radius=tube_radius

    def task_weights(self,x,q):
        began=perf_counter()
        paths,geometry=coarse_paths(np.asarray(x),np.asarray(q),self.env,self.cfg)
        need=deficit_weights(q,self.env.targets,self.env.process_noise,self.cfg.horizon)
        weights,details=candidate_path_weights(self.nodes,paths,np.asarray(self.env.cells),need,
            self.free,self.cfg.sensor_sigma,self.tube_radius)
        return weights,dict(geometric_paths=[d['path'] for d in details],route_values=details,
            geometry=geometry,tube_radius_m=self.tube_radius,
            active_cells=int(np.count_nonzero(weights)),free_cells=int(len(self.free)),
            domain_seconds=perf_counter()-began)

    def compute(self,x,q,b,cache,mean,variance,persistent_index=None,reachable=None):
        weights,metadata=self.task_weights(x,q)
        # ReachableLearning uses these exact weights with its current directed
        # geometric graph and off-grid measurement candidates. Avoid doing a
        # second, unused acquisition here merely for the display.
        priority=weights*np.asarray(variance)
        return priority,weights,dict(acquisition='clarity-valued candidate paths',
            seconds=metadata['domain_seconds'],
            weighted_vector_std_m_s=float(np.sqrt(2*np.dot(weights,variance))),
            maximum_priority=float(priority.max()),candidate_indices=[],**metadata)
