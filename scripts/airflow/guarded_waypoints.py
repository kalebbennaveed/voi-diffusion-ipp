"""Uncertainty waypoints with measured-path actuator-reserve recovery.

No true wind, new objective or artificial measurements. Enter recovery while
75% of control authority suffices for the learned 3-sigma wind envelope at
rest; return along an executed path to a previously measured low-force point.
The RPCBF still filters the resulting waypoint tracking command.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from uncertainty_waypoints import WaypointPolicy
from monitoring_routes import visible,clearance


def holding_force_bound(mean,sigma,confidence=3.):
    """Existing exact nonlinear support enclosure, NumPy for waypoint dispatch."""
    k=.1/.680;mean=np.asarray(mean);R=confidence*np.sqrt(max(float(sigma),0)**2+.1**2)
    directions=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    drift=k*np.linalg.norm(mean)*mean
    coarse=directions@drift+k*(2*np.linalg.norm(mean)*R+R*R)
    angles=np.arange(32)*2*np.pi/32;wind=mean+R*np.c_[np.cos(angles),np.sin(angles)]
    force=k*np.linalg.norm(wind,axis=1)[:,None]*wind
    remainder=2*k*R*(np.linalg.norm(mean)+R)*np.pi/32
    return float(np.minimum(coarse,np.max(directions@force.T,axis=1)+remainder).max())


class GuardedWaypointPolicy(WaypointPolicy):
    def __init__(self,env,umax=1.6,confidence=3.,**kwargs):
        super().__init__(env,**kwargs);self.umax=umax;self.confidence=confidence;self.history=[];self.retreat=None;self.trigger=None;self.retreat_cursor=0
        self.recovery_anchor=None;self.recovery_since=None;self.recovery_use_graph=False;self.recovery_goal=None
        self.recovery_failed_goals=[];self.node_tree=cKDTree(self.nodes)

    def recovery_geometric_path(self,p,goal,wind_mean,local_mean):
        """Use the existing directed geometric graph to leave a stalled return.

        A historical point can be geometrically visible but unreachable by a
        direct upwind command. Here edges require nominal compensation within
        the actual actuator limits. This is reference initialization only;
        full uncertainty and RPCBF still govern every executed command.
        """
        mu=np.asarray(wind_mean);direction=(self.nodes[self.cols]-self.nodes[self.rows])/self.weights[:,None]
        velocity=self.speed*direction;allowed=np.ones(len(self.rows),bool)
        for endpoint in (self.rows,self.cols):
            relative=mu[endpoint]-velocity
            force=(.1/.680)*np.linalg.norm(relative,axis=1)[:,None]*relative
            allowed&=np.max(np.abs(force),axis=1)<=self.umax
        radius=max(self.reference_lookahead,self.spacing*np.sqrt(2))
        distance_from_robot=np.linalg.norm(self.nodes-p,axis=1)
        starts=np.flatnonzero(self.free&(distance_from_robot>=radius)&(distance_from_robot<=radius+self.spacing*np.sqrt(2)))
        starts=starts[visible(p,self.nodes[starts],self.ob,self.bounds,self.margin)]
        # Connect the ACTUAL state, not a nearby optimistic grid cell. Keep
        # this first leg at least one tracking lookahead long, so reference
        # interpolation cannot immediately shortcut into an upwind command.
        velocity=min(self.speed,radius)*(self.nodes[starts]-p)/distance_from_robot[starts,None]
        relative=np.asarray(local_mean)[None]-velocity
        force=(.1/.680)*np.linalg.norm(relative,axis=1)[:,None]*relative
        starts=starts[np.max(np.abs(force),axis=1)<=self.umax]
        # A stalled historical endpoint need not be reachable against wind.
        # Consider all actual previously visited low-force refuges. This is
        # reference routing only; the live filter still checks every command.
        refuges=[np.asarray(goal)] if self.recovery_goal is None else [self.recovery_goal]
        if self.recovery_goal is None:
            refuges += [z for z,f in self.history if f<.5*self.umax and np.linalg.norm(z-p)>=.5]
            alternatives=[z for z in refuges if all(np.linalg.norm(z-bad)>=.3 for bad in self.recovery_failed_goals)]
            if alternatives:refuges=alternatives
        # Deduplicate measured refuges on the existing geometry grid only.
        _,indices=self.node_tree.query(refuges)
        refuges=np.asarray(refuges)[np.unique(indices,return_index=True)[1]]
        goals=[];endpoints=[];terminal_cost=[]
        for refuge in refuges:
            ids=np.asarray(self.node_tree.query_ball_point(refuge,2*self.spacing),int)
            ids=ids[self.free[ids]]
            ids=ids[visible(refuge,self.nodes[ids],self.ob,self.bounds,self.margin)]
            goals.extend(ids);endpoints.extend([refuge]*len(ids));terminal_cost.extend(np.linalg.norm(self.nodes[ids]-refuge,axis=1))
        goals=np.asarray(goals,int)
        if not len(starts) or not len(goals):return None
        start=len(self.nodes)
        graph=csr_matrix((np.r_[self.weights[allowed],distance_from_robot[starts]],
            (np.r_[self.rows[allowed],np.full(len(starts),start)],np.r_[self.cols[allowed],starts])),
            shape=(start+1,start+1))
        distance,predecessor=dijkstra(graph,indices=start,return_predecessors=True)
        cost=distance[goals]+np.asarray(terminal_cost)
        if not np.isfinite(cost).any():return None
        chosen=int(np.argmin(cost));goal=np.asarray(endpoints[chosen])
        route=[int(goals[chosen])]
        while route[-1]!=start:
            previous=int(predecessor[route[-1]])
            if previous<0:raise RuntimeError('Finite recovery route has no predecessor')
            route.append(previous)
        self.recovery_goal=goal.copy()
        return np.vstack((p,self.nodes[route[-2::-1]],goal))
    def plan(self,x,variance,clock,wind_mean=None,local_mean=None,local_std=None,priority=None,admissible_nodes=None):
        if local_mean is None or local_std is None:raise ValueError('The actual learned local wind moments are required')
        p=np.asarray(x[:2]);force=holding_force_bound(local_mean,local_std,self.confidence)
        self.history.append((p.copy(),force))
        if self.retreat is None and force>.75*self.umax:
            safe=[i for i,(z,f) in enumerate(self.history[:-1]) if f<.5*self.umax and np.linalg.norm(z-p)>=.5]
            if safe:
                idx=safe[-1];self.retreat=np.array([z for z,_ in self.history[idx:]])[::-1];self.trigger=p.copy();self.retreat_cursor=0
                self.cooldown[np.linalg.norm(self.nodes-p,axis=1)<1.0]=clock+30
                if not self.diversify_stalls:self.target=None
        if self.retreat is not None:
            goal=self.retreat[-1]
            regained=(force<.5*self.umax and np.linalg.norm(x[2:])<=self.speed+.1
                      and float(clearance(p[None],self.ob,self.bounds)[0])>=self.margin)
            if regained:
                # Wind may carry the robot to a different safe place. Returning
                # to the obsolete goal can require crossing the same jet again.
                # Entry/exit force thresholds already provide hysteresis; a
                # distance-from-trigger requirement can trap a safe robot.
                self.retreat=None
                self.recovery_anchor=None;self.recovery_since=None;self.recovery_use_graph=False;self.recovery_goal=None;self.recovery_failed_goals=[]
                if not self.diversify_stalls:self.target=None
            else:
                if self.recovery_anchor is None:
                    self.recovery_anchor=p.copy();self.recovery_since=clock
                elif clock-self.recovery_since>=15:
                    stalled=np.linalg.norm(p-self.recovery_anchor)<.15
                    self.recovery_use_graph|=stalled
                    if stalled:
                        # Retire the failed reference, not the physical region.
                        self.recovery_failed_goals.append(np.asarray(goal if self.recovery_goal is None else self.recovery_goal).copy())
                        self.recovery_goal=None
                    self.recovery_anchor=p.copy();self.recovery_since=clock
                if self.recovery_use_graph and wind_mean is not None:
                    alternate=self.recovery_geometric_path(p,goal,wind_mean,local_mean)
                    if alternate is not None:
                        return alternate,dict(reason='retreat: directed route after measured stagnation',
                            target=alternate[-1].tolist(),holding_force_bound_m_s2=force,
                            path_length_m=float(np.linalg.norm(np.diff(alternate,axis=0),axis=1).sum()),
                            target_variance=None,score=None,recovery_graph=True)
                # Follow the recorded path backwards. Visibility is checked
                # before skipping close samples; do not cut obstacle corners.
                tail=self.retreat[self.retreat_cursor:];distance=np.linalg.norm(tail-p,axis=1)
                # The last recovery point can be nearer than a normal path
                # lookahead. It must remain actionable until actually reached.
                candidates=np.flatnonzero((distance>=.02)&visible(p,tail,self.ob,self.bounds,self.margin))
                target=p.copy()
                rejoining=np.min(np.linalg.norm(self.retreat-p,axis=1))>.15
                if rejoining and len(candidates):
                    # Wind can displace the robot outside the recorded tube.
                    # Requiring the connector's *start* to already lie inside
                    # it made every connector fail and commanded hold forever.
                    # Rejoin the nearest visible remaining history point. The
                    # whole chord has analytic obstacle clearance; actual
                    # tracking still uses the unchanged full-posterior RPCBF.
                    candidate=candidates[np.argmin(distance[candidates])]
                    target=tail[candidate];self.retreat_cursor+=int(candidate)
                for candidate in ([] if rejoining else candidates[::-1]):
                    # Collapse loops only inside the already traversed tube.
                    # Farthest forward index prevents oscillating back to an
                    # earlier recovery waypoint when the history crosses itself.
                    segment=np.linspace(p,tail[candidate],max(2,int(distance[candidate]/.05)+1))
                    tube=np.linalg.norm(segment[:,None]-self.retreat[None],axis=-1).min(axis=1)
                    if np.max(tube)<=.15:
                        target=tail[candidate];self.retreat_cursor+=int(candidate);break
                return np.vstack([p,target]),dict(reason='retreat: learned wind threatens actuator reserve',
                    target=goal.tolist(),holding_force_bound_m_s2=force,path_length_m=float(np.linalg.norm(np.diff(tail,axis=0),axis=1).sum()),
                    rejoining_measured_path=bool(rejoining),
                    target_variance=None,score=None)
        path,decision=super().plan(x,variance,clock,wind_mean,priority=priority,admissible_nodes=admissible_nodes)
        decision['holding_force_bound_m_s2']=force
        return path,decision
