"""Wind-variance waypoints with a geometric path; RPCBF filters all execution.

No MBD, clarity objective, hidden wind, synthetic measurements, or route labels
are inputs to this policy. The posterior is changed by the sensor adapter only.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from monitoring_routes import clearance, visible,passage_cuts,crosses_cuts


class WaypointPolicy:
    def __init__(self, env, spacing=.125, margin=.10, speed=.4, learned_variance=.25,
                 minimum_target_distance=.35,reference_lookahead=0.,require_target_precision=False,diversify_stalls=False):
        self.ob=np.asarray(env.obstacles); self.bounds=np.asarray(env.bounds)
        self.margin=margin; self.speed=speed; self.learned_variance=learned_variance
        self.require_target_precision=require_target_precision
        self.diversify_stalls=diversify_stalls
        self.spacing=spacing;self.minimum_target_distance=minimum_target_distance;self.reference_lookahead=reference_lookahead
        nx,ny=np.rint((self.bounds[1]-self.bounds[0])/spacing).astype(int)
        xx,yy=np.meshgrid(self.bounds[0,0]+(np.arange(nx)+.5)*spacing,
                          self.bounds[0,1]+(np.arange(ny)+.5)*spacing)
        self.nodes=np.c_[xx.ravel(),yy.ravel()]
        self.free=clearance(self.nodes,self.ob,self.bounds)>=margin
        self.reachable=self.free.copy()
        ids=np.arange(nx*ny).reshape(ny,nx); rr=[];cc=[];ww=[]
        for di,dj in [(1,0),(0,1),(1,1),(-1,1)]:
            j0=max(0,-dj);j1=min(ny,ny-dj);i0=max(0,-di);i1=min(nx,nx-di)
            a=ids[j0:j1,i0:i1].ravel();b=ids[j0+dj:j1+dj,i0+di:i1+di].ravel()
            ok=self.free[a]&self.free[b]&visible(self.nodes[a],self.nodes[b],self.ob,self.bounds,margin)
            a=a[ok];b=b[ok];rr.extend([a,b]);cc.extend([b,a]);ww.extend([np.full(len(a),spacing*np.hypot(di,dj))]*2)
        self.rows=np.concatenate(rr);self.cols=np.concatenate(cc);self.weights=np.concatenate(ww)
        self.graph=csr_matrix((self.weights,(self.rows,self.cols)),shape=(nx*ny,nx*ny))
        self.cuts=passage_cuts(self.ob,self.bounds,self.margin) if diversify_stalls else np.empty((0,2,2))
        self.edge_cuts=crosses_cuts(self.nodes[self.rows],self.nodes[self.cols],self.cuts)
        self.blocked_cuts=np.zeros(len(self.cuts),bool)
        self.target=None;self.path=None;self.selected_at=0.;self.start_distance=0.;self.progress_checked_at=0.
        self.progress_route_cuts=None
        self.cooldown=np.zeros(len(self.nodes))

    def plan(self, x, variance, clock, wind_mean=None, priority=None,admissible_nodes=None):
        """Choose a high-variance reachable waypoint with travel regularization.

        Candidate score sigma^2/(1+L/1m); exclude within .35m of current
        position. Persist until arrival, substantial learning at the target,
        or measured lack of progress. Stalled targets get a temporary cooldown;
        this is an execution recovery rule, not fictitious wind information.
        """
        p=np.asarray(x[:2]);variance=np.asarray(variance)
        priority=variance if priority is None else np.asarray(priority)
        if priority.shape!=variance.shape or not np.isfinite(priority).all() or (priority<0).any():
            raise ValueError('Waypoint priority must be finite, nonnegative and match the variance grid')
        eligible=self.free if admissible_nodes is None else self.free&np.asarray(admissible_nodes,bool)
        order=np.flatnonzero(eligible);order=order[np.argsort(np.linalg.norm(self.nodes[order]-p,axis=1))]
        connected=order[visible(p,self.nodes[order],self.ob,self.bounds,0.)]
        if admissible_nodes is not None:
            connected=connected[np.linalg.norm(self.nodes[connected]-p,axis=1)<=self.spacing*np.sqrt(2)+1e-9]
        if not len(connected):return np.array([p]),dict(reason='no geometric connection',target=p.tolist())
        start=int(connected[0]);graph=self.graph;allowed=np.ones(len(self.rows),bool)
        excluded=0
        if wind_mean is not None:
            # Exact existing wind-force mapping at the proposed edge velocity.
            # A nominal actuator-feasibility screen, not a safety certificate:
            # full posterior uncertainty still enters execution RPCBF unchanged.
            wind_mean=np.asarray(wind_mean)
            direction=(self.nodes[self.cols]-self.nodes[self.rows])/self.weights[:,None]
            velocity=self.speed*direction
            relative=wind_mean[self.rows]-velocity
            force=(.1/.680)*np.linalg.norm(relative,axis=1)[:,None]*relative
            relative_end=wind_mean[self.cols]-velocity
            force_end=(.1/.680)*np.linalg.norm(relative_end,axis=1)[:,None]*relative_end
            allowed=(np.maximum(np.abs(force),np.abs(force_end)).max(axis=1)<1.2)
            if admissible_nodes is not None:allowed&=eligible[self.rows]&eligible[self.cols]
            excluded=int((~allowed).sum())
            graph=csr_matrix((self.weights[allowed],(self.rows[allowed],self.cols[allowed])),shape=self.graph.shape)
        elif admissible_nodes is not None:
            allowed=eligible[self.rows]&eligible[self.cols]
            graph=csr_matrix((self.weights[allowed],(self.rows[allowed],self.cols[allowed])),shape=self.graph.shape)
        base_allowed=allowed.copy()
        if self.blocked_cuts.any():
            allowed&=~np.any(self.edge_cuts&self.blocked_cuts[None],axis=1)
            graph=csr_matrix((self.weights[allowed],(self.rows[allowed],self.cols[allowed])),shape=self.graph.shape)
        dist,pred=dijkstra(graph,indices=start,return_predecessors=True)
        self.reachable=np.isfinite(dist)&self.free
        reason='persistent target'
        if self.target is not None:
            learned=variance[self.target]<self.learned_variance
            reached=np.linalg.norm(p-self.nodes[self.target])<.18 and (learned or not self.require_target_precision)
            # Check successive windows, not progress since initial selection:
            # one early advance must not disable future stall detection.
            stalled=False
            if clock-self.progress_checked_at>=15:
                changed_route=False
                if self.diversify_stalls and np.isfinite(dist[self.target]):
                    route=[self.target]
                    while route[-1]!=start:
                        previous=int(pred[route[-1]])
                        if previous<0:raise RuntimeError('Finite route has no predecessor')
                        route.append(previous)
                    points=self.nodes[route[::-1]]
                    signature=crosses_cuts(points[:-1],points[1:],self.cuts).any(axis=0)
                    changed_route=(self.progress_route_cuts is not None and
                        not np.array_equal(signature,self.progress_route_cuts))
                    self.progress_route_cuts=signature
                # Learned wind can replace a short passage with a longer one.
                # Distances on those different routes are not evidence of a
                # stationary robot. Start a fresh window on the new route.
                stalled=not changed_route and self.start_distance-dist[self.target]<.3
                if changed_route:reason='route changed: restart progress window'
                self.progress_checked_at=clock;self.start_distance=dist[self.target]
            if stalled and self.diversify_stalls and self.path is not None:
                crossed=crosses_cuts(self.path[:-1],self.path[1:],self.cuts)
                ordered=[]
                for segment in crossed:
                    ordered.extend(i for i in np.flatnonzero(segment&~self.blocked_cuts) if i not in ordered)
                for cut in ordered:
                    trial=self.blocked_cuts.copy();trial[cut]=True
                    edges=base_allowed&~np.any(self.edge_cuts&trial[None],axis=1)
                    alternate=csr_matrix((self.weights[edges],(self.rows[edges],self.cols[edges])),shape=self.graph.shape)
                    new_dist,new_pred=dijkstra(alternate,indices=start,return_predecessors=True)
                    if np.isfinite(new_dist[self.target]):
                        self.blocked_cuts=trial;dist,pred=new_dist,new_pred
                        self.start_distance=dist[self.target];stalled=False
                        self.progress_route_cuts=None
                        reason='stalled passage: alternate geometric route';break
            if reached or learned or stalled or not np.isfinite(dist[self.target]):
                if stalled:self.cooldown[np.linalg.norm(self.nodes-self.nodes[self.target],axis=1)<.5]=clock+30
                reason='arrived' if reached else 'target learned' if learned else 'stalled target cooldown' if stalled else 'target disconnected by learned route bounds'
                self.target=None
                self.blocked_cuts[:]=False
        if self.target is None:
            score=priority/(1+dist)
            score[~eligible|~np.isfinite(dist)|(priority<=0)|(np.linalg.norm(self.nodes-p,axis=1)<self.minimum_target_distance)|(self.cooldown>clock)]=-np.inf
            if self.require_target_precision:score[variance<self.learned_variance]=-np.inf
            if not np.isfinite(score).any():return np.array([p]),dict(reason='no eligible waypoint',target=p.tolist())
            self.target=int(np.argmax(score));self.selected_at=clock;self.start_distance=dist[self.target];self.progress_checked_at=clock
            self.progress_route_cuts=None
        route=[self.target]
        while route[-1]!=start:
            previous=int(pred[route[-1]])
            if previous<0:raise RuntimeError('Unreachable waypoint selected')
            route.append(previous)
        raw=np.vstack([p,self.nodes[route[::-1]]]);path=[raw[0]];i=0
        while i<len(raw)-1:
            valid=np.flatnonzero(visible(raw[i],raw[i+1:],self.ob,self.bounds,self.margin))
            if wind_mean is not None or admissible_nodes is not None:
                # Do not let visibility pruning cut across excluded wind cells.
                # A short local chord stays within the accepted grid corridor.
                valid=valid[valid<=2]
            j=i+1+(int(valid[-1]) if len(valid) else 0);path.append(raw[j]);i=j
        self.path=np.asarray(path)
        if self.diversify_stalls and self.progress_route_cuts is None:
            self.progress_route_cuts=crosses_cuts(self.path[:-1],self.path[1:],self.cuts).any(axis=0)
        return self.path,dict(reason=reason,target=self.nodes[self.target].tolist(),
            target_variance=float(variance[self.target]),path_length_m=float(dist[self.target]),
            score=float(priority[self.target]/(1+dist[self.target])),selected_at=self.selected_at,
            wind_infeasible_edges=excluded)

    def reference(self, x, path):
        """A local velocity reference, replanned from measured state every .2s.

        The double integrator is unchanged. This is an ordinary waypoint
        tracking setpoint; the existing feedback and full RPCBF compute u.
        """
        p=np.asarray(x[:2]);path=np.asarray(path)
        target=path[min(1,len(path)-1)]
        if self.reference_lookahead>0 and len(path)>1:
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
            distance=min(self.reference_lookahead,arc[-1])
            ahead=np.array([np.interp(distance,arc,path[:,axis]) for axis in (0,1)])
            if bool(visible(p,ahead,self.ob,self.bounds,self.margin)):
                target=ahead
            else:
                # A corner may obstruct the interpolated chord. Retain a
                # forward visible vertex; the current point is not progress.
                candidates=np.flatnonzero((arc>1e-9)&(arc<=distance)&visible(p,path,self.ob,self.bounds,self.margin))
                if len(candidates):target=path[candidates[-1]]
        delta=target-p;distance=np.linalg.norm(delta)
        speed=min(self.speed,1.0*distance)
        velocity=delta/max(distance,1e-12)*speed
        return np.r_[p,velocity]
