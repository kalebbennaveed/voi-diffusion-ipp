"""Online clarity-guided graph starts; NEVER the final planning objective.

No region names, reference routes, field truth, or prebuilt graph are used.
SciPy's compiled Dijkstra searches the graph rebuilt on every call. Temporary
cuts across discovered bottlenecks give alternative passages. Approximate no-sensing value only
ranks initial paths; full POAM/physics/filter/clarity score every MBD candidate.
"""
from time import perf_counter
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def clearance(points,obstacles,bounds):
    p=np.asarray(points)
    return np.minimum(np.minimum((p-bounds[0]).min(axis=-1),(bounds[1]-p).min(axis=-1)),
                      (np.linalg.norm(p[...,None,:]-obstacles[:,:2],axis=-1)-obstacles[:,2]).min(axis=-1))


def visible(a,b,obstacles,bounds,margin):
    a=np.asarray(a);b=np.asarray(b);d=b-a
    t=np.clip(np.sum((obstacles[:,:2]-a[...,None,:])*d[...,None,:],axis=-1)/np.maximum(np.sum(d*d,axis=-1)[...,None],1e-30),0,1)
    distance=np.linalg.norm(a[...,None,:]+t[...,None]*d[...,None,:]-obstacles[:,:2],axis=-1)-obstacles[:,2]
    return (distance.min(axis=-1)>=margin)&(clearance(a,obstacles,bounds)>=margin)&(clearance(b,obstacles,bounds)>=margin)



def passage_cuts(obstacles,bounds,margin):
    """Discover narrow free gaps between circle boundaries and workspace walls.

    These cuts are temporary search diversifiers, never physical obstacles.
    No passage name or scenario-specific waypoint is encoded.
    """
    cuts=[]
    for i,a in enumerate(obstacles):
        for b in obstacles[i+1:]:
            delta=b[:2]-a[:2];length=np.linalg.norm(delta);gap=length-a[2]-b[2]
            if 2*margin<gap<1.5:
                direction=delta/length;cuts.append([a[:2]+a[2]*direction,b[:2]-b[2]*direction])
        for axis in (0,1):
            for side in (0,1):
                sign=1 if side else -1;wall=bounds[side,axis];gap=sign*(wall-a[axis])-a[2]
                if 2*margin<gap<1.5:
                    start=a[:2].copy();start[axis]+=sign*a[2];end=a[:2].copy();end[axis]=wall;cuts.append([start,end])
    return np.asarray(cuts).reshape(-1,2,2)


def crosses_cuts(a,b,cuts):
    a=np.atleast_2d(a);b=np.atleast_2d(b)
    if not len(cuts):return np.zeros((len(a),0),bool)
    cross=lambda u,v:u[...,0]*v[...,1]-u[...,1]*v[...,0]
    p=cuts[:,0];r=cuts[:,1]-p;d=b-a
    denominator=cross(d[:,None],r[None]);safe=np.where(np.abs(denominator)>1e-12,denominator,1.)
    t=cross(p[None]-a[:,None],r[None])/safe;u=cross(p[None]-a[:,None],d[:,None])/safe
    return (np.abs(denominator)>1e-12)&(t>=-1e-10)&(t<=1+1e-10)&(u>=-1e-10)&(u<=1+1e-10)


def coarse_paths(x,q,env,cfg,count=6,value_weights=None,admissible_nodes=None):
    started=perf_counter();ob=np.asarray(env.obstacles);bounds=np.asarray(env.bounds);cells=np.asarray(env.cells)
    dx=.125;nx,ny=np.rint((bounds[1]-bounds[0])/dx).astype(int)
    xx,yy=np.meshgrid(bounds[0,0]+(np.arange(nx)+.5)*dx,bounds[0,1]+(np.arange(ny)+.5)*dx)
    nodes=np.c_[xx.ravel(),yy.ravel()];margin=max(cfg.safety_margin+.04,.10)
    free=clearance(nodes,ob,bounds)>=margin
    if admissible_nodes is not None:
        admissible_nodes=np.asarray(admissible_nodes,bool)
        if admissible_nodes.shape!=free.shape:raise ValueError('Admissibility must match the geometric grid')
        free&=admissible_nodes
    rows=[];cols=[];weights=[]
    ids=np.arange(nx*ny).reshape(ny,nx)
    for di,dj in [(1,0),(0,1),(1,1),(-1,1)]:
        j0=max(0,-dj);j1=min(ny,ny-dj);i0=max(0,-di);i1=min(nx,nx-di)
        a=ids[j0:j1,i0:i1].ravel();b=ids[j0+dj:j1+dj,i0+di:i1+di].ravel()
        ok=free[a]&free[b]&visible(nodes[a],nodes[b],ob,bounds,margin)
        a=a[ok];b=b[ok];rows.extend([a,b]);cols.extend([b,a]);weights.extend([np.full(len(a),dx*np.hypot(di,dj))]*2)
    rows=np.concatenate(rows);cols=np.concatenate(cols);weights=np.concatenate(weights)
    graph=csr_matrix((weights,(rows,cols)),shape=(nx*ny,nx*ny))
    cuts=passage_cuts(ob,bounds,margin);edge_cuts=crosses_cuts(nodes[rows],nodes[cols],cuts)
    possible=np.flatnonzero(free);order=possible[np.argsort(np.linalg.norm(nodes[possible]-x[:2],axis=1))]
    connected=order[visible(np.broadcast_to(x[:2],(len(order),2)),nodes[order],ob,bounds,0.)]
    if admissible_nodes is not None:
        connected=connected[np.linalg.norm(nodes[connected]-x[:2],axis=1)<=dx*np.sqrt(2)+1e-9]
    if not len(connected):return [np.array([x[:2]])]*count,dict(milliseconds=(perf_counter()-started)*1000,paths=0)
    start=int(connected[0]);dist,pred=dijkstra(graph,indices=start,return_predecessors=True,limit=cfg.horizon_seconds*.9)
    full_indices=np.rint((cells[:,1]-bounds[0,1])/dx-.5).astype(int)*nx+np.rint((cells[:,0]-bounds[0,0])/dx-.5).astype(int)
    # Exact no-observation Riccati solution, then approximate footprint value.
    times=np.array([.25,.5,1.])*cfg.horizon_seconds
    future=np.asarray(q)[None]/(1+times[:,None]*np.asarray(env.process_noise)[None]*np.asarray(q)[None])
    need=np.maximum(np.asarray(env.targets)[None]-future,0).mean(axis=0)
    if value_weights is not None:
        need=np.asarray(value_weights)
        if need.shape!=(len(cells),) or not np.isfinite(need).all() or np.any(need<0):
            raise ValueError('Geometric acquisition weights must match the free cells and be nonnegative')
    raw=np.zeros(nx*ny);raw[full_indices]=need
    value=gaussian_filter(raw.reshape(ny,nx),cfg.sensor_sigma/(np.sqrt(2)*dx),mode='constant').ravel()
    interior=np.clip(clearance(nodes,ob,bounds)/.5,0,1)
    score=value*interior/(1+.04*dist/.9);score[~np.isfinite(dist)|~free]=-np.inf
    selected=[];ranking=score.copy()
    for _ in range(4):
        i=int(np.argmax(ranking))
        if not np.isfinite(ranking[i]):break
        selected.append(i);ranking[np.linalg.norm(nodes-nodes[i],axis=1)<1.15]=-np.inf
    if not selected:return [np.array([x[:2]])]*count,dict(milliseconds=(perf_counter()-started)*1000,paths=0)
    def recover(tree,end,origin=start):
        route=[int(end)]
        for _ in range(nx*ny):
            if route[-1]==origin:break
            previous=int(tree[route[-1]])
            if previous<0:return []
            route.append(previous)
        return route[::-1]
    paths=[];destinations=[];mode_nodes=[];mode_cuts=[]
    for goal in selected:
        route=recover(pred,goal)
        if route:mode_nodes.append(route);destinations.append(goal);mode_cuts.append(np.zeros(len(cuts),bool))
    # Temporarily cut the bottlenecks used by each previous route. Merely
    # penalizing a thin band can return a parallel path through the SAME gap.
    # All cuts are inferred from geometry and affect initialization only.
    goal=max(selected,key=lambda g:dist[g]);initial=recover(pred,goal);route=initial
    blocked=np.zeros(len(cuts),bool)
    for _ in range(count-len(mode_nodes)):
        blocked|=crosses_cuts(nodes[route[:-1]],nodes[route[1:]],cuts).any(axis=0)
        allowed=~np.any(edge_cuts&blocked[None],axis=1)
        alternative=csr_matrix((weights[allowed],(rows[allowed],cols[allowed])),shape=graph.shape)
        _,tree=dijkstra(alternative,indices=start,return_predecessors=True,limit=cfg.horizon_seconds*.9*2)
        candidate=recover(tree,goal)
        if not candidate:
            candidate=initial;applied=np.zeros(len(cuts),bool)
        else:applied=blocked.copy()
        route=candidate;mode_nodes.append(route);destinations.append(goal);mode_cuts.append(applied)
    for mode_index,route in enumerate(mode_nodes):
        raw_path=np.vstack([x[:2],nodes[route]])
        # Visibility pruning preserves each collision-free passage exactly.
        simplified=[raw_path[0]];i=0
        while i<len(raw_path)-1:
            remaining=raw_path[i+1:];ok=visible(np.broadcast_to(raw_path[i],remaining.shape),remaining,ob,bounds,margin)
            if admissible_nodes is not None:
                # Keep the accepted grid corridor. Retiming and the subsequent
                # full filtered forecast still check the actual candidate.
                ok[1:]=False
            # Remove grid-induced detours by straight visibility chords.
            # Each chord is checked against the analytic inflated circles.
            ok &= ~crosses_cuts(np.broadcast_to(raw_path[i],remaining.shape),remaining,cuts[mode_cuts[mode_index]]).any(axis=1)
            candidates=np.flatnonzero(ok);j=i+1+(int(candidates[-1]) if len(candidates) else 0)
            simplified.append(raw_path[j]);i=j
        paths.append(np.asarray(simplified))
    return paths,dict(milliseconds=(perf_counter()-started)*1000,paths=len(paths),nodes=int(free.sum()),edges=len(rows),
                      initial_destinations=nodes[destinations].tolist(),grid=[nx,ny],search='Compiled SciPy Dijkstra; online graph; temporary cuts at discovered bottlenecks',margin_m=margin,passage_cuts=cuts.tolist(),mode_blocked_cuts=[cuts[c].tolist() for c in mode_cuts])


def reference_seed(path,x,cfg):
    """Turn rough waypoints into nominal double-integrator accelerations.

    This is initialization only. No wind plant is approximated here: it
    generates the reference to be tracked, not a predicted actual trajectory.
    All coordinates are subsequently free MBD variables.
    """
    state=np.asarray(x).copy();controls=[];target=1;path=np.asarray(path)
    for _ in range(cfg.steps):
        while target<len(path)-1 and np.linalg.norm(path[target]-state[:2])<.20:target+=1
        point=path[min(target,len(path)-1)];delta=point-state[:2];distance=np.linalg.norm(delta)
        # Slower corners keep the nominal reference inside a narrow passage.
        speed=min(.9,1.4*distance)
        desired_v=delta/max(distance,1e-12)*speed
        a=np.clip(2.5*(desired_v-state[2:]),-.8*cfg.umax,.8*cfg.umax)
        controls.append(a);state=np.r_[state[:2]+cfg.dt*state[2:]+.5*cfg.dt**2*a,state[2:]+cfg.dt*a]
    return np.arctanh(np.asarray(controls)/cfg.umax).ravel()


def seed_bank(x,q,env,cfg,previous=None):
    began=perf_counter();paths,metadata=coarse_paths(np.asarray(x),np.asarray(q),env,cfg)
    hold=reference_seed(np.asarray(x[:2])[None],x,cfg)
    if previous is None:previous=hold
    else:
        a=np.asarray(previous).reshape(cfg.steps,2)
        previous=np.concatenate((a[cfg.commit_steps:],np.zeros_like(a[:cfg.commit_steps]))).ravel()
    references=[smooth_reference_seed(p,x,env,cfg) for p in paths]
    seeds=np.stack([previous,hold,*[r[0] for r in references]])
    metadata['retiming_fallbacks']=[r[1] for r in references]
    metadata['total_seed_milliseconds']=(perf_counter()-began)*1000;metadata['coarse_paths']=[p.tolist() for p in paths]
    return seeds,metadata


def smooth_reference_seed(path,state,env,cfg):
    """Original geometric seed retiming, adapted to analytic circle clearance.

    Round corners, impose curvature/tangential acceleration limits, integrate
    constant reference accelerations exactly. This changes no physical model.
    A failed geometric retiming falls back to the reported waypoint seed.
    """
    path=np.array(path,float);state=np.array(state,float);ob=np.asarray(env.obstacles);bounds=np.asarray(env.bounds)
    margin=cfg.safety_margin+.015
    def safe(points):return np.all(visible(points[:-1],points[1:],ob,bounds,margin))
    if len(path)<2 or np.linalg.norm(path[-1]-path[0])<.05:return reference_seed(path,state,cfg),False
    n=max(1,int(np.ceil(np.max(np.abs(state[2:]))/(.75*cfg.umax*cfg.dt)))) if np.linalg.norm(state[2:])>1e-8 else 0
    prefix=np.tile(-state[2:]/max(n*cfg.dt,1e-10),(n,1))
    stopped=state.copy();stopped[:2]+=state[2:]*n*cfg.dt/2;stopped[2:]=0
    if not visible(state[:2],stopped[:2],ob,bounds,margin):return reference_seed(path,state,cfg),True
    path[0]=stopped[:2];lengths=np.linalg.norm(np.diff(path,axis=0),axis=1);path=path[np.r_[True,lengths>1e-6]]
    lengths=np.linalg.norm(np.diff(path,axis=0),axis=1);pieces=[path[:1]];last=path[0]
    for i in range(1,len(path)):
        if i<len(path)-1:
            before=(path[i]-path[i-1])/lengths[i-1];after=(path[i+1]-path[i])/lengths[i]
            distance=min(.44*lengths[i-1],.44*lengths[i],.8)
            if np.dot(before,after)<-.98:distance=0
            for _ in range(16):
                entry=path[i]-distance*before;exit=path[i]+distance*after
                t=np.linspace(0,1,max(9,int(2*distance/.025)+1))[:,None]
                curve=(1-t)**2*entry+2*(1-t)*t*path[i]+t*t*exit
                if safe(curve):break
                distance*=.65
        else:entry=exit=path[i];curve=np.array([path[i]])
        pieces.append(np.linspace(last,entry,max(2,int(np.linalg.norm(entry-last)/.025)+1))[1:]);pieces.append(curve[1:]);last=exit
    curve=np.vstack(pieces);curve=curve[np.r_[True,np.linalg.norm(np.diff(curve,axis=0),axis=1)>1e-8]]
    if len(curve)<3:return reference_seed(path,state,cfg),True
    ds=np.linalg.norm(np.diff(curve,axis=0),axis=1);arc=np.r_[0,np.cumsum(ds)]
    tangent=np.gradient(curve,arc,axis=0);tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-10)
    curvature=np.linalg.norm(np.gradient(tangent,arc,axis=0),axis=1)
    speed=np.minimum(1.,np.sqrt(.55*cfg.umax/np.maximum(curvature,1e-9)));speed[[0,-1]]=0
    for i in range(1,len(speed)):speed[i]=min(speed[i],np.sqrt(speed[i-1]**2+cfg.umax*ds[i-1]))
    for i in range(len(speed)-2,-1,-1):speed[i]=min(speed[i],np.sqrt(speed[i+1]**2+cfg.umax*ds[i]))
    times=np.r_[0,np.cumsum(2*ds/np.maximum(speed[:-1]+speed[1:],1e-10))]
    for slowdown in (1.,1.2,1.5,2.):
        count=max(2,int(np.ceil(times[-1]*slowdown/cfg.dt)));query=np.arange(count+1)*cfg.dt/slowdown
        ar=np.interp(query,times,arc);direction=np.c_[np.interp(ar,arc,tangent[:,0]),np.interp(ar,arc,tangent[:,1])]
        direction/=np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-10)
        velocity=direction*(np.interp(query,times,speed)/slowdown)[:,None]
        padded=np.pad(velocity,((1,1),(0,0)));velocity=.25*(padded[:-2]+2*padded[1:-1]+padded[2:]);velocity[[0,-1]]=0
        residual=curve[-1]-stopped[:2]-.5*cfg.dt*np.sum(velocity[:-1]+velocity[1:],axis=0)
        weight=np.sin(np.linspace(0,np.pi,count+1))**2;velocity+=weight[:,None]*residual/(cfg.dt*weight.sum())
        motion=np.diff(velocity,axis=0)/cfg.dt;all_a=np.vstack([prefix,motion]);acc=np.zeros((cfg.steps,2));acc[:min(len(acc),len(all_a))]=all_a[:cfg.steps]
        vv=np.vstack([state[2:],state[2:]+cfg.dt*np.cumsum(acc,axis=0)])
        pp=np.vstack([state[:2],state[:2]+.5*cfg.dt*np.cumsum(vv[:-1]+vv[1:],axis=0)])
        mid=pp[:-1]+.5*cfg.dt*vv[:-1]+.125*cfg.dt**2*acc
        dense=np.empty((2*len(acc)+1,2));dense[::2]=pp;dense[1::2]=mid
        if np.max(np.abs(acc))<=.96*cfg.umax and safe(dense):return np.arctanh(acc/cfg.umax).ravel(),False
    return reference_seed(path,state,cfg),True
