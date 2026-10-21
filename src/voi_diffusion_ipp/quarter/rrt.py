"""Geometric RRT-Connect demonstrations for MBD (paper Sections 4.3/5.2).

Only known geometry and the current state enter this generator. There are no
passage labels, wind values, prescribed circulation, or fixed route library.
The resulting controls initialize the full trajectory; MBD can change every
time interval independently. RRT planning time is logged separately.
"""
import numpy as np


def segment_free(a, b, rectangles, bounds, margin):
    a, b = np.asarray(a), np.asarray(b)
    if np.any(np.minimum(a,b) < bounds[0]+margin) or np.any(np.maximum(a,b) > bounds[1]-margin):
        return False
    if not len(rectangles): return True
    lo, hi = rectangles[:,:2]-margin, rectangles[:,2:4]+margin
    delta=b-a; moving=np.abs(delta)>1e-12
    near=np.full_like(lo,-np.inf); far=np.full_like(hi,np.inf)
    for d in range(2):
        if moving[d]:
            first=(lo[:,d]-a[d])/delta[d]; last=(hi[:,d]-a[d])/delta[d]
            near[:,d]=np.minimum(first,last); far[:,d]=np.maximum(first,last)
        else:
            outside=(a[d]<lo[:,d]) | (a[d]>hi[:,d])
            near[outside,d]=np.inf; far[outside,d]=-np.inf
    return not np.any(np.maximum(near.max(axis=1),0) <= np.minimum(far.min(axis=1),1))


def connect(start, goal, rectangles, bounds, rng, margin=.65, step=1.2, iterations=3000):
    """Bidirectional RRT-Connect, followed by collision-checked shortcuts."""
    free=lambda a,b:segment_free(a,b,rectangles,bounds,margin)
    if not free(start,start) or not free(goal,goal): return None
    if free(start,goal): return np.array([start,goal])
    nodes=[[np.asarray(start)], [np.asarray(goal)]]; parents=[[-1],[-1]]
    def extend(tree,target):
        points=np.asarray(nodes[tree]); i=int(np.argmin(np.sum((points-target)**2,axis=1)))
        delta=target-points[i]; distance=np.linalg.norm(delta)
        if distance<1e-10: return i,True
        point=points[i]+min(step/distance,1)*delta
        if not free(points[i],point): return None,False
        nodes[tree].append(point); parents[tree].append(i)
        return len(nodes[tree])-1, distance<=step
    def branch(tree,index):
        path=[]
        while index>=0:
            path.append(nodes[tree][index]); index=parents[tree][index]
        return path[::-1]
    for iteration in range(iterations):
        tree=iteration%2
        target=nodes[1-tree][0] if rng.random()<.15 else rng.uniform(bounds[0]+margin,bounds[1]-margin)
        first,_=extend(tree,target)
        if first is None: continue
        target=nodes[tree][first]
        for _ in range(200):
            second,reached=extend(1-tree,target)
            if second is None: break
            if reached:
                a=branch(tree,first); b=branch(1-tree,second)
                path=np.array(a+b[-2::-1])
                if tree==1: path=path[::-1]
                # Removing intermediate vertices never creates a collision.
                for _ in range(100):
                    if len(path)<3: break
                    i,j=sorted(rng.choice(len(path),2,replace=False))
                    if j>i+1 and free(path[i],path[j]): path=np.r_[path[:i+1],path[j:]]
                return path
    return None


def parameterize(path, state, cfg, cruise):
    """Time-parameterize a geometric demonstration with quintic segments.

This creates an initial acceleration at every dt, with the current velocity
at the first segment and zero velocity at geometric corners. These stops are
initialization choices, not constraints on the subsequent MBD optimization.
"""
    accelerations=[]; velocity=np.asarray(state[2:],float)
    for start,end in zip(path[:-1],path[1:]):
        distance=np.linalg.norm(end-start)
        if distance<1e-8: continue
        duration=max(1.875*distance/cruise,np.sqrt(6*distance/(.8*cfg.umax)),.3)
        for _ in range(20):
            count=int(np.ceil(duration/cfg.dt)); duration=count*cfg.dt
            # p(s)=p0 + T*v0*s + c3*s^3+c4*s^4+c5*s^5.
            delta=end-start
            rhs=np.stack([delta-duration*velocity,-duration*velocity,np.zeros(2)])
            coefficients=np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20.]]),rhs)
            query=(np.arange(count)+.5)/count
            acc=(6*query[:,None]*coefficients[0]+12*query[:,None]**2*coefficients[1]+20*query[:,None]**3*coefficients[2])/duration**2
            vel=velocity+(3*query[:,None]**2*coefficients[0]+4*query[:,None]**3*coefficients[1]+5*query[:,None]**4*coefficients[2])/duration
            if np.max(np.abs(acc))<=.9*cfg.umax and np.max(np.linalg.norm(vel,axis=1))<=cfg.speed_max: break
            duration*=1.12
        accelerations.extend(acc); velocity=np.zeros(2)
        if len(accelerations)>=cfg.steps: break
    acceleration=np.zeros((cfg.steps,2))
    if accelerations:
        values=np.asarray(accelerations)[:cfg.steps]; acceleration[:len(values)]=values
    return np.arctanh(np.clip(acceleration/cfg.umax,-.999,.999)).astype(np.float32).ravel()


def demonstrations(state, env, cfg, seed):
    """Regenerate RRT paths to all task regions from the actual reached state."""
    state=np.asarray(state,float); rectangles=np.asarray(env.rectangles); bounds=np.asarray(env.bounds)
    goals=(np.asarray(env.regions[:,:2])+np.asarray(env.regions[:,2:]))/2
    rng=np.random.default_rng(seed)
    clearance=min(float(np.min(np.maximum(rectangles[:,:2]-state[:2],state[:2]-rectangles[:,2:4]).max(axis=1))),
                  float(np.min(state[:2]-bounds[0])),float(np.min(bounds[1]-state[:2])))
    margin=max(.28,min(cfg.rrt_margin,clearance-.02))
    parameters=[]; records=[]
    # Include single-visit initializations and, optionally, paths containing
    # subsequent task-site visits. Subsequent sites are sampled without wind
    # information; no visit order is enforced on the MBD decision coordinates.
    for visits in range(1,cfg.rrt_visits+1):
        for goal_index,goal in enumerate(goals):
            for trial in range(cfg.rrt_paths):
                path=connect(state[:2],goal,rectangles,bounds,rng,margin=margin,step=max(.5,3*cfg.environment_scale))
                if path is None: continue
                visit_order=[goal_index]
                for _ in range(1,visits):
                    other=[k for k in range(len(goals)) if k!=visit_order[-1]]
                    next_goal=int(rng.choice(other))
                    extra=connect(path[-1],goals[next_goal],rectangles,bounds,rng,margin=margin,step=max(.5,3*cfg.environment_scale))
                    if extra is None:break
                    path=np.r_[path,extra[1:]];visit_order.append(next_goal)
                for cruise in (1.8,2.8):
                    parameters.append(parameterize(path,state,cfg,cruise))
                    records.append(dict(goal_index=goal_index,visit_order=visit_order,trial=trial,cruise=cruise,margin=margin,path=path.tolist()))
    # Fixed shape avoids recompiling when a particular RRT call fails.
    capacity=len(goals)*cfg.rrt_paths*2*cfg.rrt_visits
    count=len(parameters)
    while len(parameters)<capacity:
        parameters.append(np.zeros(2*cfg.steps,np.float32)); records.append(dict(unavailable=True))
    return np.asarray(parameters),dict(seed=seed,found=count,records=records)
