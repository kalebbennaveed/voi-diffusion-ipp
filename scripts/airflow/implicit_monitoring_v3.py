"""V2 planner; assimilate each actual wind measurement without a one-second delay."""
from implicit_monitoring_v2 import *
import implicit_monitoring_v2 as core


def safety_builder(points,shape,cfg):
    """Refresh only quantities consumed by execution; identical full posterior.

    Training and held-out performance fields are only needed at planning calls.
    Avoid rebuilding those unused arrays between calls. No uncertainty reduction
    or covariance approximation is introduced by this dispatch optimization.
    """
    @jax.jit
    def build(b,old,key):
        eps=jax.random.normal(key,(cfg.construction_worlds//2,*b.sample_shape),dtype=b.mean.dtype)
        coeff=b.sample_coefficients(jnp.concatenate((eps,-eps)))
        mu,cov=b.predict(points)
        safety=jax.vmap(lambda c:b.basis.evaluate(points,c))(coeff)
        return Tables(mu.reshape(shape),jnp.sqrt(cov[:,0,0]).reshape(*shape[:2],1),
                      old.support,old.planning,safety.reshape(cfg.construction_worlds,*shape),old.validation)
    return build



def rollout(theta,x0,q0,tables,world_array,basis,env,cfg,aware=True,control_fn=control):
    """Original relaxed rollout, reordered without changing any numerical step."""
    mean=TableMean(basis,tables.mean if aware else jnp.zeros_like(tables.mean))
    safety_mean=TableMean(basis,tables.mean);inner=InnerFields(basis,tables.safety)
    world=WorldField(basis,world_array if aware else jnp.zeros_like(world_array))
    refs,aa=legacy.old.nominal_reference(theta,x0,cfg)
    def motion(carry,item):
        x,penalty,slack,gap,alive=carry;ref,a=item
        nominal=m.tracking_control(x,ref,a,mean,cfg,True)
        if aware:u,sl=control_fn(x,nominal,safety_mean,inner,tables.std,basis,env,cfg)
        else:u,sl=jnp.clip(nominal,-cfg.umax,cfg.umax),jnp.array(0.)
        xn=m.physical.rk4(x,u,world,cfg.dt)
        hit=m.contact_fraction(x[:2],xn[:2],env)<=1;observing=alive&~hit
        xn=jnp.where(observing,xn,jnp.r_[x[:2],jnp.zeros(2)])
        clearance=env.clearance(xn)
        soft=cfg.violation_weight*jax.nn.relu(cfg.planning_clearance-clearance)**2
        soft+=cfg.speed_penalty*jax.nn.relu(jnp.linalg.norm(xn[2:])-cfg.speed_max)**2
        gap=jnp.where(alive,jnp.minimum(gap,jnp.where(hit,0.,clearance)),gap)
        return (xn,penalty+cfg.dt*soft*alive,jnp.maximum(slack,sl*alive),gap,observing),(xn,u,observing)
    initial=(x0,jnp.array(0.),jnp.array(0.),env.clearance(x0),jnp.array(True))
    end,(xs,us,alive)=jax.lax.scan(motion,initial,(refs[:-1],aa))
    positions=jnp.r_[x0[None],xs];p0=positions[:-1,:2];p1=positions[1:,:2]
    def sensing(p):
        C=jnp.exp(-jnp.sum((p[:,None]-env.cells[None])**2,axis=-1)/(2*cfg.sensor_sigma**2))
        return C**2/cfg.sensor_R
    c0,cm,c1=sensing(p0),sensing((p0+p1)/2),sensing(p1)
    def observe(carry,data):
        q,cost=carry;C0,CM,C1,living=data;h=cfg.dt
        f=lambda z,C:C*(1-z)**2-env.process_noise*z**2
        k1=f(q,C0);k2=f(q+h*k1/2,CM);k3=f(q+h*k2/2,CM);k4=f(q+h*k3,C1)
        qn=jnp.clip(q+h*(k1+2*k2+2*k3+k4)/6,0,1)
        qn=jnp.where(living,qn,q/(1+h*env.process_noise*q))
        inc=.5*h*(m.clarity.deficit(q,env.targets)+m.clarity.deficit(qn,env.targets))
        return (qn,cost+inc),qn
    (_,cost),qs=jax.lax.scan(observe,(q0,jnp.array(0.)),(c0,cm,c1,alive))
    return legacy.old.Rollout(positions,jnp.r_[q0[None],qs],us,refs,cost,end[1],end[2],end[3],~end[4])

def optimizer(env,cfg,basis,aware=True):
    # Compile numerical stages separately: avoid cloning the full physics scan
    # into conditional branches and skip exact duplicate finalists on the host.
    import time
    @jax.jit
    def optimize(x,q,tables,seeds,previous,key):
        def evaluate(theta):
            r=jax.vmap(lambda w:rollout(theta,x,q,tables,w,basis,env,cfg,aware))(tables.planning)
            return jnp.where(r.terminated,jnp.inf,r.monitor+r.penalty)
        return search(evaluate,seeds,previous,key,cfg,jnp.array(1.),jnp.array(0.),jnp.array(False))
    @jax.jit
    def validate(theta,x,q,tables):
        return jax.vmap(lambda w:execution_forecast(theta,x,q,tables,w,basis,env,cfg,aware))(tables.validation)
    def plan(x,q,tables,seeds,previous,key):
        begin=time.perf_counter()
        result=optimize(x,q,tables,seeds,previous,key);jax.block_until_ready(result)
        search_s=time.perf_counter()-begin
        different=np.any(np.asarray(result.initial_parameters)!=np.asarray(result.final)[None],axis=1)
        seedscore=np.asarray(result.initial_field_costs).mean(axis=1)
        second=int(np.argmin(np.where(different,seedscore,np.inf)))
        candidates=jnp.stack((result.final,result.initial_parameters[second],seeds[0]))
        seen={};forecasts=[];begin=time.perf_counter()
        for theta in candidates:
            token=np.asarray(theta).tobytes()
            if token not in seen:
                seen[token]=validate(theta,x,q,tables);jax.block_until_ready(seen[token])
            forecasts.append(seen[token])
        validation_s=time.perf_counter()-begin
        forecasts=jax.tree.map(lambda *z:jnp.stack(z),*forecasts)
        score=jnp.mean(forecasts.monitor+forecasts.penalty,axis=1)
        feasible=jnp.all(~forecasts.terminated & (forecasts.max_slack<1e-4),axis=1)
        choice=int(jnp.argmin(jnp.where(feasible,score,jnp.inf))) if bool(jnp.any(feasible)) else 2
        selected=jax.tree.map(lambda z:z[choice],forecasts)
        audit=dict(candidates=candidates,costs=score,feasible=feasible,
                   contacts=forecasts.terminated,slack=forecasts.max_slack,
                   selected=choice,no_feasible=~jnp.any(feasible),
                   unique_evaluations=len(seen),search_s=search_s,validation_s=validation_s)
        return candidates[choice],score[choice],result,selected,audit
    return plan

