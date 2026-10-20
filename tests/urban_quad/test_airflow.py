from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from voi_diffusion_ipp.quadrotor.aerodynamics import acceleration,infer_wind
from voi_diffusion_ipp.experiments.urban_airflow import environment,config,initial_belief,build_cache,wind_velocity
from voi_diffusion_ipp.planning.urban_cycle import route_library,decode,initial_parameters,rollout
from voi_diffusion_ipp.planning.urban import visible
from voi_diffusion_ipp.wind import WorldField


def test_air_drag_and_sensor_inversion():
    v=np.array([[2.,0.,0.],[-2.,0.,0.],[0.,0.,0.]])
    wind=np.broadcast_to([3.8,0,0],v.shape)
    a=acceleration(v,wind)
    assert a[1,0]>10*a[0,0]
    np.testing.assert_allclose(infer_wind(v,a),wind,atol=1e-10)
    np.testing.assert_allclose(acceleration(wind,wind),0,atol=1e-10)


def test_outward_and_return_routes_are_independent_and_clear():
    env=environment();start=np.array([6.,6.]);routes,pauses,sizes,loops,held=route_library(start,env)
    seeds=initial_parameters();indices=[int(decode(t)[0]) for t in seeds]
    assert set(indices)==set(range(20))
    # Right target, A outward/C return and C outward/A return are both present.
    assert 11 in indices and 15 in indices
    for route,n in zip(np.asarray(routes),np.asarray(sizes)):
        previous=start
        for p in route[:n]:
            assert visible(previous,p,np.asarray(env.rectangles),margin=.3)
            previous=p
    for route,n,loop in zip(np.asarray(routes),np.asarray(sizes),np.asarray(loops)):
        assert visible(route[n-1],route[loop],np.asarray(env.rectangles),margin=.3)
    points=jnp.array([[20.,9.],[20.,25.],[20.,1.5]])
    np.testing.assert_allclose(wind_velocity(points,'reverse'),-wind_velocity(points,'forward'))
    np.testing.assert_allclose(wind_velocity(points[1:2],'crosswind'),0)


def test_unknown_airflow_cannot_change_first_reference_commands():
    cfg=replace(config(),future_steps=2,backup_steps=3)
    env=environment();b=initial_belief(jax.random.PRNGKey(0))
    eps=jax.random.normal(jax.random.PRNGKey(1),(4,*b.sample_shape))
    cache=jax.jit(build_cache)(b,eps);mean,_=cache.tabulate(b)
    x=jnp.array([6.,6.,0.,0.]);q=jnp.full(len(env.cells),.4)
    library=route_library(np.array([6.,6.]),env);theta=initial_parameters()[11]
    simulate=jax.jit(lambda field:rollout(theta,x,q,b,cache,WorldField(mean.basis,field),
                          jnp.zeros((4,2)),library,env,cfg,True))
    calm=simulate(jnp.zeros_like(mean.field));windy=simulate(jnp.ones_like(mean.field)*2.)
    np.testing.assert_allclose(calm.controls[:2],windy.controls[:2],atol=1e-5)
    assert np.linalg.norm(np.asarray(calm.x[2]-windy.x[2]))>.01
    assert np.isfinite(np.asarray(windy.x)).all()


def test_airflow_policy_value_gradient_matches_perturbed_rollouts():
    from voi_diffusion_ipp.safety.flow_policy_value import values
    from voi_diffusion_ipp.wind import InnerFields
    from voi_diffusion_ipp.wind.cache import TableBasis,TableMean
    cfg=replace(config(),backup_steps=5)
    env=environment()
    basis=TableBasis(jnp.array([-4.,-4.]),jnp.array([2.,2.]))
    field=jnp.broadcast_to(jnp.array([-2.,.4]),(20,25,2))
    mean=TableMean(basis,field)
    inner=InnerFields(basis,jnp.stack([field,field+jnp.array([.3,-.1])]))
    f=jax.jit(lambda x:values(x,mean,inner,env,cfg))
    x=jnp.array([20.3,9.15,.6,-.12])
    analytical=np.asarray(jax.jacfwd(f)(x))
    eps=.002
    numerical=np.stack([np.asarray((f(x+eps*jnp.eye(4)[i])-f(x-eps*jnp.eye(4)[i]))/(2*eps))for i in range(4)],axis=1)
    np.testing.assert_allclose(analytical,numerical,atol=.012,rtol=.03)
