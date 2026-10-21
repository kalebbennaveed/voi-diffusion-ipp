import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from voi_diffusion_ipp.quarter.diffusion import lower_tail,cosine_schedule,importance_log_ratio
from voi_diffusion_ipp.quarter.filter import soft_projection
from voi_diffusion_ipp.quarter.environment import environment,Settings,wind
from voi_diffusion_ipp.quarter.planner import library,sensing_points,decode,seeds
from voi_diffusion_ipp.quarter import sensing
from voi_diffusion_ipp.planning.urban import visible


def test_tail_fractional_and_endpoints():
    c=jnp.array([20.,6.,14.,8.])
    assert np.isclose(lower_tail(c,1.),12.)
    assert np.isclose(lower_tail(c,.5),7.)
    assert np.isclose(lower_tail(c,.625),8.4)
    assert np.isclose(lower_tail(c,.1),6.)


def test_cosine_same_noise_endpoint_across_budgets():
    a,b=cosine_schedule(4);aa,bb=cosine_schedule(16)
    assert np.all(np.diff(np.asarray(b))<0)
    np.testing.assert_allclose(np.cumprod(a),b,rtol=1e-6)
    np.testing.assert_allclose(b[-1],bb[-1],rtol=1e-6)


def test_full_mixture_correction_and_bound():
    samples=jnp.array([[0.,0.],[1.,2.],[-2.,1.]])
    ordinary=jnp.array([0.,0.]);centers=jnp.array([[2.,0.],[-1.,2.]])
    nu=jnp.array([.3,.7]);rho=.35
    ratio=np.exp(importance_log_ratio(samples,ordinary,centers,nu,1.,rho))
    from scipy.stats import multivariate_normal
    phi=multivariate_normal.pdf(samples,mean=ordinary)
    psi=sum(float(w)*multivariate_normal.pdf(samples,mean=c) for w,c in zip(nu,centers))
    np.testing.assert_allclose(ratio,phi/((1-rho)*phi+rho*psi),rtol=2e-6)
    assert np.all(ratio<=1/(1-rho)+1e-6)
    np.testing.assert_allclose(importance_log_ratio(samples,ordinary,centers,nu,1.,0.),0,atol=1e-6)


def test_slack_solver_against_independent_convex_solution():
    rng=np.random.default_rng(8)
    solve=jax.jit(soft_projection)
    for _ in range(12):
        A=rng.normal(size=(5,2));b=rng.normal(size=5)-.4;nominal=rng.normal(size=2)*2
        u,slack=solve(jnp.array(nominal),jnp.array(A),jnp.array(b),1.7,15.)
        cost=lambda x:.5*np.sum((x-nominal)**2)+7.5*np.sum(np.maximum(A@x-b,0)**2)
        r=minimize(cost,np.clip(nominal,-1.7,1.7),bounds=[(-1.7,1.7)]*2,method='L-BFGS-B',options={'ftol':1e-12})
        assert cost(np.asarray(u))-r.fun<.006
        assert np.max(np.abs(u))<=1.70001


def test_both_layouts_and_sensing_footprint():
    cfg=Settings();env=environment('multi');q=jnp.zeros(len(env.cells));p=jnp.array([-28.,8.])
    rates=sensing.rate(q,p,env,cfg)
    assert np.all(np.asarray(rates)[np.linalg.norm(np.asarray(env.cells)-p,axis=1)>3.2]==0)
    assert len(env.regions)==4
    assert np.mean(np.asarray(env.regions)[1,[1,3]])==-10.
    assert np.allclose(np.asarray(wind(jnp.array([[0.,-23.]]),'crosswind')),0.)


def test_every_seed_route_is_geometrically_connected():
    for kind in ['two','multi']:
        env=environment(kind);lib=library(np.array([-28.,8.]),env,np.zeros((len(env.regions),4)))
        pts,pauses,sizes,loops,held,cursor=map(np.asarray,lib)
        for path,size in zip(pts,sizes):
            previous=np.array([-28.,8.])
            for pt in path[:size]:
                assert visible(previous,pt,np.asarray(env.rectangles),margin=.65)
                previous=pt
        indices=[int(decode(s,len(env.regions),Settings())[0]) for s in seeds(len(env.regions))]
        assert set(indices)==set(range(len(pts)))


def test_inspection_phase_does_not_reset_to_nearest_stop():
    env=environment('two');sites=sensing_points(env)
    # The drone has completed stop 0 and is still closer to it than stop 1.
    position=sites[1,0]+np.array([1.,0.])
    lib=library(position,env,np.zeros((2,4)),np.array([0,1]))
    points,pauses=map(np.asarray,lib[:2])
    route=points[19]  # East-only inspection policy.
    first_stop=np.flatnonzero(pauses[19])[0]
    np.testing.assert_allclose(route[first_stop],sites[1,1])


def test_crossing_direction_persists_on_both_sides_of_midpoint():
    for kind in ['two','multi']:
        env=environment(kind);n=len(env.regions)
        for passage,y in enumerate([8.,-23.,27.]):
            for x in [-23.,-5.,0.,5.,23.]:
                position=np.array([x,y])
                paths,_,sizes,*_=map(np.asarray,library(position,env,np.zeros((n,4)),np.zeros(n,int)))
                for target,sign in [(0,-1),(1,1)]:
                    route=paths[target*9+passage*3]
                    assert np.sign(route[0,0]-x)==sign
                for path,size in zip(paths,sizes):
                    previous=position
                    for pt in path[:size]:
                        assert visible(previous,pt,np.asarray(env.rectangles),margin=.65)
                        previous=pt


def test_replanning_preserves_whole_inspection_and_transfer_progress():
    from voi_diffusion_ipp.quarter.planner import reference_predictor
    from voi_diffusion_ipp.quarter.environment import TABLE_SHAPE
    # Hold the chosen cyclic policy fixed. Replanning every four seconds must
    # not restart its four-stop sweep indefinitely at the first inspection site.
    env=environment('multi');cfg=Settings(commit=4.)
    predict=reference_predictor(env,cfg,'blind')
    theta=seeds(4)[11];mean=jnp.zeros(TABLE_SHAPE)
    construction=jnp.zeros((2,*TABLE_SHAPE));x=jnp.array([-28.,8.,0.,0.]);q=jnp.full(len(env.cells),.35)
    continuation=None;visited=set();centers=sensing_points(env).mean(axis=1)
    for _ in range(60):
        paths=library(np.asarray(x[:2]),env,np.zeros((4,4)),np.zeros(4,int),continuation)
        pred=predict(theta,x,q,mean,construction,paths)
        x=pred.x[-1];q=pred.q[-1]
        continuation=(11,paths,int(pred.cursor),float(pred.dwell_elapsed))
        for i,center in enumerate(centers):
            if np.any(np.linalg.norm(np.asarray(pred.x[:,:2])-center,axis=1)<5):visited.add(i)
    assert visited=={0,1,2,3},visited


def test_wind_table_extension_preserves_interior_and_finite_exterior():
    from voi_diffusion_ipp.quarter.environment import TABLE_BASIS,TABLE_SHAPE
    from voi_diffusion_ipp.wind.cache import TableBasis
    ordinary=TableBasis(TABLE_BASIS.lower,TABLE_BASIS.spacing)
    field=jax.random.normal(jax.random.PRNGKey(3),TABLE_SHAPE)
    points=jnp.array([[-37.,-25.5],[37.,34.5],[0.,8.]])
    np.testing.assert_array_equal(TABLE_BASIS.evaluate(points,field),ordinary.evaluate(points,field))
    p=jnp.array([0.,8.])
    np.testing.assert_array_equal(jax.jacfwd(lambda p:TABLE_BASIS.evaluate(p,field))(p),jax.jacfwd(lambda p:ordinary.evaluate(p,field))(p))
    distant=jnp.array([1000.,-1000.])
    assert np.isfinite(TABLE_BASIS.evaluate(distant,field)).all()
    assert np.isfinite(jax.jacfwd(lambda p:TABLE_BASIS.evaluate(p,field))(distant)).all()
