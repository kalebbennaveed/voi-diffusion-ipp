"""Numerical equation checks independent of the application optimizer."""
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from voi_diffusion_ipp.quarter.mbd_core import proposal_parameters,score_and_update,search
from voi_diffusion_ipp.quarter.continuous import ControlSettings,controls,nominal_reference,continuation
from voi_diffusion_ipp.quarter.diffusion import importance_log_ratio


def test_equations_8_9_and_6_against_numpy():
    rng=np.random.default_rng(71)
    y=rng.normal(size=12);samples=rng.normal(size=(19,12));cost=rng.uniform(0,4,19)
    alpha,abar,temp=.84,.39,.7
    center,sigma=proposal_parameters(jnp.array(y),abar)
    np.testing.assert_allclose(center,y/np.sqrt(abar),rtol=1e-6)
    np.testing.assert_allclose(sigma,np.sqrt(1/abar-1),rtol=1e-6)
    weights=np.exp(-(cost-cost.min())/temp);weights/=weights.sum()
    expected_score=(-y+np.sqrt(abar)*(weights@samples))/(1-abar)
    expected_update=(y+(1-abar)*expected_score)/np.sqrt(alpha)
    update,score,w,ok=score_and_update(jnp.array(y),jnp.array(samples),jnp.array(-cost/temp),alpha,abar)
    assert bool(ok)
    np.testing.assert_allclose(w,weights,atol=1e-7)
    np.testing.assert_allclose(score,expected_score,atol=5e-7)
    np.testing.assert_allclose(update,expected_update,atol=5e-7)


def test_no_archive_is_exactly_original_proposal_weight():
    x=jnp.arange(30,dtype=float).reshape(6,5)/10
    ratio=importance_log_ratio(x,jnp.ones(5),jnp.zeros((3,5)),jnp.ones(3)/3,.7,0.)
    np.testing.assert_array_equal(ratio,np.zeros(6))


def test_single_chain_modes_initialization_uses_the_best_common_seed():
    cfg=ControlSettings(horizon=.2,commit=.1,iterations=2,proposals=8,batch_size=8,initialization='modes',cosine_end=.2)
    seeds=jnp.array([[3.,0.,0.,0.],[-2.,0.,0.,0.]])
    evaluate=lambda theta:jnp.array([jnp.sum((theta-jnp.array([-2.,0.,0.,0.]))**2)])
    key=jax.random.PRNGKey(119)
    result=search(evaluate,seeds,jnp.zeros(4),key,cfg,1.,0.,True)
    _,noise_key=jax.random.split(key)
    from voi_diffusion_ipp.quarter.diffusion import cosine_schedule
    _,abar=cosine_schedule(cfg.iterations,cfg.cosine_end)
    expected=jnp.sqrt(abar[-1])*seeds[1]+jnp.sqrt(1-abar[-1])*jax.random.normal(noise_key,(4,))
    np.testing.assert_allclose(result.before[0],expected,atol=1e-7)
    np.testing.assert_array_equal(result.initial_parameters[:2],seeds)


def test_paper_joint_trajectory_and_control_rollout_have_same_control_update():
    """Sec. 4.2 overwrites sampled states by rollout before Eq. (10d).

    Keeping those state coordinates in the diffusion state must not change its
    control coordinates. Use direct controls and an independent NumPy dynamics
    rollout here; this checks the paper's shooting representation itself.
    """
    rng=np.random.default_rng(39);horizon=6;count=23;dt=.1
    control_state=rng.normal(size=2*horizon)
    joint_state=np.r_[rng.normal(size=4*horizon),control_state]
    for alpha,abar in [(.8,.3),(.9,.65),(.95,.9)]:
        center,sigma=proposal_parameters(jnp.array(control_state),abar)
        sampled_u=np.asarray(center)+float(sigma)*rng.normal(size=(count,2*horizon))
        x=np.zeros((count,4));states=[]
        for a in sampled_u.reshape(count,horizon,2).transpose(1,0,2):
            x=np.c_[x[:,:2]+dt*x[:,2:]+.5*dt*dt*a,x[:,2:]+dt*a]
            states.append(x.copy())
        rolled_states=np.stack(states,axis=1).reshape(count,-1)
        feasible_y=np.c_[rolled_states,sampled_u]
        costs=np.sum((x[:,:2]-[.8,.4])**2,axis=1)+.02*np.sum(sampled_u**2,axis=1)
        w=np.exp(-(costs-costs.min())/.7);w/=w.sum()
        joint_score=(-joint_state+np.sqrt(abar)*(w@feasible_y))/(1-abar)
        joint_state=(joint_state+(1-abar)*joint_score)/np.sqrt(alpha)
        reduced,_,_,_=score_and_update(jnp.array(control_state),jnp.array(sampled_u),jnp.array(-costs/.7),alpha,abar)
        np.testing.assert_allclose(reduced,joint_state[-2*horizon:],atol=1e-6)
        control_state=np.asarray(reduced)


def test_every_time_coordinate_is_an_independent_control():
    cfg=ControlSettings(horizon=3,commit=.2)
    theta=jnp.zeros(2*cfg.steps);x=jnp.array([2.,3.,.5,-.2])
    ref,aa=nominal_reference(theta,x,cfg)
    t=np.arange(cfg.steps+1)*cfg.dt
    np.testing.assert_allclose(ref[:,:2],np.array(x[:2])+t[:,None]*np.array(x[2:]),atol=5e-6)
    jac=np.asarray(jax.jacfwd(lambda z:controls(z,cfg).ravel())(theta))
    np.testing.assert_allclose(jac,np.eye(theta.size)*cfg.umax,atol=1e-7)
    altered=theta.at[2*17+1].set(.3)
    changed,_=nominal_reference(altered,x,cfg)
    np.testing.assert_array_equal(changed[:18],ref[:18])
    assert float(changed[18,1])!=float(ref[18,1])
    assert float(changed[-1,1])!=float(ref[-1,1])
    np.testing.assert_array_equal(continuation(altered,cfg)[:-4],altered[4:])


def test_compact_geometry_keeps_aircraft_and_inspection_dimensions():
    from voi_diffusion_ipp.quarter.environment import environment,metadata,wind,table_basis
    env=environment('multi',.6);full=environment('multi')
    np.testing.assert_allclose(np.asarray(env.regions[:,2:]-env.regions[:,:2]),np.asarray(full.regions[:,2:]-full.regions[:,:2]),atol=3e-6)
    assert metadata('multi',.6)['drone_span']==metadata('multi')['drone_span']==.55
    np.testing.assert_array_equal(env.rectangles[:,4],full.rectangles[:,4])
    assert float(env.rectangles[1,1]-env.rectangles[0,3])==pytest.approx(3.)
    for center in (np.asarray(env.regions[:,:2])+np.asarray(env.regions[:,2:]))/2:
        assert float(env.clearance(jnp.r_[center,jnp.zeros(2)]))>.55
    assert np.all(np.asarray(env.cells)>np.asarray(env.bounds[0]))
    assert np.all(np.asarray(env.cells)<np.asarray(env.bounds[1]))
    np.testing.assert_allclose(wind(jnp.array([[0.,4.8]]),'forward',3.1,.6),wind(jnp.array([[0.,8.]]),'forward'),atol=1e-6)
    np.testing.assert_allclose(table_basis(.6).spacing,[.6,.6])


@pytest.mark.parametrize('scale', [.35, .5])
def test_smaller_inspection_sites_match_sensing_and_collision_geometry(scale):
    from voi_diffusion_ipp.quarter.environment import environment
    from voi_diffusion_ipp.quarter.sensing import rate
    cfg=ControlSettings(environment_scale=scale,inspection_scale=.55)
    env=environment('two',cfg.environment_scale,cfg.inspection_scale)
    centers=(env.regions[:,:2]+env.regions[:,2:])/2
    np.testing.assert_allclose(env.regions[:,2:]-env.regions[:,:2],[[4.95,5.5]]*2,atol=3e-6)
    for i,center in enumerate(centers):
        cells=env.cells[42*i:42*(i+1)]
        assert np.all(np.linalg.norm(cells-center,axis=1)<3.2)
        assert np.all(np.asarray(jax.vmap(env.clearance)(jnp.c_[cells,jnp.zeros_like(cells)]))>.275)
        assert np.all(np.asarray(rate(jnp.zeros(84),center,env,cfg))[42*i:42*(i+1)]>0)


def test_search_logs_reconstruct_every_reverse_step():
    cfg=ControlSettings(horizon=.3,commit=.1,iterations=6,proposals=32,batch_size=8,temperature=1.)
    target=jnp.array([.5,-.2,.7,-.4,.1,.2])
    evaluate=lambda theta:jnp.array([jnp.sum((theta-target)**2)])
    result=search(evaluate,jnp.zeros((1,6)),jnp.zeros(6),jax.random.PRNGKey(22),cfg,1.,0.,True)
    from voi_diffusion_ipp.quarter.diffusion import cosine_schedule
    alpha,abar=map(np.asarray,cosine_schedule(cfg.iterations,cfg.cosine_end))
    for j in range(cfg.iterations):
        i=cfg.iterations-1-j
        mean=np.asarray(result.weights[j])@np.asarray(result.samples[j])
        score=(-np.asarray(result.before[j])+np.sqrt(abar[i])*mean)/(1-abar[i])
        update=(np.asarray(result.before[j])+(1-abar[i])*score)/np.sqrt(alpha[i])
        # Float32 subtraction is amplified by 1/(1-abar) at the last level.
        np.testing.assert_allclose(result.scores[j],score,atol=8e-6)
        np.testing.assert_allclose(result.after[j],update,atol=2e-6)
    assert int(result.evaluations)==2+cfg.iterations*cfg.proposals+1
    assert float(evaluate(result.final)[0])<float(result.initial_cost)


def test_undefined_score_is_not_reported_as_weighted_average():
    y=jnp.ones(3)
    update,score,w,ok=score_and_update(y,jnp.zeros((4,3)),jnp.full(4,-jnp.inf),.8,.4)
    assert not bool(ok)
    np.testing.assert_array_equal(w,np.zeros(4))
    np.testing.assert_array_equal(update,y)


def test_vectorized_batch_size_preserves_mbd_samples_weights_and_selection():
    cfg=ControlSettings(horizon=.4,commit=.1,iterations=5,proposals=32,batch_size=4,temperature=.7)
    target=jnp.arange(8)/10
    evaluate=lambda theta:jnp.array([jnp.sum((theta-target)**2),jnp.sum((theta+target)**2)+.5])
    args=(evaluate,jnp.zeros((2,8)),jnp.zeros(8),jax.random.PRNGKey(123))
    first=search(*args,cfg,.5,.35,True)
    second=search(*args,replace(cfg,batch_size=32),.5,.35,True)
    for name in first._fields:
        np.testing.assert_allclose(getattr(first,name),getattr(second,name),rtol=2e-6,atol=2e-6,err_msg=name)


def test_archived_proposals_reconstruct_full_mixture_density_and_weights():
    from scipy.stats import multivariate_normal
    from voi_diffusion_ipp.quarter.diffusion import cosine_schedule
    cfg=ControlSettings(horizon=.1,commit=.1,iterations=8,proposals=64,batch_size=8,temperature=.5)
    evaluate=lambda theta:jnp.array([jnp.sum((theta-1)**2),jnp.sum((theta+2)**2)])
    result=search(evaluate,jnp.zeros((1,2)),jnp.zeros(2),jax.random.PRNGKey(12),cfg,.5,.35,True)
    _,abar=map(np.asarray,cosine_schedule(cfg.iterations,cfg.cosine_end))
    assert np.any(np.asarray(result.trace)[:,3]>.3)
    for j in range(cfg.iterations):
        rho=float(result.trace[j,3]);sigma=float(result.trace[j,5]);xs=np.asarray(result.samples[j])
        center=np.asarray(result.before[j])/np.sqrt(abar[-1-j])
        phi=multivariate_normal.pdf(xs,center,sigma*sigma*np.eye(2))
        mixture=sum(float(w)*multivariate_normal.pdf(xs,c,sigma*sigma*np.eye(2)) for w,c in zip(result.archive_probabilities[j],result.archive_centers[j]))
        ratio=np.log(phi)-np.log((1-rho)*phi+rho*mixture)
        np.testing.assert_allclose(result.log_importance_ratios[j],ratio,atol=4e-5)
        cost=np.min(np.asarray(result.field_costs[j]),axis=1)
        logw=-cost/cfg.temperature+ratio;weights=np.exp(logw-logw.max());weights/=weights.sum()
        np.testing.assert_allclose(result.weights[j],weights,atol=2e-6)


@pytest.mark.parametrize('kwargs',[dict(dt=0),dict(dt=.6),dict(dt=.125),dict(horizon=0),dict(horizon=.25),dict(temperature=0),dict(cosine_end=1),dict(commit=70)])
def test_invalid_numerics_rejected(kwargs):
    with pytest.raises(ValueError):ControlSettings(**kwargs)
