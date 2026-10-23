"""Shape-stable JAX kernels shared across distinct benchmark geometries.

Environment arrays and physical-flow parameters are dynamic arguments. No case
identifier or physical field is passed to the optimizer. The same production
MBD rollout and nonlinear execution equations are used, without route playback.
"""
from dataclasses import replace
from functools import partial
import jax
import jax.numpy as jnp
from .continuous import optimizer,rollout
from .execution import executor
from .benchmark_environments import BY_NAME,flow,flow_parameters
from . import sensing


class SuiteEngine:
    def __init__(self,cfg):
        self.cfg=cfg
        self._plans={}
        for method in ('aware','blind_filter'):
            def make(method):
                @jax.jit
                def plan(env,*args):
                    return optimizer(env,cfg,method)(*args)
                return plan
            self._plans[method]=make(method)
        execution_cfg=replace(cfg,commit=min(cfg.observation_period,cfg.commit))
        @jax.jit
        def execute(env,parameters,packed,theta,mean,construction,origin,index):
            return executor(env,execution_cfg,lambda p:flow(p,parameters),filter_on=True)(
                packed,theta,mean,construction,origin,index)
        self._execute=execute
        self._predict={}
        for blind in (False,True):
            def make(blind):
                @jax.jit
                def predict(env,th,x,q,m,c,f):
                    return rollout(th,x,q,m,c,f,env,cfg,not blind)
                return predict
            self._predict[blind]=make(blind)
        @jax.jit
        def monitor(env,q,old_position,states):
            def step(carry,packed):
                quality,p=carry
                qn=sensing.step(quality,p,packed[:2],env,cfg,.01,packed[2])
                deficit=lambda qq:jnp.maximum(env.targets-qq,0).mean()
                cost=.005*(deficit(quality)+deficit(qn))
                gap=env.clearance(jnp.r_[packed[:2],packed[3:5]])-.275
                return (qn,packed[:2]),(qn,cost,gap)
            return jax.lax.scan(step,(q,old_position),states)[1]
        self._monitor=monitor
        self._flow=jax.jit(flow)

    def planner(self,env,method):return partial(self._plans[method],env)
    def execution(self,env,case):return partial(self._execute,env,flow_parameters(BY_NAME[case]))
    def prediction(self,env,blind):return partial(self._predict[blind],env)
    def monitoring(self,env):return partial(self._monitor,env)
    def wind(self,case):
        parameters=flow_parameters(BY_NAME[case])
        return lambda p:self._flow(p,parameters)
