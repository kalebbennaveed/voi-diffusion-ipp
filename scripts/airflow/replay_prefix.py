"""Restore a recorded prefix by replaying its decisions through original physics.

No rounded robot state or clarity map is used as an initial condition. Start
from the original initial condition and regenerate the actual sensor readings.
Recorded decisions replace repeated optimization only; control, GP inference,
integration and clarity dynamics are recomputed and checked throughout.
"""
import gzip,hashlib,json
from pathlib import Path
import numpy as np


class ReplayPrefix:
    def __init__(self,path,allow_learning=False):
        self.path=Path(path)
        with gzip.open(path,'rt') as f:r=json.load(f)
        if r['summary']['contact'] is not None:raise ValueError('Cannot continue a contacted robot')
        if r['summary']['gate_time_s'] is None and not allow_learning:raise ValueError('This recovery supports prefixes already in MBD')
        self.id=r['id'];self.end=r['summary']['simulated_seconds'];self.gate=r['summary']['gate_time_s']
        self.plans={round(p['t'],8):p for p in r['plans']}
        self.tasks={round(f['t'],8):f for f in r.get('task_frames',[])}
        self.observations=np.asarray(r['observations'],float)
        self.samples=np.asarray(r['samples'],float)
        self.clarity={round(f['t'],8):np.asarray(f['q']) for f in r['clarity_frames']}
        self.handoff=r['handoff'];self.checks=r['handoff_validation']
        self.predictions=r.get('observation_predictions',[])
        self.endpoint_path=self.path.with_name(self.id+'-execution-checkpoint.npz')
        self.posterior_path=self.path.with_name(self.id+'-posterior.npz')
        self.audit=dict(source=str(path),sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            replay_through_s=self.end,checked_observations=0,max_position_error_m=0.,
            max_recorded_state_error=0.,max_recorded_clarity_error=0.,max_reading_error_m_s=0.,
            max_predictive_mean_error_m_s=0.,max_predictive_std_error_m_s=0.,
            definition='Replay saved references and MBD parameters from the exact original initial condition through unchanged dynamics, full POAM and RPCBF. Regenerate sensor noise with the original seed. Never initialize from rounded map/state recordings. Check positions against full-precision observation positions and other values against recording precision. Continue optimization only after this verified prefix.')

    def verify_state(self,t,x,q):
        if t>self.end+1e-8:return
        if abs(t-self.end)<1e-8 and self.endpoint_path.exists():
            with np.load(self.endpoint_path) as state:
                if abs(float(state['t'])-t)>1e-8:raise RuntimeError('Saved endpoint time differs')
                np.testing.assert_allclose(x,state['x'],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(q,state['q'],atol=1e-10,rtol=1e-10)
            self.audit['full_precision_endpoint_verified']=True
        index=round(t/.05);expected=self.samples[index,1:5]
        error=float(np.max(np.abs(np.asarray(x)-expected)))
        self.audit['max_recorded_state_error']=max(self.audit['max_recorded_state_error'],error)
        if not np.isfinite(error) or error>5.2e-8:raise RuntimeError(f'Replayed state differs beyond recording precision at {t}: {error}')
        if t in self.clarity:
            error=float(np.max(np.abs(np.asarray(q)-self.clarity[t])))
            self.audit['max_recorded_clarity_error']=max(self.audit['max_recorded_clarity_error'],error)
            if not np.isfinite(error) or error>5.2e-7:raise RuntimeError(f'Replayed clarity differs beyond recording precision at {t}: {error}')

    def verify_observation(self,t,p,y,mean,std):
        if t>self.end+1e-8:return
        index=round(t/.2);row=self.observations[index]
        errors={'max_position_error_m':float(np.max(np.abs(np.asarray(p)-row[1:3]))),
            'max_reading_error_m_s':float(np.max(np.abs(np.asarray(y)-row[3:5])))}
        if self.predictions:
            prediction=self.predictions[index]
            errors.update(max_predictive_mean_error_m_s=float(np.max(np.abs(np.asarray(mean)-prediction['mean']))),
                max_predictive_std_error_m_s=abs(float(std)-prediction['std']))
        for key,error in errors.items():
            self.audit[key]=max(self.audit[key],error)
            if not np.isfinite(error) or error>1e-8:raise RuntimeError(f'Replay {key} mismatch at {t}: {error}')
        self.audit['checked_observations']+=1

    def verify_posterior(self,t,b):
        if abs(t-self.end)>1e-8 or not self.posterior_path.exists():return
        with np.load(self.posterior_path) as expected:
            pairs=[('mean',b.mean),('covariance',b.covariance),('inverse',b.inverse),('centers',b.basis.centers)]
            pairs.extend(('param_'+k,v) for k,v in b.basis.parameters.items())
            for key,value in pairs:np.testing.assert_allclose(value,expected[key],atol=1e-10,rtol=1e-10)
        self.audit['full_precision_posterior_verified']=True
