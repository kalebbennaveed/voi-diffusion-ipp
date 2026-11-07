"""Offline capacity ablation using the fixed full-coverage measurement sets."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import cho_factor,cho_solve
from voi_diffusion_ipp.wind.poam import init_parameters,attentive_kernel
from diagnose_full_coverage_gp import metrics


def pivots(K,count):
    """Pivoted Cholesky; only locations and kernel determine inducing points."""
    diagonal=np.diag(K).copy();L=np.zeros((len(K),count));selected=[]
    for j in range(count):
        diagonal[selected]=-np.inf;i=int(np.argmax(diagonal));selected.append(i)
        L[:,j]=(K[:,i]-L[:,:j]@L[i,:j])/np.sqrt(max(diagonal[i],1e-12))
        diagonal-=L[:,j]**2
    return np.asarray(selected)


def sparse_fit(K,Ky,Kq,y,indices):
    Kzz=K[np.ix_(indices,indices)];Kzx=K[indices]
    C=Kzz+Kzx@Kzx.T/.01+1e-6*np.eye(len(indices))
    alpha=cho_solve(cho_factor(C,lower=True),Kzx@y/.01)
    mean=Kq[:,indices]@alpha
    A=cho_solve(cho_factor(Kzz+1e-6*np.eye(len(indices)),lower=True),Kq[:,indices].T)
    corr=cho_solve(cho_factor(C,lower=True),Kq[:,indices].T)
    variance=Ky-np.sum(Kq[:,indices]*A.T,axis=1)+np.sum(Kq[:,indices]*corr.T,axis=1)
    return mean,variance


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh output required')
    a.output.mkdir(parents=True);start=time.perf_counter()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120s capacity audit budget')));signal.alarm(120)
    reports=[]
    for scenario in (1,2):
        data=np.load(a.input/f's{scenario}-inputs.npz');x=data['x'];query=np.r_[x,data['query']];y=data['y']
        parameters={k[10:]:jnp.asarray(data[k]) for k in data.files if k.startswith('parameter_')}
        for name,params,lengths in [('trained_original',parameters,(.3,.8,1.3,1.8)),
            ('initial_original',init_parameters(jax.random.PRNGKey(731),4,prior_std=1.8),(.3,.8,1.3,1.8)),
            ('initial_resolved',init_parameters(jax.random.PRNGKey(731),4,prior_std=1.8),(.125,.3,.8,1.8))]:
            kernel=lambda u,v:attentive_kernel(params,jnp.asarray(u),jnp.asarray(v),jnp.asarray(lengths))
            K=np.asarray(kernel(x,x));Kq=np.asarray(kernel(query,x));ind=pivots(K,768)
            for count in (128,256,500,768):
                mu,var=sparse_fit(K,float(K[0,0]),Kq,y,ind[:count])
                row=dict(scenario=scenario,kernel=name,inducing=count,
                    measured_cells=metrics(mu[:len(x)],var[:len(x)],data['true_at_measurements']),
                    off_grid=metrics(mu[len(x):],var[len(x):],data['true_at_query']))
                reports.append(row);print(json.dumps(row),flush=True)
                np.savez_compressed(a.output/f's{scenario}-{name}-{count}.npz',mean=mu,variance=var,indices=ind[:count])
    (a.output/'report.json').write_text(json.dumps(dict(reports=reports,wall_seconds=time.perf_counter()-start),indent=2));signal.alarm(0)


if __name__=='__main__':main()
