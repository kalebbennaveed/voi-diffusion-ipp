"""Independent analytic Taylor-Green decay check for the MRT fluid solver."""
import os
os.environ.setdefault('PYSTENCILS_CACHE_DIR','/private/tmp/airflow-pystencils')
os.environ.setdefault('PYSTENCILS_CONFIG','/private/tmp/airflow-pystencils.json')
import json
from pathlib import Path
import numpy as np
import pystencils as ps
from lbmpy import LBMConfig
from lbmpy.enums import Stencil,Method
from lbmpy.lbstep import LatticeBoltzmannStep
checks=[]
for n in [16,32]:
    nu=.08;amplitude=.01*16/n;steps=40*(n//16)**2;k=2*np.pi/n
    flow=LatticeBoltzmannStep(domain_size=(n,n,4),periodicity=True,
        lbm_config=LBMConfig(stencil=Stencil.D3Q19,method=Method.MRT,relaxation_rate=1/(.5+3*nu),compressible=True,zero_centered=True),
        config=ps.CreateKernelConfig(cpu_openmp=False,cpu_vectorize_info=None),name=f'tgv{n}')
    for block in flow.data_handling.iterate(ghost_layers=True):
        x,y,z=block.midpoint_arrays
        u=block[flow.velocity_data_name]
        u[...,0]=amplitude*np.sin(k*x)*np.cos(k*y)
        u[...,1]=-amplitude*np.cos(k*x)*np.sin(k*y)
        u[...,2]=0
        block[flow.density_data_name][...]=1+.75*amplitude**2*(np.cos(2*k*x)+np.cos(2*k*y))
    flow.set_pdf_fields_from_macroscopic_values();flow.run(steps)
    got=flow.velocity_slice((slice(None),)*3,masked=False)
    x,y,z=np.meshgrid(np.arange(n)+.5,np.arange(n)+.5,np.arange(4)+.5,indexing='ij')
    factor=amplitude*np.exp(-2*nu*k*k*steps)
    exact=np.stack([factor*np.sin(k*x)*np.cos(k*y),-factor*np.cos(k*x)*np.sin(k*y),np.zeros_like(x)],axis=-1)
    relative=np.linalg.norm(got-exact)/np.linalg.norm(exact)
    rho=flow.density_slice((slice(None),)*3,masked=False)
    checks.append({'n':n,'steps':steps,'relative_velocity_L2_error':float(relative),'relative_mass_error':float(abs(rho.mean()-1))})
result={'test':'Analytic periodic Taylor-Green vortex, diffusive refinement, D3Q19 MRT','checks':checks,
        'acceptance':'Each velocity error < 0.02; fine error less than half coarse; periodic mass error < 1e-10.',
        'passed':all(c['relative_velocity_L2_error']<.02 and c['relative_mass_error']<1e-10 for c in checks) and checks[1]['relative_velocity_L2_error']<.5*checks[0]['relative_velocity_L2_error']}
Path('voi-diffusion-ipp/results/airflow/solver_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
if not result['passed']:raise SystemExit(1)
