"""Package the exact CFD inputs and robot implementation for the review page."""
import argparse,json,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def package(site):
    def add(z,path,name=None):
        z.write(path,name or str(path.relative_to(ROOT)))
    def inputs(z):
        for p in [site/'app.js',site/'data/validation.json',site/'data/directional-qa.json',site/'data/directional-tests.txt',
                  *[site/f'data/scenario{s}/{name}' for s in (1,2) for name in ('metadata.json','stationary-velocity.f32')]]:
            add(z,p,'scenario-site/'+str(p.relative_to(site)))
    with zipfile.ZipFile(site/'data/robot/source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'src').rglob('*.py')):add(z,p)
        for name in ['scripts/airflow/robot_review.py','scripts/airflow/ROBOT_REVIEW.txt','scripts/airflow/DIRECTIONAL_REVIEW.txt',
                     'scripts/airflow/package_review.py','tests/test_airflow_robot_review.py','tests/test_rpcbf.py','pyproject.toml']:
            add(z,ROOT/name)
        inputs(z);add(z,site/'data/robot/manifest.json','review-manifest.json')
    with zipfile.ZipFile(site/'data/airflow-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['simulate_sheltered_airflow.py','export_sheltered_wind.py','export_stationary_wind.py',
                     'verify_flow_solver.py','DIRECTIONAL_REVIEW.txt','package_review.py']:
            add(z,ROOT/'scripts/airflow'/name)
        for p in (ROOT/'results/airflow/pre_shelter_stationary').rglob('*'):
            if p.is_file():add(z,p)
        add(z,ROOT/'results/airflow/solver_verification.json');inputs(z)
        z.writestr('requirements.txt','lbmpy==1.4.1\npystencils==1.4\nnumpy==2.4.6\nscipy==1.17.1\nsympy==1.12.1\nCython==3.3.1\n')
        z.writestr('compiler.json',json.dumps({'compiler':{'command':'clang++','flags':'-O3 -DNDEBUG -fPIC -std=c++11','os':'darwin','restrict_qualifier':'__restrict__'},'cache':{'object_cache':'/private/tmp/airflow-pystencils-object'}}))
    print('Packaged exact wind inputs, solver, robot, checks, and reproduction instructions.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);package(p.parse_args().site)
