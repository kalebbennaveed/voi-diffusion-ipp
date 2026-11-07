"""Additional integration-step sensitivity audit for the two stalled RPCBF runs."""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('robot_review.py');spec=importlib.util.spec_from_file_location('r',p);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
parser=argparse.ArgumentParser()
parser.add_argument('--site',type=Path,default=r.SITE)
args=parser.parse_args();r.SITE=args.site
checks=[]
for scenario,route in [(1,'central'),(2,'lower')]:
 ref=r.Reference(route,'B-A',r.SITE);wind=r.load_wind(scenario,r.SITE)
 traces=[]
 for sub in (10,20,40):
  result=r.simulate(ref,wind,True,sub);traces.append(result)
  print(scenario,route,sub,result['summary']['status'],result['summary']['min_clearance_m'],flush=True)
 for i in (0,1):
  a,b=[np.array(t['samples']) for t in traces[i:i+2]]
  d=np.linalg.norm(a[:,1:3]-b[:,1:3],axis=1)
  checks.append(dict(scenario=scenario,route=route,method='rpcbf',coarse_substep_s=.005/(2**i),fine_substep_s=.0025/(2**i),max_position_difference_m=float(d.max()),same_outcome=traces[i]['summary']['status']==traces[i+1]['summary']['status'],min_clearance_coarse_m=traces[i]['summary']['min_clearance_m'],min_clearance_fine_m=traces[i+1]['summary']['min_clearance_m']))
manifest_path=r.SITE/'data/robot/manifest.json'
manifest=json.loads(manifest_path.read_text())
manifest['additional_refinement']=checks
manifest_path.write_text(json.dumps(manifest,indent=2))
print(json.dumps(checks,indent=2),flush=True)
