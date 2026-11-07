"""Offline domain/calibration diagnostic; no values are returned to planning."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from package_repaired_validation import wind


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    env=json.loads((args.folder/'environment.json').read_text());free=np.asarray(env['free_indices'])
    nx,ny=env['grid'];dx=env['dx_m'];xx,yy=np.meshgrid((np.arange(nx)+.5)*dx,(np.arange(ny)+.5)*dx)
    cells=np.c_[xx.ravel(),yy.ravel()];rows=[]
    for source in sorted(args.folder.glob('s*-center-*.json.gz')):
        r=json.load(gzip.open(source));field=wind(r['scenario']);truth=np.array([field(p) for p in cells])
        observations=np.asarray(r['observations']);frame=r['belief_frames'][-1]
        mean=np.asarray(frame['mean']);sigma=np.asarray(frame['std']);error=np.linalg.norm(mean-truth,axis=1)
        distance=cKDTree(observations[:,1:3]).query(cells)[0]
        region=lambda m: dict(cells=int(m.sum()),rmse_m_s=float(np.sqrt(np.mean(error[m]**2))) if m.any() else None,
                             safety_disk_coverage=float(np.mean(error[m]<=3*np.sqrt(sigma[m]**2+.01))) if m.any() else None)
        free_mask=np.zeros(len(cells),bool);free_mask[free]=True
        speed=np.linalg.norm(truth,axis=1);hold=(.1/.680)*speed[:,None]*truth
        # These are exact stationary force comparisons. They do not prove
        # dynamic passage infeasibility or safe reachability in either direction.
        hold_impossible=np.max(np.abs(hold),axis=1)>1.6
        rows.append(dict(id=r['id'],recorded_s=r['summary']['simulated_seconds'],
            diagnostic_scope='Offline truth only. A stationary holding-force limit is not a proof that a moving traversal is impossible.',
            initial_confidence_disk_m_s=float(3*np.sqrt(1.8**2+.1**2)),
            initial_disk_truth_coverage=float(np.mean(speed[free]<=3*np.sqrt(1.8**2+.01))),
            peak_air_speed_m_s=float(speed[free].max()),
            peak_stationary_force_component_m_s2=float(np.abs(hold[free]).max()),
            free_cells_without_stationary_holding_authority=int(hold_impossible[free].sum()),
            free_cells=len(free),within_20cm=region(free_mask&(distance<=.2)),
            beyond_20cm=region(free_mask&(distance>.2)),
            final_safety_disk_truth_coverage=float(np.mean(error[free]<=3*np.sqrt(sigma[free]**2+.01)))))
    (args.folder/'domain-audit.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
