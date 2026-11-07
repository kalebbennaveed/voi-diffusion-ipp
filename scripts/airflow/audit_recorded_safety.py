"""Read-only calibration and loss-of-feasibility audit of actual executions.

The true field is used only here, after execution. Logged premeasurement GP
predictions are exact queries at the actual sensor positions; they are not
substituted for the postmeasurement interpolated field used by the filter.
"""
import argparse,gzip,json
from pathlib import Path
import numpy as np
from package_repaired_validation import wind


def audit(source):
    record=json.load(gzip.open(source));samples=np.asarray(record['samples'])
    observations=np.asarray(record['observations']);field=wind(record['scenario'])
    predicted=record['observation_predictions'];assert len(predicted)==len(observations)
    truth=np.asarray([field(p) for p in observations[:,1:3]])
    mean=np.asarray([p['mean'] for p in predicted]);sigma=np.asarray([p['std'] for p in predicted])
    radius=3*np.sqrt(sigma*sigma+.01);error=np.linalg.norm(truth-mean,axis=1)
    invalid=np.flatnonzero(samples[1:,11]==0)+1
    first=float(samples[invalid[0],0]) if len(invalid) else None
    episodes=[]
    for segment in (np.split(invalid,np.flatnonzero(np.diff(invalid)>1)+1) if len(invalid) else []):
        rows=samples[segment]
        episodes.append(dict(start_s=float(rows[0,0]),end_s=float(rows[-1,0]),commands=len(segment),
            minimum_clearance_m=float(rows[:,12].min()),maximum_slack=float(rows[:,9].max()),
            maximum_speed_m_s=float(np.linalg.norm(rows[:,3:5],axis=1).max())))
    event=[]
    if first is not None:
        for i in np.flatnonzero((observations[:,0]>=first-2)&(observations[:,0]<=first+1)):
            row=samples[round(observations[i,0]/.05)]
            plan=min(record['plans'],key=lambda p:abs(p['t']-observations[i,0]))
            event.append(dict(t=float(observations[i,0]),position=observations[i,1:3].tolist(),
                actual_wind=truth[i].tolist(),prior_mean=mean[i].tolist(),prior_std=float(sigma[i]),
                premeasurement_error=float(error[i]),premeasurement_radius=float(radius[i]),
                speed=float(np.linalg.norm(row[3:5])),clearance=float(row[12]),
                command=row[5:7].tolist(),feasible=bool(row[11]),slack=float(row[9]),
                decision=plan.get('reason'),holding_force_bound=plan.get('holding_force_bound_m_s2')))
    return dict(id=record['id'],recorded_seconds=record['summary']['simulated_seconds'],
        contact=record['summary']['contact'],first_infeasible_command_s=first,
        infeasible_commands=len(invalid),infeasible_episodes=episodes,premeasurement_wind_rmse_m_s=float(np.sqrt(np.mean(error*error))),
        premeasurement_confidence_disk_coverage=float(np.mean(error<=radius)),
        maximum_standardized_prediction_error=float(np.max(error/np.sqrt(sigma*sigma+.01))),
        first_failure_trace=event,
        interpretation='Premeasurement coverage is an offline estimator diagnostic, not a reconstruction of the postmeasurement RPCBF envelope. A feasible finite-sample QP is not a global safety guarantee. Contacts and infeasible commands remain failures.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    rows=[audit(p) for p in sorted(args.folder.glob('s*-center-*.json.gz'))]
    (args.folder/'recorded-safety-audit.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    print(json.dumps([{k:v for k,v in r.items() if k!='first_failure_trace'} for r in rows],indent=2))
