"""Offline numerical and 2D diagnosis of the recorded quarter experiments."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PASSAGES = dict(A=8., B=-23., C=27.)

def crossings(path, scale):
    out=[]
    for a,b in zip(path[:-1],path[1:]):
        if (a[0]<0<=b[0]) or (a[0]>0>=b[0]):
            y=a[1]+(b[1]-a[1])*(-a[0])/(b[0]-a[0])
            out.append(min(PASSAGES,key=lambda k:abs(PASSAGES[k]*scale-y)))
    return out

def draw_geometry(ax,env):
    for x,y,X,Y,_ in env['buildings']:
        ax.add_patch(Rectangle((x,y),X-x,Y-y,facecolor='.65',edgecolor='.3'))
    for i,(x,y,X,Y) in enumerate(env['regions']):
        ax.add_patch(Rectangle((x,y),X-x,Y-y,facecolor='#edac3260',edgecolor='#b57812'))
        ax.text((x+X)/2,(y+Y)/2,str(i+1),ha='center',va='center')
    lo,hi=np.array(env['bounds']);ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',xlabel='x (m)',ylabel='y (m)')
    for k,y in PASSAGES.items():ax.text(0,y*env['horizontal_scale'],k,weight='bold',ha='center',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('output/european_quarter/rrt_mbd_execution_300'));p.add_argument('--output',type=Path,default=Path('output/european_quarter/planner_investigation/original'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=[]
    for source in sorted(a.results.glob('*_proposed_8101.json')):
        d=json.loads(source.read_text());env=d['environment'];scale=d['config']['environment_scale'];frames=d['frames'];t=np.array([f['t'] for f in frames]);q=np.array([f['q'] for f in frames]);pos=np.array([f['p'][:2] for f in frames]);plans=d['plans']
        route_counts=[]
        for plan in plans:
            counts=Counter(k for r in plan['rrt']['records'] if 'path' in r for k in set(crossings(np.array(r['path']),scale)))
            route_counts.append(dict(t=plan['t'],**{k:counts[k] for k in PASSAGES}))
        grid=np.array(env['grid']);w=np.array(env['wind']);samples={}
        for k,y in PASSAGES.items():
            mask=(abs(grid[:,1]-y*scale)<scale*2)&(abs(grid[:,0])<18*scale)
            samples[k]=dict(mean_velocity=w[mask].mean(axis=0).tolist(),peak_speed=float(np.linalg.norm(w[mask],axis=1).max()))
        row=dict(file=source.name,passages_y_m={k:v*scale for k,v in PASSAGES.items()},wind_by_passage=samples,rrt_crossing_counts=route_counts,
                 selected_origins=dict(Counter(v['decision_origin'] for v in plans)),
                 validation_mean_over_optimistic=np.array([v['predicted_mean']/max(v['predicted_optimistic'],1e-6) for v in plans]).tolist(),
                 clarity_range=[float(q.min()),float(q.max())],region_cell_counts=env['region_cell_counts'],
                 sensor_radius_m=3.2,initial_q_by_region=[float(np.mean(q[0,np.all(np.array(env['cells'])>=r[:2],axis=1)&np.all(np.array(env['cells'])<=r[2:],axis=1)])) for r in np.array(env['regions'])])
        report.append(row)
        fig,axs=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
        draw_geometry(axs[0,0],env);skip=7;axs[0,0].quiver(grid[::skip,0],grid[::skip,1],w[::skip,0],w[::skip,1],color='#245d85',scale=35);axs[0,0].plot(*pos.T,color='#16745b',lw=1.8);axs[0,0].set_title('Recorded motion and true wind')
        draw_geometry(axs[0,1],env)
        for r in plans[0]['rrt']['records']:
            if 'path' in r:
                path=np.array(r['path']);cs=crossings(path,scale);axs[0,1].plot(*path.T,lw=1,alpha=.45,color={'A':'#be4b42','B':'#294fb3','C':'#3c994a'}.get(cs[0] if cs else '', '.5'))
        axs[0,1].set_title('RRT paths at t=0; blue=B')
        audit=np.load(source.parent/plans[0]['audit']);draw_geometry(axs[0,2],env)
        # Exact double-integrator nominal positions from logged acceleration parameters.
        for j,params in enumerate(audit['samples']):
            weights=audit['weights'][j];theta=weights@params;acc=d['config']['umax']*np.tanh(theta.reshape(-1,2));x=audit['initial_state'].copy();path=[x[:2].copy()];dt=d['config']['dt']
            for u in acc:x=np.r_[x[:2]+dt*x[2:]+.5*dt**2*u,x[2:]+dt*u];path.append(x[:2])
            axs[0,2].plot(*np.array(path).T,alpha=.65,label=f'level {j+1}')
        axs[0,2].set_title('Weighted nominal path at each MBD level');axs[0,2].legend(fontsize=8)
        for method in ('proposed','blind','blind_filter'):
            other=json.loads(source.with_name(source.name.replace('proposed',method)).read_text());tt=np.array([f['t'] for f in other['frames']]);qq=np.array([f['q'] for f in other['frames']]);axs[1,0].plot(tt,np.maximum(.75-qq,0).mean(axis=1),label=method)
        axs[1,0].set(title='Measured mean clarity deficit',xlabel='Mission time (s)',ylabel='Deficit');axs[1,0].legend()
        for k in PASSAGES:axs[1,1].plot([r['t'] for r in route_counts],[r[k] for r in route_counts],label=k)
        axs[1,1].set(title='RRT trajectories covering each passage',xlabel='Mission time (s)',ylabel='Count');axs[1,1].legend()
        axs[1,2].semilogy([r['t'] for r in plans],[r['predicted_optimistic'] for r in plans],label='Optimistic total');axs[1,2].semilogy([r['t'] for r in plans],[r['predicted_mean'] for r in plans],label='Mean total');axs[1,2].set(title='Selection: favorable cost versus mean',xlabel='Mission time (s)',ylabel='Predicted cost (log scale)');axs[1,2].legend()
        fig.suptitle(f"Original implementation: {d['layout']} / {d['case']} · seed {d['seed']}",fontsize=16)
        fig.savefig(a.output/f"{d['layout']}_{d['case']}.png",dpi=150);plt.close(fig)
    (a.output/'structural_audit.json').write_text(json.dumps(report,indent=2))
    print('Saved structural audit and',len(report),'2D diagnostic figures to',a.output)

if __name__=='__main__':main()
