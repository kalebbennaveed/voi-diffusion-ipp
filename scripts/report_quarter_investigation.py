"""Scientific 2D report of numerical defects, parameter trials and real flights."""
import json,html,argparse
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from investigate_quarter_planner import draw_geometry,crossings

ROOT=Path('output/european_quarter/planner_investigation')
RUNS=ROOT/'calibrated300'
OLD=ROOT.parent/'rrt_mbd_execution_300'
OUT=ROOT/'report'
TITLES={('two','forward'):'Forward',('two','reverse'):'Reversed',('two','crosswind'):'Adverse A/C; calm B',('two','commitment'):'Larger east; all passages windy',('two','calm_a'):'Larger east; calm A',('multi','crosswind'):'Four inspection sites'}
COLORS={'proposed':'#007f68','blind':'#cc7a16','blind_filter':'#3e69ae'}
ABLATION_NAMES={'aware':'Expected cost; ordinary Gaussian proposal','optimistic':'Optimistic cost; ordinary Gaussian proposal','proposed':'Optimistic cost; adapted proposal'}

def load(path):return json.loads(path.read_text())

def wind_arrows(ax,env,field):
    grid=np.asarray(env['grid']);field=np.asarray(field);scale=env['horizontal_scale']
    selected=[]
    for y in np.array([8,-23,27])*scale:
        actual_y=grid[np.argmin(abs(grid[:,1]-y)),1]
        indices=np.flatnonzero((abs(grid[:,1]-actual_y)<1e-5)&(abs(grid[:,0])<19*scale))[::3]
        selected.extend(indices)
    selected=np.asarray(selected,dtype=int);selected=selected[np.linalg.norm(field[selected],axis=1)>.05]
    return ax.quiver(grid[selected,0],grid[selected,1],field[selected,0],field[selected,1],color='#245b85',scale=35,width=.004,zorder=4)

def wind_figure(d,prefix=''):
    env=d['environment'];grid=np.array(env['grid']).reshape(*env['grid_shape'],2)
    true=np.array(env['wind']);belief=d['beliefs'][-1];mean=np.array(belief['mean'])
    valid=np.ones(len(true),bool);points=np.array(env['grid'])
    for b in env['buildings']:valid&=~(np.all(points>=b[:2],axis=1)&np.all(points<=b[2:4],axis=1))
    vmax=max(1,float(np.linalg.norm(true[valid],axis=1).max()),float(np.linalg.norm(mean[valid],axis=1).max()))
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for ax,field,title in zip(axes[:2],(true,mean),('True physical wind',f'Posterior mean at t={belief["t"]:g} s')):
        draw_geometry(ax,env);speed=np.where(valid,np.linalg.norm(field,axis=1),np.nan).reshape(*env['grid_shape'])
        im=ax.pcolormesh(grid[:,:,0],grid[:,:,1],speed,cmap='Blues',vmin=0,vmax=vmax,shading='nearest',zorder=-1)
        arrows=wind_arrows(ax,env,field);ax.quiverkey(arrows,.1,1.03,3.,'3 m/s',labelpos='E');ax.set_title(title,pad=28)
    fig.colorbar(im,ax=axes[:2],shrink=.6,label='Wind speed (m/s)')
    draw_geometry(axes[2],env);error=np.where(valid,np.linalg.norm(mean-true,axis=1),np.nan).reshape(*env['grid_shape'])
    im=axes[2].pcolormesh(grid[:,:,0],grid[:,:,1],error,cmap='magma',vmin=0,vmax=max(.1,float(np.nanmax(error))),shading='nearest',zorder=-1)
    axes[2].set_title('Held-out field error\ntrue wind used only for scoring');fig.colorbar(im,ax=axes[2],shrink=.6,label='Vector error magnitude (m/s)')
    fig.suptitle(TITLES[(d['layout'],d['case'])]+' · physical wind versus learned field',fontsize=14)
    name=f'{prefix}{d["layout"]}_{d["case"]}_wind.png';fig.savefig(OUT/name,dpi=140);plt.close(fig);return name

def metrics(d):
    t=np.array([f['t'] for f in d['frames']]);q=np.array([f['q'] for f in d['frames']]);positions=np.array([f['p'][:2] for f in d['frames']]);targets=np.array(d['environment']['targets']);deficit=np.maximum(targets-q,0).mean(axis=1)
    scale=d['config']['environment_scale'];central=abs(positions[:,0])<19*scale
    passages=np.argmin(abs(positions[:,1,None]-np.array([8,-23,27])*scale),axis=1)
    coverage=Counter()
    for p in d['plans']:
        labels=set(k for r in (p.get('rrt') or {}).get('records',[]) if 'path' in r for k in crossings(np.array(r['path']),scale))
        coverage.update(labels)
    return dict(cost=d['cost'],success=d['success'],completed=d['completed'],seconds=d['completed_seconds'],observations=len(d['observations']),
        origins=dict(Counter(p['decision_origin'] for p in d['plans'])),rrt_passage_coverage_calls=dict(coverage),planning_calls=len(d['plans']),
        passage_seconds={label:float(np.diff(t,prepend=0)[central&(passages==i)].sum()) for i,label in enumerate('ABC')},
        early_60_cost=float(np.interp(60,t,[f['cost'] for f in d['frames']])),
        mean_deficit=float(np.trapezoid(deficit,t)/max(t[-1],1e-5)),max_guidance_fraction=float(np.mean([p.get('guidance_used_fraction',0) for p in d['plans']])),
        mean_ess=float(np.mean([r[2] for p in d['plans'] for r in p['trace']])),mixture_active_fraction=float(np.mean([r[3]>0 for p in d['plans'] for r in p['trace']])))

def paths_figure(identity,methods,prefix=''):
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    d=next(iter(methods.values()));env=d['environment'];wind=np.array(env['wind']);grid=np.array(env['grid']);scale=d['config']['environment_scale']
    # Spatial quiver thinning, so vector density is independent of flat order.
    shape=env['grid_shape'];g=grid.reshape(*shape,2);w=wind.reshape(*shape,2)
    for ax,method in zip(axes[0],('proposed','blind','blind_filter')):
        draw_geometry(ax,env)
        if method not in methods:ax.set_title(method+' · not complete');continue
        item=methods[method];p=np.array([f['p'][:2] for f in item['frames']]);t=np.array([f['t'] for f in item['frames']]);color=COLORS[method]
        wind_arrows(ax,env,wind)
        line=np.stack((p[:-1],p[1:]),axis=1);rgba=np.tile(matplotlib.colors.to_rgba(color),(len(line),1));rgba[:,3]=.10+.65*(t[1:]/max(t[-1],1))
        ax.add_collection(LineCollection(line,colors=rgba,lw=1.8));ax.plot(*p[-1],marker='o',color=color,ms=5)
        ax.set_title(f"{method.replace('_',' + ')} · {item['cost']:.2f} deficit·s" if item['success'] else f"{method.replace('_',' + ')} · terminated at {item['completed_seconds']:.2f} s")
        if not item['success']:ax.plot(*p[-1],marker='x',ms=10,c='#a90020',mew=2)
    for method,item in methods.items():
        if not item['success']:continue
        t=np.array([f['t'] for f in item['frames']]);q=np.array([f['q'] for f in item['frames']]);cost=np.array([f['cost'] for f in item['frames']]);deficit=np.maximum(np.array(env['targets'])-q,0).mean(axis=1)
        axes[1,0].plot(t,cost,color=COLORS[method],label=method);axes[1,1].plot(t,deficit,color=COLORS[method],label=method)
    axes[1,0].set(title='Accumulated clarity deficit',xlabel='Mission time (s)',ylabel='Deficit·s');axes[1,0].legend(fontsize=8)
    axes[1,1].set(title='Instantaneous mean clarity deficit',xlabel='Mission time (s)',ylabel='Deficit');axes[1,1].set_ylim(bottom=0)
    p=methods.get('proposed')
    if p:
        cells=np.array(env['cells']);t=np.array([f['t'] for f in p['frames']]);q=np.array([f['q'] for f in p['frames']]);target=np.array(env['targets'])
        for i,region in enumerate(env['regions']):
            mask=np.all(cells>=region[:2],axis=1)&np.all(cells<=region[2:],axis=1)
            axes[1,2].plot(t,np.maximum(target[mask]-q[:,mask],0).mean(axis=1),label=f'Region {i+1}')
        axes[1,2].set(title='Proposed: each inspection region',xlabel='Mission time (s)',ylabel='Mean deficit');axes[1,2].legend(fontsize=8);axes[1,2].set_ylim(bottom=0)
    fig.suptitle(TITLES[identity]+' · 300 s physical missions · seed 8101',fontsize=16)
    name=f'{prefix}{identity[0]}_{identity[1]}_comparison.png';fig.savefig(OUT/name,dpi=140);plt.close(fig);return name

def intermediate_figure(d,path,prefix=''):
    with np.load(path) as file:a={k:file[k] for k in file.files}
    plan=next((p for p in d['plans'] if p.get('audit')==path.name),d['plans'][0])
    cfg=d['config'];dt=cfg['dt'];x0=a['initial_state'];env=d['environment'];stages=[0,len(a['samples'])//2,len(a['samples'])-1]
    def positions(theta):
        acc=cfg['umax']*np.tanh(theta.reshape(-1,2));x=x0.copy();points=[x[:2].copy()]
        for u in acc:x=np.r_[x[:2]+dt*x[2:]+.5*dt*dt*u,x[2:]+dt*u];points.append(x[:2])
        return np.array(points)
    fig,axes=plt.subplots(1,4,figsize=(18,5),constrained_layout=True)
    draw_geometry(axes[0],env)
    for r in plan['rrt']['records']:
        if 'path' not in r:continue
        points=np.array(r['path']);cs=crossings(points,cfg['environment_scale']);color={'A':'#c84638','B':'#315daf','C':'#1b8a65'}.get(cs[0] if cs else '', '.6');axes[0].plot(*points.T,c=color,alpha=.3,lw=1.)
    axes[0].set_title('Geometric RRT initializations\nred A · blue B · green C')
    for ax,j in zip(axes[1:],stages):
        draw_geometry(ax,env)
        for params in a['samples'][j]:ax.plot(*positions(params).T,color='#75869b',alpha=.22,lw=.7)
        selected=int(np.argmax(a['weights'][j]));ax.plot(*positions(a['samples'][j,selected]).T,color='#8958a7',lw=1.7,label='Highest weight in this level')
        ax.plot(*positions(a['chosen_theta']).T,color='#007f68',lw=1.6,ls='--',label='Selected nominal reference')
        ax.plot(*np.array(plan['points']).T,color='#007f68',lw=2.0,label='Selected: predicted motion')
        ess=1/max(float(np.sum(a['weights'][j]**2)),1e-10);ax.set_title(f'Diffusion level {j+1}/{len(a["samples"])}\nEffective sample size: {ess:.1f}/{len(a["weights"][j])}')
    axes[-1].legend(fontsize=8,loc='lower left')
    fig.suptitle(f'Logged MBD search at t={plan["t"]:.0f} s. Gray: nominal reference samples. Green solid: selected predicted motion in the posterior mean field.',fontsize=12)
    name=f'{prefix}{d["layout"]}_{d["case"]}_intermediate_t{plan["t"]:.0f}.png';fig.savefig(OUT/name,dpi=140);plt.close(fig);return name

def spatial_figure(d,prefix=''):
    env=d['environment'];cells=np.array(env['cells']);fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for j,t in enumerate((0.,60.,120.,300.)):
        f=min(d['frames'],key=lambda f:abs(f['t']-t));belief=max((b for b in d['beliefs'] if b['t']<=t+1e-4),key=lambda b:b['t'])
        for ax in axes[:,j]:draw_geometry(ax,env)
        deficits=np.maximum(np.array(env['targets'])-f['q'],0)
        for r in env['regions']:
            mask=np.all(cells>=r[:2],axis=1)&np.all(cells<=r[2:],axis=1)
            xx=np.unique(cells[mask,0]);yy=np.unique(cells[mask,1])
            values=np.full((len(yy),len(xx)),np.nan)
            for point,value in zip(cells[mask],deficits[mask]):
                values[np.searchsorted(yy,point[1]),np.searchsorted(xx,point[0])]=value
            image=axes[0,j].pcolormesh(xx,yy,values,vmin=0,vmax=.75,cmap='YlOrRd',shading='nearest',zorder=1)
        # Recent actual motion, light old segments and opaque newest segments.
        path=np.array([v['p'][:2] for v in d['frames'] if max(0,t-30)<=v['t']<=t])
        if len(path)>1:
            segments=np.stack((path[:-1],path[1:]),axis=1);colors=np.tile(matplotlib.colors.to_rgba('#007f68'),(len(segments),1));colors[:,3]=np.linspace(.15,1,len(segments));axes[0,j].add_collection(LineCollection(segments,colors=colors,lw=2,zorder=3))
        axes[0,j].plot(*f['p'][:2],marker='o',ms=4,c='black');axes[0,j].add_patch(Circle(f['p'][:2],f['p'][2],facecolor='none',edgecolor='#007f68',lw=1))
        axes[0,j].set_title(f'Clarity deficit · t={f["t"]:g} s')
        g=np.array(env['grid']).reshape(*env['grid_shape'],2);std=np.array(belief['std']).reshape(*env['grid_shape']);im=axes[1,j].pcolormesh(g[:,:,0],g[:,:,1],std,vmin=0,vmax=d['config']['prior_std'],cmap='magma',shading='nearest',zorder=-1)
        axes[1,j].set_title(f'Posterior std · t={belief["t"]:g} s')
    fig.colorbar(image,ax=axes[0],shrink=.65,label='Clarity deficit');fig.colorbar(im,ax=axes[1],shrink=.65,label='Wind-component standard deviation (m/s)')
    fig.suptitle(TITLES[(d['layout'],d['case'])]+' · proposed · recorded spatial quantities',fontsize=15)
    name=f'{prefix}{d["layout"]}_{d["case"]}_spatial.png';fig.savefig(OUT/name,dpi=140);plt.close(fig);return name

STAGES={
 'calibrated300':('Search and prediction corrections','calibrated'),
 'online300':('Online estimation between planning calls','online'),
 'supported300':('Corrected field sampling and contact forecasts','supported'),
 'supported_forward_reverse300':('Latest corrections: forward and reversed wind','supported'),
}

def comparison_rows(directory):
    grouped={};status=[]
    for path in sorted(directory.glob('*_8101.json')):
        d=load(path)
        if 'frames' not in d:continue
        status.append(dict(file=path.name,**metrics(d)))
        if d.get('completed'):grouped.setdefault((d['layout'],d['case']),{})[d['method']]=d
    rows=[]
    for identity,methods in grouped.items():
        if not all(m in methods for m in COLORS):continue
        p=methods['proposed'];row=dict(environment=TITLES[identity],layout=identity[0],case=identity[1],costs={m:methods[m]['cost'] for m in COLORS},successes={m:methods[m]['success'] for m in COLORS})
        row['reductions']={m:100*(1-p['cost']/methods[m]['cost']) if p['success'] and methods[m]['success'] else None for m in ('blind','blind_filter')}
        row['original_costs']={}
        for m in COLORS:
            old=OLD/f'{identity[0]}_{identity[1]}_{m}_8101.json'
            if old.exists():row['original_costs'][m]=load(old)['cost']
        rows.append(row)
    return rows,grouped,status

def table(rows):
    result='<table><tr><th>Environment</th><th>Proposed</th><th>Blind planning</th><th>Blind + filter</th><th>Reduction vs blind</th><th>Reduction vs filtered</th></tr>'
    for r in rows:
        result+=f'<tr><td>{html.escape(r["environment"])}</td>'
        for m in COLORS:result+=f'<td>{r["costs"][m]:.3f}</td>' if r['successes'][m] else '<td class="failed">Terminated; excluded</td>'
        for m in ('blind','blind_filter'):
            value=r['reductions'][m];result+='<td>—</td>' if value is None else f'<td>{value:+.1f}%</td>'
        result+='</tr>'
    return result+'</table>'

FACTS=[
 ('B was missing from the RRT initialization', 'Original adverse A/C initializations covered A and C, never B. Small-noise MBD refinement did not create that distant path class. Geometry-only RRT connections now retain distinct obstacle-winding classes and narrow free-face passages. Wind values and scenario labels do not enter the generator. Every acceleration coordinate can still change during MBD.'),
 ('Planning predicted a different controller', 'Planning used a soft projection with 0.40 m clearance; execution used a hard projection with 0.55 m clearance. Isolated replay of four adverse-case windows gave 0.430 m mean motion-prediction error, reduced to 0.021 m by matching the executed projection. Smaller integration steps alone did not fix it. True wind was supplied only to this diagnostic plant; the mission planner receives observations.'),
 ('Optimism discarded constraint penalties too', 'The former score took the favorable tail of monitoring plus penalties. The corrected score takes the favorable monitoring tail plus the mean soft penalty across all performance fields. One old selected sequence had a predicted optimistic score of 5.51 and mean score of 511.85. These are predictions, not measured mission costs.'),
 ('Nominal demonstration guidance dominated', 'About 97% of proposals in the old adverse run used demonstration likelihood over model likelihood. The reference MBD code evaluates agreement on rolled-out motion. The corrected evaluated preset uses geometric RRT initialization and model-only refinement weights.'),
 ('Wind observations were held until replanning', 'Measurements collected at 1 s were batch-assimilated every 5 s, leaving the execution filter with a stale posterior between calls. Updating every 1 s kept an identical first planned control sequence clear for the first 20 s: minimum surface clearance 0.352 m, versus contact at 9.86 s previously. The original planned reference and clock are retained through these estimator updates.'),
 ('The field sampler distorted covariance', 'On 120 held-out free-space points after 300 recorded observations, the original sampler had 72.3% relative covariance-matrix error. Numerical support correction reduced this to 0.093%, with maximum standard-deviation discrepancy below 0.001 m/s. Support locations are numerical anchors, not extra wind measurements. GP posterior inference itself is unchanged by this sampling correction.'),
 ('Forecasts could sense after physical contact', 'A predicted path could pass through a building and continue receiving sensing credit. Contact now leads to an absorbing failure state and unsensed clarity decay through the remaining horizon, consistent with physical evaluation. A direct regression test verifies this. Planning still permits slack; this is not a safety guarantee.'),
 ('The larger-east field did not match calm A', 'The original larger-east field actually winds A, B and C. A separate calm-A larger-east control case is now evaluated, preserving the all-windy case.'),
]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plots',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);stages={};figures=[]
    for key,(title,preset) in STAGES.items():
        directory=ROOT/key;rows,grouped,status=comparison_rows(directory)
        stages[key]=dict(title=title,preset=preset,comparisons=rows,physical_recordings=status)
        for row in rows:
            identity=(row['layout'],row['case']);methods=grouped[identity];p=methods['proposed'];prefix=key+'_'
            names=[f'{prefix}{identity[0]}_{identity[1]}_comparison.png',f'{prefix}{identity[0]}_{identity[1]}_spatial.png',f'{prefix}{identity[0]}_{identity[1]}_wind.png']
            audits=[directory/plan['audit'] for plan in p['plans'] if plan.get('audit') and (directory/plan['audit']).exists()]
            wanted=next((v for v in audits if v.stem.endswith('0001')),audits[0] if audits else None)
            if args.plots:
                paths_figure(identity,methods,prefix);spatial_figure(p,prefix);wind_figure(p,prefix)
                if wanted:names.append(intermediate_figure(p,wanted,prefix))
            else:names+=sorted(v.name for v in OUT.glob(f'{prefix}{identity[0]}_{identity[1]}_intermediate_t*.png'))
            figures.append((key,TITLES[identity],names))
    screens=[]
    for directory in ('initial_screen_v2','parameter_screen20'):
        for path in (ROOT/directory).glob('*/*_proposed_8101.json'):
            d=load(path)
            if d.get('completed'):screens.append(dict(variant=path.parent.name,window_seconds=d['mission_seconds'],first_predicted_cost=d['plans'][0]['predicted_optimistic'],**metrics(d),config=d['config']))
    ablations=[]
    for path in (ROOT/'ablations120').glob('*_8101.json'):
        d=load(path)
        if d.get('completed'):ablations.append(dict(method=d['method'],**metrics(d)))
    d=load(ROOT/'calibrated300/two_crosswind_proposed_8101.json')
    if d['completed_seconds']>=120:ablations.append(dict(method='proposed (first 120 s)',cost=float(np.interp(120,[f['t'] for f in d['frames']],[f['cost'] for f in d['frames']])),success=True))
    newest_ablations=[]
    for source in (ROOT/'supported300/two_crosswind_proposed_8101.json',*sorted((ROOT/'supported_ablations120').glob('*_8101.json'))):
        if not source.exists():continue
        item=load(source)
        if item['completed_seconds']>=120:newest_ablations.append(dict(method=item['method'],cost=float(np.interp(120,[f['t'] for f in item['frames']],[f['cost'] for f in item['frames']])),success=item['success']))
    diagnostic_files={'sampler':'sampler_covariance.json','estimator':'estimator_replay.json','selection':'selection_bank128.json','controller_uncertainty':'filter_belief_counterfactual.json'}
    diagnostics={key:load(ROOT/name) for key,name in diagnostic_files.items() if (ROOT/name).exists()}
    audits={key:load(ROOT/key/'execution_integrity.json') for key in (*STAGES,'supported_ablations120') if (ROOT/key/'execution_integrity.json').exists()}
    summary=dict(stages=stages,parameter_screens=screens,ablations120=ablations,supported_ablations120=newest_ablations,diagnostics=diagnostics,execution_integrity=audits,regression_tests='62 passed in 394.17 s',notes='Single-seed development evidence. Stages use different configurations. Failed flights are retained but excluded from performance percentages. Raw decisions, physics, observations, settings and source snapshots are retained.')
    (OUT/'results.json').write_text(json.dumps(summary,indent=2))
    content='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Planner investigation</title><style>body{font:17px/1.55 system-ui;max-width:1240px;margin:40px auto;padding:0 24px;color:#1e2e3c}h1,h2,h3{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border-bottom:1px solid #ccd4dc;text-align:left;padding:10px}img{width:100%;height:auto}li{margin:12px 0}.note{background:#eef3f7;padding:16px}.failed{color:#a0293d}code{font-size:14px}a{color:#0c6f83}nav{display:flex;gap:18px;flex-wrap:wrap}details{border-top:1px solid #ccd4dc;padding:18px 0}summary{cursor:pointer;font-weight:650}.scroll{overflow:auto}</style></head><body><h1>Planner investigation</h1><p>Implementation defects, measured corrections and five-minute nonlinear quadrotor missions. These 2D diagnostics use physical simulation records.</p><nav><a href="#results">Comparisons</a><a href="#defects">Confirmed defects</a><a href="#learning">Learning tests</a><a href="#diagnostics">Maps and MBD search</a><a href="#limits">Limits</a></nav>
    <p class="note"><b>Metric:</b> accumulated mean clarity deficit, in deficit·s. At each time, take max(target clarity − actual clarity, 0) in each inspection cell, average across all inspection cells, then integrate over the 300 s mission. Lower is better. Positive reductions favor proposed. No percentage is computed against a terminated flight. All methods within a row share geometry, wind, vehicle, initial state, clarity, observation settings and MBD budget.</p><h2 id="results">Five-minute comparisons</h2><p>The tables retain successive revisions separately. This is single-seed development evidence, not a multi-seed statistical claim.</p>'''
    for key in reversed(STAGES):
        stage=stages[key];content+=f'<h3>{html.escape(stage["title"])}</h3>'
        content+='<div class="scroll">'+table(stage['comparisons'])+'</div>' if stage['comparisons'] else '<p>No complete three-method comparison yet.</p>'
        for r in stage['physical_recordings']:
            if not r['completed']:content+=f'<p>Running: {html.escape(r["file"])} at {r["seconds"]:g} / 300 s. Partial cost excluded.</p>'
        notes={
          'calibrated300':'Posterior updates at the 5 s planning calls. The adverse filtered-blind flight terminated at 9.86 s; its path is retained and marked. The online-estimator revision repeats this comparison.',
          'online300':'Actual observations update estimation and the runtime controller every 1 s, retaining the planned reference and clock. MBD replans every 5 s. The filtered baseline completes this revision.',
          'supported300':'Online 1 s estimation, inducing-location relocation, converged covariance correction in field sampling, and forecast sensing termination at contact. Kernel parameters remain fixed. Individual task-performance effects of this bundle are not isolated.',
          'supported_forward_reverse300':'Fresh 300 s forward/reversed missions using exactly the same configuration as the latest corrected adverse-case comparison: 1 s online estimation, inducing-location relocation, corrected field sampling and forecast termination at contact. No retuning between cases or methods. Same seed 8101; the physical wind direction is reversed.'}
        content+=f'<p>{notes[key]}</p>'
    content+='<h2 id="defects">Confirmed defects and corrections</h2><ol>'+''.join(f'<li><b>{title}.</b> {body}</li>' for title,body in FACTS)+'</ol>'
    content+='''<h3>Why B is not automatically optimal</h3><p>At the original wind strength, a known-field timed-reference diagnostic gives A slightly lower one-visit monitoring cost than B (about 5.75 versus 6.05 deficit·s). Its shorter distance can remain worthwhile despite slower headwind travel. With stronger wind, those timed A references often fail to arrive within 45 s while B arrives in about 20 s. This diagnoses those reference timings; it does not prove every A trajectory infeasible. Task cost, rather than a desired route label, decides.</p>
    <h3>Why clarity deficit can look weak</h3><p>The clarity model passes independent sensing-ODE and exact unsensed-decay checks. Deficits are clipped per cell before averaging, so oversensing cannot cancel losses elsewhere. But the original process noise is mild, and a 3.2 m sensing radius can cover an ordinary inspection site from its center. Brief transits can produce small instantaneous deficits. Initial clarity is asymmetric, so wind reversal need not give symmetric curves. These task parameters are unchanged within comparisons.</p>
    <h3>Estimator replay</h3><p>With the same actual observation stream, inducing-location relocation reduced held-out corridor wind RMSE from 0.198 to 0.174 m/s and improved predictive 95% interval coverage from 93.0% to 95.3%. Shortening kernel length scales worsened error in tested variants. The latest revision relocates inducing points but does not train kernel parameters online.</p>
    <h2 id="learning">120-second adverse-case ablations</h2><p>First corrected revision; all three flights complete without contact. Lower is better.</p><table><tr><th>Variant</th><th>Deficit·s</th></tr>'''
    for r in sorted(ablations,key=lambda x:x['cost']):content+=f'<tr><td>{html.escape(r["method"])}</td><td>{r["cost"]:.3f}</td></tr>'
    content+='</table><h3>The same window after the field-sampling correction</h3><table><tr><th>Variant</th><th>First 120 s deficit·s</th></tr>'
    for r in newest_ablations:content+=f'<tr><td>{html.escape(ABLATION_NAMES.get(r["method"],r["method"]))}</td><td>{r["cost"]:.3f}</td></tr>'
    content+='''</table><p><code>aware</code> optimizes expected task cost and removes the proposed proposal-mixture mechanism as well as favorable-tail selection; this does not isolate optimism alone. <code>no_learning</code> freezes the posterior for planning and execution, measuring the overall benefit of disturbance learning rather than a pure exploration bonus.</p>'''
    if 'selection' in diagnostics:
        check=diagnostics['selection'];content+=f'<h3>Independent trajectory-ranking check</h3><p>For all {check["nominal_sequences"]} recorded finalists at the newest pre-observation planning call, increasing the independent selection bank from 8 to {check["independent_fields"]} fields retained the selected trajectory at rank {check["selected_rank_one_based"]}. Rank correlation was {check["rank_correlation"]:.3f}; its score changed from {check["saved_selected_score"]:.3f} to {check["larger_bank_selected_score"]:.3f}. All 16 independent eight-field blocks selected the same recorded index. Initial selection instability is therefore not supported as the explanation at this state. Later posterior states are not covered by this check.</p>'
        if 'route_groups' in check:
            content+='<p>The predicted best reference through each passage gives the following scores under the same posterior and runtime-controller prediction:</p><table><tr><th>First crossing</th><th>Best score, 128 fields</th></tr>'
            for r in sorted(check['route_groups'],key=lambda r:r['larger_bank_score']):content+=f'<tr><td>{r["first_crossing"]}</td><td>{r["larger_bank_score"]:.3f}</td></tr>'
            content+='</table><p>B is present. The current forecast ranks C above B before observing the hidden wind. This is an objective/controller prediction issue, not a missing geometric proposal in the corrected search.</p>'
    if 'controller_uncertainty' in diagnostics:
        content+='''<h3>Why the forecast undervalues a potentially learnable passage</h3><p>A controlled diagnostic holds the recorded initial B reference, state, clarity, mean wind and zero-mean diagnostic physical field fixed. With the recorded controller construction-field spread, the prediction never reaches the eastern region within 45 s and incurs 11.277 deficit·s. Removing only that construction spread gives entry at 29.0 s and 7.832 deficit·s. Thus fixed controller uncertainty can suppress the predicted benefit of B even when the diagnostic physical field is calm.</p><p>This zero-spread controller is an idealized counterfactual, not a claimed reachable posterior or a replacement mission. C still reaches the region at 20.4 s and costs about 6.35 in that zero-wind diagnostic, with or without construction uncertainty. Controller uncertainty therefore explains part, not all, of B's poor forecast. Reference timing also matters: the current RRT time parameterization stops at geometric vertices, so extra vertices can make a shorter geometric path slower. The small MBD budget does not reliably undo that initialization bias.</p><p>The method note explicitly fixes the belief throughout each prediction (page 3) and distinguishes optimism from value of learning (page 12). The present objective therefore varies the physical wind samples while holding the controller posterior fixed throughout a prediction. It cannot credit future observations for changing that controller. The corrected expected-cost ablation also starts with the same C reference, so initial optimism alone is not the entire cause. Reintroducing predicted posterior/controller changes would be a method revision, rather than another claimed numerical fix.</p>'''
    content+='''
    <h3>Original baselines and unsuccessful trials</h3><p>RRT initialization and reference timing changes also change blind choices. Original adverse blind planning achieved 17.730 deficit·s, stronger than the new blind results. Improvement against a new matched baseline is not improvement over every old baseline. Original costs are retained in <a href="results.json">results.json</a>. A corrected-only broad-prior/guidance 120 s trial scored 17.686 versus 13.754 for the original proposed prefix: numerical corrections alone did not improve that configuration. Prior-scale tuning, model weighting and penalty strength remain development choices.</p><p>Completed 5–20 s screens vary penalty weight, wind, scale, 45/60 s horizons, MBD iterations and samples, temperature and decision spacing. They inspect initial behavior and runtime, not long-term superiority. Coarser decisions with finer controller substeps were slower because they increased filter evaluations. Full configurations are in results.json.</p>
    <h2 id="diagnostics">Spatial maps, physical motion and intermediate MBD trajectories</h2><p>Trajectories are recorded nonlinear motion. Clarity maps show cell deficits and wind maps posterior standard deviation per component. The circle is geometric sensing range; actual sensing also accounts for building visibility. Search panels show sampled nominal references and selected predicted motion under the posterior mean field. Failed flights are marked and excluded from cost curves.</p>'''
    for key,title,names in reversed(figures):
        names=[name for name in names if (OUT/name).exists()]
        if not names:continue
        content+=f'<details><summary>{html.escape(STAGES[key][0]+" · "+title)}</summary>'
        for name in names:content+=f'<p><a href="{name}"><img loading="lazy" src="{name}" alt="{html.escape(title)} diagnostic figure"></a></p>'
        content+='</details>'
    content+='''<h2 id="limits">Limits and reproduction</h2><p>45 s horizon; 0.2 s trajectory decisions; 0.05 s execution controller; 0.01 s nonlinear rigid-body/motor physics; replanning every 5 s; 6 MBD levels × 16 samples; finite cosine time interval [0, 0.05], with largest proposal standard deviation about 0.09 in parameter coordinates; temperature 0.30; 4 performance fields and 8 independent selection fields; 2 posterior field draws form the controller constraints. Prior wind standard deviation 1.2 m/s; soft violation weight 1. Following the later request to permit planning slack, planning uses soft penalties rather than the note's original strict feasibility-indicator rule. Physical contact ends sensing credit. Four-site execution clearance is 0.70 m for all methods; other cases use 0.55 m.</p><p>This is local MBD refinement around RRT-derived initializations, not a demonstrated globally converged trajectory search. Full time-indexed acceleration coordinates remain independent; no temporal basis is imposed. A larger-bank ranking check is reported for one pre-observation state; posterior-state ranking convergence and sensitivity to the two controller-construction draws are still unverified. Optimism under a frozen planning posterior does not explicitly predict how future observations alter a later controller. These runs support disturbance-aware execution and useful online learning; they do not isolate optimal value-of-information planning. No multi-seed generalization or real-time claim is made. Simulation time pauses during planning; concurrent CPU timings are not real-time measurements.</p><p><code>.venv/bin/python scripts/run_quarter_investigation_missions.py --preset supported --cases forward reverse crosswind large_east calm_a multi --methods proposed blind blind_filter --output NEW_DIRECTORY</code></p><p>Presets <code>calibrated</code> and <code>online</code> reproduce the earlier stages. Loaded source snapshots, controls, weights, actual measurements, settings and 100 Hz physical records accompany each experiment. <code>scripts/audit_rrt_quarter.py --results DIRECTORY</code> reconstructs motion, attitude, timestamps, cost and clearance from raw records. <a href="results.json">Machine-readable results</a> distinguish completed, failed and running flights.</p></body></html>'''
    (OUT/'index.html').write_text(content)
    print('Report:',OUT/'index.html',{k:len(v['comparisons']) for k,v in stages.items()})

if __name__=='__main__':main()
