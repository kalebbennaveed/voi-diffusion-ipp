"""Fine geometric task routes and full-GP survey starts for original MBD.

No true wind field is queried. Clarity remains on its original physical grid;
only geometric initialization uses a finer grid to represent the upper passage.
Route-target information and frontier scores rank starts, never replace the
full physical/clarity/RPCBF objective used by the optimizer.
"""
from time import perf_counter
import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from monitoring_routes import clearance,visible,passage_cuts,crosses_cuts,smooth_reference_seed,reference_seed
from survey_information import prepare_full_targets,full_target_variance_reduction_components

_prepare = jax.jit(prepare_full_targets)
_gains = jax.jit(jax.vmap(full_target_variance_reduction_components, in_axes=(None, 0, None)))

def task_paths(x,q,env,cfg,count=6):
    started=perf_counter();ob=np.asarray(env.obstacles);bounds=np.asarray(env.bounds);cells=np.asarray(env.cells)
    nx,ny=128,74;dx=.0625;xx,yy=np.meshgrid((np.arange(nx)+.5)*dx,(np.arange(ny)+.5)*dx)
    nodes=np.c_[xx.ravel(),yy.ravel()];margin=.125
    free=clearance(nodes,ob,bounds)>=margin
    rows=[];cols=[];weights=[]
    ids=np.arange(nx*ny).reshape(ny,nx)
    for di,dj in [(1,0),(0,1),(1,1),(-1,1)]:
        j0=max(0,-dj);j1=min(ny,ny-dj);i0=max(0,-di);i1=min(nx,nx-di)
        a=ids[j0:j1,i0:i1].ravel();b=ids[j0+dj:j1+dj,i0+di:i1+di].ravel()
        ok=free[a]&free[b]&visible(nodes[a],nodes[b],ob,bounds,margin)
        a=a[ok];b=b[ok];rows.extend([a,b]);cols.extend([b,a]);weights.extend([np.full(len(a),dx*np.hypot(di,dj))]*2)
    rows=np.concatenate(rows);cols=np.concatenate(cols);lengths=np.concatenate(weights)
    edge_clearance=np.minimum(clearance(nodes[rows],ob,bounds),clearance(nodes[cols],ob,bounds))
    weights=lengths*(1+(.18/np.maximum(edge_clearance,.125))**4)
    graph=csr_matrix((weights,(rows,cols)),shape=(nx*ny,nx*ny))
    cuts=passage_cuts(ob,bounds,margin);edge_cuts=crosses_cuts(nodes[rows],nodes[cols],cuts)
    possible=np.flatnonzero(free);order=possible[np.argsort(np.linalg.norm(nodes[possible]-x[:2],axis=1))]
    connected=order[visible(np.broadcast_to(x[:2],(len(order),2)),nodes[order],ob,bounds,0.)]
    if not len(connected):return [np.array([x[:2]])]*count,dict(milliseconds=(perf_counter()-started)*1000,paths=0)
    start=int(connected[0]);dist,pred=dijkstra(graph,indices=start,return_predecessors=True,limit=np.inf)
    # Clarity lives on the unchanged original .125 m grid. Interpolate only
    # its initializer ranking values to the finer geometric search grid.
    from scipy.spatial import cKDTree
    _,nearest_cells=cKDTree(cells).query(nodes)
    # Exact no-observation Riccati solution, then approximate footprint value.
    times=np.array([.25,.5,1.])*cfg.horizon_seconds
    future=np.asarray(q)[None]/(1+times[:,None]*np.asarray(env.process_noise)[None]*np.asarray(q)[None])
    need=np.maximum(np.asarray(env.targets)[None]-future,0).mean(axis=0)
    raw=need[nearest_cells];raw[~free]=0
    value=gaussian_filter(raw.reshape(ny,nx),cfg.sensor_sigma/(np.sqrt(2)*dx),mode='constant').ravel()
    interior=np.clip(clearance(nodes,ob,bounds)/.5,0,1)
    score=value*interior/(1+.012*dist/.9);score[~np.isfinite(dist)|~free]=-np.inf
    selected=[];ranking=score.copy()
    for _ in range(4):
        i=int(np.argmax(ranking));selected.append(i);ranking[np.linalg.norm(nodes-nodes[i],axis=1)<1.15]=-np.inf
    def recover(tree,end,origin=start):
        route=[int(end)]
        for _ in range(nx*ny):
            if route[-1]==origin:break
            previous=int(tree[route[-1]])
            if previous<0:return []
            route.append(previous)
        return route[::-1]
    paths=[];destinations=[];mode_nodes=[];mode_cuts=[]
    for goal in selected:
        route=recover(pred,goal)
        if route:mode_nodes.append(route);destinations.append(goal);mode_cuts.append(np.zeros(len(cuts),bool))
    # Temporarily cut the bottlenecks used by each previous route. Merely
    # penalizing a thin band can return a parallel path through the SAME gap.
    # All cuts are inferred from geometry and affect initialization only.
    goal=max(selected,key=lambda g:dist[g]);initial=recover(pred,goal);route=initial
    blocked=np.zeros(len(cuts),bool)
    for _ in range(count-len(mode_nodes)):
        blocked|=crosses_cuts(nodes[route[:-1]],nodes[route[1:]],cuts).any(axis=0)
        allowed=~np.any(edge_cuts&blocked[None],axis=1)
        alternative=csr_matrix((weights[allowed],(rows[allowed],cols[allowed])),shape=graph.shape)
        _,tree=dijkstra(alternative,indices=start,return_predecessors=True,limit=np.inf)
        candidate=recover(tree,goal)
        if not candidate:
            candidate=initial;applied=np.zeros(len(cuts),bool)
        else:applied=blocked.copy()
        route=candidate;mode_nodes.append(route);destinations.append(goal);mode_cuts.append(applied)
    for mode_index,route in enumerate(mode_nodes):
        raw_path=np.vstack([x[:2],nodes[route]])
        # Visibility pruning preserves each collision-free passage exactly.
        simplified=[raw_path[0]];i=0
        while i<len(raw_path)-1:
            remaining=raw_path[i+1:]
            # Preserve the clearance-weighted graph's safe interior tendency
            # when pruning long chords. An adjacent original edge remains a
            # valid fallback where the passage is narrower than this margin.
            ok=visible(np.broadcast_to(raw_path[i],remaining.shape),remaining,ob,bounds,.155)
            # Remove grid-induced detours by straight visibility chords.
            # Each chord is checked against the analytic inflated circles.
            ok &= ~crosses_cuts(np.broadcast_to(raw_path[i],remaining.shape),remaining,cuts[mode_cuts[mode_index]]).any(axis=1)
            candidates=np.flatnonzero(ok);j=i+1+(int(candidates[-1]) if len(candidates) else 0)
            simplified.append(raw_path[j]);i=j
        paths.append(np.asarray(simplified))
    return paths,dict(milliseconds=(perf_counter()-started)*1000,paths=len(paths),nodes=int(free.sum()),edges=len(rows),
                      initial_destinations=nodes[destinations].tolist(),grid=[nx,ny],search='Compiled SciPy Dijkstra; online 128x74 grid; clearance-weighted edges and temporary inferred passage cuts',margin_m=margin,passage_cuts=cuts.tolist(),mode_blocked_cuts=[cuts[c].tolist() for c in mode_cuts])


def _arc(path):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]


def _sample(path, distances):
    arc = _arc(path)
    return np.c_[np.interp(distances, arc, path[:, 0]), np.interp(distances, arc, path[:, 1])]


def _prefix(path, distance):
    arc = _arc(path)
    return np.vstack((path[arc < distance - 1e-8], _sample(path, [distance])))


def remaining_route(path, position, obstacles, bounds):
    """Reconnect to the nearest visible suffix of an already selected route.

    This preserves the route's destination and untraversed passage. It never
    substitutes a newly shorter route through another obstacle gap. A current
    state with less than the initializer's .125 m clearance can reconnect at
    its existing clearance; full physical forecasts still validate the seed.
    """
    path, position = np.asarray(path, float), np.asarray(position, float)
    if len(path) < 2:
        return np.vstack((position, path[-1])) if len(path) else None
    delta = np.diff(path, axis=0)
    length2 = np.sum(delta * delta, axis=1)
    fraction = np.clip(np.sum((position-path[:-1])*delta, axis=1)/np.maximum(length2, 1e-30), 0, 1)
    projections = path[:-1] + fraction[:, None]*delta
    margin = min(.125, max(0., float(clearance(position, obstacles, bounds))-1e-7))
    allowed = visible(np.broadcast_to(position, projections.shape), projections, obstacles, bounds, margin)
    distances = np.where(allowed, np.linalg.norm(projections-position, axis=1), np.inf)
    if not np.any(np.isfinite(distances)):
        return None
    index = int(np.argmin(distances))
    route = np.vstack((position, projections[index], path[index+1:]))
    keep = np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-8]
    return route[keep]


def _frontier_distance(path, point):
    """Arc position of a stored frontier; None when it is behind the suffix."""
    if len(path) < 2:
        return None
    delta = np.diff(path, axis=0)
    fraction = np.clip(np.sum((np.asarray(point)-path[:-1])*delta, axis=1)
                       / np.maximum(np.sum(delta*delta, axis=1), 1e-30), 0, 1)
    projections = path[:-1] + fraction[:, None]*delta
    index = int(np.argmin(np.linalg.norm(projections-point, axis=1)))
    if np.linalg.norm(projections[index]-point) > .15:
        return None
    return float(_arc(path)[index] + fraction[index]*np.linalg.norm(delta[index]))


def seed_bank(x, q, env, cfg, belief, previous=None, *, relevance_weights=None,
              frontier_state=None, survey_enabled=True):
    """Return seven full-control starts, incumbent, metadata, and 48 targets.

    Three task starts are the direct and two alternative passages to the
    geometrically discovered distant valuable destination. The highest-value
    original clarity route supplies a local monitoring alternative. Two survey
    starts end at distinct uncertainty frontiers. A hold start completes the
    original seven-mode MBD initialization.

    ``relevance_weights(positions48, clarity_weights48, task_seeds3,
    task_indices3)`` may apply the planner's physical/task relevance test.
    Its exact returned weights score frontiers and are returned for MBD: a
    different passage cannot silently become the information objective later.
    The callback sees full task references before a persistent frontier prefix
    replaces one task seed. Its three paired relevance forecasts can therefore
    be computed once and reused by the caller's optimizer.

    Frontier information uses 25 geometric samples and only the owner's eight
    target points. It ranks initialization, not physical predictions. An
    optional JSON-compatible frontier state preserves an unfinished route and
    waypoint in its original passage slot. The caller commits one of the
    returned ``seed_frontier_states`` only when choosing that survey motion.
    """
    began = perf_counter()
    x, q = np.asarray(x), np.asarray(q)
    paths, metadata = task_paths(x, q, env, cfg, count=6)
    hold = reference_seed(x[:2][None], x, cfg)
    if previous is None:
        shifted = hold
    else:
        controls = np.asarray(previous).reshape(cfg.steps, 2)
        shifted = np.concatenate((controls[cfg.commit_steps:], np.zeros_like(controls[:cfg.commit_steps]))).ravel()
    if not paths or 'initial_destinations' not in metadata:
        metadata.update(coarse_paths=[], total_seed_milliseconds=(perf_counter()-began)*1000,
                        survey_frontiers=[], task_route_indices=[], retiming_fallbacks=[False]*6,
                        seed_frontier_states=[None]*7, seed_path_indices=[None]*7,
                        next_frontier_state=None, persistent_frontier_status='unreachable')
        return (jnp.asarray(np.tile(hold, (7, 1))), jnp.asarray(shifted), metadata,
                jnp.asarray(np.tile(x[:2], (48, 1))), jnp.zeros(48))

    destinations = np.asarray(metadata['initial_destinations'])
    direct = int(np.argmin(np.linalg.norm(destinations[:4]-destinations[-1], axis=1)))
    task_indices = [direct, 4, 5]
    clarity_start = paths[0].copy()
    persistent_slot, persistent_distance = None, None
    persistent_status = 'none'
    # Retain geometry before making targets or evaluating task relevance.
    # A nearby clarity peak must not erase the still-uncertain remote passage.
    if survey_enabled and frontier_state is not None:
        slot = int(frontier_state.get('passage_slot', 0))
        stored = np.asarray(frontier_state.get('route_points', []), float)
        if slot in (1, 2, 3) and stored.ndim == 2 and stored.shape[1] == 2 and len(stored):
            if np.linalg.norm(x[:2]-stored[-1]) <= .35:
                persistent_status = 'destination_reached'
            else:
                suffix = remaining_route(stored, x[:2], np.asarray(env.obstacles), np.asarray(env.bounds))
                if suffix is not None and len(suffix) > 1:
                    persistent_slot = slot
                    paths[task_indices[slot-1]] = suffix
                    point = np.asarray(frontier_state.get('frontier_point', stored[-1]), float)
                    persistent_distance = _frontier_distance(suffix, point)
                    persistent_status = ('frontier_reached' if np.linalg.norm(x[:2]-point) <= .35
                                         or persistent_distance is None or persistent_distance <= .35 else 'retained')
                else:
                    persistent_status = 'route_disconnected'

    # Build task targets on complete routes; targets are not limited to the
    # committed prefix, so local survey motion can value useful distant paths.
    from scipy.spatial import cKDTree
    cells = np.asarray(env.cells)
    future = q / (1 + cfg.horizon_seconds * np.asarray(env.process_noise) * q)
    need = np.maximum(np.asarray(env.targets) - future, 0)
    tree = cKDTree(cells)
    targets, weights = [], []
    persistent_target = None
    for path_index, path in enumerate(paths):
        length = _arc(path)[-1]
        target_distances = np.linspace(length/8, length, 8)
        if (persistent_slot is not None and path_index == task_indices[persistent_slot-1]
                and persistent_status == 'retained'):
            # Include the exact retained frontier in the finite target set so
            # its learning/release test uses its own full predictive variance.
            index = int(np.argmin(np.abs(target_distances-persistent_distance)))
            target_distances[index] = persistent_distance
            target_distances.sort()
            persistent_target = path_index*8 + int(np.argmin(np.abs(target_distances-persistent_distance)))
        points = _sample(path, target_distances)
        _, endpoint = tree.query(path[-1])
        # Modest progress weighting avoids spending the entire target budget
        # on already visited starts shared by every possible task route.
        targets.append(points)
        weights.append(float(need[endpoint]) * np.linspace(.25, 1., 8))
    targets = np.concatenate(targets)
    weights = np.concatenate(weights)
    weights = weights / max(weights.sum(), 1e-30)

    task_starts = [paths[i] for i in task_indices]
    task_references = [smooth_reference_seed(path, x, env, cfg) for path in task_starts]
    clarity_weights = weights.copy()
    if relevance_weights is not None:
        weights = np.asarray(relevance_weights(jnp.asarray(targets, jnp.float32),
                             jnp.asarray(weights, jnp.float32),
                             jnp.asarray(np.stack([r[0] for r in task_references]), jnp.float32),
                             jnp.asarray(task_indices, jnp.int32)), dtype=np.float32)
        if weights.shape != (48,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError('Relevance callback must return 48 finite nonnegative weights')
    candidates, observed, distances, owners = [], [], [], []
    # A fixed-size batch bounds compute and keeps the same JAX executable.
    for i in task_indices:
        path = paths[i]
        length = _arc(path)[-1]
        for desired in (.5, 1., 2., 3., 4.5, 6.):
            distance = min(length, desired)
            candidates.append(_prefix(path, distance))
            observed.append(_sample(path, np.linspace(distance/25, distance, 25)))
            distances.append(distance)
            owners.append(i)
    started = perf_counter()
    # Actual POAM conditioning may deliberately use float64. Initialization
    # uses the same explicitly float32 planning posterior as MBD and does not
    # change the retained execution belief.
    planning_belief = jax.tree.map(
        lambda value: value.astype(jnp.float32)
        if hasattr(value, 'dtype') and jnp.issubdtype(value.dtype, jnp.floating)
        else value, belief)
    gains_np = np.zeros(len(candidates))
    target_variance = np.zeros(48)
    if survey_enabled and np.any(weights > 0):
        with jax.experimental.disable_x64():
            prepared = _prepare(planning_belief, jnp.asarray(targets, jnp.float32), jnp.asarray(weights, jnp.float32))
            components = _gains(planning_belief, jnp.asarray(observed, jnp.float32), prepared)
            jax.block_until_ready(components)
        # Owner-only covariance reduction prevents a lower-route seed from
        # winning because it incidentally correlates with a relevant upper path.
        route_components = np.asarray(components).reshape((len(candidates), 6, 8)).sum(axis=2)
        gains_np = route_components[np.arange(len(candidates)), owners]
        target_variance = np.asarray(prepared.variance)
    information_seconds = perf_counter() - started
    scores = gains_np / (1 + .15 * np.asarray(distances))
    endpoints = np.asarray([p[-1] for p in candidates])
    # Exclude effectively zero-length frontiers. Keep full task paths as a
    # deterministic fallback if a model has no useful information anywhere.
    scores[(np.asarray(distances) < .20) | (gains_np <= 1e-12)] = -np.inf
    next_state = None
    if persistent_slot is not None:
        owner = task_indices[persistent_slot-1]
        if np.sum(weights[owner*8:(owner+1)*8]) <= 1e-12 or not survey_enabled:
            persistent_status = 'irrelevant'
        else:
            if persistent_target is not None and target_variance[persistent_target] <= .30**2:
                persistent_status = 'frontier_learned'
            if persistent_status != 'retained':
                route_options = np.flatnonzero(np.asarray(owners) == owner)
                best = int(route_options[np.argmax(scores[route_options])])
                if np.isfinite(scores[best]):
                    persistent_distance = distances[best]
                else:
                    persistent_distance = None
            if persistent_distance is not None:
                retained_path = _prefix(paths[owner], persistent_distance)
                task_starts[persistent_slot-1] = retained_path
                task_references[persistent_slot-1] = smooth_reference_seed(retained_path, x, env, cfg)
                next_state = dict(route_points=paths[owner].tolist(), frontier_point=retained_path[-1].tolist(),
                                  passage_slot=persistent_slot, task_destination=paths[owner][-1].tolist())
    chosen, ranking = [], scores.copy()
    for _ in range(2):
        index = int(np.argmax(ranking))
        if not np.isfinite(ranking[index]):
            break
        chosen.append(index)
        ranking[np.linalg.norm(endpoints-endpoints[index], axis=1) < .45] = -np.inf
        # Keep survey alternatives on different inferred passages rather than
        # endpoints on the same central route with identical prefixes.
        ranking[np.asarray(owners) == owners[index]] = -np.inf
    survey_starts = [candidates[i] for i in chosen]
    survey_owners = [owners[i] for i in chosen]
    while len(survey_starts) < 2:
        survey_starts.append(task_starts[len(survey_starts)])
        survey_owners.append(task_indices[len(survey_owners)])
    seed_paths=task_starts+[clarity_start]+survey_starts
    references = task_references + [smooth_reference_seed(path, x, env, cfg) for path in seed_paths[3:]]
    seeds = np.stack([hold, *[reference[0] for reference in references]])
    def state_for(owner, frontier):
        if not survey_enabled or np.sum(weights[owner*8:(owner+1)*8]) <= 1e-12:
            return None
        return dict(route_points=paths[owner].tolist(), frontier_point=np.asarray(frontier).tolist(),
                    passage_slot=task_indices.index(owner)+1, task_destination=paths[owner][-1].tolist())
    states = [None]
    for slot, owner in enumerate(task_indices, 1):
        options = np.flatnonzero(np.asarray(owners) == owner)
        best = int(options[np.argmax(scores[options])])
        state = state_for(owner, endpoints[best]) if np.isfinite(scores[best]) else None
        states.append(next_state if slot == persistent_slot else state)
    states.append(None)
    states.extend(state_for(owner, path[-1]) for owner, path in zip(survey_owners, survey_starts))
    metadata.update(
        coarse_paths=[p.tolist() for p in paths],
        seed_paths=[p.tolist() for p in seed_paths],
        task_route_indices=task_indices,
        clarity_route_index=0,
        clarity_seed_path=clarity_start.tolist(),
        seed_path_indices=[None, *task_indices, 0, *survey_owners],
        seed_frontier_states=states,
        next_frontier_state=next_state,
        persistent_frontier_status=persistent_status,
        clarity_target_weights=clarity_weights.tolist(),
        target_weights_already_gated=relevance_weights is not None,
        survey_frontiers=[dict(path_index=int(owners[i]), length_m=float(distances[i]),
                               point=endpoints[i].tolist(), variance_reduction=float(gains_np[i]),
                               initialization_score=float(scores[i])) for i in chosen],
        survey_frontier_candidates=len(candidates),
        survey_initialization_seconds=information_seconds,
        survey_target_definition='48 geometric-route points weighted by future endpoint clarity deficit and the shared relevance callback; frontier gain uses only its own passage targets',
        retiming_fallbacks=[reference[1] for reference in references],
        total_seed_milliseconds=(perf_counter()-began)*1000,
    )
    return (jnp.asarray(seeds), jnp.asarray(shifted), metadata,
            jnp.asarray(targets, jnp.float32), jnp.asarray(weights, jnp.float32))

