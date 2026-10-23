"""Fast, known-geometry initializations for the full MBD control sequence.

A cached visibility graph supplies several obstacle winding classes. Rounded
corners and curvature-aware time scaling turn each geometric path into a
moving double-integrator reference. Every returned coordinate is an ordinary
MBD acceleration decision; these paths impose no route or waypoint constraints.
There is deliberately no access to wind, beliefs, task values, or case labels.
"""

import heapq
from functools import lru_cache

import numpy as np

from .rrt import segment_free, winding_signature

AIRCRAFT_RADIUS = 0.275  # The physical 0.55 m aircraft never scales with the map.


def _geometry(rectangles, bounds):
    rectangles = np.asarray(rectangles, float)
    if not rectangles.size:
        rectangles = np.empty((0, 4))
    else:
        rectangles = rectangles.reshape((-1, rectangles.shape[-1]))
    return rectangles[:, :4], np.asarray(bounds, float)


@lru_cache(maxsize=32)
def _roadmap(rectangles_key, bounds_key, margin):
    """Cache only static known geometry; current states and goals stay dynamic."""
    rectangles = np.asarray(rectangles_key, float).reshape((-1, 4))
    bounds = np.asarray(bounds_key, float).reshape((2, 2))
    corners = []
    for rectangle in rectangles:
        lo, hi = rectangle[:2] - margin - 0.025, rectangle[2:] + margin + 0.025
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                point = np.array([x, y])
                if segment_free(point, point, rectangles, bounds, margin):
                    corners.append(point)
    points = np.asarray(corners, float).reshape((-1, 2))
    graph = np.full((len(points), len(points)), np.inf)
    for i in range(len(points)):
        for j in range(i):
            if segment_free(points[i], points[j], rectangles, bounds, margin):
                graph[i, j] = graph[j, i] = np.linalg.norm(points[i] - points[j])
    return points, graph


def _distances(graph, source):
    distance = np.full(len(graph), np.inf)
    parent = np.full(len(graph), -1, int)
    distance[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distance[node]:
            continue
        for other in np.flatnonzero(np.isfinite(graph[node])):
            candidate = cost + graph[node, other]
            if candidate + 1e-10 < distance[other]:
                distance[other], parent[other] = candidate, node
                heapq.heappush(queue, (candidate, int(other)))
    return distance, parent


def _branch(parent, node):
    result = []
    while node >= 0:
        result.append(int(node))
        node = parent[node]
    return result[::-1]


def visibility_paths(start, goal, rectangles, bounds, margin=0.65, count=3):
    """Return up to ``count`` short, distinct obstacle winding classes.

    Candidate routes contain a visible edge with shortest collision-free
    prefixes to either endpoint. This finite geometric construction includes
    narrow passages without random sampling or hand-specified passage gates.
    It is not an exhaustive homotopy enumerator for arbitrary obstacle fields.
    """
    rectangles, bounds = _geometry(rectangles, bounds)
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if not segment_free(start, start, rectangles, bounds, margin):
        return []
    if not segment_free(goal, goal, rectangles, bounds, margin):
        return []
    if np.linalg.norm(goal - start) < 1e-4:
        return [start[None]]
    static, static_graph = _roadmap(tuple(rectangles.ravel()), tuple(bounds.ravel()), float(margin))
    points = np.vstack((start, goal, static))
    graph = np.full((len(points), len(points)), np.inf)
    graph[2:, 2:] = static_graph
    for i in range(2):
        for j in range(i + 1, len(points)):
            if segment_free(points[i], points[j], rectangles, bounds, margin):
                graph[i, j] = graph[j, i] = np.linalg.norm(points[i] - points[j])
    distance, forward = _distances(graph, 0)
    _, backward = _distances(graph, 1)
    if not np.isfinite(distance[1]):
        return []
    classes = {}
    candidates = [_branch(forward, 1)]
    for i, j in zip(*np.nonzero(np.isfinite(graph))):
        candidates.append(_branch(forward, i) + _branch(backward, j)[::-1])
    for nodes in candidates:
        # A forced edge can produce a backtracking loop. It is not another
        # useful passage mode and must not displace a simple geometric route.
        if nodes[0] != 0 or nodes[-1] != 1 or len(set(nodes)) != len(nodes):
            continue
        path = points[nodes]
        signature = winding_signature(path, rectangles)
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        if signature not in classes or length < classes[signature][0]:
            classes[signature] = (length, path)
    return [path.copy() for _, path in sorted(classes.values(), key=lambda item: item[0])[:count]]


def _polyline_free(points, rectangles, bounds, margin):
    """Vectorized exact line/AABB intersection, including boundary contact."""
    if np.any(points < bounds[0] + margin) or np.any(points > bounds[1] - margin):
        return False
    if not len(rectangles):
        return True
    if len(points) == 1:
        return segment_free(points[0], points[0], rectangles, bounds, margin)
    a, delta = points[:-1, None], np.diff(points, axis=0)[:, None]
    lo, hi = rectangles[None, :, :2] - margin, rectangles[None, :, 2:] + margin
    moving = np.abs(delta) > 1e-12
    denominator = np.where(moving, delta, 1.0)
    first, last = (lo - a) / denominator, (hi - a) / denominator
    near, far = np.minimum(first, last), np.maximum(first, last)
    outside = (a < lo) | (a > hi)
    near = np.where(moving, near, np.where(outside, np.inf, -np.inf))
    far = np.where(moving, far, np.where(outside, -np.inf, np.inf))
    return not np.any(np.maximum(near.max(axis=-1), 0.0) <= np.minimum(far.min(axis=-1), 1.0))


def _rounded_path(path, rectangles, bounds, clearance, spacing=0.035):
    """Quadratic corner blends with tangent continuity, checked against geometry."""
    path = np.asarray(path, float)
    if len(path) <= 1:
        return path.copy()
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    path = path[np.r_[True, lengths > 1e-5]]
    if len(path) <= 1:
        return path
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    entries, exits, curves = {}, {}, {}
    for i in range(1, len(path) - 1):
        before = (path[i] - path[i - 1]) / lengths[i - 1]
        after = (path[i + 1] - path[i]) / lengths[i]
        distance = min(0.44 * lengths[i - 1], 0.44 * lengths[i], 2.2)
        if np.dot(before, after) < -0.98:
            distance = 0.0
        for _ in range(18):
            entry, exit = path[i] - distance * before, path[i] + distance * after
            u = np.linspace(0.0, 1.0, max(9, int(2 * distance / spacing) + 1))[:, None]
            curve = (1 - u) ** 2 * entry + 2 * (1 - u) * u * path[i] + u**2 * exit
            # Extra numerical buffer covers finite-dt integration error.
            if _polyline_free(curve, rectangles, bounds, clearance + 0.025):
                break
            distance *= 0.65
        entries[i], exits[i], curves[i] = entry, exit, curve
    pieces = [path[:1]]
    last = path[0]
    for i in range(1, len(path)):
        end = entries.get(i, path[i])
        count = max(2, int(np.ceil(np.linalg.norm(end - last) / spacing)))
        pieces.append(np.linspace(last, end, count + 1)[1:])
        if i in curves:
            pieces.append(curves[i][1:])
            last = exits[i]
        else:
            last = path[i]
    points = np.vstack(pieces)
    return points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]]


def _integrate(acceleration, state, dt):
    velocity = np.vstack((state[2:], state[2:] + dt * np.cumsum(acceleration, axis=0)))
    position = np.vstack(
        (state[:2], state[:2] + 0.5 * dt * np.cumsum(velocity[:-1] + velocity[1:], axis=0))
    )
    return np.c_[position, velocity]


def _safe_reference(reference, acceleration, rectangles, bounds, clearance, cfg):
    # Curved constant-acceleration intervals need a midpoint check in addition
    # to their endpoint chord. The extra spatial buffer is retained as well.
    mid = reference[:-1, :2] + 0.5 * cfg.dt * reference[:-1, 2:] + 0.125 * cfg.dt**2 * acceleration
    positions = np.empty((2 * len(acceleration) + 1, 2))
    positions[::2], positions[1::2] = reference[:, :2], mid
    return _polyline_free(positions, rectangles, bounds, clearance)


def _braking_prefix(state, cfg, rectangles, bounds, clearance):
    if np.linalg.norm(state[2:]) < 1e-8:
        return np.empty((0, 2)), state.copy()
    count = max(1, int(np.ceil(np.max(np.abs(state[2:])) / (0.75 * cfg.umax * cfg.dt))))
    acceleration = np.tile(-state[2:] / (count * cfg.dt), (count, 1))
    reference = _integrate(acceleration, state, cfg.dt)
    if not _safe_reference(reference, acceleration, rectangles, bounds, clearance, cfg):
        return None, None
    return acceleration, reference[-1]


def _physical_margin(clearance):
    # Planning clearance is a preferred buffer, not the aircraft radius. A
    # reached state may legitimately be inside that buffer; the recovery may
    # never shrink the 0.275 m physical radius.
    return min(clearance, AIRCRAFT_RADIUS + 0.005)


def _transfer_prefix(state, target, cfg):
    """Exact rest-to-rest discrete double-integrator transfer along one line."""
    delta = target - state[:2]
    count = max(
        1,
        int(np.ceil(np.sqrt(np.max(np.abs(delta)) / (0.7 * cfg.umax * cfg.dt**2)))),
        int(np.ceil(np.linalg.norm(delta) / (min(1.8, 0.65 * cfg.speed_max) * cfg.dt))),
    )
    acceleration = delta / (count * cfg.dt) ** 2
    return np.vstack((np.tile(acceleration, (count, 1)), np.tile(-acceleration, (count, 1))))


def _initial_prefix(state, cfg, rectangles, bounds, clearance):
    """Brake physically safely, then leave a preferred-clearance violation.

    The recovery target is the nearest visible point on an expanded obstacle
    or boundary face. Only known geometry enters it. Checking the transition
    at physical clearance and the following route at preferred clearance is
    essential: requiring the initial state to already satisfy the latter
    discards every geometric mode near an obstacle or boundary.
    """
    physical = _physical_margin(clearance)
    prefix, stopped = _braking_prefix(state, cfg, rectangles, bounds, physical)
    details = {
        "physical_clearance": physical,
        "requested_clearance": clearance,
        "braking_time": 0.0,
        "recovery_time": 0.0,
        "recovery_distance": 0.0,
        "recovery_target": None,
        "recovery_required": False,
        "initial_speed_excess": max(0.0, float(np.linalg.norm(state[2:])) - cfg.speed_max),
    }
    if prefix is None:
        details["unavailable_reason"] = "No physically safe nominal braking prefix"
        return None, None, details
    details["braking_time"] = len(prefix) * cfg.dt
    if segment_free(stopped[:2], stopped[:2], rectangles, bounds, clearance):
        return prefix, stopped, details
    details["recovery_required"] = True
    for target_margin in (
        max(clearance, float(getattr(cfg, "rrt_margin", 0.65))) + 0.025,
        clearance + 0.025,
    ):
        xs = [stopped[0], bounds[0, 0] + target_margin, bounds[1, 0] - target_margin]
        ys = [stopped[1], bounds[0, 1] + target_margin, bounds[1, 1] - target_margin]
        for rectangle in rectangles:
            xs.extend((rectangle[0] - target_margin, rectangle[2] + target_margin))
            ys.extend((rectangle[1] - target_margin, rectangle[3] + target_margin))
        candidates = np.array([(x, y) for x in xs for y in ys])
        order = np.argsort(np.linalg.norm(candidates - stopped[:2], axis=1), kind="stable")
        for target in candidates[order]:
            if not segment_free(target, target, rectangles, bounds, clearance):
                continue
            if not segment_free(stopped[:2], target, rectangles, bounds, physical):
                continue
            extra = _transfer_prefix(stopped, target, cfg)
            reference = _integrate(extra, stopped, cfg.dt)
            if not _safe_reference(reference, extra, rectangles, bounds, physical, cfg):
                continue
            details.update(
                recovery_time=len(extra) * cfg.dt,
                recovery_distance=float(np.linalg.norm(target - stopped[:2])),
                recovery_target=target.tolist(),
            )
            return np.vstack((prefix, extra)), reference[-1], details
    details["unavailable_reason"] = "No physically safe local transition to planning clearance"
    return None, None, details


def parameterize(path, state, cfg, cruise, rectangles, bounds, clearance=None, prefix=None):
    """Return ``(full_theta, diagnostics)`` for one rounded geometric route.

    Motion uses curvature speed limits and a forward/backward acceleration
    pass, keeping nonzero speed at ordinary corners. ``initialization_effort``
    optionally sets both acceleration fractions; None preserves the original
    .50 tangential/.58 curvature fractions. All final physical checks remain.
    Sampling velocities at dt
    and differencing gives exactly the same constant-acceleration integration
    as ``continuous.nominal_reference``. Every integrated segment is checked.
    Unsafe references raise ``ValueError`` instead of silently supplying an
    obstacle-intersecting demonstration. Long routes may extend past horizon;
    diagnostics explicitly report that the goal has not yet been reached.
    """
    rectangles, bounds = _geometry(rectangles, bounds)
    state = np.asarray(state, float)
    clearance = max(
        AIRCRAFT_RADIUS,
        float(
            getattr(cfg, "planning_clearance", AIRCRAFT_RADIUS) if clearance is None else clearance
        ),
    )
    if prefix is None:
        prefix, stopped, _ = _initial_prefix(state, cfg, rectangles, bounds, clearance)
        if prefix is None:
            raise ValueError("No physically safe transition to geometric initialization")
        path = np.asarray(path, float).copy()
        path[0] = stopped[:2]
    else:
        stopped = _integrate(prefix, state, cfg.dt)[-1] if len(prefix) else state.copy()
    curve = _rounded_path(path, rectangles, bounds, clearance)
    if len(curve) == 1:
        acceleration = np.zeros((cfg.steps, 2))
        used = min(len(prefix), cfg.steps)
        acceleration[:used] = prefix[:used]
        return np.arctanh(np.clip(acceleration / cfg.umax, -0.999, 0.999)).astype(
            np.float32
        ).ravel(), {
            "arrival_time": len(prefix) * cfg.dt,
            "reached_goal": len(prefix) <= cfg.steps,
            "path_length": 0.0,
            "reference_clearance": clearance,
            "corner_stops": 0,
        }
    ds = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    arclength = np.r_[0.0, np.cumsum(ds)]
    tangent = np.gradient(curve, arclength, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    cusp = norm < 1e-8
    # An exact reversal between successive visits is a physical stop, rather
    # than an undefined tangent. Ordinary rounded corners remain moving.
    outgoing = np.vstack((np.diff(curve, axis=0), curve[-1]-curve[-2]))
    outgoing /= np.maximum(np.linalg.norm(outgoing, axis=1)[:, None], 1e-12)
    tangent = np.where(cusp[:, None], outgoing, tangent/np.maximum(norm[:, None], 1e-12))
    curvature = np.linalg.norm(np.gradient(tangent, arclength, axis=0), axis=1)
    effort = getattr(cfg, "initialization_effort", None)
    tangential_fraction = 0.50 if effort is None else float(effort)
    corner_fraction = 0.58 if effort is None else float(effort)
    speed = np.minimum(
        min(cruise, 0.93 * cfg.speed_max),
        np.sqrt(corner_fraction * cfg.umax / np.maximum(curvature, 1e-9)),
    )
    tangential = tangential_fraction * cfg.umax
    speed[cusp] = 0.0
    speed[0], speed[-1] = 0.0, 0.0
    for i in range(1, len(speed)):
        speed[i] = min(speed[i], np.sqrt(speed[i - 1] ** 2 + 2 * tangential * ds[i - 1]))
    for i in range(len(speed) - 2, -1, -1):
        speed[i] = min(speed[i], np.sqrt(speed[i + 1] ** 2 + 2 * tangential * ds[i]))
    times = np.r_[0.0, np.cumsum(2 * ds / np.maximum(speed[:-1] + speed[1:], 1e-10))]
    prefix_time = len(prefix) * cfg.dt
    for slowdown in (1.0, 1.15, 1.4, 1.8, 2.5):
        duration = times[-1] * slowdown
        count = max(1, int(np.ceil(duration / cfg.dt)))
        query = np.arange(count + 1) * cfg.dt / slowdown
        s = np.interp(query, times, arclength)
        direction = np.c_[
            np.interp(s, arclength, tangent[:, 0]), np.interp(s, arclength, tangent[:, 1])
        ]
        direction /= np.maximum(np.linalg.norm(direction, axis=1)[:, None], 1e-10)
        velocity = direction * (np.interp(query, times, speed) / slowdown)[:, None]
        # One short symmetric velocity filter softens changes in curvature
        # acceleration at line/curve joins, while preserving movement through
        # corners. A following displacement correction closes the endpoint.
        padded = np.pad(velocity, ((1, 1), (0, 0)))
        velocity = 0.25 * (padded[:-2] + 2 * padded[1:-1] + padded[2:])
        velocity[[0, -1]] = 0.0
        # The trapezoidal displacement has a small quadrature residual. A
        # smooth zero-endpoint velocity correction closes it exactly, without
        # alternating inverse-position acceleration or corner stops.
        residual = (
            curve[-1] - stopped[:2] - 0.5 * cfg.dt * np.sum(velocity[:-1] + velocity[1:], axis=0)
        )
        weight = np.sin(np.linspace(0.0, np.pi, count + 1)) ** 2
        if weight.sum() > 1e-10:
            velocity += weight[:, None] * residual / (cfg.dt * weight.sum())
        motion = np.diff(velocity, axis=0) / cfg.dt
        full = np.vstack((prefix, motion))
        acceleration = np.zeros((cfg.steps, 2))
        used = min(len(full), cfg.steps)
        acceleration[:used] = full[:used]
        reference = _integrate(acceleration, state, cfg.dt)
        if (
            np.max(np.abs(acceleration)) <= 0.96 * cfg.umax
            # An already reached state can slightly exceed the preferred
            # speed. Its braking prefix must not accelerate farther; the
            # geometric route itself still obeys speed_max.
            and np.max(np.linalg.norm(reference[:, 2:], axis=1))
            <= max(cfg.speed_max, float(np.linalg.norm(state[2:]))) + 1e-7
            and (
                len(prefix) >= cfg.steps
                or np.max(np.linalg.norm(reference[len(prefix) :, 2:], axis=1))
                <= cfg.speed_max + 1e-7
            )
            and _safe_reference(
                reference, acceleration, rectangles, bounds, _physical_margin(clearance), cfg
            )
            and (
                len(prefix) >= cfg.steps
                or _safe_reference(
                    reference[len(prefix) :],
                    acceleration[len(prefix) :],
                    rectangles,
                    bounds,
                    clearance,
                    cfg,
                )
            )
        ):
            theta = (
                np.arctanh(np.clip(acceleration / cfg.umax, -0.999, 0.999))
                .astype(np.float32)
                .ravel()
            )
            return theta, {
                "arrival_time": prefix_time + count * cfg.dt,
                "reached_goal": len(full) <= cfg.steps,
                "path_length": float(arclength[-1]),
                "reference_clearance": clearance,
                "corner_stops": 0,
            }
    raise ValueError("No collision-free bounded-acceleration time parameterization found")


def _state_clearance(point, rectangles, bounds):
    obstacle = (
        np.min(np.maximum(rectangles[:, :2] - point, point - rectangles[:, 2:]).max(axis=1))
        if len(rectangles)
        else np.inf
    )
    return min(float(obstacle), float(np.min(point - bounds[0])), float(np.min(bounds[1] - point)))


def demonstrations(state, env, cfg, seed=0):
    """Drop-in geometry seed bank using existing ``rrt_paths/visits`` budgets.

    Returns a fixed ``(sites * rrt_paths * 2 * rrt_visits, 2*steps)`` float32
    array and JSON-compatible diagnostics. Missing classes are repeated with
    their same signature; truly unavailable seeds are explicitly marked.
    ``seed`` only samples subsequent visit orders, never hidden environment
    information. The cached graph and route choice are otherwise deterministic.
    """
    state = np.asarray(state, float)
    rectangles, bounds = _geometry(env.rectangles, env.bounds)
    goals = (np.asarray(env.regions[:, :2]) + np.asarray(env.regions[:, 2:])) / 2
    clearance = max(AIRCRAFT_RADIUS, float(getattr(cfg, "planning_clearance", AIRCRAFT_RADIUS)))
    prefix, stopped, recovery = _initial_prefix(state, cfg, rectangles, bounds, clearance)
    paths_per_goal, visits_count = (
        int(getattr(cfg, "rrt_paths", 3)),
        int(getattr(cfg, "rrt_visits", 1)),
    )
    capacity = len(goals) * paths_per_goal * 2 * visits_count
    parameters, records = [], []
    cache = {}
    if stopped is not None:
        available = _state_clearance(stopped[:2], rectangles, bounds)
        margin = max(clearance, min(float(getattr(cfg, "rrt_margin", 0.65)), available - 0.02))

        def connections(start, goal):
            key = tuple(np.r_[start, goal])
            if key not in cache:
                cache[key] = visibility_paths(
                    start, goal, rectangles, bounds, margin, paths_per_goal
                )
            return cache[key]

        for visits in range(1, visits_count + 1):
            for goal_index, goal in enumerate(goals):
                options = connections(stopped[:2], goal)
                for trial in range(paths_per_goal):
                    if not options:
                        continue
                    path = options[trial % len(options)].copy()
                    signature = winding_signature(path, rectangles)
                    order = [goal_index]
                    for extra_visit in range(1, visits):
                        other = [k for k in range(len(goals)) if k != order[-1]]
                        if not other:
                            break
                        other.sort(key=lambda k: (np.linalg.norm(goals[k]-path[-1]), k))
                        next_goal = int(other[(trial+extra_visit-1) % len(other)])
                        extra = connections(path[-1], goals[next_goal])
                        if not extra:
                            break
                        path = np.vstack((path, extra[(trial + extra_visit) % len(extra)][1:]))
                        order.append(next_goal)
                    for cruise in (min(1.8, 0.65 * cfg.speed_max), min(2.8, 0.9 * cfg.speed_max)):
                        try:
                            theta, diagnostics = parameterize(
                                path, state, cfg, cruise, rectangles, bounds, clearance, prefix
                            )
                        except ValueError:
                            continue
                        parameters.append(theta)
                        records.append(
                            dict(
                                goal_index=goal_index,
                                visit_order=order.copy(),
                                trial=trial,
                                cruise=float(cruise),
                                margin=float(margin),
                                path=path.tolist(),
                                signature=[int(value) for value in signature],
                                distinct_classes=len(options),
                                **diagnostics,
                            )
                        )
    found = len(parameters)
    while len(parameters) < capacity:
        parameters.append(np.zeros(2 * cfg.steps, np.float32))
        records.append({"unavailable": True})
    return np.asarray(parameters, np.float32).reshape((capacity, 2 * cfg.steps)), {
        "method": "visibility-rounded",
        "initialization_effort": (
            None
            if getattr(cfg, "initialization_effort", None) is None
            else float(cfg.initialization_effort)
        ),
        "seed": int(seed),
        "found": found,
        "recovery": recovery,
        "records": records,
    }


def select_initial_modes(parameters, record, state, env, cfg, q=None):
    """Select exactly ``cfg.mode_count`` seeds, with one braking/hold baseline.

    The remaining slots use the fastest representative of each distinct
    ``(goal_index, winding signature)``. Near one site, other sites take
    priority; between sites, each goal gets a representative before extra
    classes. Optional clarity ``q`` can rank goals by their known task deficit.
    If fewer distinct classes exist than the fixed budget, best representatives
    repeat explicitly. This does not add route constraints to later optimization.

    Returns ``(seeds, selection_record)``. ``chosen_indices`` indexes the input
    bank, with ``-1`` denoting the baseline. ``records`` describes selected modes
    in the same order and marks any repeats with ``duplicate=True``.
    """
    count = int(getattr(cfg, "mode_count", 4))
    if count < 1:
        raise ValueError("mode_count must be positive")
    parameters = np.asarray(parameters, np.float32)
    if parameters.ndim != 2 or parameters.shape[1] != 2 * cfg.steps:
        raise ValueError("Mode bank must contain full 2*steps acceleration vectors")
    if len(record["records"]) != len(parameters):
        raise ValueError("Mode diagnostics must match the seed bank")
    state = np.asarray(state, float)
    rectangles, bounds = _geometry(env.rectangles, env.bounds)
    clearance = max(AIRCRAFT_RADIUS, float(getattr(cfg, "planning_clearance", AIRCRAFT_RADIUS)))
    physical_margin = _physical_margin(clearance)
    prefix, _ = _braking_prefix(state, cfg, rectangles, bounds, physical_margin)
    baseline_acceleration = np.zeros((cfg.steps, 2))
    if prefix is not None:
        used = min(len(prefix), cfg.steps)
        baseline_acceleration[:used] = prefix[:used]
    baseline = (
        np.arctanh(np.clip(baseline_acceleration / cfg.umax, -0.999, 0.999))
        .astype(np.float32)
        .ravel()
    )
    baseline_safe = prefix is not None and _safe_reference(
        _integrate(baseline_acceleration, state, cfg.dt),
        baseline_acceleration,
        rectangles,
        bounds,
        physical_margin,
        cfg,
    )
    baseline_record = {
        "kind": "hold" if np.linalg.norm(state[2:]) < 1e-8 else "braking",
        "source_index": -1,
        "safe_reference": bool(baseline_safe),
        "reference_clearance": physical_margin,
        "preferred_clearance": clearance,
        "duplicate": False,
    }
    representatives = {}
    for index, item in enumerate(record["records"]):
        if item.get("unavailable") or "goal_index" not in item:
            continue
        # A local zero-length route is already represented by the baseline.
        if item.get("path_length", np.inf) < 1e-4:
            continue
        identity = (int(item["goal_index"]), tuple(item.get("signature", ())), tuple(item.get("visit_order", ())))
        rank = (float(item.get("arrival_time", np.inf)), len(item.get("visit_order", ())), index)
        if identity not in representatives or rank < representatives[identity][0]:
            representatives[identity] = (rank, index)
    groups = {}
    for (goal, _, _), (_, index) in representatives.items():
        groups.setdefault(goal, []).append(index)
    for indices in groups.values():
        indices.sort(
            key=lambda index: (record["records"][index].get("arrival_time", np.inf), index)
        )
        # Reserve a second alternative for a genuine multi-site reference when
        # present. Single and multi-site routes share the same overall budget.
        # All coordinates remain free during diffusion; no order is imposed.
        multi=[i for i in indices if len(record["records"][i].get("visit_order",()))>1]
        single=[i for i in indices if i not in multi]
        if single and multi:
            indices[:]=[single[0],multi[0],*single[1:],*multi[1:]]
    regions = np.asarray(env.regions, float)
    distance = np.linalg.norm(
        np.maximum(np.maximum(regions[:, :2] - state[:2], state[:2] - regions[:, 2:]), 0.0), axis=1
    )
    deficits = np.zeros(len(regions))
    if q is not None and hasattr(env, "cells") and hasattr(env, "targets"):
        cells, targets, quality = np.asarray(env.cells), np.asarray(env.targets), np.asarray(q)
        if quality.shape == targets.shape == (len(cells),):
            for goal, region in enumerate(regions):
                belongs = ((cells >= region[:2]) & (cells <= region[2:])).all(axis=1)
                if belongs.any():
                    deficits[goal] = np.maximum(targets[belongs] - quality[belongs], 0.0).mean()
    goal_order = sorted(groups, key=lambda goal: (-deficits[goal], distance[goal], goal))
    near = np.flatnonzero(distance <= max(0.3, 0.5 * float(getattr(cfg, "sensor_sigma", 2.2))))
    preferred = [goal for goal in goal_order if len(near) != 1 or goal != near[0]]
    deferred = [goal for goal in goal_order if goal not in preferred]
    unique = []
    for goals in (preferred, deferred):
        for level in range(max((len(groups[goal]) for goal in goals), default=0)):
            unique.extend(groups[goal][level] for goal in goals if level < len(groups[goal]))
    selected = unique[: max(0, count - 1)]
    repeats = [False] * len(selected)
    while len(selected) < count - 1:
        selected.append(unique[(len(selected) - len(unique)) % len(unique)] if unique else -1)
        repeats.append(True)
    seeds, chosen_records = [baseline], [baseline_record]
    for index, duplicate in zip(selected, repeats):
        if index < 0:
            seeds.append(baseline.copy())
            chosen_records.append({**baseline_record, "duplicate": duplicate})
        else:
            seeds.append(parameters[index])
            chosen_records.append(
                {
                    **record["records"][index],
                    "kind": "geometry",
                    "source_index": int(index),
                    "duplicate": duplicate,
                }
            )
    return np.asarray(seeds, np.float32), {
        "method": "geometry-mode-selection",
        "mode_count": count,
        "chosen_indices": [-1, *selected],
        "distinct_geometry_modes": len(unique),
        "goal_priority": goal_order,
        "prioritized_away_from_site": int(near[0]) if len(near) == 1 else None,
        "records": chosen_records,
    }
