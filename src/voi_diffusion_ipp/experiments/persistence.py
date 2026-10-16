"""Persistent-coverage diagnostics from executed QoI and position histories."""

import numpy as np


def summarize(positions, clarity, cells, targets, process_noise, sigma, dt):
    x, q, cells, targets = map(np.asarray, (positions, clarity, cells, targets))
    visits = np.zeros(len(cells), int)
    recovered = np.zeros(len(cells), int)
    last_near = np.full(len(cells), -np.inf)
    run = np.zeros(len(cells), int)
    eligible = np.zeros(len(cells), bool)
    last_pass = np.full(len(cells), np.nan)
    met = q[0] >= targets
    loss_start = np.full(len(cells), np.nan)
    intervals = []
    for i, p in enumerate(x):
        near = np.exp(-np.sum((cells - p[:2]) ** 2, axis=1) / sigma**2) >= 0.25
        onset = near & (run == 0)
        eligible[onset] = (visits[onset] == 0) | ((i - last_near[onset]) * dt >= 3)
        run = np.where(near, run + 1, 0)
        new = (run == 2) & eligible
        repeated = new & np.isfinite(last_pass)
        intervals.extend(((i - 1) * dt - last_pass[repeated]).tolist())
        visits += new
        last_pass[new] = (i - 1) * dt
        last_near[near] = i
        below = q[i] < targets
        loss_start = np.where(below & met & np.isnan(loss_start), i, loss_start)
        recovered += (~below) & met & np.isfinite(loss_start) & ((i - loss_start) * dt >= 1)
        met |= ~below
        loss_start = np.where(~below, np.nan, loss_start)
    deficit = np.maximum(targets - q, 0).mean(1)
    times = np.arange(len(q)) * dt
    late = times >= 30
    unattended = q[0] / (1 + np.asarray(process_noise)[None] * q[0][None] * times[:, None])
    no_sensing = np.maximum(targets - unattended, 0).mean(1)
    return {
        "sensing_threshold_squared": 0.25,
        "minimum_pass_samples": 2,
        "minimum_pass_gap_seconds": 3.0,
        "minimum_loss_seconds": 1.0,
        "visited_cells": int(np.sum(visits > 0)),
        "revisited_cells": int(np.sum(visits > 1)),
        "recovered_cells": int(np.sum(recovered > 0)),
        "sensing_pass_counts": visits.tolist(),
        "clarity_recovery_counts": recovered.tolist(),
        "median_revisit_seconds": float(np.median(intervals)) if intervals else None,
        "late_mean_deficit_after_30s": float(
            np.trapezoid(deficit[late], times[late]) / (times[late][-1] - times[late][0])
        )
        if late.sum() > 1
        else None,
        "no_sensing_integrated_deficit": float(np.trapezoid(no_sensing, times)),
        "mean_fraction_at_target": float(np.mean(q >= targets)),
        "process_noise_min": float(np.min(process_noise)),
        "process_noise_max": float(np.max(process_noise)),
        "duration_seconds": float(times[-1]),
    }
