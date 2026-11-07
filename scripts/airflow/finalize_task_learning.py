"""Audit the final three matched cohorts and write an immutable input inventory.

This reports recorded outcomes, including failures. It never edits a recording,
changes a gate, extrapolates a partial run, or substitutes a successful trial.
"""
from pathlib import Path
import hashlib
import json
from audit_full_learning import audit
from evaluate_task_learning import evaluate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/task_learning_20261105'
COHORTS = {
    'final_baseline': 'Baseline 1: learn everywhere',
    'final_targeted': 'Reference-weighted relearning + MBD',
    'final_clarity': 'Clarity-weighted relearning + MBD',
}


def main():
    results = {}
    inventory = []
    protocols = []
    for cohort, label in COHORTS.items():
        folder = OUT / cohort
        if len(list(folder.glob('s*-center-*.json.gz'))) != 2:
            raise RuntimeError(f'{cohort}: both scenario recordings are required')
        checks = audit(folder)
        comparison = evaluate(folder)
        (folder / 'numerical-audit.json').write_text(json.dumps(checks, indent=2, allow_nan=False))
        (folder / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False))
        results[cohort] = dict(label=label, runs=comparison)
        protocols.extend(json.loads(p.read_text()) for p in sorted(folder.glob('*-protocol.json')))
        for path in sorted(folder.iterdir()):
            if path.suffix in ('.json', '.gz', '.npz', '.zip'):
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                inventory.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                    sha256=digest))
    shared_settings = ('settings', 'physical_settings', 'gp_profile', 'sensor_period_s',
                       'control_period_s', 'integration_period_s', 'noise_std_m_s', 'safety')
    for key in shared_settings:
        assert all(p[key] == protocols[0][key] for p in protocols), key
    physical_modules = {
        'full_learning_monitoring.py', 'efficient_learning_monitoring.py', 'spatial_poam.py',
        'cached_poam_fields.py', 'supported_poam.py', 'reachable_learning.py',
        'uncertainty_waypoints.py', 'guarded_waypoints.py', 'monitoring_routes.py',
        'wind_model_trust.py', 'task_wind_acquisition.py',
    }
    shared_sources = [name for name in protocols[0]['source_sha256']
                      if name.startswith('src/') or Path(name).name in physical_modules]
    for name in shared_sources:
        assert all(p['source_sha256'][name] == protocols[0]['source_sha256'][name]
                   for p in protocols), name
    contrasts = []
    for scenario in (1, 2):
        base = next(r for r in results['final_baseline']['runs'] if r['scenario'] == scenario)
        for cohort in ('final_targeted', 'final_clarity'):
            candidate = next(r for r in results[cohort]['runs'] if r['scenario'] == scenario)
            comparable = base['complete_300'] and candidate['complete_300']
            base_cost = base['full_mission_monitoring_cost']
            cost = candidate['full_mission_monitoring_cost']
            contrasts.append(dict(scenario=scenario, cohort=cohort, complete_pair=comparable,
                monitoring_cost_reduction_percent=100*(base_cost-cost)/base_cost if comparable else None,
                contact_free=candidate['contact'] is None,
                visits_A_and_returns_B=candidate['first_A_s'] is not None and candidate['return_B_s'] is not None,
                executes_mbd=candidate['accepted_mbd_execution_seconds'] > 0,
                all_warm_decision_cycles_within_one_second=candidate['warm_actual_decision_deadline_misses'] == 0))
    payload = dict(cohorts=results, contrasts=contrasts, horizon_seconds=15,
        mission_seconds=300, initial_condition='center',
        notes=[
            'One recorded initial condition and noise seed per scenario; this is not a multi-seed robustness claim.',
            'Baseline 1 switches only at whole-map true wind RMSE <= 0.20 m/s. No switch means no MBD execution.',
            'Targeted variants use finite-route posterior precision <= 0.20 m/s, followed by per-trajectory model checks.',
            'Acquisition weights and handoff criteria both differ; this is not an acquisition-only ablation.',
            'Wind truth is used only for plant, sensors and reported diagnostics; only Baseline 1 has the authorized oracle RMSE gate.',
            'No recorded contact does not establish robust safety. Infeasible commands and deadline misses remain failures.',
            'Timing separates warm trajectory requests from POAM updates, startup compilation and whole-mission wall time.',
        ])
    (OUT / 'final-results.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
    (OUT / 'final-run-manifest.json').write_text(json.dumps(dict(
        recorded_inputs=inventory, numerical_audits_passed=True,
        shared_model_settings_verified=list(shared_settings),
        identical_model_source_files=shared_sources,
        all_six_recordings_complete=all(r['complete_300'] for v in results.values() for r in v['runs']),
        all_six_recordings_contact_free=all(r['contact'] is None for v in results.values() for r in v['runs']),
    ), indent=2, allow_nan=False))
    print(json.dumps(dict(contrasts=contrasts, files=len(inventory)), indent=2))


if __name__ == '__main__':
    main()
