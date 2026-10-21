"""A prolonged mission must reproduce uninterrupted planning and execution."""
import importlib.util
from pathlib import Path
import numpy as np
from voi_diffusion_ipp.quarter.continuous import ControlSettings

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('quarter_runner',ROOT/'scripts/run_full_control_quarter.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_continuation_preserves_physics_belief_and_random_streams(tmp_path):
    cfg=ControlSettings(horizon=1.,commit=.2,iterations=2,proposals=4,
                        worlds=2,validation_worlds=2,batch_size=4,
                        environment_scale=.35,inspection_scale=.55,
                        initialization='rrt',rrt_paths=1,common_construction=True,
                        cosine_end=.05,planning_clearance=.6,slack_weight=50.)
    full_dir=tmp_path/'full';prefix_dir=tmp_path/'prefix';long_dir=tmp_path/'continued'
    for directory in (full_dir,prefix_dir,long_dir):directory.mkdir()
    full=runner.run('two','forward','proposed',.6,8101,cfg,full_dir)
    prefix=runner.run('two','forward','proposed',.4,8101,cfg,prefix_dir)
    continued=runner.run('two','forward','proposed',.6,8101,cfg,long_dir,
                         resume=prefix_dir/'two_forward_proposed_8101.json')
    assert full['success'] and prefix['success'] and continued['success']
    assert full['frames']==continued['frames']
    assert full['observations']==continued['observations']
    assert full['beliefs']==continued['beliefs']
    assert full['cost']==continued['cost']
    assert full['min_clearance']==continued['min_clearance']
    assert full['rms_tracking_error']==continued['rms_tracking_error']
    for direct,resumed in zip(full['plans'],continued['plans']):
        for key in ('policy','trace','points','predicted_optimistic','best_training_cost'):
            assert direct[key]==resumed[key],key
        with np.load(full_dir/direct['execution_audit']) as a,np.load(long_dir/resumed['execution_audit']) as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
