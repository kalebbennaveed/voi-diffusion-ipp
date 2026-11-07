"""Physics and export contracts for the stationary-wind robot diagnostic."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
import pytest

PATH=Path(__file__).resolve().parents[1]/'scripts/airflow/robot_review.py'
spec=importlib.util.spec_from_file_location('robot_review',PATH)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def test_wind_mapping_units_and_relative_velocity():
    np.testing.assert_allclose(r.acceleration(np.zeros(2),np.array([-4.,0.])),[-16*.1/.680,0.])
    np.testing.assert_allclose(r.acceleration(np.array([-4.,0.]),np.array([-4.,0.])),0.)
    np.testing.assert_allclose(r.acceleration(np.array([1.,0.]),np.zeros(2)),[-.1/.680,0.])
    field=r.FixedWind(jnp.tile(jnp.array([2.,-3.]),(4,4,1)),.125)
    x=jnp.array([.2,.2,2.,-3.]);u=jnp.zeros(2)
    np.testing.assert_allclose(r.derivative(x,u,field),[2.,-3.,0.,0.])
    np.testing.assert_allclose(r.microstep(x,u,field,.01),x+.01*jnp.array([2.,-3.,0.,0.]),atol=1e-12)

def test_field_is_exact_display_bilinear_interpolation():
    rng=np.random.default_rng(47)
    for scenario in (1,2):
        field=r.load_wind(scenario);g=np.array(field.grid);ny,nx,_=g.shape
        p=rng.uniform([0,0],[8,4.625],size=(100,2))
        expected=[]
        for point in p:
            x,y=np.clip(point/field.dx-.5,[0,0],[nx-1,ny-1])
            i,j=min(nx-2,int(np.floor(x))),min(ny-2,int(np.floor(y)))
            fx,fy=x-i,y-j
            expected.append((1-fy)*((1-fx)*g[j,i]+fx*g[j,i+1])+fy*((1-fx)*g[j+1,i]+fx*g[j+1,i+1]))
        np.testing.assert_allclose(field.at(p),expected,atol=1e-12)
        np.testing.assert_array_equal(field.at(p),field.at(p))

@pytest.mark.parametrize('route',['central','lower','upper'])
def test_references_are_c2_feasible_and_reversible(route):
    ab,ba=r.Reference(route,'A-B'),r.Reference(route,'B-A')
    assert ab.duration==ba.duration
    t=np.linspace(.1,ab.duration-.1,101);h=1e-6
    p,v,a=ab.at(t)
    np.testing.assert_allclose((ab.at(t+h)[0]-ab.at(t-h)[0])/(2*h),v,atol=2e-7)
    # Jerk jumps at C2 spline knots give O(h) finite-difference error there.
    np.testing.assert_allclose((ab.at(t+h)[1]-ab.at(t-h)[1])/(2*h),a,atol=1e-6)
    np.testing.assert_allclose(ba.at(ab.duration-t)[0],p,atol=1e-10)
    for endpoint in (0,ab.duration):
        np.testing.assert_allclose(ab.at(endpoint)[1],0,atol=1e-12)
        np.testing.assert_allclose(ab.at(endpoint)[2],0,atol=1e-12)
    assert ab.checks['min_clearance_m']>=.06

def test_swept_contact_finds_thin_crossing():
    c=r.OBSTACLES[2,:2];radius=r.OBSTACLES[2,2]
    a=c+np.array([-radius-.01,0]);b=c+np.array([radius+.01,0])
    t=r.first_contact(a,b)
    np.testing.assert_allclose(a+t*(b-a),c+[-radius,0],atol=1e-12)

def test_actual_passage_classification_distinguishes_detours():
    for route in ('central','lower','upper'):
        for direction in ('A-B','B-A'):
            ref=r.Reference(route,direction);t=np.linspace(0,ref.duration,1001)
            p=ref.at(t)[0];events=r.passage_crossings(np.c_[t,p])
            assert len(events)==1
            assert events[0]['route']==route
            assert events[0]['direction']==direction

def test_full_filter_gradient_and_bounds():
    field=r.load_wind(1);belief=r.KnownWind(r.KnownBasis(field));inner=belief.inner_fields(jnp.zeros((3,1,2)))
    x=jnp.array([1.45,.875,.1,.05]);nom=jnp.array([1.3,.5])
    u,info=r.filtered_control(x,nom,field)
    assert np.max(np.abs(u))<=1.60001
    assert np.all(np.isfinite(info.gradient))
    from voi_diffusion_ipp.safety.flow_policy_value import values
    h=2e-5;eye=np.eye(4)
    fd=np.stack([(np.array(values(x+h*e,belief,inner,r.ENV,r.CFG))-np.array(values(x-h*e,belief,inner,r.ENV,r.CFG)))/(2*h) for e in eye],axis=1)
    np.testing.assert_allclose(info.gradient,fd,atol=5e-3,rtol=5e-3)

def test_export_all_cases_and_truthful_status():
    folder=r.SITE/'data/robot';manifest=json.loads((folder/'manifest.json').read_text())
    assert len(manifest['cases'])==12
    assert len({(c['scenario'],c['route'],c['direction']) for c in manifest['cases']})==12
    for c in manifest['cases']:
        data=json.loads((folder/(c['id']+'.json')).read_text())
        assert data['reference']['checks']['min_clearance_m']>=.06
        for method in ('unfiltered','rpcbf','still_air'):
            out=data[method];samples=np.array(out['samples']);summary=out['summary']
            assert np.all(np.isfinite(samples))
            assert np.all(np.diff(samples[:,0])>0)
            assert np.max(np.abs(samples[:,9:11]))<=1.60001
            assert summary['status']=='collision' if summary['contact'] else summary['status']!='collision'
            assert np.min(samples[:,12])>=-1e-6
            if method=='rpcbf':
                feasible=samples[:,14]>0
                assert np.max(samples[feasible,16],initial=0)<=1e-4
                assert summary['infeasible_steps']>=np.sum(samples[:,14]==0)-1
        np.testing.assert_allclose(data['unfiltered']['samples'][0][1:5],data['rpcbf']['samples'][0][1:5])
    assert all(c['same_outcome'] for c in manifest['timestep_refinement'])


def test_redesign_shelters_monitoring_regions_without_erasing_route_wind():
    validation=json.loads((r.SITE/'data/validation.json').read_text())
    assert validation['environment_revision']=='directional-sheltered-ports-v11'
    for row in validation['scenarios']:
        for region in row['monitoring_regions'].values():
            assert region['mean_m_s'] < .5
            assert region['p95_m_s'] < 1.
        meta=json.loads((r.SITE/f"data/scenario{row['scenario']}/metadata.json").read_text())
        assert meta['openings']['central_inlet']['side']=='north'
        assert meta['openings']['central_inlet']['z_m'][0] > .75
        assert meta['openings']['outlet']['x_m'][0] > 2.9
        assert meta['openings']['outlet']['side']=='floor'
        assert meta['openings']['outlet']['z_m']==0
    by_id={s['scenario']:s for s in validation['scenarios']}
    assert by_id[1]['fixed_field_mean_speeds_m_s']['central_passage'] > 1.4
    assert by_id[1]['fixed_field_mean_speeds_m_s']['lower_bypass'] < .5
    assert by_id[2]['fixed_field_mean_speeds_m_s']['central_passage'] > 1.8
    assert by_id[2]['fixed_field_mean_speeds_m_s']['lower_bypass'] > 3.


def test_central_and_scenario_two_lower_wind_oppose_return():
    manifest=json.loads((r.SITE/'data/robot/manifest.json').read_text())
    cases={c['id']:c for c in manifest['cases']}
    for scenario,route in ((1,'central'),(2,'central'),(2,'lower')):
        ab=cases[f's{scenario}-{route}-A-B']['directional_authority']
        ba=cases[f's{scenario}-{route}-B-A']['directional_authority']
        assert ab['air_work_per_mass_m2_s2']>0
        assert ba['air_work_per_mass_m2_s2']<0
        assert ba['max_opposing_air_acceleration_m_s2']>4
        assert ba['max_required_control_component_m_s2']>2*r.CFG.umax
        assert ba['reference_time_over_actuator_limit_s']>.5
    assert cases['s2-lower-B-A']['directional_authority']['max_opposing_air_acceleration_m_s2']>8


def test_central_return_collides_without_filter_and_slows_with_filter():
    manifest=json.loads((r.SITE/'data/robot/manifest.json').read_text())
    for case in manifest['cases']:
        if case['route']!='central' or case['direction']!='B-A':continue
        assert case['unfiltered']['contact'] is not None
        safe=case['rpcbf'];assert safe['contact'] is None
        assert safe['min_clearance_m']>=.06
        assert safe['infeasible_steps']==0
        crossing=[e for e in safe['passage_crossings'] if e['direction']=='B-A']
        assert crossing and all(e['route']=='central' for e in crossing)
        baseline=case['still_air']['passage_crossings'][0]
        assert crossing[0]['time_s']>baseline['time_s']+1.5
        # A delayed crossing alone was too weak. Require substantial sustained
        # slowing over the same fixed 1.2 m throat, and successful traversal.
        transit=safe['central_transit'];still=case['still_air']['central_transit']
        assert safe['target_region_reached']
        assert transit['duration_s'] is not None
        assert transit['mean_forward_speed_m_s'] <= .20
        assert transit['duration_s'] >= 4*still['duration_s']


def test_transit_measurement_rejects_a_detour_and_unfinished_crossing():
    for route in ('central','upper','lower'):
        ref=r.Reference(route,'B-A');t=np.linspace(0,ref.duration,2001)
        p,v,_=ref.at(t);rows=np.c_[t,p,v]
        transit=r.central_transit(rows,'B-A')
        if route=='central':
            assert transit['duration_s']>0
            np.testing.assert_allclose(transit['duration_s']*transit['mean_forward_speed_m_s'],1.2)
            truncated=rows[rows[:,0]<r.passage_crossings(rows)[0]['time_s']]
            assert r.central_transit(truncated,'B-A')['duration_s'] is None
        else:assert transit is None


def test_scenario_one_lower_remains_easy_in_both_directions():
    manifest=json.loads((r.SITE/'data/robot/manifest.json').read_text())
    for case in manifest['cases']:
        if case['route']!='lower' or case['scenario']!=1:continue
        plain=case['unfiltered'];assert plain['target_region_reached']
        assert plain['min_clearance_m']>=.06
        assert plain['max_tracking_error_m']<.3
        assert plain['max_control_component_m_s2']<1.
        assert all(e['route']=='lower' for e in plain['passage_crossings'])


def test_both_upper_bypasses_remain_smooth_and_clear():
    manifest=json.loads((r.SITE/'data/robot/manifest.json').read_text())
    cases=[c for c in manifest['cases'] if c['route']=='upper']
    assert len(cases)==4
    for case in cases:
        for method in ('unfiltered','rpcbf'):
            result=case[method]
            assert result['target_region_reached'] and result['contact'] is None
            assert result['min_clearance_m']>.06
            assert result['endpoint_error_m']<.02
            if method=='unfiltered':assert result['max_tracking_error_m']<.3
        # Smooth nominal passage uses less than 1 m/s^2 per axis, well within
        # the unchanged 1.6 limit; no artificially enlarged actuator is used.
        assert case['unfiltered']['max_control_component_m_s2']<1.
    refinement=[c for c in manifest['timestep_refinement'] if c['route']=='upper']
    assert len(refinement)==8
    assert all(c['same_outcome'] for c in refinement)
    assert max(c['max_common_time_position_difference_m'] for c in refinement)<.005
