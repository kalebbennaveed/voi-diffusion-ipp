"""Recording optimization must preserve data and failed-write recovery."""
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/airflow'))
import recording_io


def fixture():
    return dict(id='trial', samples=[[.05, np.float64(1.23), 0, True]],
                frames=[dict(t=.2, value=np.array([.3, np.nan, np.inf])),
                        dict(t=.4, value=[None, 'wind → measurement', -np.inf])],
                plan=dict(parameters=np.array([[1, 2], [3, 4]])),
                completed=False, empty=[])


def test_stream_matches_previous_json_bytes():
    record=fixture()
    expected=json.dumps(recording_io._clean(record),separators=(',',':'),allow_nan=False,
                        default=lambda v:np.asarray(v).tolist()).encode()
    for size in (1, 32, 262144):
        assert b''.join(recording_io.recording_chunks(record,size))==expected
    assert np.isnan(record['frames'][0]['value'][1])  # Source data not mutated.


def test_roundtrip_retains_all_values(tmp_path):
    path=tmp_path/'record.json.gz'
    recording_io.write_recording(path,fixture())
    assert json.loads(gzip.decompress(path.read_bytes()))==json.loads(recording_io._encode(fixture()))
    assert not list(tmp_path.glob('*.tmp'))


def test_compact_rounded_map_frames_encode_identically_to_boxed_lists():
    rng=np.random.default_rng(17)
    arrays=dict(t=180.,mean=np.round(rng.normal(size=(3072,2)),5),
        std=np.round(rng.random(3072),5),q=np.round(rng.random(3072),6),
        weights=np.round(rng.random(3072),8))
    boxed={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in arrays.items()}
    assert b''.join(recording_io.recording_chunks({'frames':[arrays]}))==b''.join(recording_io.recording_chunks({'frames':[boxed]}))


def test_disk_failure_preserves_previous_checkpoint(tmp_path,monkeypatch):
    path=tmp_path/'record.json.gz'
    recording_io.write_recording(path,fixture());before=path.read_bytes()
    original=recording_io.recording_chunks
    def interrupted(record):
        yield next(original(record,1))
        raise OSError(28,'No space left on device')
    monkeypatch.setattr(recording_io,'recording_chunks',interrupted)
    with pytest.raises(OSError,match='No space'):
        recording_io.write_recording(path,dict(samples=[[9,9]]))
    assert path.read_bytes()==before
    assert not list(tmp_path.glob('*.tmp'))
