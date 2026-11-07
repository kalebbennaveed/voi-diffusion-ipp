"""Lossless compressed inference checkpoints; no model or array conversions."""
from pathlib import Path
import gzip,pickle,os
import jax


def load_profile(path):
    path=Path(path)
    if not path.exists() and path.suffix!='.gz':path=path.with_suffix(path.suffix+'.gz')
    with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as stream:
        # JAX array unpickling otherwise honors the caller's current x64 flag
        # and can silently truncate a saved float64 posterior to float32.
        with jax.experimental.enable_x64():return pickle.load(stream)


def save_profile(path,profile):
    path=Path(path)
    if path.suffix!='.gz':path=path.with_suffix(path.suffix+'.gz')
    temporary=path.with_name(path.name+'.tmp')
    with gzip.open(temporary,'wb',compresslevel=1) as stream:pickle.dump(profile,stream)
    os.replace(temporary,path)
    return path
