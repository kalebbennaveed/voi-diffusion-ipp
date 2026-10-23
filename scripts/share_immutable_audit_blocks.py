"""Lossless APFS block sharing for immutable duplicate-prefix NPZ archives.

Search and per-chain audit ZIPs contain the same large sampling arrays followed
by different small metadata members. Keep every file's SHA-256 unchanged and
use independent copy-on-write clones, so subsequent writes cannot alias files.
Only historical output directories are considered; active investigations are
excluded. The original path is replaced only after full byte verification.
"""
from pathlib import Path
import ctypes,os,hashlib,json,shutil,re

ROOT=Path('output/european_quarter');BLOCK=1024*1024
lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
clone=lib.clonefile;clone.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int]
record_path=ROOT/'planner_investigation'/'lossless_audit_block_sharing.json'
report=json.loads(record_path.read_text()) if record_path.exists() else []
processed={r['path'] for r in report}
for target in ROOT.rglob('*_chain_*.npz'):
    if 'planner_investigation' in target.parts:continue
    if str(target) in processed:continue
    source=target.with_name(re.sub(r'_chain_\d+','',target.name))
    if not source.exists() or target.stat().st_size<500_000:continue
    digest=hashlib.sha256();same=0;different=[]
    with target.open('rb') as old,source.open('rb') as original:
        offset=0
        while data:=old.read(BLOCK):
            digest.update(data);other=original.read(len(data))
            if data==other:same+=len(data)
            else:different.append((offset,len(data)))
            offset+=len(data)
    size=target.stat().st_size;changed=sum(n for _,n in different)
    if same<.8*size or changed>min(32*BLOCK,shutil.disk_usage(ROOT).free//4):continue
    temporary=target.with_name(target.name+'.cow-audit')
    if clone(os.fsencode(source.resolve()),os.fsencode(temporary.resolve()),0):continue
    try:
        with temporary.open('r+b') as new,target.open('rb') as old:
            for offset,n in different:
                old.seek(offset);new.seek(offset);new.write(old.read(n))
            new.truncate(size)
        check=hashlib.sha256()
        with temporary.open('rb') as new:
            while data:=new.read(BLOCK):check.update(data)
        if check.digest()!=digest.digest():raise RuntimeError('Byte verification failed; original remains unchanged')
        os.replace(temporary,target)
        record=dict(path=str(target),shared_from=str(source),sha256=digest.hexdigest(),identical_block_bytes=same,changed_block_bytes=changed)
        report.append(record);print(record,flush=True)
    finally:
        if temporary.exists():temporary.unlink()
record_path.write_text(json.dumps(report,indent=2))
print('Retained every archive unchanged; shared logical bytes',sum(r['identical_block_bytes'] for r in report))
