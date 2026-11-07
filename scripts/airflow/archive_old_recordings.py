"""Losslessly archive old, untracked output JSON; never touch served demo files."""
import argparse,gzip,hashlib,json,os,shutil,subprocess
from pathlib import Path


def digest(path,compressed=False):
    h=hashlib.sha256()
    with (gzip.open(path,'rb') if compressed else path.open('rb')) as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true')
    p.add_argument('--restore-source',help='Restore only this original path from the manifest')
    p.add_argument('--free-gib',type=float,default=3.)
    p.add_argument('--manifest',type=Path,default=Path('output/old-recording-archives-20261105.json'))
    a=p.parse_args();rows=json.loads(a.manifest.read_text()) if a.manifest.exists() else []
    if a.restore:
        if a.restore_source:
            rows=[row for row in rows if row['source']==a.restore_source]
            if not rows:raise ValueError('Requested original path is not in the archive manifest')
        needed=sum(row['original_bytes'] for row in rows if not Path(row['source']).exists())
        if needed+1048576>shutil.disk_usage('.').free:
            raise RuntimeError(f'Restoration needs {needed} bytes; restore a single file with --restore-source or first free sufficient space')
        for row in rows:
            source=Path(row['source']);archive=Path(row['archive'])
            if source.exists():continue
            temporary=source.with_suffix(source.suffix+'.restoring')
            with gzip.open(archive,'rb') as r,temporary.open('wb') as w:shutil.copyfileobj(r,w)
            if digest(temporary)!=row['sha256']:raise RuntimeError('Restore checksum mismatch')
            temporary.replace(source);os.utime(source,ns=(row['mtime_ns'],row['mtime_ns']))
        return
    tracked=set(subprocess.check_output(['git','ls-files','-z','output']).decode().split('\0'))
    candidates=sorted((f for folder in ('output/disturbance_suite','output/compact_mbd',
        'output/european_quarter','output/hard_passage_mbd','output/all_methods_demo','output/urban_quad')
        for f in Path(folder).rglob('*.json') if str(f) not in tracked and f.stat().st_size>1000000),
        key=lambda f:f.stat().st_size,reverse=True)
    for source in candidates:
        if shutil.disk_usage('.').free>=a.free_gib*1024**3:break
        archive=source.with_suffix(source.suffix+'.gz');stat=source.stat();sha=digest(source)
        if archive.exists():
            if digest(archive,True)!=sha:continue
        else:
            temporary=archive.with_suffix(archive.suffix+'.archiving')
            with source.open('rb') as r,gzip.open(temporary,'wb',compresslevel=1) as w:shutil.copyfileobj(r,w,1048576)
            if digest(temporary,True)!=sha:raise RuntimeError('Archive checksum mismatch')
            temporary.replace(archive)
        row=dict(source=str(source),archive=str(archive),sha256=sha,mtime_ns=stat.st_mtime_ns,
            original_bytes=stat.st_size,archive_bytes=archive.stat().st_size)
        if not any(r['source']==str(source) for r in rows):rows.append(row)
        temporary=a.manifest.with_suffix('.json.tmp');temporary.write_text(json.dumps(rows,indent=2));temporary.replace(a.manifest)
        source.unlink()  # Verified archive and restore manifest already exist.
        if len(rows)%20==0:print(json.dumps(dict(archived=len(rows),free_gib=shutil.disk_usage('.').free/1024**3)),flush=True)
    print(json.dumps(dict(archived=len(rows),verified_logical_saving_bytes=sum(r['original_bytes']-r['archive_bytes'] for r in rows),free_gib=shutil.disk_usage('.').free/1024**3)),flush=True)


if __name__=='__main__':main()
