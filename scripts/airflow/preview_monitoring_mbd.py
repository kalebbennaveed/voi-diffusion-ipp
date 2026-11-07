"""Expose only already-computed mission records while the workers continue."""
import argparse
import json
import time
from pathlib import Path


def preview(directory, site, name="monitoring"):
    out = site / 'data' / name
    out.mkdir(parents=True, exist_ok=True)
    ids = [f's{s}-{m}' for s in (1, 2) for m in ('aware', 'baseline')]
    stamps = {}
    while True:
        summaries = []
        for name in ['environment.json', *[i + suffix for i in ids for suffix in ('.json', '-protocol.json')]]:
            source = directory / name
            if not source.exists():
                continue
            stamp = source.stat().st_mtime_ns
            if stamps.get(name) != stamp:
                data = source.read_bytes()
                json.loads(data)
                tmp = out / (name + '.tmp')
                tmp.write_bytes(data)
                tmp.replace(out / name)
                stamps[name] = stamp
        for ident in ids:
            path = out / (ident + '.json')
            if path.exists():
                record = json.loads(path.read_text())
                summaries.append(dict(id=ident, scenario=record['scenario'], method=record['method'], summary=record['summary']))
        protocols=list(out.glob('*-protocol.json'))
        settings=max((json.loads(p.read_text())['settings'] for p in protocols),key=lambda c:c['mission_seconds']) if protocols else {}
        manifest = dict(schema=1, state='computing', duration_s=settings.get('mission_seconds',60), planner_settings=settings, runs=summaries,
                        note='Only completed integration intervals are shown. Final mission audit and source package are pending.')
        tmp = out / 'manifest.json.tmp'
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(out / 'manifest.json')
        print(json.dumps({r['id']:r['summary']['simulated_seconds'] for r in summaries}), flush=True)
        if len(summaries) == 4 and all(r['summary']['completed'] and r['summary']['simulated_seconds']==settings.get('mission_seconds',60) for r in summaries):
            break
        time.sleep(10)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--site', type=Path, required=True)
    p.add_argument('--name',default='monitoring')
    args = p.parse_args()
    preview(args.directory, args.site,args.name)
