"""Local design preview only; does not publish or alter earlier simulation assets."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'output/scenario4_design_20261107'
SITE = ROOT.parent/'sites/belief-adaptive-monitoring/dist/revised'


def main():
    data = json.loads((SOURCE/'geometry-audit.json').read_text())
    # Refresh design-only metadata without repeating or changing the measured audit.
    data['environment'] = json.loads((ROOT/'configs/scenarios/west_monitoring_yard_20261107.json').read_text())
    (SOURCE/'geometry-audit.json').write_text(json.dumps(data, separators=(',', ':'), allow_nan=False))
    dest = SITE/'data/scenario4-design-20261107'
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE/'geometry-audit.json', dest/'geometry-audit.json')
    shutil.copy2(ROOT/'configs/scenarios/west_monitoring_yard_20261107.json', dest/'environment-spec.json')
    for name in ('final-results.html', 'index.html'):
        path = SITE/name
        html = path.read_text()
        if 'id="scenario-four-design-link"' not in html:
            addition = '''<section id="scenario-four-design-link" style="margin:32px 0;padding:28px;border:1px solid #ccd9d5;border-radius:14px;background:#fff"><p style="font-size:12px;letter-spacing:.12em;color:#087f74">NEW · GEOMETRY-SCREENED DESIGN</p><h2>Scenario 4 · Monitoring bays and a connected service yard</h2><p>Eight obstacles, competing monitoring routes, and a large accessible area outside the task. Inspect the actual geometric seeds and learning masks across 44 cases.</p><p><a href="scenario-four-design.html">Open the interactive Scenario 4 design →</a></p><p style="font-size:13px;color:#596a66">Wind placement is a proposal. No Scenario 4 performance result is claimed. All earlier simulations above remain unchanged.</p></section>'''
            html = html.replace('</main>', addition+'</main>', 1)
            path.write_text(html)
    print('Added design links without replacing earlier result containers or recordings.')


if __name__ == '__main__':
    main()
