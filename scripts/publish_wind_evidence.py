"""Publish all eight declared low-value outcomes and diagnostic figures to the 2D demo."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.mplconfig'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/wind_route_study'
BEGIN = '<!-- BEGIN FRESH WIND EVIDENCE -->'
END = '<!-- END FRESH WIND EVIDENCE -->'


def main():
    source = DOC / 'low_value_confirmation_results_01.json'
    result = json.loads(source.read_text())
    rows = result['rows']
    assert len(rows) == 8 and result['n'] == 8
    assert result['all_missions_no_dense_contact']
    assert all(r['methods']['proposed']['cycle_2s_miss_fraction'] == 0 for r in rows)
    values = np.array([r['paired_regret_percent'] for r in rows])
    labels = [str(r['seed']) for r in rows]
    mean = result['mean_paired_regret_percent']
    lo, hi = result['paired_t_90_percent_interval']
    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    ax.axvspan(-5, 5, color='#edf3ef', label='Declared ±5% equivalence band')
    ax.axvline(0, color='#64737e', linewidth=0.8)
    ax.axvline(5, color='#b3743b', linestyle='--', linewidth=1)
    ax.scatter(values, np.arange(8), color='#087f83', s=40, label='Each fresh paired seed')
    ax.errorbar(mean, 9, xerr=[[mean-lo], [hi-mean]], fmt='D', color='#243e50', capsize=5,
                label='Mean and paired 90% t interval')
    ax.set(yticks=list(range(8))+[9], yticklabels=labels+['Mean'], ylim=(10, -1),
           xlabel='Proposed cost relative to current-belief cost (%) · lower is better',
           title='Wind away from an open main passage · eight fresh 120 s pairs')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.legend(loc='lower right', fontsize=8)
    fig.savefig(DOC / 'low_value_confirmation_01.png', dpi=170)
    plt.close(fig)
    table = ''.join(
        f'<tr><td>{r["seed"]}</td><td>{r["methods"]["proposed"]["integrated_whole_grid_deficit"]:.4f}</td>'
        f'<td>{r["methods"]["current_belief"]["integrated_whole_grid_deficit"]:.4f}</td>'
        f'<td>{r["paired_regret_percent"]:+.2f}%</td></tr>' for r in rows
    )
    block = f'''{BEGIN}
 <details id="fresh-wind-evidence" class="fresh-wind-evidence"><summary>Eight fresh seeds · low-value check remains inconclusive</summary>
 <p>Open passage, wind elsewhere: mean proposed cost <strong>{mean:.2f}% higher</strong>; paired 90% interval <strong>{lo:.2f}% to +{hi:.2f}%</strong>. The one-sided 95% upper bound is +{hi:.2f}%, exceeding the declared +5% limit. Neither noninferiority nor equivalence was established.</p>
 <p>All 16 missions lasted 120 seconds with no contact in the independent dense checks. Each proposed planner generated 10-second trajectories in under 2 seconds after the first two startup cycles. These are recorded simulation timings, not an asynchronous flight-controller demonstration.</p>
 <div class="evidence-table-scroll"><table><caption>Integrated whole-field clarity-deficit cost · lower is better</caption><thead><tr><th scope="col">Seed</th><th scope="col">Proposed</th><th scope="col">Current belief</th><th scope="col">Relative cost</th></tr></thead><tbody>{table}</tbody></table></div>
 <p class="help">Fresh seeds 18101–18108 were declared before execution; development seed 8101 is excluded. This tests randomness in one fixed environment and prior. Playable cards above show development recordings, not a selected fresh-seed success. A substantial isolated anticipation benefit remains unproven. Results source SHA-256: <code>{hashlib.sha256(source.read_bytes()).hexdigest()[:16]}</code>.</p>
 </details>
{END}'''
    page = ROOT / 'demo/index.html'
    html = page.read_text()
    if BEGIN in html:
        first, rest = html.split(BEGIN, 1)
        _, last = rest.split(END, 1)
        html = first + block + last
    else:
        marker = ' <details id="environment-gallery"'
        assert marker in html
        html = html.replace(marker, block + '\n' + marker, 1)
    page.write_text(html)
    print('Published eight outcomes and saved low_value_confirmation_01.png')


if __name__ == '__main__':
    main()
