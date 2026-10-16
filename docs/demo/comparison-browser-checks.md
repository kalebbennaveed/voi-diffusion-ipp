# Final-result demo verification

- Default seed 200 opens on Final result with proposed cost 3.378 and comparator cost 5.422.
- The displayed 37.7% reduction is unchanged when the comparator recording is selected.
- Both completed trajectories, arrival times, cost partitions and slack metrics are visible.
- Inspecting the comparator opens its own MBD candidates, not the proposed trace.
- Show final selected plan advances to level 4/4 and displays its final evaluated cost.
- Synchronized mission playback reaches the end and returns to Final result.
- Seed 104 correctly shows 14.0% higher proposed cost.
- Seed 204 separately shows 0.8935 peak allowed slack, 1.20 seconds using slack, and no physical collision.
- The no-measurement run explicitly says its belief is unchanged and has no matched comparator.
- Browser console returned no errors or warnings.
- Current 491-pixel viewport has no horizontal page overflow; both result maps fit together.
- Saved preview: comparison-preview.jpg. Seven numerical frontend checks are in comparison-tests.txt.

The two newly exported seed-200 missions exactly match their archived validation costs and
bitwise match x, q, control, slack and feasible arrays. All 23 frozen numerical-source hashes
are unchanged. These are reproducibility exports, not additional independent validation seeds.
