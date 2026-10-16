# Animation-first browser verification

The landing page opens Compare & play and starts both recorded robots automatically.
The displayed clock advanced from 0.04 to 2.39 seconds with changing costs and positions.
Pause held the clock at 0.24 seconds across subsequent work. Jump to first ROI arrival
paused at 7.05 seconds and showed proposed arrival at 6.90 seconds while the comparator
had not arrived. With looping disabled, playback stopped at exactly 12.00 seconds and
showed the archived final costs 3.378 and 5.422. Denoising opens its own animation with
Play and the reverse-level slider above the map. Final result retains the full table.

Animated position/clarity interpolation and the playback clock are tested independently
in model.test.mjs. The complete nine-test frontend result is in animation-tests.txt.
No numerical planner or experiment data were changed.

Looping was also verified in the browser: after 11.98 seconds the clock passed through
the final-frame hold and returned to 6.05 seconds on the next cycle. The page was left
playing at half speed with Loop enabled. Console errors/warnings were empty.
