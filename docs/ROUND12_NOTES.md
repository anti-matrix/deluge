# Notes for Round 12 (calibration)

## Carrying capacity is about 10x too high

Measured after the Round 1 rework, seed 3, year 0: total capacity is about 58 million foragers and 1.7 billion farmers. Both are roughly ten times too high for the scenario: forager bands of a few hundred people should face a world that supports millions, not tens of millions, and early farming should not support a modern-scale population.

Likely contributors to check first:

- Cell area: hexes are about 176 km across (26,700 km2), so per-km2 densities from `capacity` multiply into large per-cell numbers.
- `capacity.forage.max_density` (1.0) and `capacity.farm.max_density` (50.0) are the upper ends of the spec's ranges and apply to large shares of land.

Not changed yet, by decision: calibrate in Round 12 against the section 14 targets.

**Round 3 interim calibration (foraging only):** `capacity.forage.density_scale: 0.11` scales foraging capacity so world seed 1 supports about 7.6 M foragers at year 0 (68.8 M before, measured with history seed 1's cold-snap anomaly of -3.1 °C). Effective forager densities are now 0.011-0.11 people/km². **Accepted range (approved after Round 3): 0.01-0.1 forager per km²**, replacing the spec's 0.1-1 for this map scale.

**Round 4 calibration (farming and herding switch on):**

- `capacity.farm.density_scale: 0.21` (0.3 at first, 0.25 after tuning iteration 1, 0.21 after iteration 2) and `capacity.herd.density_scale: 0.3`. Target: total population at year 11,000 between 50 and 300 million on world seed 1, history seeds 1-5. With 0.3 the first full run (history seed 1) ended at 317 M; with 0.25 one run ended at 348 M (accepted, but lowered once more); with 0.21 the five runs end at 156-243 M.
- The scale multiplies the spec's 10-50 people/km² farming range, so the configured range becomes 2.1-10.5/km²; technique multipliers (plough 1.4, calendar 1.1, irrigation 1.8 on river cells) lift good river land back into the spec range. Measured on the five runs, settlements on river cells with fertility above 0.6 and at least 80% farmers: median 10.4-10.8 people/km² at year 3,000 and at year 11,000; 10th-90th percentile about 4-30. The median sits just inside the 10-50 target; the lower part of the distribution (young or climate-limited settlements) is below it. Lowering the scale further would push the median out of the target, so 0.21 is about as low as these two targets allow together.
- The farming capacity still uses `min_density` 10 on all farmable land before the scale; poor land therefore carries 2.1/km² at least. Revisit with the section 14 targets.
