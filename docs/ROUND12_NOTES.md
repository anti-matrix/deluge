# Notes for Round 12 (calibration)

## Carrying capacity is about 10x too high

Measured after the Round 1 rework, seed 3, year 0: total capacity is about 58 million foragers and 1.7 billion farmers. Both are roughly ten times too high for the scenario: forager bands of a few hundred people should face a world that supports millions, not tens of millions, and early farming should not support a modern-scale population.

Likely contributors to check first:

- Cell area: hexes are about 176 km across (26,700 km2), so per-km2 densities from `capacity` multiply into large per-cell numbers.
- `capacity.forage.max_density` (1.0) and `capacity.farm.max_density` (50.0) are the upper ends of the spec's ranges and apply to large shares of land.

Not changed yet, by decision: calibrate in Round 12 against the section 14 targets.
