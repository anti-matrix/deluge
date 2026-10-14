# Mesa API notes

Verified against the installed library on 2026-10-14 (Round 0), by introspection and small probe scripts. Where SPECS.md and the library disagree, the library wins; deviations are listed at the end.

## Environment

| Item | Version |
| --- | --- |
| Python | 3.14.6 (venv at `.venv`, created with `python -m venv`; uv not installed) |
| Mesa | 3.5.1 (installed as `mesa[rec]`) |
| solara | 1.64.0 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| pandas | 3.0.6 |
| pyarrow | 26.0.0 |
| matplotlib | 3.11.2 |
| altair | 6.3.0 |
| networkx | 3.7 (pulled in by Mesa) |
| opensimplex | 0.4.5.1 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| hypothesis | 6.169.1 |
| git | 2.53.0.windows.4 (bundled with GitHub Desktop, not on PATH) |

Not installed yet, declared as extras: `jit` = numba (round 11), `earth` = rasterio, xarray (round 12).

## Names and import paths

| Spec name | Exact import | Signature |
| --- | --- | --- |
| Model | `mesa.Model` | `Model(*args, seed=None, rng=None, scenario=None, **kwargs)` |
| Agent | `mesa.Agent` | `Agent(model, *args, **kwargs)` (unique_id assigned automatically) |
| CellAgent | `mesa.discrete_space.CellAgent` | `CellAgent(model, *args, **kwargs)`; set `self.cell = cell` to place it |
| HexGrid | `mesa.discrete_space.HexGrid` | `HexGrid(dimensions, torus=False, capacity=None, random=None, cell_klass=Cell)` |
| PropertyLayer | `mesa.discrete_space.PropertyLayer` | `PropertyLayer(name, dimensions, default_value=0.0, dtype=float)`; also `PropertyLayer.from_data(...)` |
| DataCollector | `mesa.DataCollector` (module `mesa.datacollection`) | `DataCollector(model_reporters=None, agent_reporters=None, agenttype_reporters=None, tables=None)` |
| SolaraViz | `mesa.visualization.SolaraViz` | `SolaraViz(model, renderer=None, components=[], *, play_interval=100, render_interval=1, simulator=None, model_params=None, name=None, use_threads=False, **console_kwargs)` |

Also available: `mesa.discrete_space.FixedAgent` (an agent that never moves), `mesa.visualization.SpaceRenderer`, `make_space_component`, `make_plot_component`, `Slider`, `mesa.batch_run`.

## Behavior that matters for Deluge

**Model construction and RNG**
- Call `super().__init__(rng=...)`. Passing `seed=` still works but emits a `FutureWarning` (deprecated); passing both raises `ValueError`.
- `rng` may be an int seed or a `numpy.random.Generator`. With a Generator, Mesa uses it as `model.rng` and seeds `model.random` (stdlib `random.Random`) from an integer drawn from it.
- Deluge passes its `"mesa"` child stream (see `deluge/rng.py`), so Mesa's internal randomness also derives from the run's single seed.

**Stepping**
- In 3.5, `Model.__init__` replaces `self.step` with `_wrapped_step`, driven by an internal event list: each `model.step()` advances `model.time` by 1.0, increments `model.steps`, then calls the user's `step()`. Don't increment `steps` yourself.
- Overhead: 11,000 calls of an empty `step()` plus `DataCollector.collect` take about 0.19 s.
- `model.running` is honored by `run_model()`. Also new in 3.5: `run_for`, `run_until`, `schedule_event`, `schedule_recurring`. Deluge uses its own `run()` loop over `step()`.

**HexGrid**
- `HexGrid((100, 60))` builds 6,000 cells in about 0.12 s; `grid.width == 100`, `grid.height == 60`.
- Coordinates are `(x, y)` tuples in offset layout. An interior cell has 6 neighbors, e.g. `(5, 5)` → `(4,4) (4,5) (4,6) (5,4) (5,6) (6,5)`. A corner `(0, 0)` has 3 when non-torus.
- Cells are in `grid._cells[(x, y)]` and `grid.all_cells`; neighbors via `cell.neighborhood`. Each cell has a `position` (2D float array of its hex center), but see the mismatch below.
- Property layers: `grid.create_property_layer(name, default_value=0.0, dtype=float)` or `grid.add_property_layer(PropertyLayer)`. The layer is then reachable as `grid.<name>` (a `PropertyLayer` whose `.data` is a NumPy array of shape `(width, height)`), and each cell exposes the value as `cell.<name>`. The data shape is (width, height), not (rows, cols).
- `PropertyLayer.from_data(name, array)` **copies** the array. The per-cell accessors (`cell.<name>`) are bound to `layer.data` when the layer is added, so replacing `layer.data` with a new array breaks them; update in place (`layer.data[...] = ...`) instead.

**HexGrid geometry mismatch (verified in Round 1)**
- Connectivity (`_connect_cells_2d`): a cell on an **odd** row connects to columns `x-1` and `x` in the rows above and below; an **even** row connects to `x` and `x+1`. That is the layout where even rows are shifted right by half a hex.
- `cell.position` (`_init_hex_geometry`) shifts **odd** rows right: `x = sqrt(3) * (col + 0.5 * (row % 2))`. With these positions, two of each cell's six connected neighbors are 3.0 apart instead of sqrt(3), and two true geometric neighbors are not connected. `find_nearest_cell` uses the same KD-tree of these positions.
- Mesa's own hex drawer (`space_drawers.HexSpaceDrawer._get_hexmesh`) shifts **even** rows right, which matches the connectivity.
- Deluge follows the connectivity: `deluge/world/grid.py` copies Mesa's neighbor offsets, places centers at `(sqrt(3) * (x + 0.5 * (y even)), 1.5 * y)` like the drawer, and never uses `cell.position` or `find_nearest_cell`. `tests/test_world_grid.py` checks that our neighbor table equals Mesa's `cell.neighborhood` and that every neighbor pair is exactly sqrt(3) apart in our positions.

**Visualization**
- `SpaceRenderer` and the matplotlib and altair backends have explicit HexGrid support: hexagons are drawn as a `PolyCollection`, and property layers can be drawn as colored hexes. Round 2 should try this before writing a custom renderer.

## Deviations from SPECS.md

- **Mesa RNG argument:** spec text says "seeded"; Mesa 3.5 deprecates `seed=` in favor of `rng=`. Deluge uses `rng=`.
- **Child streams:** spec says `rng.spawn`. Deluge builds each child `SeedSequence` with a spawn key derived from the stream's name (crc32) instead of its spawn order. This is the same mechanism `spawn` uses, but adding or reordering streams can never shift the others.
- **Repo root:** the spec's top-level `deluge/` directory is this repository's root (`civsim/`), because `docs/` and `CLAUDE.md` already live there. The package is `civsim/deluge/`.
- **Hex positions (Round 1):** see the geometry mismatch above; Deluge uses Mesa's connectivity with drawer-consistent positions rather than `cell.position`.

## opensimplex notes (Round 1)

- opensimplex 0.4.5.1: `OpenSimplex(seed)` with `noise2(x, y)` and `noise2array(xs, ys)`. Without numba each point is evaluated in pure Python (about 25 us), and `noise2array` evaluates the full outer grid of `xs` and `ys`, which on hex centers is twice the needed points.
- `deluge/world/noise.py` is a NumPy port of `opensimplex.internals._noise2` that uses the library's own permutation table (`OpenSimplex(seed)._perm`, a private attribute) and constants. `tests/test_world_noise.py` checks it matches `OpenSimplex.noise2` to 1e-12. World generation went from about 1.7 s to 0.14 s.
