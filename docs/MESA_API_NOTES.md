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

**Visualization (verified in Round 2)**
- `SpaceRenderer(model, backend="matplotlib")` draws a HexGrid correctly for Deluge. `HexSpaceDrawer._get_hexmesh` builds the mesh row by row with centers `(col * sqrt(3) + (row even) * sqrt(3) / 2, 1.5 * row)`, which is exactly `HexGeometry.positions`. Property layer colours are taken as `layer.data.T.ravel()` (row-major, the mesh order). Agent markers use the same even-row shift. None of it uses `cell.position`, so the position mismatch above does not affect drawing. `tests/test_viz.py` checks the mesh centers against our geometry and that every connected neighbour pair is drawn as touching hexes.
- `renderer.setup_propertylayer(fn)` with `fn(layer) -> PropertyLayerStyle | None` draws every layer in `grid._mesa_property_layers` whose style is not None, each as its own `PolyCollection` (zorder -1, in insertion order). `PropertyLayerStyle(colormap=..., color=..., alpha, colorbar, vmin, vmax)`: with `color`, the value maps to opacity; `colormap` may be a `Colormap` object.
- `SpaceRendererComponent` redraws only the meshes that were drawn once before (`renderer.draw_propertylayer()` etc.). `renderer.post_process(ax)` runs once per renderer; the axis limits persist because each frame only removes artists. `draw_structure()` draws every hex edge as a dotted line (unused: at 15,360 cells it hides the map).
- Frame cost at 160 x 96 with two layers: about 1 s to draw and encode the PNG; about 1.5 s per frame in the browser.
- `ModelController` Reset calls `type(model)(**params)`. `mesa.Model` sets a default `scenario`, but since `DelugeModel.__init__` takes no `scenario` argument, the params (e.g. `seed`) are passed straight through.
- **SolaraViz layout does not render with the installed frontend.** solara 1.64 with ipyvue 3.1.1 / ipyvuetify 3.1.0 serves the Vue 3 app. `SolaraViz` puts all components, including the map, in `solara.GridDraggable`, whose `vue-grid-layout` plugin is Vue 2 only. It fails in the browser (`TypeError: GlobalVue.use is not a function`) and the main area stays blank. `solara.Sidebar` also overlays the main area instead of pushing it aside. Deluge's `viz/app.py` therefore builds its own Solara `Page` from Mesa's pieces (`SpaceRendererComponent`, `ModelController`, `ModelCreator`, `make_plot_component`) in plain `Columns`.

## Deviations from SPECS.md

- **Mesa RNG argument:** spec text says "seeded"; Mesa 3.5 deprecates `seed=` in favor of `rng=`. Deluge uses `rng=`.
- **Child streams:** spec says `rng.spawn`. Deluge builds each child `SeedSequence` with a spawn key derived from the stream's name (crc32) instead of its spawn order. This is the same mechanism `spawn` uses, but adding or reordering streams can never shift the others.
- **Repo root:** the spec's top-level `deluge/` directory is this repository's root (`civsim/`), because `docs/` and `CLAUDE.md` already live there. The package is `civsim/deluge/`.
- **Hex positions (Round 1):** see the geometry mismatch above; Deluge uses Mesa's connectivity with drawer-consistent positions rather than `cell.position`.
- **WorldData (Round 1 rework):** adds a `flow_to` field (downstream cell per land cell). Flow directions are drawn at random among lower neighbours, so they cannot be re-derived from elevation and must be stored.
- **Terrain (Round 1 rework):** `coast` is no longer a class; it is a per-year flag (land next to that year's sea). `tundra` and `ice` are added.
- **Cell size (Round 1 rework):** the cell size comes from the grid and the latitude span: 160 x 96 over 130 degrees gives hexes about 176 km across (26,700 km2), larger than the spec's 50 to 100 km. Capacity uses this geometric area.

- **Browser UI (Round 2):** the spec names `SolaraViz` as the UI. Its layout grid does not render with the installed solara / ipyvue 3 (see Visualization above), so `deluge/viz/app.py` is a Solara `Page` built from Mesa's visualization components; the map itself is Mesa's `SpaceRenderer`, not a custom renderer.
- **Map renderer (Round 9):** the map is no longer Mesa's `SpaceRenderer` (matplotlib PNGs: no hover, zoom or pan, and slow at 15,360 hexes). It is a custom ipyvue `VueTemplate` widget (`deluge/viz/hexmap.py` + `hexmap.vue`, the same mechanism as Solara's own echarts component) that draws the hexes on an HTML canvas; Python sends one RGB colour per cell, the border segments and the overlay marks per frame. Mesa's `ModelController` and plot components are replaced too: a Solara task steps the model N years per frame in a worker thread and builds each frame between steps under one lock; charts and the lineage tree are SVG strings. Mesa's `_build_model_init_kwargs` is still how Reset builds the model's keyword arguments (tests).
- **Sea flag layer (Round 2):** the model keeps the year's sea flag as a `sea` PropertyLayer on the grid (spec section 4 lists the sea flag as a cell field), updated in place by the climate phase. The UI draws it over the terrain layer.

- **Two seeds (Round 3):** `run.world_seed` draws the map (the "world" stream), `run.history_seed` every other stream including Mesa's. Both are in run metadata; the UI has a control for each.
- **Settlement state (Round 3):** settlements are Mesa `CellAgent`s (placed with `agent.cell = grid[(x, y)]`, moved the same way, removed with `agent.remove()`), but their numbers live in a NumPy `SettlementTable` so yearly demography is vectorized. At most one settlement per cell (the spec's "free neighboring cell").
- **Drowned bands (Round 3):** bands whose cell the sea took move at the start of phase 3, before they eat, rather than in phase 4; otherwise a slow sea rise would starve them for a year.
- **Famine migration (Round 3):** starving forager bands move whole to a free cell or, where the land around is full, into a band of their own lineage with more food per head. Bands never merge across lineages.
- **Migration danger (Round 3):** the spec's cell utility includes danger; there is no war until Round 6, so the utility uses food, distance and crowding only.
- **Forager density (Round 3):** foraging capacity is scaled by `capacity.forage.density_scale` (0.11), below the spec's 0.1-1 person/km2; see docs/ROUND12_NOTES.md.

- **Event verbs (Round 4):** the spec's verb list has no word for gaining or losing a technique; the event log adds `learned` and `forgot`. Actors are typed entity ids (teachers, lineage, settlement, technique, technology, artifact), so a teaching event names the techniques passed on.
- **Event log pulled forward (Round 4):** `deluge/data/events.py` and `events.parquet` exist from Round 4 (spec: Round 8); the memory system itself stays in Round 8. Run directories also hold `entities.parquet` (kind and label of every entity id in the log).
- **Culture vectors (Round 4):** the spec gives each founding group a random culture vector at year 0. It is drawn now, per group (history seed), and only openness is used (teaching and diffusion); polities, drift and the other traits come in Round 5.
- **Subsistence mode choice (Round 4):** rule-based until Round 7 (now biased by the brains, below): each year a settlement moves `mode_shift_rate` of its people toward the mode whose shift promises the fastest growth; leaving farming is sticky. Production is per mode: each mode's workers are capped by that mode's capacity on the cell.
- **Latent techniques (Round 4, user design):** a technique held where it cannot be used (no coast for sailing, no capacity for its mode on the cell) is latent: much slower decay, none while a neighbour practises it, and no `learned`/`forgot` events. Size-tied techniques are usable everywhere and decay normally in small groups. `forgot` is logged only for a practised technique.
- **Sedentary farmers (Round 4):** bands farming at least `sedentary_farm_share` move whole only after `farmer_famine_years` famine years in a row (spec section 6). Sailing lets migration paths cross a few sea cells.
- **Settlement size caps (Round 4):** the spec caps settlement size per cell with towns for larger populations. With 176 km cells one settlement stands for a cell's population, and the cap is not implemented; carrying capacity limits size.
- **Teachers' weapons (Round 4):** the hook `teachers_state.attacked(model, settlement)` logs a failed attack and the weapons display; nothing calls it until war exists (Round 6).

- **Late techniques off by default (after Round 4b, user decision):** `knowledge.late_techniques_enabled` (default false) stops the 14 techniques marked `late` in `configs/knowledge.yaml` from being invented or adopted. They still exist in the graph; tests that exercise them switch it on. See docs/ROUND12_NOTES.md.

- **Polities (Round 5):** `Polity` is a `mesa.Agent` (no cell) whose numbers live in a NumPy `PolityTable`, like settlements. Settlements carry a `polity` column. Ids are never reused; every polity records its parent, and `LineageRegistry.polity_tree` exports the tree (`polity_tree.parquet` from `scripts/run.py`; the spec's `polities.parquet` time series comes in Round 10).
- **Mergers inactive (Round 5, user decision):** the merger mechanism is built (larger partner absorbs the smaller, keeps its id and culture, uniform crossover with chance `merge_crossover_chance`), but its two inputs, shared-memory overlap (Round 8) and a common enemy (Round 6), are zero, so no merger happens yet. The spec speaks of "the new polity" on a merger; Deluge keeps the larger partner's id so lifespans stay readable.
- **Tribute and monuments (Round 5, user decision):** rule-based placeholders, replaced by brain outputs in Round 7 (below): tribute rate = `tribute_max_rate` x hierarchy, monument spending = `monument_max_share` x ritual_investment of the treasury (once the capital holds megalithic construction). Unpaid upkeep leaves a monument unmaintained (oldest are paid first).
- **Openness (Round 5):** diffusion and teaching now read the openness of the settlement's polity (identical to the founding group's until the first split).
- **Contact (Round 5):** "neighbours" for frontier pressure, secession groups, culture drift and appearance/dialect mixing are settlements within `polities.contact_radius` hex steps (2), since settlements are often a few cells apart. Knowledge diffusion keeps strict adjacency.
- **Deposit access (Round 5):** a polity controls its settlements' cells and those within `territory_radius` (1); a deposit there gives every member access. The Round 4b rule (a deposit within `access_radius_cells` of the settlement's own cell) is kept as the stand-in for local reach and trade until trade exists.
- **Cohesion and pressure terms (Round 5):** f is the share of border settlements facing a culturally distant polity (distance ramping between two config thresholds); war joins it in Round 6 and the shared-memory term in Round 8. Polity size in the decay term is counted in settlements (populations differ by orders of magnitude between foragers and farmers), scaled by hierarchy and the `admin_cap` effect of writing and telecommunication. State strength ((tribute - spending) per head) lowers pressure. Population pressure is clipped to 0..1.
- **Fragmentation regions (Round 5):** the spec does not define a polity's regions. Regional seats are its largest settlements at least `region_spacing` steps apart (capital first); members join the nearest seat. On fragmentation the old polity ends and every region, the capital's included, becomes a new child polity. "Collapse" in the chronicle is a polity's end (`died`): fragmentation or extinction.
- **Polity events (Round 5):** no new verbs. `founded` (founding polity from its band at year 0; a split from its parent polity), `joined` (a merger), `died` (fragmentation or extinction), `built` (a monument). Entity kinds `polity` and `monument` are added to the event log.
- **Appearance and dialect (Round 5):** stored per settlement in the settlement table; only `systems/identity.py` reads or writes them (test-enforced). Perception, the brain inputs and the evolvable culture weights on appearance and dialect distance come in Round 7; until then no system uses them, so their inside-polity effects are zero by construction. The intermarriage rate is an evolvable polity parameter drawn per founding polity (uniform in config bounds) and mutated at splits. Dialects converge between settlements in contact of any polity; appearance mixes inside polities and when bands join. Since bands only join their own lineage and mergers and conquest are not active yet, appearance blends across founding peoples only after a union (tested with a forced merger). The teachers' appearance is drawn and kept on the model for the memory records of Round 8.

- **Brains (Round 7):** no new Mesa API is used; brains are NumPy arrays (`brains/network.py` `BrainTable`, indexed by polity id like the `PolityTable`) evaluated in one batched einsum pass per year in a new phase 6 (`systems/decide.py`), between knowledge and war. MLP: 28 inputs (25 features of spec section 10, 3 memory slots that stay zero until Round 8; recurrent units add more, off by default) -> 16 tanh -> 11 sigmoid outputs, plus the per-neighbour target head (spec "target preference"): the hidden state gives one weight per target feature (`Q h + q0`), and a candidate's score is sigmoid(features . weights). 804 weights. Founders' weights ~ N(0, 2 / sqrt(fan in)): `init_sigma` 2.0 spreads the founders' outputs about as widely as the uniform 0..1 culture traits did (sd ~0.26; 1.0 gave ~0.12).
- **Marker weights start at zero (Round 7):** the spec's hard rule ("an evolved brain weight or an evolved culture parameter whose sign starts at zero") is read literally: in the founders' random brains the weights that read appearance or dialect distance (their input rows and target-head entries) are exactly zero (`brains.marker_init_sigma` 0.0); mutation moves them either way. The marker-effect culture parameters (`agents/polity.py` `MARKER_EFFECTS`: cohesion, grievance, tribute and secession weights on appearance and dialect distance from the capital, plus secession on culture distance) also start at zero and mutate at splits only (no drift toward neighbours or conquerors).
- **Perception (Round 7):** only `systems/identity.py` reads the markers (`pair_distances`, `core_distances`: tanh(euclidean distance / reference)), as before. `ancestry.blind_perception` zeroes every perceived distance, so it blanks the brain inputs and target features and also the inside-polity marker effects (spec: "a config switch blanks these inputs to the brains"; extended so a blind run has no marker channel at all).
- **Brain outputs wired (Round 7):** war choice (aggression, campaign, target head) replaces the rule-based choice (`war.weight_*` and `campaign_share` are gone); tribute rate = `tribute_max_rate` x tribute output and monument share = `monument_max_share` x monument output replace hierarchy and ritual_investment there (user instruction; those traits still scale elite positions, the size term and mobilization). Expand moves the split threshold (`split_threshold` +/- `split_threshold_range`/2). Subsistence: the farming / herding outputs add a bias (`mode_brain_bias`) to a shift's expected growth, scaled to one shift step so the brain tips close calls rather than dictating (0.01 kept half the founding peoples from ever farming). Openness to newcomers: a starving band joins a band of another polity only with chance = that polity's newcomers output. Core migration (spec gives no definition; chosen here): the capital band, unless sedentary, moves whole with chance `core_move_rate` x the output if a better free cell is in reach. Trade openness and alliance seeking are computed and logged (`brain_*` model columns) and drive nothing.
- **Inheritance (Round 7):** splits and fragmentation give each child a mutated copy (sigma 0.05, 1% jumps of sd 0.5) and the parent's priorities until its first decision; conquest ends the loser's brain; the merger crossover (inactive) files the replaced genome. `brains.frozen` copies brains exactly (founders' brains throughout). Every brain born is kept and written by `scripts/run.py` to `brains/brains.parquet` with `brains/layout.json`.

- **Memory substrate (Round 8):** no new Mesa API. `systems/memory.py` is phase 9: it reads the year's new event-log entries (the event log is the only input) and writes a `Memory` record (spec fields plus the actors' appearance and dialect signatures from `identity.actor_signatures`, detail levels, holders, parent record) into the store of every living polity that took part. Vividness = magnitude / `memory.magnitude_reference[verb]`, a per-verb unit conversion (people and deaths per 1,000, rations per 10,000 or 100,000, counts per 1), capped at `vividness_max`; the spec's "proportional to magnitude" is kept, so late events of big populations are far more vivid than the teachers' events (whose raw magnitudes are a few people, techniques or groups).
- **Holders (Round 8):** "held across most of the polity's settlements" needs memory per settlement, which the spec does not define. Records are owned by polity stores; settlements are holders of records (the settlement actors of an event, else the capital; daughters inherit, joining bands hand theirs on; each generation records spread to members in contact and are lost by holders). Conquered settlements keep holding their old records; when a store ends, records still held live on with their holders only (the source for borrowing by the conqueror).
- **Operators (Round 8):** detail loss coarsens one field (year to the next span in `year_spans`, place to its `region_cells` block then none, magnitude to one significant figure). A fused record keeps the more vivid one's actors and symbols and adds the other's up to `max_actors` / `max_symbols` (without a bound, fused records grew to 40+ actors in 3,000 years); provenance is the full union. Substitution draws from the store's entities by total vividness (any entity kind). Amplification by repetition: a new record sharing its verb and a symbol with a held one adds `repeat_gain` x its vividness. Borrowing copies one record per pair in contact (x the receiver's openness) and from subjects to their ruler, with one detail-loss draw on the copy. Transformations are logged when they happen (copies at splits, unions and borrowing) and on transmission years (operators, summed amplification), in `model.memory.ops`.
- **Symbols (Round 8):** integer handles for an entity, a cell or an event, created when the key first enters memory at `symbol_min_vividness`; signatures ride along with each actor in the record, and no symbol is keyed on appearance, so any clustering by appearance is read from the output only.
- **Consequences (Round 8):** derived values (symbol weights, shared-memory term, legitimacy, brain summary, subject overlap) are recomputed once per generation, not yearly. Legitimacy counts the symbols of the capital, the polity, its parent polities and its founding lineage. Invocation happens at the call sites before a battle (attacker), a monument and a union, from the polity's records involving the other party. Affinity drives mergers and assimilation; there is no alliance mechanic yet (alliance seeking still drives nothing), so the spec's "raises alliance chances" is not wired. `war.assimilation_memory_weight` moved to `memory.consequences.assimilation_weight`. `memory.control` zeroes every consequence weight; `memory.enabled: false` removes memory. Control runs match memory-off runs exactly (test and acceptance), since memory draws only from the `memory` stream.
- **Mergers active (Round 8):** odds = `merge_rate` x min(1, `merger_weight` x weighted Jaccard overlap of the two stores' heaviest symbols) x common enemy (both at war with one third polity within `war.war_memory_years`). The teachers have no dialect; their signature carries NaN there.
- **Brain memory inputs (Round 8):** the three memory slots (unchanged count, so genomes keep their size and founders' draws are unchanged) carry the heaviest symbol weight, the vividness share of records with an actor of another founding group (the teachers included) and the mean age of records, x `brain_input_weight`; their founder weights were already drawn like every non-marker input.
- **Outputs (Round 8):** `scripts/run.py` writes `memory.parquet` (live records at the end) and `memory_ops.parquet`; the spec's 500-year `memory_snapshots/` come with Round 10's tables. `analysis/myths.py` renders the store report.

## Mesa grid performance at 160 x 96 (Round 1 rework)

- `HexGrid((160, 96))` with seven static property layers is built once per model; most model setup time is world generation, not Mesa.

## opensimplex notes (Round 1)

- opensimplex 0.4.5.1: `OpenSimplex(seed)` with `noise2(x, y)` and `noise2array(xs, ys)`. Without numba each point is evaluated in pure Python (about 25 us), and `noise2array` evaluates the full outer grid of `xs` and `ys`, which on hex centers is twice the needed points.
- `deluge/world/noise.py` is a NumPy port of `opensimplex.internals._noise2` that uses the library's own permutation table (`OpenSimplex(seed)._perm`, a private attribute) and constants. `tests/test_world_noise.py` checks it matches `OpenSimplex.noise2` to 1e-12. World generation went from about 1.7 s to 0.14 s.
