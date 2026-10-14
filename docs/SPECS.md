# Deluge: Civilization Neuroevolution Sim — Build Spec

Oct 14, 2026 · @Japeth

## Purpose and how to use this doc

This spec builds a Python simulation of 11,000 years of history after a global catastrophe, where civilizations rise, split, conquer and form their own founding myths through neuroevolution and selection, with no scripted outcomes. It is written to be executed round by round in Claude Code.

**Design principles**

- **Emergence over scripting.** The code supplies mechanisms (substrates, operators, consequences), never content. No outcome, motif or historical pattern is hard-coded. If teachers become gods, states rise on rivers or floods dominate myths, it must emerge.
- **No LLM agents.** Every decision comes from small neural networks evolved inside the sim. The sim runs fully offline on a CPU and costs no tokens once built.
- **Groups, not individuals.** Agents are settlements and polities. Individual people are only numbers inside them.
- **Deterministic per seed.** Same seed and config = identical history. All randomness flows from one seeded generator.
- **Every layer runs alone.** Each build round produces a runnable sim that is tested before the next layer is added.

**Rules for the Claude Code build**

1. Build one round at a time (section 13). Do not start a round until the previous one passes its acceptance criteria and the user approves.
2. Verify every Mesa API call against the installed version before using it. Mesa's API changed substantially between 2.x and 3.x; this spec targets 3.x. Where this doc and the installed library disagree, the library wins, and the deviation is reported.
3. End each round with a local git commit and a short report: what was built, test results, deviations from spec, open problems.
4. Never invent mechanics that inject content (named myths, scripted events beyond those listed in section 2). If a mechanic seems to need content, stop and ask.
5. Keep all tunable numbers in config files, never as literals in code.

## Scenario

The run starts 1,000 years after a nuclear catastrophe melted the ice caps, with five forager groups on a flooded world, and a sixth group carrying the knowledge of civilization appearing among them. It then runs for 11,000 years.

**Backstory (fixed initial conditions, not simulated)**

- A global nuclear war melted ice caps comparable to Earth's at the start of the Younger Dryas, causing a sea-level rise and deluge.
- One thousand years later, the survivors are scattered hunter-gatherer bands. Year 0 of the sim is this moment.

**Initial agents**

| Group | Count | Starting state |
| --- | --- | --- |
| Forager groups A to E | 5 | Bands of 100 to 250 people, clustered in a home region, foraging only, a random culture vector, a random brain |
| The sixth group ("the teachers") | 1 | 30 to 60 people, with advanced technology, highly mobile, holding the early knowledge graph: agriculture, storage, sedentary building, division of labor, megalithic construction, sky observation and calendars |

The teachers appear at a configurable year (default: year 0) and location. They are humans carrying knowledge and technology from before the catastrophe, not a competing group: they appear, teach, interact with the five groups for a contact period, then depart by air and leave the sim permanently (section 7).

**What the run measures at year 11,000**

- Where each of the five founding lineages stands: territory, population, number of successor polities, extinct or surviving.
- What the five groups did with the teachers' knowledge: which techniques each received, kept, lost or passed on, and how far it spread.
- What each lineage's surviving memory and symbol structure looks like, and where it came from (section 11).

**Events the sim may contain without breaking the no-script rule**

Only physical and starting conditions are given: the post-deluge map, the climate curve, sea level, the initial groups and the teachers' arrival, contact phase and departure. Everything social, political, technological and mythological after year 0 must emerge.

## Tech stack and repo layout

Python 3.12+ with Mesa 3.x for the model and browser visualization, NumPy for all numeric state and the brains, and Parquet for output.

| Package | Role |
| --- | --- |
| `mesa[rec]` 3.x | Model and agent classes, `discrete_space.HexGrid`, `PropertyLayer` for cell fields, `DataCollector`, `SolaraViz` browser UI |
| `numpy` | Cell fields, brains, mutation, all vectorized math |
| `numba` (optional, round 11) | JIT for hot loops if profiling shows need |
| `opensimplex` | Noise for procedural terrain |
| `scipy` | Distance transforms, smoothing, flow accumulation for rivers |
| `pyarrow`, `pandas` | Parquet output and analysis |
| `pyyaml` | Config files |
| `rasterio`, `xarray` (optional) | Reading real elevation and paleoclimate rasters |
| `matplotlib` | Static plots in analysis notebooks |
| `pytest`, `hypothesis` | Tests, property tests for invariants |

**Repo layout**

```
deluge/
  pyproject.toml
  configs/
    default.yaml          # every tunable number
    worlds/               # world-source configs (procedural, earth)
  deluge/
    __init__.py
    model.py              # DelugeModel: the yearly step loop
    rng.py                # single seeded generator, child streams
    world/
      grid.py             # hex grid + cell field registry
      procedural.py       # noise, erosion, rivers, climate, biomes
      earth.py            # real-topography importer
      climate.py          # climate curve, shocks, sea level
      capacity.py         # carrying capacity per subsistence mode
    agents/
      settlement.py
      polity.py
      lineage.py          # bookkeeping registry, not a Mesa agent
      teachers.py
    systems/              # one module per yearly phase (section 4)
      demography.py
      subsistence.py
      knowledge.py
      cohesion.py
      politics.py         # formation, splitting, collapse
      war.py
      memory.py           # events, transmission, symbols
    brains/
      network.py          # fixed-topology MLP in NumPy
      evolution.py        # inheritance and mutation
      features.py         # input vector assembly
    data/
      collectors.py
      events.py           # global event log
    viz/
      app.py              # SolaraViz entry point
      portrayal.py
  scripts/
    run.py                # headless single run
    batch.py              # multi-seed runs
  analysis/               # notebooks and reports
  tests/
  runs/                   # outputs, git-ignored
```

Run commands: `python scripts/run.py --config configs/default.yaml --seed 42` for headless runs, `solara run deluge/viz/app.py` for the browser UI.

## Core architecture

One step is one year, the world is a hex grid of about 6,000 cells, and the agents are settlements grouped into polities, each traced to one of six founding lineages.

**Time and space**

- 11,000 steps, one per year. Some systems run less often (configurable): memory transmission every 25 years, one generation; climate shocks drawn yearly.
- Default grid 100 × 60 hex cells using `mesa.discrete_space.HexGrid`, non-torus. A cell represents a region of roughly 50 to 100 km across.
- Cell fields (elevation, terrain, fertility, rainfall, river flag, wild food, sea flag) live in Mesa `PropertyLayer`s or parallel NumPy arrays indexed by cell, never as per-cell Python objects in hot loops.

**Agent hierarchy**

| Level | Mesa class | Holds | Lifetime |
| --- | --- | --- | --- |
| Settlement | `CellAgent` on one cell | population, food store, subsistence mode, known techniques, polity id, lineage id | founded, grows, migrates, abandoned |
| Polity | `Agent` (no cell) | member settlements, culture vector, cohesion, brain, memory store, capital | forms, expands, splits, conquered, collapses |
| Lineage | plain registry object | tree of descent from the six founders | permanent |
| Teachers | `CellAgent`, mobile | contact-phase agent: schedule, technology, early knowledge graph | removed when they depart |

Every settlement and polity carries a lineage id and a parent id, so descent is a tree that can be rebuilt at any year.

**Yearly step order**

Each system is a function `run(model)` in `deluge/systems/`. Order is fixed so results are deterministic:

1. **Climate:** update the year's temperature and rainfall anomaly, draw shocks (drought, cold snap), apply sea level.
2. **Capacity:** recompute carrying capacity per cell for each subsistence mode.
3. **Subsistence and demography:** food production, consumption, storage, births, deaths, famine.
4. **Migration and settlement:** surplus population founds new settlements; starving settlements move or dissolve.
5. **Knowledge:** invention, diffusion between neighbors, teacher contact, loss of unused techniques.
6. **Brains decide:** each polity's network reads its features and outputs priorities for the year.
7. **Diplomacy and war:** resolve attacks, alliances and conquest chosen in step 6.
8. **Cohesion and politics:** update cohesion and pressure; polity formation, secession, collapse; brain inheritance on splits.
9. **Memory:** log the year's salient events; on transmission years, run lossy transmission.
10. **Data collection:** record model and agent metrics.

**Config and randomness**

- All parameters in `configs/default.yaml`, loaded into frozen dataclasses. Model inputs are keyword arguments so `SolaraViz` can instantiate the model.
- One `numpy.random.Generator` seeded per run, with child streams per system (`rng.spawn`) so adding randomness to one system doesn't shift the others.
- Checkpoint the full model state with pickle every 500 years, so a long run can be resumed or forked into counterfactuals.

## World layer

Every world source, procedural or real, outputs the same `WorldData` object, so the rest of the sim never knows where the map came from.

**The common interface**

```python
@dataclass(frozen=True)
class WorldData:
    width: int; height: int
    elevation: np.ndarray      # metres, relative to present sea level
    terrain: np.ndarray        # int enum: sea, coast, plain, forest, hills, mountain, desert, steppe, wetland
    river: np.ndarray          # flow accumulation, 0 = no river
    base_temp: np.ndarray      # mean annual degC at year 0
    base_rain: np.ndarray      # mm/year at year 0
    fertility: np.ndarray      # 0..1 soil quality
    wild_food: np.ndarray      # 0..1 forage richness
```

Sea is derived each year from `elevation < sea_level(year)`, so coastlines move with the sea-level curve.

**Procedural generator** (`world/procedural.py`)

1. Elevation from layered simplex noise with a continent mask (a few low-frequency blobs), scaled so about 35% of cells start as land.
2. Optional simple tectonics: random plate seeds, mountain ridges along converging boundaries.
3. Hydraulic erosion, a light particle-based pass, then flow accumulation to mark rivers and valleys.
4. Temperature from latitude and elevation lapse rate (−6.5 °C per km).
5. Rainfall from prevailing wind bands by latitude, reduced in the rain shadow behind mountains.
6. Terrain and biome from temperature and rainfall (Whittaker-style lookup).
7. Fertility high on river floodplains and young sediment, low in mountains and deserts; wild food highest in forest edges, wetlands and coasts.

Each stage is a pure function of the previous arrays and an RNG stream, testable alone.

**Real-topography importer** (`world/earth.py`, optional)

- Input: a region bounding box plus local files the user downloads (the sandbox has no network): ETOPO or GEBCO elevation, HydroSHEDS rivers, WorldClim or CHELSA climate.
- Resample to the hex grid by area-weighted averaging, then run steps 6 and 7 above.
- Ships with a tiny bundled test raster so tests never need the real files.

**Sea level and climate over time** (`world/climate.py`)

- `sea_level(year)`: a piecewise curve from config. Default: starts 60 m below present and rises to 0 m over the first 3,000 years with two faster pulses, so coasts keep drowning early in the run. A "modern melt" preset starts at +70 m instead.
- `temp_anomaly(year)`: a cold snap of −3 to −5 °C in years 0 to 1,200 (the Younger Dryas analogue), a warm optimum around years 3,000 to 6,000, then mild cooling, plus red noise (AR(1)) for decades-long swings.
- Shocks: yearly draws for regional droughts (2 to 30 years long, spatially correlated) and rare large events such as volcanic winters. Probabilities in config.

**Carrying capacity** (`world/capacity.py`)

People per cell, recomputed yearly:

```latex
K_{mode} = A_{cell} \cdot Y_{mode}(fertility, wild\_food, temp, rain) \cdot T_{mode}(techniques)
```

where `Y` is the yield for foraging, herding or farming and `T` is a multiplier from known techniques (irrigation, plough, storage). Foraging supports roughly 0.1 to 1 person per km², early farming 10 to 50 on good land. Exact values are calibration parameters (section 14).

## Population and subsistence

Settlements grow toward the carrying capacity of their cell under their subsistence mode, store surplus, starve in bad years, and send out daughter settlements when crowded.

**Subsistence modes**

| Mode | Requires | Mobility | Storage | Notes |
| --- | --- | --- | --- | --- |
| Foraging | nothing | high: a band can move each year | minimal | yields from `wild_food`, degraded by overuse |
| Herding | animal domestication | medium | on the hoof | good on steppe and hills, poor in forest |
| Farming | agriculture | settled | grain stores | yields from `fertility`, river floodplains best, irrigation multiplies |

A settlement can mix modes (shares summing to 1). Shifting to farming is a choice the polity brain influences, and it is sticky: fields and stores make moving costly.

**Yearly demography per settlement**

1. Production = sum over modes of share × yield × labor, times the year's climate factor for the cell.
2. Food available = production + store × (1 − spoilage). Need = population × ration.
3. Growth rate rises with food per head up to a max (about 1% per year for farmers, 0.3% for foragers); deaths rise sharply below the ration.
4. Surplus goes to the store up to a cap set by storage techniques. Part of the surplus can be taken by the polity (tribute, section 8).
5. Overuse feedback: foraging above capacity lowers `wild_food` locally; intensive farming without fallow techniques slowly lowers `fertility`.

**Migration and new settlements**

- When population exceeds a threshold of capacity, a fraction splits off and founds a settlement on the best free neighboring cell within a search radius (a utility over capacity, distance, crowding and danger).
- Starving settlements migrate whole if foragers or herders; farmers lose people first and abandon only after repeated famine.
- Settlement size has a cap per cell; large populations become multiple settlements or one town with a tech-dependent cap (towns need storage, division of labor and sedentary building).
- Daughter settlements inherit the parent's polity, lineage, techniques and a copy of its memory.

## Knowledge and the teachers

Techniques form a graph with prerequisites; settlements gain them by rare invention, by diffusion from neighbors, or by contact with the teachers, and lose them when they go unused.

**The knowledge graph** (data file `configs/knowledge.yaml`, not code)

| Technique | Prerequisites | Effect |
| --- | --- | --- |
| Animal domestication | none | unlocks herding |
| Agriculture | none | unlocks farming |
| Storage (pottery, granaries) | agriculture | higher store cap, lower spoilage |
| Sedentary building | agriculture | town cap up, defense bonus |
| Division of labor | sedentary building | production bonus at larger sizes, enables specialists |
| Sky observation and calendar | none | farming yield bonus, reduces climate-shock losses |
| Megalithic construction | division of labor | enables monument investment (section 8) |
| Irrigation | agriculture, division of labor | farming yield multiplier on river cells |
| Sailing | none | movement across sea cells, coastal trade |
| Copper and bronze | division of labor | military and tool bonus |
| Writing | division of labor, calendar | memory capacity up, administration (polity size cap up) |
| Plough | agriculture, domestication | farming yield bonus |
| Horse riding | domestication | movement and raid range up, steppe military bonus |
| Iron | bronze | military bonus, cheap tools |

The list is a starting set and can grow. Effects are numeric multipliers in config.

**Acquisition and loss, per settlement per year**

- **Invention:** small base probability, scaled by population, division of labor, and how many prerequisites are known. Most techniques should be invented independently a few times across the map in a long run.
- **Diffusion:** for each neighbor or trade partner that knows a technique, probability scales with contact intensity and the receiver's openness trait (culture vector). Enemies diffuse military techniques faster.
- **Loss:** a technique not used by its holder (for example agriculture in a settlement that went back to foraging) decays and can be forgotten after generations, more easily in small populations. This is the mechanism by which knowledge can be lost after a collapse.

**The teachers**

- **Who they are:** 30 to 60 humans carrying knowledge and technology from before the catastrophe, more advanced than today's. They are not aliens and not a sixth civilization. Their origin is a starting condition the sim does not explain.
- **Their role is a contact phase only.** They appear, visit the five groups, teach and interact for a while, then depart by air and are removed from the sim. They never take part in demography, politics, war, migration or evolution, and leave no agent behind.
- **Technology:** aircraft (the source of their mobility and their departure), weapons, tools and medicine. These do not compete with the groups. Weapons are used only if a group attacks them, and such an attack always fails. All technology acts on the sim only through what the groups witness and receive.
- **Schedule** (all config): arrival year (default 0) and place, length of stay (default 50 years), which groups they visit and in what order (default: all five, nearest first), years spent with each group.
- **Teaching:** each year a group is in contact, each technique of the early graph transfers with a probability scaled by the group's openness and the time spent with it. Groups can come away with different subsets, so the starting advantage varies by group without being scripted.
- **Witnessed events:** their arrival by air, each teaching contact, each use of their technology the groups see (flight, weapons in self-defense, medicine), and their departure by air are logged as raw events with high magnitude, like any other event. What the groups remember of them is left entirely to the memory system (section 11).
- **After they leave**, the knowledge they gave spreads, decays or is lost through the normal mechanics above.

**Artifacts left behind**

- When the teachers finish their stay with a group, there is a 30% chance (config, `artifact_chance`) that they leave artifacts with it. Drawn independently per group, so a run can have none, one or several groups with artifacts.
- If they do, the number (default 1 to 4) and the types are drawn from their inventory: tools (production bonus), medicine (lower mortality in famine and disease years), weapons (military multiplier). Bonus sizes are config values.
- Each artifact has its own lifespan drawn at handover (default 30 to 300 years). It gives its bonus only while it lasts and cannot be reproduced, repaired or taught, since nothing in the knowledge graph leads to it.
- Artifacts belong to the settlement that received them and pass with it through splits, conquest and raids, so they can change hands.
- Handover, each use in a logged event (a battle won with weapons, a famine survived with medicine) and the moment an artifact breaks are raw events in the memory system, with no special tags.

## Polities, cohesion and collapse

Polities are the decision-making units; they hold together through cohesion, which grows under external pressure and erodes with size, comfort and elite competition, so large states eventually split without any scripted cycle.

**Formation**

- At year 0 each founding group is one polity (a band-level polity over its settlements).
- A settlement whose polity collapses becomes its own polity, inheriting the old brain and culture with mutation.
- Polities grow by founding daughter settlements, by voluntary union (two polities with high shared memory and a common enemy can merge), and by conquest (section 9).

**Culture vector** (per polity, about 6 floats in 0..1)

`hierarchy`, `militarism`, `openness`, `kin_vs_civic`, `ritual_investment`, `mobility_preference`. These are traits, not labels: each one multiplies specific mechanics (openness scales diffusion and assimilation, hierarchy scales tribute and the maximum polity size, ritual investment scales monument building). They change by:

- inheritance with small mutation at splits,
- drift toward neighbors through contact (weighted by trade and alliance),
- shifts toward the conqueror in assimilated settlements, slowly.

**Cohesion (asabiya)**, after Turchin's metaethnic frontier model

```latex
C_{t+1} = C_t + r_0 \cdot f \cdot C_t (1 - C_t) - \delta \cdot C_t
```

- `f` = frontier pressure: share of the polity's border facing culturally distant polities, plus recent war.
- `δ` (decay) rises with polity size, distance of settlements from the capital, internal inequality and elite pressure.
- Shared memory (section 11) adds to cohesion as a separate term, its weight a config parameter. The memory system decides whether that term grows; the code only connects it.

**Structural-demographic pressure**, after Turchin and Goldstone

- Population pressure: population over capacity across the polity.
- Elite pressure: tribute-takers grow faster in good times than positions available; tracked as an `elite_ratio`.
- State strength: tribute collected minus spending on war and monuments.
- Pressure lowers cohesion and raises the per-settlement probability of secession.

**Splitting and collapse**

- Each year, each settlement can secede with probability rising with distance from capital, low cohesion, high pressure, plus whatever weight the polity's evolved culture parameters give to cultural, dialect and appearance distance from the core (starting at zero; see Ancestry, appearance and identity).
- Seceding settlements adjacent to each other form one new polity, inheriting a mutated copy of the parent brain and culture and a copy of the parent's memory.
- If cohesion falls below a threshold and pressure is high, the polity can fragment entirely into its regions.
- Every split records parent and child ids, so the lineage tree stays complete.

**Monuments**

With megalithic construction known, a polity can spend surplus on monuments (a brain output). Monuments cost labor, raise cohesion and prestige while maintained, and are logged as events. Whether they pay off is left to selection.

## Ancestry, appearance and identity

Settlements carry heritable appearance traits and a drifting dialect, which polities can perceive but which have no built-in effect on anything; whether ethnic identity, in-group preference or appearance-based status emerges is left entirely to selection and memory.

**Hard rule for the build:** no trait in this section may change productivity, fertility, health, intelligence, military strength or any other capability. They are markers only. Every effect they have must pass through an evolved brain weight or an evolved culture parameter whose sign starts at zero and is free to go either way. Tests enforce this by checking that no system other than perception and mixing reads these fields.

**Appearance** (per settlement, population-level)

- A small vector of visible traits, for example skin tone, hair color, eye color and stature, each stored as a population mean and variance on a 0..1 scale.
- At year 0, each of the five founding peoples gets its own random mean per trait, drawn far enough apart that the peoples are visibly distinct, with small variance inside each. The teachers get their own random draw (config can fix it).
- **Drift:** each generation the means wander randomly, faster in small populations.
- **Mixing:** migration, conquest, assimilation and unions move the means of the settlements involved toward each other, weighted by the numbers of people exchanged. The rate of intermarriage between groups inside a polity is an evolvable culture parameter, not a fixed number.
- **Environmental selection** (optional, off by default): a slow pull of skin tone toward a latitude-dependent optimum, as in real human populations. Kept off by default so drift and mixing alone can be studied first.

**Dialect**

- A short vector that drifts each generation and converges between settlements in frequent contact. It stands for language and accent: distance in dialect space is how mutually strange two groups sound.
- Splits inherit the parent's dialect, so successor polities slowly diverge into distinct speech communities.

**Perception and response**

- The appearance and dialect distance between a polity and each neighbor, and between its core and its own peripheral settlements, are inputs to the polity brain.
- War target choice, alliance seeking, trade openness and acceptance of migrants use a per-neighbor scoring head in the brain, so a polity can evolve to treat neighbors differently by any feature, including appearance and dialect, or ignore them.
- Inside a polity, how much appearance and dialect distance from the core affects cohesion, grievance, tribute rates and secession is set by evolvable culture parameters that start at zero. Ethnic tension, tolerance or a ranked hierarchy by appearance can each evolve, or none of them.
- **Control condition:** a config switch blanks these inputs to the brains, so runs with and without perceivable differences can be compared.

**In memory**

- Every memory record stores the appearance and dialect signature of each actor at the time of the event.
- Symbols can therefore become attached to how a group looked or sounded, including the teachers. Whether memories of outsiders cluster by appearance, and whether those memories are mostly of war or of trade, is read from the output.

**Questions this section lets the analysis ask**

- Does in-group preference by appearance or dialect evolve, and under what conditions (frontier wars, scarcity, conquest)?
- Do ruling cores end up differing in appearance from the peripheries they rule, and does that gap persist or blend away?
- Do the peoples who mixed most do better or worse than those who stayed apart?
- Does the teachers' appearance become linked to prestige in memory?
- Do polities whose identity tracks appearance last longer than those whose identity tracks dialect or shared memory?

## War and conquest

Wars are raids or campaigns chosen by polity brains and resolved probabilistically from strength, terrain and distance; conquest moves settlements between polities, and assimilation is slow enough that resentment lingers.

**Choosing war**

The brain outputs an aggression score and a target preference (section 10). A polity attacks the neighbor with the highest product of aggression, target attractiveness (its stores, fertile land, weakness) and the attacker's surplus capacity to fight. A cap on simultaneous wars keeps it bounded.

**Strength**

```latex
S = P_{mobilized} \cdot M_{tech} \cdot (0.5 + C) \cdot D_{distance}
```

- `P_mobilized` = population × mobilization share (rises with militarism and hierarchy).
- `M_tech` = military multiplier from bronze, horses, iron.
- `C` = cohesion.
- `D_distance` = decays with distance from the attacker's capital (logistics), extended by horses, sailing and writing.
- Defender bonus for terrain (hills, mountains, wetlands), fortified towns (sedentary building) and fighting at home.

**Resolution**

- Win probability from a logistic function of the strength ratio. Both sides lose population proportionally to the opponent's strength.
- **Raid:** the winner takes part of the loser's stores; no territory changes.
- **Campaign:** the winner takes the target settlement and possibly its neighbors; repeated defeats can end with the loser's capital taken and the polity destroyed.
- Every battle with losses above a threshold, every captured capital and every destroyed polity is logged as an event.

**After conquest**

- A conquered settlement keeps its own culture vector, techniques and memory, and gets a `grievance` value.
- Its culture drifts toward the conqueror's at a rate set by the conqueror's openness and the shared-memory overlap; grievance decays over generations.
- High grievance raises its secession probability and lowers polity cohesion, so conquest without assimilation sets up later breakup.
- Techniques the conquered held diffuse to the conqueror at a high rate.

## Polity brains and neuroevolution

Each polity is driven by a small fixed-topology network whose weights are inherited with mutation when polities split, so strategies evolve by cultural group selection with no fitness function defined by the code.

**Network**

- Multilayer perceptron in NumPy: about 24 inputs → 16 hidden (tanh) → 12 outputs. Roughly 600 weights per brain.
- All polity brains are evaluated in one batched matrix multiply per year (stack inputs into a matrix, weights into a 3-D array), so cost stays trivial even with hundreds of polities.
- Optional recurrent state (a few hidden units fed back each year) so a polity can carry intentions across years. Off by default, a config switch.

**Inputs** (`brains/features.py`, normalized to about −1..1)

- Own state: log population, food per head, store level, cohesion, pressure, elite ratio, number of settlements, territory growth over 25 years.
- Knowledge: counts of known techniques by category (subsistence, building, military).
- Neighbors: strength ratio to the strongest and weakest neighbor, number of hostile neighbors, cultural, dialect and appearance distance to neighbors and to the polity's own periphery, recent war wins and losses.
- Environment: climate anomaly this year, drought flag, capacity headroom.
- Memory: a fixed-length summary vector of the polity's memory store (section 11), so myths can feed decisions without the code knowing what they mean.

**Outputs**

Priorities squashed to 0..1: expand (found settlements), aggression, raid vs campaign, target preference (a per-neighbor scoring head over each neighbor's features, including appearance and dialect distance), trade openness, alliance seeking, tribute rate, monument spending, subsistence shift toward farming, toward herding, openness to newcomers, migration of the core.

**Inheritance and mutation** (`brains/evolution.py`)

- On a split or a new polity from collapse: child weights = parent weights + Gaussian noise (σ from config, default 0.05), with a small chance per weight of a larger jump.
- On a merger: the new polity inherits the brain of the larger partner, with a configurable chance of uniform crossover with the smaller one.
- On conquest: the conquered polity's brain dies; the settlements now follow the conqueror's brain.
- Founding groups start with independent random brains (seeded), so the five lineages begin with different temperaments.

**Why no fitness function**

Selection is the sim itself. A brain spreads when its polity grows, splits and leaves descendants, and disappears when its polity is conquered or collapses. This is slow compared with game-playing neuroevolution: in 11,000 years there may be only a few thousand polity births. Two settings help it work:

- Splits must happen often enough to give selection material. Calibration target: hundreds to low thousands of polity births per run.
- Multi-run evolution (optional, round 12): seed a new run's founding brains from the most successful lineages of previous runs, to study what strategies evolve over many histories.

## Emergent memory and myth

The code supplies a memory substrate, generic copying errors and consequences for what is remembered, and nothing else; any myth-like structure must arise from transmission and selection. No motif, label or story template exists anywhere in the code.

**Hard rule for the build:** no code path may reference a specific kind of story (flood, hero, god, founder). Event records carry only the raw fields below. Tests enforce this by checking that no myth or motif vocabulary appears in `memory.py`.

**Event records** (`systems/memory.py`)

```python
@dataclass
class Memory:
    mem_id: int
    actors: tuple[int, ...]       # entity ids: polities, settlements, the teachers, a lineage
    action: int                   # generic verb code: arrived, taught, displayed, departed, gave, used, broke, fought, won, lost, moved, founded, built, starved, died, joined
    place: int                    # cell id
    year: int                     # year of the original event
    magnitude: float              # size of the effect (people lost, techniques gained, land taken)
    vividness: float              # salience, decays over time
    symbols: tuple[int, ...]      # links to symbols (below)
    provenance: tuple[int, ...]   # ids of original events this memory descends from
```

The action codes are generic verbs that describe what the sim did. They are the only vocabulary the system has.

**What gets recorded**

Each year, events above a magnitude threshold are written to the memory of the polities and settlements that took part: the teachers' arrival, each teaching contact, each display of their technology and their departure, migrations, foundings, big battles, conquests, famines, splits, monuments, deaths of polities. Vividness starts proportional to magnitude.

**Lossy transmission** (every generation, 25 years)

Each polity holds at most `K` memories (default 40, raised by writing). Transmission copies the store to the next generation and applies generic operators with probabilities from config:

- **Decay:** vividness falls; memories below a floor drop out when the store is full.
- **Detail loss:** a field is coarsened (exact year becomes a broad age, place becomes the region, magnitude rounds).
- **Merge:** two memories with overlapping actors, place or symbols fuse into one, its provenance the union of both.
- **Actor substitution:** an actor is swapped for another entity in memory, weighted by that entity's total vividness. Credit for an event can thus drift to the most remembered actor.
- **Amplification:** vividness of a memory rises when it is reinforced by repetition (similar events recur) or by use (it was invoked, below).
- **Borrowing:** through contact and conquest, memories copy between polities with the same operators applied.

Splits copy the parent's store to the child, so successors share an origin store that then diverges.

**Symbols**

- A symbol is an integer id created when an entity, place or event first enters memory with high vividness. It is a handle, not a meaning.
- Memories point to symbols. Polities can share symbols through borrowing, and the same symbol can come to anchor different memories in different polities.
- A symbol's weight in a polity = summed vividness of the memories attached to it.

**Consequences** (what makes selection act on memory)

- **Cohesion:** the shared-memory term in the cohesion equation grows with the vividness of memories held across most of the polity's settlements.
- **Legitimacy:** a polity whose capital or ruling lineage is an actor in heavy symbols gets a stability bonus that lowers secession odds.
- **Affinity:** the overlap of heavy symbols between two polities raises alliance and union chances and speeds assimilation after conquest.
- **Invocation:** before a war, a monument or a union, the polity "uses" its heaviest related symbols, amplifying them. Behavior shapes memory and memory shapes behavior.
- **Brain input:** a fixed-length summary of the store (for example, the top symbols' weights, the share of memories involving outsiders, the mean age of memories) is part of the brain's inputs.

The weights of these effects are config parameters. Setting them all to zero must give a sim where memory exists but has no effect, the control condition for checking whether myth-making changes history.

**Provenance and reading results**

- The global event log keeps every original event. Every memory carries its provenance chain back to them.
- `analysis/myths.py` renders a polity's store for humans: each heavy symbol, the memories attached, and for each memory the original events it descends from, with years and actors, plus the transformations that changed it (logged on transmission years in a compact table).
- Cross-run analysis looks for recurring structures: for example, how often the teachers become the substituted actor in memories of events they had no part in, or how often catastrophe memories survive longest. These are questions asked of the output, never built into the code.

## Visualization, data and analysis

The browser UI is for watching one history unfold; headless runs write Parquet files that analysis scripts compare across many seeds.

**Browser UI** (`viz/app.py`, run with `solara run`)

- **Map view:** the hex grid drawn as polygons, colored by a selectable layer: polity (color per polity, shades per lineage), terrain, population density, subsistence mode, cohesion, or a knowledge technique's spread. Sea updates with the sea-level curve.
- **Overlays:** capitals, monuments, the teachers' position and path during the contact phase, active wars as arrows between capitals.
- **Controls:** play, pause, step 1, 10 or 100 years, seed, world source, key parameters as sliders.
- **Side panels:** population per lineage over time, number of polities over time, a lineage tree snapshot, and an inspector for a clicked polity: culture vector, brain outputs, top symbols and their provenance.
- The SolaraViz default renderer may not draw hex polygons well. If so, build a custom Solara component with matplotlib or a lightweight SVG renderer. Verify what the installed Mesa offers before writing custom code.
- The UI only reads model state. It never changes model logic, and headless results must match UI results for the same seed.

**Data collection** (`data/collectors.py`)

| Table | Granularity | Contents |
| --- | --- | --- |
| `model.parquet` | every year | total population, number of polities, mean cohesion, wars active, sea level, climate anomaly |
| `lineages.parquet` | every 10 years | per lineage: population, territory, polities, techniques known |
| `polities.parquet` | every 25 years | per polity: lineage, parent, size, cohesion, culture vector, brain output means |
| `events.parquet` | every event | year, action, actors, place, magnitude |
| `memory_snapshots/` | every 500 years and at the end | each polity's full memory store with provenance |
| `brains/` | at polity birth | weights and parent id, for later analysis of what evolved |

Run metadata (config hash, seed, git commit, runtime) goes into `runs/<run_id>/meta.json`.

**Multi-run analysis**

- `scripts/batch.py` runs N seeds in parallel processes (multiprocessing, one run per core).
- Analysis questions to support from the start:
  - How often does each founding lineage end up largest, and what predicts it (start location, early contact with the teachers, climate luck)?
  - When does farming spread, how fast, and from where?
  - How long do polities last, and what is the distribution of their sizes?
  - Which memories survive longest, which actors get substituted in, and do the same structures recur across runs?
  - What do surviving brains converge on, compared with the founders' random brains?

## Build rounds

The build runs in 13 rounds (0 to 12), each ending with a runnable sim, passing tests, a local commit and a report; no round starts before the previous one is approved.

**Round 0. Scaffold**

- Environment first. Assume nothing is installed. Check the Python version (3.12+ required; if missing or older, stop and tell the user how to install it). Create a virtual environment in the repo (`uv venv` if uv is available, otherwise `python -m venv .venv`), declare all dependencies from section 3 in `pyproject.toml`, and install them into the venv (`uv pip install -e .[dev]` or `pip install -e .[dev]`). Optional extras (`numba`, `rasterio`, `xarray`) go in separate extras groups and are not installed until their round. Never install packages globally. On Windows, use the venv's `Scripts` paths and PowerShell-compatible commands. Record the exact installed versions in `docs/MESA_API_NOTES.md`.

* Deliverables: repo layout from section 3, `pyproject.toml`, config loader with frozen dataclasses, seeded RNG with child streams, an empty `DelugeModel` that steps 11,000 times, pytest setup.
* Verify: installed Mesa version, the exact names of `HexGrid`, `CellAgent`, `PropertyLayer`, `DataCollector`, `SolaraViz`. Record them in `docs/MESA_API_NOTES.md`.
* Accept: `run.py` completes 11,000 empty steps in under 5 seconds; two runs with the same seed produce identical output.

**Round 1. World generation**

- Deliverables: `WorldData`, procedural pipeline (section 5), sea-level and climate curves, carrying capacity, a static map renderer (PNG) per layer.
- Accept: generated maps show continents, mountain ranges, rivers flowing downhill to the sea, plausible biome bands; land share near target; sea level rise visibly floods coastal cells across years 0 to 3,000; tests for each pipeline stage with fixed seeds.

**Round 2. Browser UI skeleton**

- Deliverables: SolaraViz app showing the hex map with terrain and sea level, play and step controls, year counter.
- Accept: `solara run deluge/viz/app.py` opens in the browser and animates sea-level rise; UI and headless runs agree for the same seed.

**Round 3. Foragers and demography**

- Deliverables: settlements, foraging, growth, famine, migration, daughter settlements; the five founding groups placed; lineage registry.
- Accept: populations grow toward capacity and stabilize; bands spread across land over centuries; a drought visibly causes famine and migration; no settlement exists on a sea cell; total population never negative (property test).

**Round 4. Knowledge and the teachers**

- Deliverables: knowledge graph from config, invention, diffusion, loss; the teachers' contact phase (arrival, visits, teaching, departure); farming and herding modes.
- Accept: farming spreads outward from contact points; independent invention happens occasionally without the teachers; with the teachers disabled (config), the run still produces farming eventually but later; knowledge loss observed after a population crash; the teachers depart on schedule, never enter war, demography or politics, and leave no agent behind.

**Round 5. Polities and cohesion**

- Deliverables: polity agent, culture vector, appearance and dialect traits with drift and mixing (perception wired in round 7), tribute, cohesion and pressure equations, secession, fragmentation, mergers, the lineage tree.
- Accept: polities grow and split; polity count rises and falls over time rather than growing monotonically; every polity traces to one of the six founders; lineage tree export works.

**Round 6. War and conquest**

- Deliverables: war choice (rule-based placeholder until round 7), strength, resolution, raid and campaign, conquest, assimilation, grievance.
- Accept: conquest produces larger states in some runs; overextended states break up; terrain and distance measurably affect outcomes (a test with controlled scenarios).

**Round 7. Brains and neuroevolution**

- Deliverables: batched MLP, feature assembly, outputs wired into expansion, war, tribute, monuments, subsistence shift; inheritance and mutation on splits and mergers. Replace the round 6 placeholder.
- Accept: brain outputs vary across polities and shift over the run; brains of surviving lineages differ measurably from the founders' (weight distance and output behavior); a control run with frozen brains differs in outcome statistics across 20 seeds.

**Round 8. Memory and myth**

- Deliverables: event log, memory records, transmission operators, symbols, consequences, provenance, the no-vocabulary test, `analysis/myths.py`.
- Accept: memory stores stay within capacity; provenance chains resolve back to logged events for every memory; with consequence weights at zero, memory has no effect on outcomes (control test); a human-readable report shows each lineage's heaviest symbols and their origins.

**Round 9. Full UI**

- Deliverables: all map layers, overlays, side panels and the polity inspector from section 12.
- Accept: a full 11,000-year run can be watched and inspected in the browser at usable speed.

**Round 10. Data and batch runs**

- Deliverables: all Parquet tables, run metadata, checkpointing, `batch.py` with multiprocessing, the analysis scripts for the questions in section 12.
- Accept: 20 seeds run in parallel unattended; an analysis report is generated from their outputs.

**Round 11. Performance**

- Deliverables: profile a full run, vectorize or JIT the hottest paths.
- Accept: one full run of 11,000 years in under 10 minutes on a typical laptop CPU (target; revise after round 10 measurements).

**Round 12. Calibration and experiments**

- Deliverables: calibration against the targets in section 14; optional real-topography import; optional multi-run brain seeding; experiment configs (no teachers, harsher climate, modern-melt sea level).
- Accept: default runs meet the calibration targets; experiment results documented in `analysis/`.

## Calibration, performance and open questions

Calibration targets are broad patterns from the real Holocene record, used to tune parameters so the sim produces plausible dynamics, without forcing any specific outcome.

**Calibration targets** (approximate, from general knowledge of the archaeological record; tune ranges, not points)

| Pattern | Target in a default run |
| --- | --- |
| Farming adoption | first farming communities within 1,000 to 3,000 years where the teachers make contact; independent invention elsewhere later |
| Population growth | total population grows roughly 10 to 100 times over the run, with crashes after climate shocks |
| Polity size | most polities stay small; a few reach regional scale after several thousand years |
| Polity lifespan | median in centuries, a long tail beyond a millennium |
| Collapse | large polities fragment repeatedly; some regions see rise and fall cycles |
| Lineage survival | at least one founding lineage goes extinct in some runs; no lineage wins in every run |
| Selection material | hundreds to low thousands of polity births per run |

**Performance budget**

- Target: one 11,000-year run in under 10 minutes on a laptop CPU, a few hundred MB of RAM.
- Keep cell fields in NumPy arrays and brains batched. Avoid per-cell Python loops in yearly systems; loop over settlements (thousands at most), not cells.
- The UI may run slower than headless; it can render every N years.

**Open questions** (decide before or during the round named)

- Round 1: procedural world size and land share, or start straight with a real region?
- Round 3: where the five founding groups start, random land cells or chosen regions?
- Round 4: the teachers' contact phase: arrival year and place, length of stay, which groups they visit, and the artifact chance (default 30% per group)?
- Round 7: recurrent brain state on or off by default?
- Round 8: memory capacity `K` and the consequence weights, the main levers for whether memory matters at all.
- Round 12: run counterfactual experiments with real Earth topography, and which region?

**Reading list**

- Peter Turchin et al., "War, space, and the evolution of Old World complex societies," PNAS, 2013: an agent-based grid model of state formation over 3,000 years, the closest precedent for this sim. [paper](https://www.pnas.org/doi/10.1073/pnas.1308825110)
- Peter Turchin, *War and Peace and War* (2006): asabiya and the metaethnic frontier, in book form.
- Joshua Epstein and Robert Axtell, *Growing Artificial Societies* (1996): Sugarscape, the founding text of artificial societies. Mesa ships a Sugarscape example.
- Robert Axelrod, *The Evolution of Cooperation* (1984).
  - Ross Hammond and Robert Axelrod, "The Evolution of Ethnocentrism," *Journal of Conflict Resolution* (2006): an agent-based model in which in-group favoritism by arbitrary tags evolves without being programmed.
- Robert Boyd and Peter Richerson, *Culture and the Evolutionary Process* (1985): the theory behind cultural group selection and transmission biases.
- Mesa documentation, for the 3.x API used throughout. [docs](https://mesa.readthedocs.io/latest/overview.html)

## Sources

- [Mesa overview (discrete spaces, HexGrid, AgentSet, SolaraViz)](https://mesa.readthedocs.io/latest/overview.html)
- [Mesa Schelling example (CellAgent, discrete\_space, SolaraViz usage)](https://mesa.readthedocs.io/latest/examples/basic/schelling.html)
