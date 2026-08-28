# Neuroevolution Racing

Neuroevolution Racing is a university AI project that evolves the weights and biases of car-controlling Multilayer Perceptrons (MLPs) with a manually implemented Genetic Algorithm (GA). The application runs the population in a shared racing simulation and exposes live generation, fitness, telemetry, and current-leader network data in its dashboard.

> The neural network and genetic algorithm are implemented independently from the visualization stack. Bevy, Avian2D and egui are used only for simulation, collision detection, rendering and visualization.

## Architecture

This repository is a Cargo workspace with a one-way dependency boundary:

```text
crates/app  ──────>  crates/neuroevolution
 Bevy app              pure Rust AI boundary
```

- `crates/neuroevolution`: pure-Rust implementation of dense MLP inference, genomes, populations, tournament selection, uniform crossover, Gaussian mutation, and elitism. It does **not** depend on Bevy, Avian2D, egui, or an ML/GA library.
- `crates/app`: fixed-timestep simulation, sampled centerline tracks, Avian2D collisions and spatial queries, MLP-controlled cars, continuous progress evaluation, generation orchestration, rendering, and the egui dashboard.

`CarController` receives an explicit six-value `CarObservation`. Its stable MLP input order is:

```text
[0] left +60° sensor     [3] right -30° sensor
[1] left +30° sensor     [4] right -60° sensor
[2] front 0° sensor      [5] normalized speed
```

The stable network output order is `[0] steering, [1] acceleration`; a small app-side adapter maps that order to the existing `CarControls::new(acceleration, steering)` API. Each population member owns an MLP reconstructed from its genome and receives only the six observation values above.

## Technology

- Rust (edition 2024)
- Bevy 0.19
- Avian2D 0.7
- bevy_egui 0.42 / egui_plot 0.37
- serde + RON for simulation and track data

No machine-learning or genetic-algorithm framework is used.

## Run

On Ubuntu/WSL2, install the X11 keyboard runtime once:

```bash
sudo apt install libxkbcommon-x11-0
```

```bash
cargo run -p neuroevolution-racing-app
```

The first native build can take several minutes. Use the right dashboard to change tracks without restarting, choose the simulation or Test Drive mode, pause/resume, choose 1×, 2×, 10×, or 25× simulation speed, and switch between windowed, 1080p, 1440p, and native fullscreen display modes. The space bar toggles pause and F11 toggles native fullscreen/windowed mode.

The Linux build intentionally uses X11/XWayland. WSLg's native Wayland path can lose its Vulkan surface while a window is manually resized; Bevy 0.19.1 cannot recover when the driver consequently reports no presentation modes.

## Training and genetic algorithm

Training starts with a deterministic seeded population of 500 genomes. Each genome contains all 74 parameters of the `6 → 8 → 2` dense MLP: weights and biases for both layers. Every individual starts from the same position, heading, speed, observation, and controls. Its independent episode ends on lap completion, a real translational wall collision, lack of significant track progress for 2.5 seconds, or a 60-second safety timeout.

Each generation selects one reproducible three-track subset from the training suite and uses the same tracks for every individual. A track score combines progress since the episode's spawn point normalized by the remaining lap with normalized useful progress speed, subtracts a bounded collision penalty, and gives completed laps a bonus. Training fitness is the mean of those track scores. The evaluated generation's champion then runs once on a randomly selected held-out validation track; that validation score is recorded but never enters fitness or evolution. Only after validation does the application preserve elites, select parents by tournament, apply uniform crossover and Gaussian mutation, and increment the generation.

The default episode formula is `1.0 × normalized_progress + 0.20 × normalized_useful_speed + 0.25 if completed - 0.08 if collided`. The configuration requires the completion bonus to exceed the entire speed term, so every completed episode scores above every possible non-completed episode. Useful speed is new best track distance per elapsed second, clamped after division by the configurable `120 u/s` normalization scale; raw absolute car speed is never rewarded.

The dashboard displays:

- the current generation, phase, active/finished episode counts and safety timer;
- real best/average fitness history from completed generations;
- the live first-place individual and its current normalized progress;
- that leader's actual neural network, including every neuron activation, weight sign, and relative weight magnitude.

The selected car is reassigned continuously to whichever individual has reached the greatest forward distance in the current generation. Network activations are captured by a generic MLP forward-trace API and exposed through controller telemetry, keeping egui and visualization details out of the neural implementation. Completed generations still contribute best and average samples to the fitness plot.

## Camera controls

- Mouse wheel: smooth bounded zoom toward the world point under the cursor.
- `Q` / `E`: rotate the view counter-clockwise / clockwise.
- Middle- or right-mouse drag: pan the free camera.
- **Fit Track**: preserve the current rotation and fit the active track (or Open Field) to the simulation area.
- **Reset View**: restore zero rotation and fit the scene again.

The application automatically fits the view at startup, after a track or Test Drive environment change, and after a significant window, resolution, or fullscreen size change. Camera rotation affects only the view: track coordinates, cars, sensors, collisions, and progress remain in unchanged world coordinates. The camera offers **Free** and **Follow Leader** behavior during population simulation, switching targets whenever first place changes. Test Drive uses the same mechanism as **Follow Car** for the manual vehicle. Both follow modes preserve interactive zoom and camera rotation.

Camera input is suppressed while egui is using the corresponding pointer or keyboard input.

## Test Drive

Test Drive is a developer physics playground, not a training or genetic-algorithm mode. It always spawns exactly one manually controlled car and offers two environments:

- **Track** uses the currently selected data-driven track with the same road, walls, collisions, five sensors, dimensions, timestep, and continuous progress logic as simulated cars.
- **Open Field** provides a large empty area with a non-colliding square gizmo grid, emphasized major lines, highlighted axes, and an origin marker for spatial reference. It has no walls. With no obstacles in sensor range, all five wall sensor values naturally read `1.0`; track progress is not attached.

Keyboard controls are `W` = acceleration `+1`, `S` = acceleration `-1`, `A` = steering `+1` (left/counter-clockwise), `D` = steering `-1` (right/clockwise), and `R` = reset car. Opposing keys cancel to zero. The dashboard's Sliders input mode supplies intermediate values directly in `[-1, +1]`. Reset restores position, heading, speed, controls, sensor values, and track progress where applicable.

Vehicle speed has no hard maximum. Propulsion efficiency decreases continuously as speed magnitude grows, so holding acceleration can keep increasing speed but produces progressively smaller gains. With neutral acceleration, a constant coasting loss gradually brings the car toward rest without reversing it. Opposing acceleration retains the full configured rate for responsive braking. The physical world scale is `1 unit = 16.07142 cm`, and the dashboard shows speed in `u/s` with an optional `km/h` conversion. `1 unit/s = 0.57852 km/h`. The observation uses `normalized_speed = abs(v) / (abs(v) + scale)` with a default scale of `250.0`; this asymptotic mapping stays in `[0, 1]` and does not clamp or otherwise alter the unbounded physical velocity.

The five wall rays keep the fixed `+60°, +30°, 0°, -30°, -60°` angles and normalize hit distance so `0` means a wall at the origin and `1` means no wall within the default `750.0`-unit maximum range.

Steering requests an angular rate, but shared vehicle physics limits the achievable rate using lateral acceleration: `a_lateral = |v| * |omega|`, so `omega_max = a_lateral_max / |v|`. Normal turn rate remains dominant at low speed; at high speed the grip limit increases the minimum turning radius and makes braking necessary for tight corners. This is still a simplified kinematic model: the car moves along its heading, with no tire slip, drifting, or full vehicle dynamics.

Both keyboard and sliders are only control sources. They write the same canonical component consumed by every vehicle:

```text
MLP controller ──────┐
Manual controller ───┴──> CarControls { steering, acceleration }
                                          v
                              shared fixed-step vehicle physics
```

The human never modifies a transform or receives privileged movement mechanics. `CarControls` clamps both scalar channels to `[-1, +1]`; the one shared integration/collision system consumes it regardless of source. Therefore an MLP that supplies the same two values from the same initial state and timestep receives identical vehicle-state evolution. The Test Drive dashboard displays the final component values actually reaching physics, the exact six-scalar `CarObservation`, raw speed/heading/position/progress, and the shared `SimulationConfig` values.

## Car sprite assets

Cars render as child sprites centered on the physical vehicle transform. The
population deterministically cycles through `car_01` through `car_05` and
`car_07` through `car_11`. The defective `car_06` was removed and `car_03`
takes its former population slot; `car_12` remains Test Drive-only. Every
visual is normalized to fill the same 28x15 physical footprint, collision
query, sensor origin, controls, and progress behavior.

All non-manual population members spawn on top of one another from the same
track-derived position and heading, with identical speed, controls,
observation, and progress state. Future intentional start perturbations may
vary lateral offset, heading, or speed, but each compared individual must
receive the same perturbation within an evaluation.

The 11 ready-to-load RGBA assets under `assets/cars` are generated once from
the supplied source sheet, never cropped at runtime:

```bash
python -m pip install Pillow
python tools/extract_car_sprites.py
python tools/extract_car_sprites.py --check
```

Each PNG has a 280x150 transparent canvas, fills the physical footprint, and
faces +X/right. The
generated contact sheet is inspection-only and is not loaded by the game. See
[`assets/cars/README.md`](assets/cars/README.md) for the source/license note;
the supplied stock-image-derived artwork should be treated as a development
placeholder unless it is appropriately licensed for distribution.

## Current status

Implemented infrastructure:

- a data-driven track library loaded from human-readable RON files;
- five training circuits and three architecturally held-out validation circuits;
- generic Catmull–Rom geometry generation from authored control points;
- a road ribbon and matching collision walls derived from the centerline and configurable width;
- deterministic runtime track replacement that despawns old cars, progress components, road visuals, mesh assets, and wall colliders before rebuilding;
- responsive rotation-aware camera fitting plus bounded cursor-centered zoom, Q/E rotation, middle-drag pan, reset, manual-car follow, and live leader follow;
- resizable window and queried monitor/video-mode fullscreen support, with graceful closest-mode/native fallbacks;
- a configurable population of arcade-like kinematic cars driven by genome-backed MLP controllers through the canonical `CarControls` boundary;
- eleven normalized transparent car sprite assets, with a deterministic ten-car training set and cosmetic Test Drive selection;
- a one-car Test Drive mode with Track/Open Field environments, keyboard/analog actuator sources, reset, and live physics/observation telemetry;
- five Avian2D raycast sensors per car, explicitly filtered to track walls;
- continuous distance-along-track progress, normalized progress, and maximum reached distance;
- local centerline projection continuity and one-lap start/finish wrap handling;
- optional debug drawing for the generated centerline, dense sampled points, larger authored control points, and selected-car projection;
- fixed 60 Hz simulation behavior with pause and speed controls;
- independent episode termination, multi-track training fitness, held-out champion validation, and automatic GA evolution in Training mode;
- real generation counter and best/average fitness history;
- live visualization of the current generation leader's `6 → 8 → 2` network using its actual weights and per-inference neuron activations;
- Champion mode groundwork and a reserved Race mode;
- unit-tested RON parsing, malformed definitions, role filtering, all bundled track files, generated geometry and borders, arc-length projection/progress, runtime replacement, camera fitting, video-mode selection, and observation shape.

Intentionally **not implemented**: gradient-based training/backpropagation, persistence/checkpoint loading, a completed Champion showcase, and Race mode. Learning is performed exclusively through the manually implemented genetic algorithm.

## Data-driven tracks

Track definitions live in [`assets/tracks`](assets/tracks). A definition contains generic metadata and geometry input: `id`, display `name`, optional `country`, generic `category`, `role`, `difficulty`, constant `width`, `samples_per_control_point`, `start_index`, and a list of planar `(x, y)` control points. Optional description and approximation notes are displayed or retained with the definition. There are no circuit-specific geometry branches in Rust.

At startup, `TrackLibrary` scans every track RON file and loads the explicit [`training.ron`](assets/tracks/training.ron) and [`validation.ron`](assets/tracks/validation.ron) suites. Every definition must appear exactly once in the suite matching its declared role. `training_tracks()` therefore cannot return a validation definition, and `validation_tracks()` cannot return a training definition.

| Track | Role | Difficulty |
|---|---|---|
| Monza | Training | Easy |
| Red Bull Ring | Training | Easy / Medium |
| Interlagos | Training | Medium |
| Barcelona-Catalunya | Training | Medium |
| Silverstone | Training | Medium / Hard |
| Spa-Francorchamps | Validation | Hard |
| Suzuka | Validation | Hard |
| Monaco | Validation | Very Hard |

These are project-owned, simplified 2D coordinate approximations designed as environments for the AI experiment. Their plan-view proportions were refined against the public [CC0 F1 Track Layouts SVG reference set](https://github.com/MasterPlay007/F1-Track-Layouts-SVG); no source SVG, raster map, or logo is shipped as an application asset. They are not GPS-, survey-, width-, elevation-, banking-, or FIA-accurate recreations. Monaco is intentionally narrower but still navigable by the rectangular car. Suzuka's real grade-separated crossover cannot exist in this planar simulator, so its crossing approaches reconnect as deliberately offset at-grade bends: the result evokes the figure-eight while keeping centerline, borders, collision, and progress unambiguous.

### Adding or editing a track

1. Copy an existing circuit RON file in `assets/tracks` and choose a unique file name and matching `id`.
2. Author at least four finite control points around one closed lap. `start_index` chooses which authored point becomes sample zero/start-finish.
3. Choose a positive width and sampling count, then add the id to exactly one suite file.
4. Run `cargo test --workspace`. The all-assets test generates every spline and rejects near-zero length, centerline self-intersection, either border self-intersection, or left/right border intersection.
5. Run the app with Track Debug enabled. Small yellow dots are generated samples; large orange/yellow rings are authored control points.

The runtime `Track` remains separate from `TrackDefinition`. For each adjacent quartet of closed-loop control points it evaluates the Catmull–Rom spline, samples the centerline, computes tangents and cumulative arc length, and offsets the tangents to form left/right borders. Projection and normalized progress operate only on this generated data and never need to know the active circuit id.

Outside Training mode, selecting another dashboard entry updates the active definition in place. The simulation removes the old cars, progress state, colliders, road mesh, and wall visuals; generates a fresh `Track`; respawns controllers from the relevant genomes; and lets the changed track bounds trigger a camera refit. During Training, the explicit training/validation cycle owns track selection so a manual switch cannot leak held-out data into evolution.

## Development checks

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
