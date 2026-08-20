# Neuroevolution Racing

Neuroevolution Racing is a university AI project that will evolve the weights and biases of car-controlling Multilayer Perceptrons (MLPs) with a manually implemented Genetic Algorithm (GA). This first iteration is the visual simulation playground around that future work.

> The neural network and genetic algorithm are implemented independently from the visualization stack. Bevy, Avian2D and egui are used only for simulation, collision detection, rendering and visualization.

## Architecture

This repository is a Cargo workspace with a one-way dependency boundary:

```text
crates/app  ──────>  crates/neuroevolution
 Bevy app              pure Rust AI boundary
```

- `crates/neuroevolution`: deliberately minimal pure-Rust library. It contains only the `neural` and `genetic` module boundaries and student TODOs. It does **not** depend on Bevy, Avian2D, egui, or an ML/GA library.
- `crates/app`: fixed-timestep simulation, sampled centerline track, Avian2D collisions and spatial queries, rendering, temporary controller, continuous progress tracking, and egui dashboard.

`CarController` receives an explicit six-value `CarObservation`: five local wall sensors and normalized speed. The temporary controller receives centerline look-ahead through a separate navigation context, exists only to exercise the simulation, and performs no learning.

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

## Camera controls

- Mouse wheel: smooth bounded zoom toward the world point under the cursor.
- `Q` / `E`: rotate the view counter-clockwise / clockwise.
- Middle- or right-mouse drag: pan the free camera.
- **Fit Track**: preserve the current rotation and fit the active track (or Open Field) to the simulation area.
- **Reset View**: restore zero rotation and fit the scene again.

The application automatically fits the view at startup, after a track or Test Drive environment change, and after a significant window, resolution, or fullscreen size change. Camera rotation affects only the view: track coordinates, cars, sensors, collisions, and progress remain in unchanged world coordinates. Test Drive additionally offers **Free** and **Follow Car** behavior; follow mode centers the manual car while preserving interactive zoom and camera rotation.

Camera input is suppressed while egui is using the corresponding pointer or keyboard input.

## Test Drive

Test Drive is a developer physics playground, not a training or genetic-algorithm mode. It always spawns exactly one manually controlled car and offers two environments:

- **Track** uses the currently selected data-driven track with the same road, walls, collisions, five sensors, dimensions, timestep, and continuous progress logic as simulated cars.
- **Open Field** provides a large empty area with a non-colliding square gizmo grid, emphasized major lines, highlighted axes, and an origin marker for spatial reference. It has no walls. With no obstacles in sensor range, all five wall sensor values naturally read `1.0`; track progress is not attached.

Keyboard controls are `W` = acceleration `+1`, `S` = acceleration `-1`, `A` = steering `+1` (left/counter-clockwise), `D` = steering `-1` (right/clockwise), and `R` = reset car. Opposing keys cancel to zero. The dashboard's Sliders input mode supplies intermediate values directly in `[-1, +1]`. Reset restores position, heading, speed, controls, sensor values, and track progress where applicable.

Vehicle speed has no hard maximum. Propulsion efficiency decreases continuously as speed magnitude grows, so holding acceleration can keep increasing speed but produces progressively smaller gains. With neutral acceleration, a constant coasting loss gradually brings the car toward rest without reversing it. Opposing acceleration retains the full configured rate for responsive braking. The observation's normalized speed uses an independent asymptotic curve in `[0, 1]`; this normalization does not clamp or otherwise alter physical velocity.

Both keyboard and sliders are only control sources. They write the same canonical component consumed by every vehicle:

```text
TemporaryController ─┐
ManualController ────┼──> CarControls { steering, acceleration }
future MLP ──────────┘                    │
                                          v
                              shared fixed-step vehicle physics
```

The human never modifies a transform or receives privileged movement mechanics. `CarControls` clamps both scalar channels to `[-1, +1]`; the one shared integration/collision system consumes it regardless of source. Therefore a future MLP that supplies the same two values from the same initial state and timestep receives identical vehicle-state evolution. The Test Drive dashboard displays the final component values actually reaching physics, the exact six-scalar `CarObservation`, raw speed/heading/position/progress, and the shared `SimulationConfig` values.

## Current status

Implemented infrastructure:

- a data-driven track library loaded from human-readable RON files;
- five training circuits and three architecturally held-out validation circuits;
- generic Catmull–Rom geometry generation from authored control points;
- a road ribbon and matching collision walls derived from the centerline and configurable width;
- deterministic runtime track replacement that despawns old cars, progress components, road visuals, mesh assets, and wall colliders before rebuilding;
- responsive rotation-aware camera fitting plus bounded cursor-centered zoom, Q/E rotation, middle-drag pan, reset, and Test Drive follow behavior;
- resizable window and queried monitor/video-mode fullscreen support, with graceful closest-mode/native fallbacks;
- ten arcade-like kinematic cars driven by a deterministic temporary controller through the canonical `CarControls` boundary;
- a one-car Test Drive mode with Track/Open Field environments, keyboard/analog actuator sources, reset, and live physics/observation telemetry;
- five Avian2D raycast sensors per car, explicitly filtered to track walls;
- continuous distance-along-track progress, normalized progress, and maximum reached distance;
- local centerline projection continuity and one-lap start/finish wrap handling;
- optional debug drawing for the generated centerline, dense sampled points, larger authored control points, and selected-car projection;
- fixed 60 Hz simulation behavior with pause and speed controls;
- training/champion mode groundwork and a reserved race mode;
- fitness history/plot component populated with explicitly labeled preview data;
- isolated static `6 → 8 → 2` network visualization placeholder;
- unit-tested RON parsing, malformed definitions, role filtering, all bundled track files, generated geometry and borders, arc-length projection/progress, runtime replacement, camera fitting, video-mode selection, observation shape, and controller angle helpers.

Intentionally **not implemented**: MLP layers, weights, feed-forward evaluation, training/backpropagation, genomes, selection, crossover, mutation, elitism, or any other GA/ML technique. Those academically relevant pieces remain marked for manual student implementation in `crates/neuroevolution`.

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

Selecting another dashboard entry updates the active definition in place. The simulation removes the old cars, progress state, colliders, road mesh, and wall visuals; generates a fresh `Track`; respawns the temporary-controller cars at its start; and lets the changed track bounds trigger a camera refit. This clean reset is the boundary the future multi-track evaluator can reuse, but no evaluation loop or fitness aggregation exists yet.

## Development checks

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
