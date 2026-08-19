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
- `crates/app`: fixed-timestep simulation, hard-coded track, Avian2D collisions and spatial queries, rendering, temporary controller, checkpoint tracking, and egui dashboard.

The temporary checkpoint-following controller is isolated behind `CarController` in the app. It exists only to exercise the simulation and performs no learning.

## Technology

- Rust (edition 2024)
- Bevy 0.19
- Avian2D 0.7
- bevy_egui 0.42 / egui_plot 0.37
- serde for simulation configuration types

No machine-learning or genetic-algorithm framework is used.

## Run

```bash
cargo run -p neuroevolution-racing-app
```

The first native build can take several minutes. Use the right dashboard to pause/resume and choose 1×, 2×, 10×, or 25× simulation speed. The space bar also toggles pause.

## Current status

Implemented infrastructure:

- one geometric closed circuit with static wall collision;
- ten arcade-like kinematic cars driven by a deterministic temporary controller;
- five Avian2D raycast sensors per car and debug rays for the selected car;
- ordered checkpoints, completed-checkpoint count, lap count, and fractional progress toward the next checkpoint;
- fixed 60 Hz simulation behavior with pause and speed controls;
- training/champion mode groundwork and a reserved race mode;
- fitness history/plot component populated with explicitly labeled preview data;
- isolated static `6 → 8 → 2` network visualization placeholder;
- unit-tested checkpoint crossing and controller angle helpers.

Intentionally **not implemented**: MLP layers, weights, feed-forward evaluation, training/backpropagation, genomes, selection, crossover, mutation, elitism, or any other GA/ML technique. Those academically relevant pieces remain marked for manual student implementation in `crates/neuroevolution`.

## Development checks

```bash
cargo fmt --all --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
