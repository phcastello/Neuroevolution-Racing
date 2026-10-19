//! Integration checks against real forward activations, with a finite egui
//! viewport. Timing is reported for inspection, never used as a pass threshold.

use std::time::Instant;

use racing_rl::{
    backend::{Cpu, matrix, read},
    config::Config,
    network::Network,
};

use super::*;

fn viewport_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1000.0, 700.0),
        )),
        ..Default::default()
    }
}

fn canvas_frame(
    context: &egui::Context,
    trace: &PpoNetworkTrace,
    hidden: &[usize],
    parameters: &[Vec<f32>],
    settings: &mut PpoNetworkSettings,
    graph: &layout::NetworkLayout,
) -> Vec<edges::EdgeRenderStats> {
    // egui may request more than one layout pass. Retain each pass so a first
    // build is observable without requiring a particular egui pass count.
    let mut stats = Vec::new();
    let mut output = context.run_ui(viewport_input(), |ui| {
        stats.push(draw_canvas(ui, trace, hidden, parameters, settings, graph));
    });
    output.textures_delta.clear();
    assert!(!stats.is_empty());
    stats
}

fn exercise_real_forward(name: &str, hidden: &[usize], expected_connections: usize) {
    let started = Instant::now();
    let config = Config {
        hidden: hidden.to_vec(),
        ..Default::default()
    };
    let device = Default::default();
    let network = Network::<Cpu>::new(&config, &device).unwrap();
    let width = config.observation.dimension();
    assert_eq!(width, 25);
    let inputs: Vec<_> = (0..width)
        .map(|index| (index as f32 - 12.0) / 24.0)
        .collect();
    let input = matrix(&inputs, 1, width, &device).unwrap();
    let cache = network.forward(input.clone()).unwrap();
    let names: Vec<_> = config
        .observation
        .features()
        .into_iter()
        .map(|feature| feature.name)
        .collect();
    let trace = PpoNetworkTrace::from_forward(&cache, 0, &names).unwrap();
    let original_trace = trace.clone();
    let parameters = network.snapshot().unwrap();
    let original_parameters = parameters.clone();
    assert!(trace.matches_architecture(hidden, &parameters));
    assert_eq!(trace.input, inputs);
    for (values, layer) in trace.trunk.iter().zip(&cache.trunk) {
        assert_eq!(*values, read(layer.activation.clone()).unwrap());
    }
    let context = egui::Context::default();
    let mut settings = PpoNetworkSettings::default();

    for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
        settings.mode = mode;
        settings.max_nodes_per_layer = 12;
        settings.edge_budget = 360;
        settings.zoom = 1.0;
        settings.show_edges = true;
        let graph = layout::build_layout(width, hidden, mode, 12, 1.0);
        if mode == PpoNetworkMode::Expanded || hidden == [8, 4] {
            for layer in &graph.layers {
                assert_eq!(layer.nodes.len(), layer.width);
                assert!(layer.omission.is_none());
                assert_eq!(
                    layer.nodes.iter().map(|node| node.0).collect::<Vec<_>>(),
                    (0..layer.width).collect::<Vec<_>>()
                );
            }
        }

        let first_started = Instant::now();
        let first = canvas_frame(&context, &trace, hidden, &parameters, &mut settings, &graph);
        let first_elapsed = first_started.elapsed();
        assert!(first.iter().any(|stats| stats.geometry_rebuilt));
        assert!(first.iter().any(|stats| stats.view_rebuilt));
        let first_stats = *first.last().unwrap();
        assert_eq!(first_stats.total_connections, expected_connections);
        if mode == PpoNetworkMode::Expanded || hidden == [8, 4] {
            assert_eq!(first_stats.selected_connections, expected_connections);
        } else {
            assert!(first_stats.selected_connections > 0);
            assert!(first_stats.selected_connections <= settings.edge_budget);
        }
        assert!(first_stats.visible_connections > 0);
        assert!(first_stats.visible_connections <= first_stats.selected_connections);
        if mode == PpoNetworkMode::Expanded && hidden == [256, 256, 256, 256, 128] {
            assert!(graph.canvas_size.y > 700.0);
            assert!(first_stats.visible_connections < expected_connections);
        }

        let stationary_started = Instant::now();
        let stationary = canvas_frame(&context, &trace, hidden, &parameters, &mut settings, &graph);
        let stationary_elapsed = stationary_started.elapsed();
        assert!(
            stationary
                .iter()
                .all(|stats| !stats.geometry_rebuilt && !stats.view_rebuilt)
        );
        assert_eq!(
            stationary.last().unwrap().selected_connections,
            first_stats.selected_connections
        );
        assert_eq!(
            stationary.last().unwrap().visible_connections,
            first_stats.visible_connections
        );

        if mode == PpoNetworkMode::Expanded {
            // Neither of these condensed-only controls may truncate or rebuild
            // an expanded graph, even with an explicitly zero line budget.
            settings.max_nodes_per_layer = 32;
            settings.edge_budget = 0;
            let unchanged_graph = layout::build_layout(width, hidden, mode, 32, 1.0);
            assert_eq!(graph, unchanged_graph);
            for stats in canvas_frame(
                &context,
                &trace,
                hidden,
                &parameters,
                &mut settings,
                &unchanged_graph,
            ) {
                assert_eq!(stats.selected_connections, expected_connections);
                assert!(!stats.geometry_rebuilt && !stats.view_rebuilt);
            }
        }

        settings.zoom = 0.5;
        let zoomed = layout::build_layout(width, hidden, mode, settings.max_nodes_per_layer, 0.5);
        let zoom_stats = canvas_frame(
            &context,
            &trace,
            hidden,
            &parameters,
            &mut settings,
            &zoomed,
        );
        assert!(zoom_stats.iter().any(|stats| stats.geometry_rebuilt));
        if mode == PpoNetworkMode::Expanded {
            assert_eq!(
                zoom_stats.last().unwrap().selected_connections,
                expected_connections
            );
        }

        settings.show_edges = false;
        assert!(
            canvas_frame(
                &context,
                &trace,
                hidden,
                &parameters,
                &mut settings,
                &zoomed,
            )
            .iter()
            .all(|stats| *stats == edges::EdgeRenderStats::default())
        );
        settings.show_values = false;
        settings.show_omitted_count = false;
        // Also exercise the actual controls/ScrollArea entry point in the same
        // finite viewport, independently of the canvas-statistics checks.
        let mut output = context.run_ui(viewport_input(), |ui| {
            draw_ppo_network(ui, &trace, hidden, &parameters, &mut settings);
        });
        output.textures_delta.clear();
        assert_eq!(trace, original_trace);
        assert_eq!(parameters, original_parameters);
        eprintln!(
            "{name} {mode:?}: build={first_elapsed:?}, stationary={stationary_elapsed:?}, selected={}, visible={}",
            first_stats.selected_connections, first_stats.visible_connections
        );
    }

    let after = network.infer(input).unwrap();
    assert_eq!(trace.mean.as_slice(), read(after.mean).unwrap());
    assert_eq!(trace.value, read(after.value).unwrap()[0]);
    assert_eq!(trace.log_std, config.initial_log_std);
    assert_eq!(network.snapshot().unwrap(), original_parameters);
    assert_eq!(trace, original_trace);
    eprintln!("{name}: total integration check {:?}", started.elapsed());
}

#[test]
fn base_real_forward_renders_both_modes_without_mutating_policy() {
    exercise_real_forward("Base", &[64, 64], 5_888);
}

#[test]
fn stress_real_forward_renders_full_expanded_network_and_reuses_cache() {
    exercise_real_forward("Stress", &[256, 256, 256, 256, 128], 236_160);
}

#[test]
fn small_real_forward_needs_no_condensation_and_preserves_every_connection() {
    exercise_real_forward("Small", &[8, 4], 244);
}

fn append_painted_text(shape: &egui::Shape, output: &mut String) {
    match shape {
        egui::Shape::Text(text) => {
            output.push_str(&text.galley.job.text);
            output.push('\n');
        }
        egui::Shape::Vec(shapes) => {
            for child in shapes {
                append_painted_text(child, output);
            }
        }
        _ => {}
    }
}

#[test]
fn global_actor_parameter_tooltip_receives_hover_inside_the_canvas() {
    let config = Config {
        hidden: vec![64, 64],
        ..Default::default()
    };
    let device = Default::default();
    let network = Network::<Cpu>::new(&config, &device).unwrap();
    let width = config.observation.dimension();
    let input = matrix(&vec![0.5; width], 1, width, &device).unwrap();
    let trace =
        PpoNetworkTrace::from_forward(&network.forward(input.clone()).unwrap(), 0, &[]).unwrap();
    let original_trace = trace.clone();
    let parameters = network.snapshot().unwrap();
    let context = egui::Context::default();
    let mut settings = PpoNetworkSettings::default();
    let graph = layout::build_layout(
        width,
        &config.hidden,
        settings.mode,
        settings.max_nodes_per_layer,
        settings.zoom,
    );
    let mut pointer = None;
    let mut painted_text = String::new();
    // First register the canvas and its child hitboxes. Then move, without a
    // click: egui deliberately suppresses tooltips after click-and-rest until
    // the pointer moves again. Advancing time covers its normal 0.5 s delay.
    for frame in 0..5 {
        let mut raw_input = viewport_input();
        raw_input.time = Some(frame as f64);
        if frame == 1 {
            raw_input
                .events
                .push(egui::Event::PointerMoved(pointer.unwrap()));
        }
        let mut output = context.run_ui(raw_input, |ui| {
            pointer = Some(graph.log_std[0] + ui.next_widget_position().to_vec2());
            draw_canvas(
                ui,
                &trace,
                &config.hidden,
                &parameters,
                &mut settings,
                &graph,
            );
        });
        for shape in &output.shapes {
            append_painted_text(&shape.shape, &mut painted_text);
        }
        output.textures_delta.clear();
    }
    assert!(
        painted_text.contains("Actor · log_std steering (direção)"),
        "Global parameter hover did not produce its tooltip: {painted_text}"
    );
    assert!(painted_text.contains("log_std = -0.500000"));
    assert!(painted_text.contains("Desvio padrão σ = exp(log_std) = 0.606531"));
    assert!(painted_text.contains("independente da observação"));
    assert!(painted_text.contains("Não recebe conexões do tronco."));
    assert_eq!(trace, original_trace);
    assert_eq!(network.snapshot().unwrap(), parameters);
    let inferred = network.infer(input).unwrap();
    assert_eq!(trace.mean.as_slice(), read(inferred.mean).unwrap());
    assert_eq!(trace.value, read(inferred.value).unwrap()[0]);
}

/// Exercise the actual ScrollArea and pointer responses, rather than calling
/// the navigation arithmetic directly. A separate button exposes accidental
/// capture of controls outside the diagram.
struct NavigationHarness {
    context: egui::Context,
    trace: PpoNetworkTrace,
    hidden: Vec<usize>,
    parameters: Vec<Vec<f32>>,
    settings: PpoNetworkSettings,
    time: f64,
    in_window: bool,
}

struct NavigationFrame {
    viewport: egui::Rect,
    external_button: egui::Rect,
    graph: layout::NetworkLayout,
    text: String,
    stats: Vec<edges::EdgeRenderStats>,
    window: Option<egui::Rect>,
}

impl NavigationHarness {
    fn new(mode: PpoNetworkMode, zoom: f32) -> Self {
        Self::with_hidden(mode, zoom, &[64, 64])
    }

    fn with_hidden(mode: PpoNetworkMode, zoom: f32, hidden: &[usize]) -> Self {
        let config = Config {
            hidden: hidden.to_vec(),
            ..Default::default()
        };
        let device = Default::default();
        let network = Network::<Cpu>::new(&config, &device).unwrap();
        let input = matrix(&vec![0.5; config.observation.dimension()], 1, 25, &device).unwrap();
        let trace =
            PpoNetworkTrace::from_forward(&network.forward(input).unwrap(), 0, &[]).unwrap();
        Self {
            context: egui::Context::default(),
            trace,
            hidden: config.hidden,
            parameters: network.snapshot().unwrap(),
            settings: PpoNetworkSettings {
                mode,
                zoom,
                ..Default::default()
            },
            time: 0.0,
            in_window: false,
        }
    }

    fn origin(&self) -> egui::Vec2 {
        self.settings.views[usize::from(self.settings.mode == PpoNetworkMode::Expanded)].origin
    }

    fn frame(&mut self, events: Vec<egui::Event>) -> NavigationFrame {
        let mut input = viewport_input();
        input.time = Some(self.time);
        input.events = events;
        self.time += 1.0 / 60.0;
        let mut frame = None;
        let mut stats = Vec::new();
        let mut window = None;
        let mut output = self.context.run_ui(input, |ui| {
            let mut contents = |ui: &mut egui::Ui| {
                let button = ui.button("External dashboard control");
                let viewport = egui::Rect::from_min_size(
                    ui.next_widget_position(),
                    egui::vec2(ui.available_width(), 440.0),
                );
                let mut graph = layout::build_layout(
                    self.trace.input.len(),
                    &self.hidden,
                    self.settings.mode,
                    self.settings.max_nodes_per_layer,
                    self.settings.zoom,
                );
                stats.push(show_canvas(
                    ui,
                    &self.trace,
                    &self.hidden,
                    &self.parameters,
                    &mut self.settings,
                    &mut graph,
                    440.0,
                    false,
                ));
                frame = Some((viewport, button.rect, graph));
            };
            if self.in_window {
                window = egui::Window::new("Live PPO network")
                    .default_pos(egui::pos2(18.0, 18.0))
                    .default_size(egui::vec2(950.0, 620.0))
                    .show(ui.ctx(), &mut contents)
                    .map(|output| output.response.rect);
            } else {
                contents(ui);
            }
        });
        let mut text = String::new();
        for shape in &output.shapes {
            append_painted_text(&shape.shape, &mut text);
        }
        output.textures_delta.clear();
        let (viewport, external_button, graph) = frame.unwrap();
        NavigationFrame {
            viewport,
            external_button,
            graph,
            text,
            stats,
            window,
        }
    }

    fn prime(&mut self) -> NavigationFrame {
        self.frame(Vec::new());
        self.frame(Vec::new());
        self.frame(Vec::new());
        self.frame(Vec::new())
    }

    fn press(&mut self, point: egui::Pos2) {
        self.frame(vec![egui::Event::PointerMoved(point)]);
        self.frame(vec![primary_button(point, true)]);
    }

    fn release(&mut self, point: egui::Pos2) {
        self.frame(vec![primary_button(point, false)]);
    }
}

fn primary_button(pos: egui::Pos2, pressed: bool) -> egui::Event {
    egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

fn assert_point_delta(actual: egui::Vec2, expected: egui::Vec2) {
    assert!(
        (actual - expected).abs().max_elem() < 0.02,
        "Screen-space movement differed: actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn primary_drag_tracks_both_axes_without_zoom_scaling_or_release_drift() {
    for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
        for zoom in [0.001, 0.1, 1.0, 2.5] {
            let mut harness = NavigationHarness::new(mode, zoom);
            let frame = harness.prime();
            if zoom <= 0.1 {
                assert!(frame.graph.canvas_size.x < frame.viewport.width());
                assert!(frame.graph.canvas_size.y < frame.viewport.height());
            }
            // Prefer an actual visible neuron, so hover-only child widgets are
            // covered as well as the empty background used on the return drag.
            let point = frame
                .graph
                .layers
                .iter()
                .flat_map(|layer| &layer.nodes)
                .map(|node| node.1 + harness.origin() + frame.viewport.min.to_vec2())
                .find(|point| frame.viewport.shrink(60.0).contains(*point))
                .unwrap_or(frame.viewport.center());
            harness.press(point);
            let before = harness.origin();
            let delta = egui::vec2(37.0, -23.0);
            let moved = harness.frame(vec![egui::Event::PointerMoved(point + delta)]);
            assert_point_delta(harness.origin() - before, delta);
            assert!(moved.stats.iter().all(|stats| !stats.geometry_rebuilt));
            let second_delta = egui::vec2(-19.0, 31.0);
            harness.frame(vec![egui::Event::PointerMoved(
                point + delta + second_delta,
            )]);
            assert_point_delta(harness.origin() - before, delta + second_delta);
            let stopped = harness.origin();
            harness.release(point + delta + second_delta);
            for _ in 0..4 {
                harness.frame(Vec::new());
                assert_point_delta(harness.origin(), stopped);
            }
            assert_eq!(harness.settings.zoom, zoom);
        }
    }
}

#[test]
fn drag_wheel_and_scroll_bar_share_position_without_input_method_jumps() {
    let mut harness = NavigationHarness::new(PpoNetworkMode::Condensed, 1.0);
    let frame = harness.prime();
    let point = frame.viewport.center();
    harness.press(point);
    harness.frame(vec![egui::Event::PointerMoved(
        point + egui::vec2(25.0, 18.0),
    )]);
    harness.release(point + egui::vec2(25.0, 18.0));
    let before_wheel = harness.origin();
    harness.frame(vec![
        egui::Event::PointerMoved(point),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(-44.0, -66.0),
            phase: egui::TouchPhase::Move,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    // Let egui's normal wheel smoothing finish before beginning another drag.
    for _ in 0..90 {
        harness.frame(Vec::new());
    }
    let after_wheel = harness.origin();
    assert!(after_wheel.x < before_wheel.x - 10.0);
    assert!(after_wheel.y < before_wheel.y - 10.0);
    harness.press(point);
    let delta = egui::vec2(-27.0, 13.0);
    harness.frame(vec![egui::Event::PointerMoved(point + delta)]);
    assert_point_delta(harness.origin() - after_wheel, delta);
    harness.release(point + delta);

    // The floating horizontal scrollbar occupies the bottom ten points. Start
    // on its real handle, computed from the public virtual-content dimensions.
    let view = &harness.settings.views[0];
    let content_width = frame.graph.canvas_size.x + view.leading.x + view.trailing.x;
    let offset = view.leading.x - view.origin.x;
    let handle_center = frame.viewport.left()
        + (offset + frame.viewport.width() * 0.5) / content_width * frame.viewport.width();
    let bar_point = egui::pos2(handle_center, frame.viewport.bottom() - 5.0);
    harness.press(bar_point);
    let before_bar = harness.origin();
    harness.frame(vec![egui::Event::PointerMoved(
        bar_point + egui::vec2(35.0, 0.0),
    )]);
    let after_bar = harness.origin();
    assert!(
        after_bar.x < before_bar.x,
        "Scrollbar gesture was captured as canvas pan"
    );
    assert_point_delta(
        egui::vec2(0.0, after_bar.y - before_bar.y),
        egui::Vec2::ZERO,
    );
    harness.release(bar_point + egui::vec2(35.0, 0.0));
    harness.press(point);
    harness.frame(vec![egui::Event::PointerMoved(point + delta)]);
    assert_point_delta(harness.origin() - after_bar, delta);
}

#[test]
fn navigation_is_scoped_to_canvas_and_retained_independently_per_mode() {
    let mut harness = NavigationHarness::new(PpoNetworkMode::Condensed, 1.0);
    let frame = harness.prime();
    harness.press(frame.viewport.center());
    harness.frame(vec![egui::Event::PointerMoved(
        frame.viewport.center() + egui::vec2(51.0, -29.0),
    )]);
    harness.release(frame.viewport.center() + egui::vec2(51.0, -29.0));
    let condensed_origin = harness.origin();
    let button_point = frame.external_button.center();
    harness.press(button_point);
    harness.frame(vec![egui::Event::PointerMoved(
        button_point + egui::vec2(40.0, 0.0),
    )]);
    harness.release(button_point + egui::vec2(40.0, 0.0));
    assert_point_delta(harness.origin(), condensed_origin);

    harness.settings.mode = PpoNetworkMode::Expanded;
    let expanded = harness.prime();
    harness.press(expanded.viewport.center());
    harness.frame(vec![egui::Event::PointerMoved(
        expanded.viewport.center() + egui::vec2(-35.0, 47.0),
    )]);
    harness.release(expanded.viewport.center() + egui::vec2(-35.0, 47.0));
    let expanded_origin = harness.origin();
    harness.settings.mode = PpoNetworkMode::Condensed;
    harness.frame(Vec::new());
    assert_point_delta(harness.origin(), condensed_origin);
    harness.settings.mode = PpoNetworkMode::Expanded;
    harness.frame(Vec::new());
    assert_point_delta(harness.origin(), expanded_origin);
}

#[test]
fn global_parameter_tooltip_follows_canvas_after_drag_and_pointer_zoom() {
    for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
        let mut harness = NavigationHarness::new(mode, 1.0);
        let frame = harness.prime();
        let local = frame.graph.log_std[0].to_vec2();
        let target = frame.viewport.center() + egui::vec2(160.0, -25.0);
        let before = harness.origin();
        let delta = target - (frame.viewport.min + before + local);
        let start = frame.viewport.left_top() + egui::vec2(90.0, 60.0);
        harness.press(start);
        harness.frame(vec![egui::Event::PointerMoved(start + delta)]);
        assert_point_delta(harness.origin() - before, delta);
        harness.release(start + delta);

        harness.frame(vec![egui::Event::PointerMoved(target)]);
        let before_zoom = harness.origin();
        harness.frame(vec![egui::Event::Zoom(1.5)]);
        let anchor = target - frame.viewport.min;
        assert_point_delta(harness.origin(), anchor - (anchor - before_zoom) * 1.5);
        assert_eq!(harness.settings.zoom, 1.5);
        // The original parameter remains under the zoom anchor. Moving away
        // and back clears egui's intentional click-and-rest tooltip suppression.
        // Egui ignores motion within 0.1 seconds of the last click; waiting
        // after an early move would leave that intentional suppression active.
        harness.time += 0.2;
        harness.frame(vec![egui::Event::PointerMoved(
            target + egui::vec2(40.0, 0.0),
        )]);
        harness.frame(vec![egui::Event::PointerMoved(target)]);
        harness.time += 1.0;
        let mut hovered = harness.frame(Vec::new());
        // A newly opened tooltip Area has an invisible sizing pass on its
        // first frame. Keep the pointer still while egui measures and paints it.
        for _ in 0..3 {
            let next = harness.frame(Vec::new());
            hovered.text.push_str(&next.text);
        }
        assert!(
            hovered.text.contains("Actor · log_std steering (direção)"),
            "Tooltip failed after pan/zoom: {}",
            hovered.text
        );
        assert!(hovered.text.contains("log_std = -0.500000"));
        assert!(hovered.text.contains("Não recebe conexões do tronco."));
    }
}

#[test]
fn dragging_a_neuron_inside_movable_window_pans_without_moving_window() {
    for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
        for zoom in [0.1, 1.0, 2.5] {
            let mut harness = NavigationHarness::new(mode, zoom);
            harness.in_window = true;
            let frame = harness.prime();
            let original_window = frame.window.unwrap();
            let point = frame
                .graph
                .layers
                .iter()
                .flat_map(|layer| &layer.nodes)
                .map(|node| node.1 + harness.origin() + frame.viewport.min.to_vec2())
                .find(|point| frame.viewport.shrink(60.0).contains(*point))
                .expect("No visible neuron available for the window integration check");
            harness.press(point);
            let before = harness.origin();
            let delta = egui::vec2(83.0, -47.0);
            let moved = harness.frame(vec![egui::Event::PointerMoved(point + delta)]);
            assert_point_delta(harness.origin() - before, delta);
            assert_point_delta(
                moved.window.unwrap().min - original_window.min,
                egui::Vec2::ZERO,
            );
            harness.release(point + delta);
            assert_point_delta(harness.origin() - before, delta);
        }
    }
}

#[test]
fn coalesced_arrival_press_and_final_move_release_do_not_jump_or_lose_delta() {
    for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
        let mut harness = NavigationHarness::new(mode, 1.0);
        harness.in_window = true;
        let frame = harness.prime();
        let point = frame.viewport.center();
        // Mouse events can be coalesced between Bevy/egui frames. Entering the
        // graph must not treat movement before the actual press as dragging.
        harness.frame(vec![egui::Event::PointerMoved(
            frame.external_button.center(),
        )]);
        let before = harness.origin();
        harness.frame(vec![
            egui::Event::PointerMoved(point),
            primary_button(point, true),
        ]);
        assert_point_delta(harness.origin(), before);
        let first_delta = egui::vec2(43.0, -31.0);
        harness.frame(vec![egui::Event::PointerMoved(point + first_delta)]);
        assert_point_delta(harness.origin() - before, first_delta);
        let final_delta = egui::vec2(27.0, 19.0);
        let released_at = point + first_delta + final_delta;
        harness.frame(vec![
            egui::Event::PointerMoved(released_at),
            primary_button(released_at, false),
        ]);
        assert_point_delta(harness.origin() - before, first_delta + final_delta);
        harness.frame(Vec::new());
        assert_point_delta(harness.origin() - before, first_delta + final_delta);
    }
}

/// Use the complete production entry point and its real nested vertical
/// ScrollArea. Read the registered pan response instead of guessing where the
/// controls above the diagram place its viewport.
fn full_window_frame(
    harness: &mut NavigationHarness,
    events: Vec<egui::Event>,
) -> (egui::Rect, egui::Rect, egui::Rect) {
    let mut input = viewport_input();
    input.screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1402.0, 882.0),
    ));
    input.time = Some(harness.time);
    input.events = events;
    harness.time += 1.0 / 60.0;
    let context = harness.context.clone();
    let mut pan_rect = None;
    let mut window_rect = None;
    let mut external_button = None;
    let mut output = context.run_ui(input, |ui| {
        external_button = Some(
            egui::Area::new(egui::Id::new("external_dashboard_test_control"))
                .fixed_pos(egui::pos2(1100.0, 80.0))
                .constrain(false)
                .order(egui::Order::Foreground)
                .show(ui.ctx(), |ui| ui.button("Dashboard control"))
                .inner
                .rect,
        );
        window_rect = egui::Window::new("PPO Actor-Critic · live activations")
            .id(egui::Id::new("ppo_network_window"))
            .default_pos(egui::pos2(18.0, 54.0))
            .default_size(egui::vec2(980.0, 760.0))
            .resizable(true)
            .vscroll(true)
            .show(ui.ctx(), |ui| {
                ui.label("Rollout 0 · car #0 · last policy decision");
                let pan_id = ui.id().with("ppo_network_pan");
                draw_ppo_network(
                    ui,
                    &harness.trace,
                    &harness.hidden,
                    &harness.parameters,
                    &mut harness.settings,
                );
                pan_rect = ui.ctx().read_response(pan_id).map(|response| response.rect);
            })
            .map(|output| output.response.rect);
    });
    output.textures_delta.clear();
    (
        window_rect.unwrap(),
        pan_rect.unwrap(),
        external_button.unwrap(),
    )
}

#[test]
fn full_network_window_with_vertical_scroll_preserves_pan_for_base_and_small() {
    for hidden in [&[64, 64][..], &[8, 4][..]] {
        for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
            for zoom in [0.1, 1.0, 2.5] {
                let mut harness = NavigationHarness::with_hidden(mode, zoom, hidden);
                let mut frame = full_window_frame(&mut harness, Vec::new());
                for _ in 0..4 {
                    frame = full_window_frame(&mut harness, Vec::new());
                }
                let graph = layout::build_layout(25, hidden, mode, 12, zoom);
                let neuron = graph
                    .layers
                    .iter()
                    .flat_map(|layer| &layer.nodes)
                    .map(|node| node.1 + harness.origin() + frame.1.min.to_vec2())
                    .find(|point| frame.1.shrink(50.0).contains(*point))
                    .expect("Full network window has no visible neuron to drag");
                for (point, delta) in [
                    (neuron, egui::vec2(120.0, 80.0)),
                    (frame.1.center(), egui::vec2(-83.0, -47.0)),
                ] {
                    full_window_frame(&mut harness, vec![egui::Event::PointerMoved(point)]);
                    full_window_frame(&mut harness, vec![primary_button(point, true)]);
                    let before = harness.origin();
                    let moved = full_window_frame(
                        &mut harness,
                        vec![egui::Event::PointerMoved(point + delta)],
                    );
                    assert_point_delta(harness.origin() - before, delta);
                    assert_point_delta(moved.0.min - frame.0.min, egui::Vec2::ZERO);
                    full_window_frame(&mut harness, vec![primary_button(point + delta, false)]);
                    full_window_frame(&mut harness, Vec::new());
                    assert_point_delta(harness.origin() - before, delta);
                }
            }
        }
    }
}

#[test]
fn single_frame_full_drag_pans_but_click_bar_and_external_control_do_not() {
    for hidden in [&[64, 64][..], &[8, 4][..]] {
        for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
            let mut harness = NavigationHarness::with_hidden(mode, 1.0, hidden);
            let mut frame = full_window_frame(&mut harness, Vec::new());
            for _ in 0..4 {
                frame = full_window_frame(&mut harness, Vec::new());
            }
            let point = frame.1.center();
            assert!(
                !frame.1.intersects(frame.2),
                "The external test control overlaps the canvas"
            );
            full_window_frame(&mut harness, vec![egui::Event::PointerMoved(point)]);
            full_window_frame(&mut harness, Vec::new());
            let before = harness.origin();
            let delta = egui::vec2(120.0, 80.0);
            let moved = full_window_frame(
                &mut harness,
                vec![
                    primary_button(point, true),
                    egui::Event::PointerMoved(point + delta),
                    primary_button(point + delta, false),
                ],
            );
            assert_point_delta(harness.origin() - before, delta);
            assert_point_delta(moved.0.min - frame.0.min, egui::Vec2::ZERO);
            full_window_frame(&mut harness, Vec::new());
            assert_point_delta(harness.origin() - before, delta);

            // A normal click, with no displacement, cannot change navigation.
            full_window_frame(&mut harness, vec![egui::Event::PointerMoved(point)]);
            let after_drag = harness.origin();
            full_window_frame(
                &mut harness,
                vec![primary_button(point, true), primary_button(point, false)],
            );
            assert_point_delta(harness.origin(), after_drag);

            // Start on the scrollbar and leave it during the same-frame gesture.
            // The canvas must never claim that press as a fallback drag.
            let bar = egui::pos2(frame.1.center().x, frame.1.bottom() - 5.0);
            full_window_frame(&mut harness, vec![egui::Event::PointerMoved(bar)]);
            full_window_frame(
                &mut harness,
                vec![
                    primary_button(bar, true),
                    egui::Event::PointerMoved(bar + egui::vec2(35.0, -23.0)),
                    primary_button(bar + egui::vec2(35.0, -23.0), false),
                ],
            );
            assert_point_delta(harness.origin(), after_drag);

            let external = frame.2.center();
            assert!(!frame.1.contains(external));
            full_window_frame(&mut harness, vec![egui::Event::PointerMoved(external)]);
            full_window_frame(
                &mut harness,
                vec![
                    primary_button(external, true),
                    egui::Event::PointerMoved(external + egui::vec2(42.0, 29.0)),
                    primary_button(external + egui::vec2(42.0, 29.0), false),
                ],
            );
            assert_point_delta(harness.origin(), after_drag);
            assert_eq!(harness.settings.zoom, 1.0);
        }
    }
}
