//! Presentation of the real PPO forward cache. Sampling is exclusively visual:
//! every activation and both heads have already been computed by `Network`.
use bevy_egui::egui;
use racing_rl::{
    Result,
    backend::{Backend, Tensor, read},
    ensure,
    network::ForwardCache,
};

use super::network_view::activation_fill;

mod edges;
mod layout;
#[cfg(test)]
mod presentation_tests;
use layout::PpoNetworkMode;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PpoNetworkTrace {
    pub input: Vec<f32>,
    pub input_names: Vec<String>,
    pub trunk: Vec<Vec<f32>>,
    pub mean: [f32; 2],
    pub value: f32,
    pub log_std: [f32; 2],
    pub version: u64,
}

impl PpoNetworkTrace {
    /// Capture one selected car from the exact batched forward that supplied its
    /// action. No diagnostic forward or separately implemented neural network.
    pub(crate) fn from_forward<B: Backend>(
        cache: &ForwardCache<B>,
        row: usize,
        input_names: &[String],
    ) -> Result<Self> {
        let batch = validate_forward(cache, input_names)?;
        ensure(row < batch, "PPO activation trace row")?;
        let input = tensor_row(forward_input(cache), row)?;
        let mean = tensor_row(&cache.mean, row)?;
        let log_std = tensor_row(&cache.log_std, 0)?;
        Ok(Self {
            input_names: observation_names(input.len(), input_names),
            input,
            trunk: cache
                .trunk
                .iter()
                .map(|layer| tensor_row(&layer.activation, row))
                .collect::<Result<_>>()?,
            mean: [mean[0], mean[1]],
            value: tensor_row(&cache.value, row)?[0],
            log_std: [log_std[0], log_std[1]],
            version: cache.version,
        })
    }

    /// Capture an entire car batch with one download per tensor, rather than
    /// repeatedly downloading every car's activations for each individual row.
    pub(crate) fn batch_from_forward<B: Backend>(
        cache: &ForwardCache<B>,
        input_names: &[String],
    ) -> Result<Vec<Self>> {
        let batch = validate_forward(cache, input_names)?;
        if batch == 1 {
            return Ok(vec![Self::from_forward(cache, 0, input_names)?]);
        }
        let input_tensor = forward_input(cache);
        let input_width = input_tensor.dims()[1];
        let inputs = read(input_tensor.clone())?;
        let trunk = cache
            .trunk
            .iter()
            .map(|layer| Ok((layer.activation.dims()[1], read(layer.activation.clone())?)))
            .collect::<Result<Vec<_>>>()?;
        let means = read(cache.mean.clone())?;
        let values = read(cache.value.clone())?;
        let log_std = read(cache.log_std.clone())?;
        let names = observation_names(input_width, input_names);
        Ok((0..batch)
            .map(|row| Self {
                input: inputs[row * input_width..(row + 1) * input_width].to_vec(),
                input_names: names.clone(),
                trunk: trunk
                    .iter()
                    .map(|(width, values)| values[row * width..(row + 1) * width].to_vec())
                    .collect(),
                mean: [means[row * 2], means[row * 2 + 1]],
                value: values[row],
                log_std: [log_std[0], log_std[1]],
                version: cache.version,
            })
            .collect())
    }

    fn matches_architecture(&self, hidden: &[usize], parameters: &[Vec<f32>]) -> bool {
        if self.input.is_empty()
            || self.input_names.len() != self.input.len()
            || self.trunk.len() != hidden.len()
            || hidden.contains(&0)
            || self
                .trunk
                .iter()
                .zip(hidden)
                .any(|(values, &width)| values.len() != width)
            || parameters.len() != 2 * (hidden.len() + 2) + 1
        {
            return false;
        }
        let mut input = self.input.len();
        for (layer, output) in hidden.iter().copied().chain([2, 1]).enumerate() {
            // The two heads share the final trunk input; Critic is not fed Actor.
            if layer == hidden.len() + 1 {
                input = hidden.last().copied().unwrap_or(self.input.len());
            }
            if input.checked_mul(output) != Some(parameters[2 * layer].len())
                || parameters[2 * layer + 1].len() != output
            {
                return false;
            }
            input = output;
        }
        parameters.last().is_some_and(|values| values.len() == 2)
    }
}

fn forward_input<B: Backend>(cache: &ForwardCache<B>) -> &Tensor<B, 2> {
    cache
        .trunk
        .first()
        .map_or(&cache.head_input, |layer| &layer.input)
}

fn validate_forward<B: Backend>(cache: &ForwardCache<B>, input_names: &[String]) -> Result<usize> {
    let [batch, actions] = cache.mean.dims();
    ensure(
        batch > 0 && actions == 2 && cache.value.dims() == [batch, 1],
        "PPO activation trace head dimensions",
    )?;
    let input = forward_input(cache);
    ensure(
        input.dims()[0] == batch
            && cache
                .trunk
                .iter()
                .all(|layer| layer.activation.dims()[0] == batch)
            && cache.log_std.dims() == [1, 2],
        "PPO activation trace batch dimensions",
    )?;
    ensure(
        input_names.is_empty() || input_names.len() == input.dims()[1],
        "PPO activation trace observation metadata",
    )?;
    Ok(batch)
}

fn observation_names(width: usize, names: &[String]) -> Vec<String> {
    if names.is_empty() {
        (0..width).map(|i| format!("input_{i}")).collect()
    } else {
        names.to_vec()
    }
}

fn tensor_row<B: Backend>(tensor: &Tensor<B, 2>, row: usize) -> Result<Vec<f32>> {
    let [rows, width] = tensor.dims();
    ensure(row < rows, "PPO activation trace tensor row")?;
    read(tensor.clone().slice([row..row + 1, 0..width]))
}

#[derive(Clone, Debug)]
pub(crate) struct PpoNetworkSettings {
    pub mode: PpoNetworkMode,
    /// Density of the largest layer in Condensed mode only.
    pub max_nodes_per_layer: usize,
    pub edge_budget: usize,
    pub show_edges: bool,
    pub show_values: bool,
    pub show_omitted_count: bool,
    pub zoom: f32,
    views: [NetworkView; 2],
    edges: edges::EdgeCache,
    inspected_block: usize,
    inspected_source: usize,
    inspected_target: usize,
}

impl Default for PpoNetworkSettings {
    fn default() -> Self {
        Self {
            mode: PpoNetworkMode::Condensed,
            max_nodes_per_layer: 12,
            edge_budget: 360,
            show_edges: true,
            show_values: true,
            show_omitted_count: true,
            zoom: 0.9,
            views: Default::default(),
            edges: edges::EdgeCache::default(),
            inspected_block: 0,
            inspected_source: 0,
            inspected_target: 0,
        }
    }
}

/// Screen-space origin shared by dragging, the wheel and the scroll bars.
/// Margins grow only when dragging/zooming needs more room; ordinary scrolling
/// never changes their extent or fights a moving scroll-bar handle.
#[derive(Clone, Debug, Default)]
struct NetworkView {
    origin: egui::Vec2,
    leading: egui::Vec2,
    trailing: egui::Vec2,
    zoom: f32,
    unscaled_size: egui::Vec2,
    viewport: egui::Vec2,
    drag_pointer: Option<egui::Pos2>,
    finished_drag_frame: Option<u64>,
}

impl NetworkView {
    fn frame(&mut self, size: egui::Vec2, viewport: egui::Vec2, zoom: f32, reset: bool) {
        let unscaled_size = size / zoom;
        let changed = reset
            || self.zoom != zoom
            || (self.unscaled_size - unscaled_size).abs().max_elem() > 0.1
            || self.viewport != viewport;
        if reset || self.zoom == 0.0 || (self.unscaled_size - unscaled_size).abs().max_elem() > 0.1
        {
            self.origin = (viewport - size) * 0.5;
            self.leading = viewport;
            self.trailing = viewport;
        } else if self.zoom != zoom {
            self.zoom_at(zoom, viewport * 0.5);
        }
        self.zoom = zoom;
        self.unscaled_size = unscaled_size;
        self.viewport = viewport;
        if changed {
            self.make_room(viewport);
        }
    }

    fn zoom_at(&mut self, zoom: f32, anchor: egui::Vec2) {
        self.origin = anchor - (anchor - self.origin) * (zoom / self.zoom);
        self.zoom = zoom;
    }

    fn make_room(&mut self, viewport: egui::Vec2) {
        self.leading = self.leading.max(self.origin + viewport);
        self.trailing = self.trailing.max(viewport - self.origin);
    }
}

pub(crate) fn draw_ppo_network(
    ui: &mut egui::Ui,
    trace: &PpoNetworkTrace,
    hidden: &[usize],
    parameters: &[Vec<f32>],
    settings: &mut PpoNetworkSettings,
) {
    if !trace.matches_architecture(hidden, parameters) {
        ui.colored_label(
            egui::Color32::LIGHT_RED,
            "Ativações e parâmetros não correspondem à arquitetura PPO carregada.",
        );
        return;
    }
    ui.small(format!(
        "Última decisão · versão {} · tronco ReLU → Actor / Critic",
        trace.version
    ));
    ui.horizontal(|ui| {
        ui.selectable_value(&mut settings.mode, PpoNetworkMode::Condensed, "Condensada");
        ui.selectable_value(&mut settings.mode, PpoNetworkMode::Expanded, "Expandida");
        ui.separator();
        ui.checkbox(&mut settings.show_edges, "Conexões");
    });
    let mut fit = false;
    ui.horizontal(|ui| {
        ui.add(
            egui::Slider::new(&mut settings.zoom, 0.001..=2.5)
                .logarithmic(true)
                .text("Zoom"),
        );
        if ui.small_button("Enquadrar").clicked() {
            fit = true;
        }
        if ui.small_button("100%").clicked() {
            settings.zoom = 1.0;
        }
    });
    egui::CollapsingHeader::new("Detalhes da visualização").id_salt("ppo_network_details").show(ui, |ui| {
        if settings.mode == PpoNetworkMode::Condensed {
            ui.add(egui::Slider::new(&mut settings.max_nodes_per_layer, 8..=32).text("Densidade proporcional"));
            ui.checkbox(&mut settings.show_omitted_count, "Quantidade de neurônios omitidos");
            ui.add_enabled(settings.show_edges, egui::Slider::new(&mut settings.edge_budget, 0..=2000).text("Conexões na seleção"));
        } else {
            ui.small("Todos os neurônios e todas as conexões reais estão disponíveis, sem amostragem. O desenho descarta somente segmentos fora da área visível.");
        }
        ui.checkbox(&mut settings.show_values, "Valores nas entradas e no tronco");
        ui.small("As saídas e os parâmetros do Actor sempre mostram seus valores. Estes controles afetam somente a apresentação.");
    });
    settings.max_nodes_per_layer = settings.max_nodes_per_layer.clamp(8, 32);
    settings.zoom = settings.zoom.clamp(0.001, 2.5);
    let graph_height = (ui.available_height() - 118.0).clamp(200.0, 800.0);
    if fit {
        let full = layout::build_layout(
            trace.input.len(),
            hidden,
            settings.mode,
            settings.max_nodes_per_layer,
            1.0,
        );
        settings.zoom = ((ui.available_width() - 16.0) / full.canvas_size.x)
            .min(graph_height / full.canvas_size.y)
            .clamp(0.001, 2.5);
    }
    let mut graph = layout::build_layout(
        trace.input.len(),
        hidden,
        settings.mode,
        settings.max_nodes_per_layer,
        settings.zoom,
    );
    let shown: usize = graph.layers.iter().map(|layer| layer.nodes.len()).sum();
    let total: usize = graph.layers.iter().map(|layer| layer.width).sum();
    ui.small(format!("{shown}/{total} neurônios + 2 parâmetros globais · verde + / vermelho − · passe o mouse para inspecionar"));
    if graph.scale < 0.6 {
        ui.small("Aproxime para ler os rótulos e valores; todos os neurônios e pesos continuam disponíveis.");
    }
    let stats = show_canvas(
        ui,
        trace,
        hidden,
        parameters,
        settings,
        &mut graph,
        graph_height,
        fit,
    );
    if settings.show_edges {
        ui.small(format!(
            "Conexões: {} / {} representadas · {} na área visível",
            stats.selected_connections, stats.total_connections, stats.visible_connections
        ));
    } else {
        ui.small("Conexões ocultas · arraste com o botão esquerdo ou use as barras para navegar.");
    }
    draw_connection_inspector(ui, trace, hidden, parameters, settings);
    egui::CollapsingHeader::new("Todas as entradas normalizadas da última decisão")
        .id_salt("ppo_network_inputs")
        .show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("ppo_network_input_values")
                .max_height(180.0)
                .show(ui, |ui| {
                    egui::Grid::new("ppo_normalized_inputs")
                        .striped(true)
                        .show(ui, |ui| {
                            for (index, (name, value)) in
                                trace.input_names.iter().zip(&trace.input).enumerate()
                            {
                                ui.monospace(format!("{index:02}"));
                                ui.label(name);
                                ui.monospace(format!("{value:+.5}"));
                                ui.end_row();
                            }
                        });
                });
        });
}

#[allow(clippy::too_many_arguments)]
fn show_canvas(
    ui: &mut egui::Ui,
    trace: &PpoNetworkTrace,
    hidden: &[usize],
    parameters: &[Vec<f32>],
    settings: &mut PpoNetworkSettings,
    graph: &mut layout::NetworkLayout,
    height: f32,
    reset: bool,
) -> edges::EdgeRenderStats {
    let mode_index = usize::from(settings.mode == PpoNetworkMode::Expanded);
    let mut view = settings.views[mode_index].clone();
    let rect = egui::Rect::from_min_size(
        ui.next_widget_position(),
        egui::vec2(ui.available_width(), height),
    );
    // Register the viewport before its children. Hover-only neuron hitboxes do
    // not steal dragging, while scroll bars and other controls retain priority.
    let pan = ui.interact(
        rect.intersect(ui.clip_rect()),
        ui.id().with("ppo_network_pan"),
        egui::Sense::drag(),
    );
    view.frame(graph.canvas_size, rect.size(), graph.scale, reset);
    let frame = ui.ctx().cumulative_frame_nr();
    let dragging = pan.dragged_by(egui::PointerButton::Primary);
    if (dragging || pan.drag_stopped_by(egui::PointerButton::Primary))
        && view.finished_drag_frame != Some(frame)
    {
        let (pointer, press) =
            ui.input(|input| (input.pointer.interact_pos(), input.pointer.press_origin()));
        // A move into the viewport and the press may arrive in the same frame.
        // Start at the press position, not at last frame's unrelated cursor.
        // Remembering the position also makes egui's extra layout passes safe.
        if let (Some(pointer), Some(previous)) = (pointer, view.drag_pointer.or(press)) {
            view.origin += pointer - previous;
        }
        view.drag_pointer = pointer.filter(|_| dragging);
        if !dragging {
            view.finished_drag_frame = Some(frame);
        }
        view.make_room(rect.size());
        ui.set_cursor_icon(egui::CursorIcon::Grabbing);
    } else if pan.hovered() {
        view.drag_pointer = None;
        ui.set_cursor_icon(egui::CursorIcon::Grab);
    } else {
        view.drag_pointer = None;
    }
    // egui intentionally does not report a drag when press and release arrive
    // in one frame. Retain that short gesture too, without handling it twice
    // or capturing buttons/scroll bars outside the diagram's interaction area.
    if !dragging && !pan.drag_stopped() && view.finished_drag_frame != Some(frame) {
        let quick = ui.input(|input| {
            let mut start = None;
            for event in &input.events {
                if let egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    ..
                } = event
                {
                    if *pressed {
                        start = Some(*pos);
                    } else if let Some(start) = start {
                        return Some((start, *pos));
                    }
                }
            }
            None
        });
        if let Some((start, end)) = quick {
            let style = ui.spacing().scroll;
            let inset = egui::Vec2::splat(
                style
                    .allocated_width()
                    .max(style.bar_width + style.bar_inner_margin + style.bar_outer_margin),
            );
            let visible = rect.intersect(ui.clip_rect());
            let area = egui::Rect::from_min_max(visible.min, visible.max - inset);
            if area.contains(start) && ui.ctx().layer_id_at(start) == Some(ui.layer_id()) {
                view.origin += end - start;
                view.make_room(rect.size());
                view.finished_drag_frame = Some(frame);
            }
        }
    }
    if pan.hovered() && !pan.dragged() {
        let factor = ui.input(|input| input.zoom_delta());
        if factor != 1.0 {
            let zoom = (settings.zoom * factor).clamp(0.001, 2.5);
            let anchor = ui.input(|input| input.pointer.hover_pos()).unwrap() - rect.min;
            view.zoom_at(zoom, anchor);
            view.make_room(rect.size());
            settings.zoom = zoom;
            *graph = layout::build_layout(
                trace.input.len(),
                hidden,
                settings.mode,
                settings.max_nodes_per_layer,
                zoom,
            );
            // Ctrl+wheel/pinch is zoom here; a plain wheel remains scrolling.
            ui.input_mut(|input| input.smooth_scroll_delta = egui::Vec2::ZERO);
        }
    }
    let scroll = egui::ScrollArea::both()
        .id_salt(("ppo_network_graph", settings.mode))
        .scroll_source(
            egui::scroll_area::ScrollSource::SCROLL_BAR
                | egui::scroll_area::ScrollSource::MOUSE_WHEEL,
        )
        .auto_shrink([false, false])
        .max_height(height)
        .min_scrolled_height(180.0)
        .scroll_offset(view.leading - view.origin)
        .show_viewport(ui, |ui, _| {
            let size = graph.canvas_size + view.leading + view.trailing;
            let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
            let offset = response.rect.min.to_vec2() + view.leading;
            paint_canvas(
                ui, trace, hidden, parameters, settings, graph, response, painter, offset,
            )
        });
    // This includes wheel/bar changes made by egui after painting. Next frame
    // starts from exactly this origin, so switching input methods cannot jump.
    view.origin = view.leading - scroll.state.offset;
    settings.views[mode_index] = view;
    scroll.inner
}

#[cfg(test)]
fn draw_canvas(
    ui: &mut egui::Ui,
    trace: &PpoNetworkTrace,
    hidden: &[usize],
    parameters: &[Vec<f32>],
    settings: &mut PpoNetworkSettings,
    graph: &layout::NetworkLayout,
) -> edges::EdgeRenderStats {
    let (response, painter) = ui.allocate_painter(graph.canvas_size, egui::Sense::hover());
    let offset = response.rect.min.to_vec2();
    paint_canvas(
        ui, trace, hidden, parameters, settings, graph, response, painter, offset,
    )
}

#[allow(clippy::too_many_arguments)]
fn paint_canvas(
    ui: &mut egui::Ui,
    trace: &PpoNetworkTrace,
    hidden: &[usize],
    parameters: &[Vec<f32>],
    settings: &mut PpoNetworkSettings,
    graph: &layout::NetworkLayout,
    response: egui::Response,
    painter: egui::Painter,
    offset: egui::Vec2,
) -> edges::EdgeRenderStats {
    let viewport = ui.clip_rect().intersect(response.rect);
    painter.rect_filled(response.rect, 5.0, egui::Color32::from_rgb(18, 22, 28));
    draw_head_group(
        &painter,
        graph.actor_bounds.translate(offset),
        "Actor",
        "médias e parâmetros da política",
        graph.scale,
        egui::Color32::from_rgb(104, 204, 184),
    );
    draw_head_group(
        &painter,
        graph.critic_bounds.translate(offset),
        "Critic",
        "estimativa do retorno",
        graph.scale,
        egui::Color32::from_rgb(136, 179, 229),
    );
    let hover = ui
        .input(|input| input.pointer.hover_pos())
        .filter(|point| viewport.contains(*point));
    let focused = hover.and_then(|point| {
        graph.layers.iter().enumerate().find_map(|(column, layer)| {
            layer
                .nodes
                .iter()
                .find(|(_, position)| {
                    (*position + offset).distance(point) <= graph.node_radius + 3.0
                })
                .map(|(neuron, _)| (column, *neuron))
        })
    });
    let stats = if settings.show_edges {
        settings.edges.paint(
            &painter,
            graph,
            hidden.len(),
            parameters,
            trace.version,
            offset,
            viewport,
            settings.mode,
            settings.edge_budget.min(2000),
            focused,
        )
    } else {
        edges::EdgeRenderStats::default()
    };
    let critic_values = [trace.value];
    let activations: Vec<&[f32]> = std::iter::once(trace.input.as_slice())
        .chain(trace.trunk.iter().map(Vec::as_slice))
        .chain([trace.mean.as_slice(), critic_values.as_slice()])
        .collect();
    let head_column = hidden.len() + 1;
    for (column, layer) in graph.layers.iter().enumerate() {
        for &(neuron, local_position) in &layer.nodes {
            let position = local_position + offset;
            let hitbox = egui::Rect::from_center_size(
                position,
                egui::Vec2::splat(2.0 * (graph.node_radius + 3.0)),
            );
            if !viewport.intersects(hitbox) {
                continue;
            }
            let value = activations[column][neuron];
            let bias = (column > 0).then(|| parameters[2 * (column - 1) + 1][neuron]);
            let ring = if column == head_column {
                egui::Color32::from_rgb(104, 204, 184)
            } else if column > head_column {
                egui::Color32::from_rgb(136, 179, 229)
            } else {
                egui::Color32::from_gray(160)
            };
            paint_neuron(
                &painter,
                position,
                graph.node_radius,
                value,
                ring,
                settings.show_values || column >= head_column,
                false,
            );
            if column < head_column && graph.scale >= 0.55 {
                painter.text(
                    position - egui::vec2(graph.node_radius + 5.0, 0.0),
                    egui::Align2::RIGHT_CENTER,
                    neuron.to_string(),
                    egui::FontId::monospace((10.0 * graph.scale).clamp(8.0, 15.0)),
                    egui::Color32::from_gray(175),
                );
            } else if column >= head_column {
                let (name, meaning) = if column == head_column {
                    (
                        if neuron == 0 {
                            "μ steering"
                        } else {
                            "μ acceleration"
                        },
                        format!("tanh(μ) = {:+.3}", value.tanh()),
                    )
                } else {
                    ("V(s)", "retorno esperado".into())
                };
                paint_head_label(
                    &painter,
                    position,
                    graph.node_radius,
                    name,
                    &meaning,
                    graph.scale,
                );
            }
            ui.interact(hitbox, response.id.with((column, neuron)), egui::Sense::hover()).on_hover_ui(|ui| {
                if column == 0 {
                    ui.strong(format!("Entrada #{neuron}: {}", trace.input_names[neuron]));
                    ui.label("Observação real normalizada usada na última decisão.");
                } else if column <= hidden.len() {
                    ui.strong(format!("Tronco compartilhado · camada {column} · neurônio #{neuron}"));
                    ui.label("Ativação ReLU do forward real.");
                } else if column == head_column {
                    ui.strong(if neuron == 0 { "Actor · μ steering (direção)" } else { "Actor · μ acceleration (aceleração)" });
                    ui.label("Média da Gaussiana latente. O playback executa tanh(μ).");
                } else {
                    ui.strong("Critic · V(s)");
                    ui.label("Estimativa do retorno descontado a partir da observação; recebe o tronco compartilhado.");
                }
                ui.monospace(format!("Valor: {value:+.6}"));
                if let Some(bias) = bias { ui.monospace(format!("Bias: {bias:+.6}")); }
            });
        }
        if column < head_column && graph.scale >= 0.55 {
            let position = layer.nodes[0].1 + offset - egui::vec2(0.0, 30.0 * graph.scale);
            let title = if column == 0 {
                format!("Entradas · {}", layer.width)
            } else {
                format!("ReLU {column} · {}", layer.width)
            };
            painter.text(
                position,
                egui::Align2::CENTER_BOTTOM,
                title,
                egui::FontId::proportional((13.0 * graph.scale).clamp(10.0, 18.0)),
                egui::Color32::from_rgb(175, 191, 210),
            );
        }
        if let Some(omission) = layer.omission {
            let position = omission.position + offset;
            let area =
                egui::Rect::from_center_size(position, egui::vec2(110.0, 55.0) * graph.scale);
            if viewport.intersects(area) {
                // Draw the ⋮ as three dots so it remains clear with every font/scale.
                for delta in [-6.0, 0.0, 6.0] {
                    painter.circle_filled(
                        position + egui::vec2(0.0, delta * graph.scale),
                        1.6 * graph.scale,
                        egui::Color32::from_gray(175),
                    );
                }
                if settings.show_omitted_count && graph.scale >= 0.55 {
                    painter.text(
                        position + egui::vec2(0.0, 18.0 * graph.scale),
                        egui::Align2::CENTER_TOP,
                        format!("{} omitidos", omission.count),
                        egui::FontId::proportional((10.0 * graph.scale).max(8.0)),
                        egui::Color32::from_gray(150),
                    );
                }
                ui.interact(area, response.id.with(("omission", column)), egui::Sense::hover()).on_hover_text(format!("{} neurônios intermediários omitidos somente nesta apresentação. Expandida mostra todos os índices e conexões.", omission.count));
            }
        }
    }
    for (dimension, local_position) in graph.log_std.iter().enumerate() {
        let position = *local_position + offset;
        let hitbox = egui::Rect::from_center_size(
            position,
            egui::Vec2::splat(2.0 * (graph.node_radius + 4.0)),
        );
        if !viewport.intersects(hitbox) {
            continue;
        }
        let value = trace.log_std[dimension];
        paint_neuron(
            &painter,
            position,
            graph.node_radius,
            value,
            egui::Color32::from_rgb(226, 187, 111),
            true,
            true,
        );
        paint_head_label(
            &painter,
            position,
            graph.node_radius,
            if dimension == 0 {
                "log_std steering"
            } else {
                "log_std acceleration"
            },
            "global treinável",
            graph.scale,
        );
        ui.interact(hitbox, response.id.with(("log_std", dimension)), egui::Sense::hover()).on_hover_ui(|ui| {
            ui.strong(if dimension == 0 { "Actor · log_std steering (direção)" } else { "Actor · log_std acceleration (aceleração)" });
            ui.monospace(format!("log_std = {value:+.6}"));
            ui.monospace(format!("Desvio padrão σ = exp(log_std) = {:.6}", value.exp()));
            ui.label("Parâmetro global treinável da distribuição Gaussiana do Actor, independente da observação. Não recebe conexões do tronco. Não é amostrado no playback determinístico.");
        });
    }
    stats
}

fn draw_head_group(
    painter: &egui::Painter,
    bounds: egui::Rect,
    title: &str,
    subtitle: &str,
    scale: f32,
    color: egui::Color32,
) {
    if !painter.clip_rect().intersects(bounds) {
        return;
    }
    painter.rect_filled(bounds, 8.0 * scale, color.gamma_multiply(0.065));
    painter.rect_stroke(
        bounds,
        8.0 * scale,
        egui::Stroke::new(scale, color.gamma_multiply(0.45)),
        egui::StrokeKind::Inside,
    );
    if scale < 0.35 {
        return;
    }
    painter.text(
        bounds.left_top() + egui::vec2(13.0, 10.0) * scale,
        egui::Align2::LEFT_TOP,
        title,
        egui::FontId::proportional((15.0 * scale).max(10.0)),
        color,
    );
    if scale >= 0.65 {
        painter.text(
            bounds.left_top() + egui::vec2(13.0, 30.0) * scale,
            egui::Align2::LEFT_TOP,
            subtitle,
            egui::FontId::proportional((10.0 * scale).max(8.0)),
            egui::Color32::from_gray(155),
        );
    }
}

fn paint_head_label(
    painter: &egui::Painter,
    position: egui::Pos2,
    radius: f32,
    name: &str,
    meaning: &str,
    scale: f32,
) {
    if scale < 0.55 {
        return;
    }
    let anchor = position + egui::vec2(radius + 10.0 * scale, 0.0);
    painter.text(
        anchor - egui::vec2(0.0, 7.0 * scale),
        egui::Align2::LEFT_CENTER,
        name,
        egui::FontId::proportional((12.0 * scale).max(9.0)),
        egui::Color32::from_gray(225),
    );
    painter.text(
        anchor + egui::vec2(0.0, 10.0 * scale),
        egui::Align2::LEFT_CENTER,
        meaning,
        egui::FontId::monospace((9.0 * scale).max(8.0)),
        egui::Color32::from_gray(155),
    );
}

fn paint_neuron(
    painter: &egui::Painter,
    position: egui::Pos2,
    radius: f32,
    value: f32,
    ring: egui::Color32,
    show_value: bool,
    global: bool,
) {
    let fill = if global {
        egui::Color32::from_rgb(49, 42, 30)
    } else {
        activation_fill(value)
    };
    let scale = radius / 14.0;
    painter.circle_filled(position, radius, fill);
    painter.circle_stroke(position, radius, egui::Stroke::new(1.2 * scale, ring));
    if global {
        painter.circle_stroke(
            position,
            radius * (11.0 / 14.0),
            egui::Stroke::new(0.65 * scale, ring.gamma_multiply(0.6)),
        );
    }
    if show_value && radius >= 8.5 {
        let text = format!("{value:+.2}");
        let font = ((radius * 3.0) / text.len() as f32).clamp(5.0, 12.0);
        painter.text(
            position,
            egui::Align2::CENTER_CENTER,
            text,
            egui::FontId::monospace(font),
            egui::Color32::WHITE,
        );
    }
}

fn draw_connection_inspector(
    ui: &mut egui::Ui,
    trace: &PpoNetworkTrace,
    hidden: &[usize],
    parameters: &[Vec<f32>],
    settings: &mut PpoNetworkSettings,
) {
    egui::CollapsingHeader::new("Inspecionar conexão real").id_salt("ppo_weight_inspector").show(ui, |ui| {
        settings.inspected_block = settings.inspected_block.min(hidden.len() + 1);
        let block_name = |index: usize| {
            if index < hidden.len() { format!("{} → ReLU {}", if index == 0 { "Entradas".into() } else { format!("ReLU {index}") }, index + 1) }
            else if index == hidden.len() { "Tronco → Actor μ".into() } else { "Tronco → Critic V(s)".into() }
        };
        egui::ComboBox::from_id_salt("ppo_weight_block").selected_text(block_name(settings.inspected_block)).show_ui(ui, |ui| {
            for index in 0..hidden.len() + 2 { ui.selectable_value(&mut settings.inspected_block, index, block_name(index)); }
        });
        let block = settings.inspected_block;
        let input = if block == 0 { trace.input.len() } else { hidden[(block - 1).min(hidden.len() - 1)] };
        let output = if block < hidden.len() { hidden[block] } else if block == hidden.len() { 2 } else { 1 };
        settings.inspected_source = settings.inspected_source.min(input - 1);
        settings.inspected_target = settings.inspected_target.min(output - 1);
        ui.horizontal(|ui| {
            ui.label("Origem"); ui.add(egui::DragValue::new(&mut settings.inspected_source).range(0..=input - 1));
            ui.label("Destino"); ui.add(egui::DragValue::new(&mut settings.inspected_target).range(0..=output - 1));
            let value = parameters[2 * block][settings.inspected_target * input + settings.inspected_source];
            ui.monospace(format!("W[{}, {}] = {value:+.7}", settings.inspected_target, settings.inspected_source));
        });
        ui.small("Todos os índices e pesos reais são consultáveis, inclusive quando estão fora da área visível. log_std é global e não pertence a nenhuma matriz de conexões.");
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use racing_rl::{
        backend::{Cpu, matrix},
        config::Config,
        network::Network,
    };
    #[test]
    fn trace_is_selected_row_of_real_forward_and_agrees_with_inference() {
        let config = Config {
            hidden: vec![8, 4],
            ..Default::default()
        };
        let device = Default::default();
        let network = Network::<Cpu>::new(&config, &device).unwrap();
        let width = config.observation.dimension();
        let input: Vec<_> = (0..2 * width).map(|i| (i as f32 - 17.0) / 50.0).collect();
        let x = matrix(&input, 2, width, &device).unwrap();
        let inferred = network.infer(x.clone()).unwrap();
        let cache = network.forward(x).unwrap();
        let names: Vec<_> = config
            .observation
            .features()
            .into_iter()
            .map(|f| f.name)
            .collect();
        let trace = PpoNetworkTrace::from_forward(&cache, 1, &names).unwrap();
        let batch = PpoNetworkTrace::batch_from_forward(&cache, &names).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[1], trace);
        assert_eq!(
            batch[0],
            PpoNetworkTrace::from_forward(&cache, 0, &names).unwrap()
        );
        assert_eq!(trace.input, input[width..]);
        assert_eq!(trace.mean.as_slice(), &read(inferred.mean).unwrap()[2..4]);
        assert_eq!(trace.value, read(inferred.value).unwrap()[1]);
        assert_eq!(
            trace.trunk[0],
            tensor_row(&cache.trunk[0].activation, 1).unwrap()
        );
        assert_eq!(
            trace.trunk[1],
            tensor_row(&cache.trunk[1].activation, 1).unwrap()
        );
        assert_eq!(trace.log_std, config.initial_log_std);
        assert!(trace.matches_architecture(&config.hidden, &network.snapshot().unwrap()));
        assert!(PpoNetworkTrace::from_forward(&cache, 2, &names).is_err());
        assert!(PpoNetworkTrace::from_forward(&cache, 0, &names[..1]).is_err());
    }

    #[test]
    fn architecture_mismatch_is_rejected_without_linearizing_heads() {
        let config = Config {
            hidden: vec![8, 4],
            ..Default::default()
        };
        let device = Default::default();
        let network = Network::<Cpu>::new(&config, &device).unwrap();
        let x = matrix(
            &vec![0.0; config.observation.dimension()],
            1,
            config.observation.dimension(),
            &device,
        )
        .unwrap();
        let trace = PpoNetworkTrace::from_forward(&network.forward(x).unwrap(), 0, &[]).unwrap();
        let mut parameters = network.snapshot().unwrap();
        assert!(!trace.matches_architecture(&[8, 5], &parameters));
        let critic_weights = 2 * (config.hidden.len() + 1);
        parameters[critic_weights].truncate(2);
        assert!(!trace.matches_architecture(&config.hidden, &parameters));
    }

    #[test]
    fn visual_detail_settings_never_change_live_trace_or_policy_outputs() {
        let config = Config {
            hidden: vec![64, 64],
            ..Default::default()
        };
        let device = Default::default();
        let network = Network::<Cpu>::new(&config, &device).unwrap();
        let x = matrix(
            &vec![0.5; config.observation.dimension()],
            1,
            config.observation.dimension(),
            &device,
        )
        .unwrap();
        let trace =
            PpoNetworkTrace::from_forward(&network.forward(x.clone()).unwrap(), 0, &[]).unwrap();
        let before = network.snapshot().unwrap();
        let context = egui::Context::default();
        for show_edges in [false, true] {
            let mut settings = PpoNetworkSettings {
                max_nodes_per_layer: 4,
                edge_budget: 5,
                show_edges,
                show_values: false,
                ..Default::default()
            };
            let mut output = context.run_ui(Default::default(), |ui| {
                draw_ppo_network(ui, &trace, &config.hidden, &before, &mut settings);
            });
            // This test exercises presentation without a renderer. Explicitly
            // discard font texture uploads instead of leaving unapplied deltas.
            output.textures_delta.clear();
        }
        let after = network.infer(x).unwrap();
        assert_eq!(trace.trunk[0].len(), 64);
        assert_eq!(trace.trunk[1].len(), 64);
        assert_eq!(trace.mean.as_slice(), read(after.mean).unwrap());
        assert_eq!(trace.value, read(after.value).unwrap()[0]);
        assert_eq!(before, network.snapshot().unwrap());
    }
}
