//! Real dense connections, cached independently from the changing activations.
//!
//! Expanded mode has no sampling budget. We clip segments (not just endpoints)
//! to the viewport and submit cached triangle meshes, avoiding hundreds of
//! thousands of per-frame egui line shapes. Panning/zooming invalidates only the
//! visible mesh; playback and neuron hover retain it.

use std::{ops::Range, sync::Arc};

use bevy_egui::egui::{self, Mesh, Painter, Pos2, Rect, Stroke, Vec2};

use super::super::network_view::weight_stroke;
use super::layout::{LayerLayout, NetworkLayout, PpoNetworkMode};

/// A chunk is an optimization only: every selected connection belongs to one.
const CHUNK_CONNECTIONS: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EdgeBlock {
    from: usize,
    to: usize,
    parameter: usize,
}

fn topology(hidden_layers: usize) -> Vec<EdgeBlock> {
    (0..hidden_layers)
        .map(|layer| EdgeBlock {
            from: layer,
            to: layer + 1,
            parameter: layer * 2,
        })
        .chain([
            EdgeBlock {
                from: hidden_layers,
                to: hidden_layers + 1,
                parameter: hidden_layers * 2,
            },
            EdgeBlock {
                from: hidden_layers,
                to: hidden_layers + 2,
                parameter: (hidden_layers + 1) * 2,
            },
        ])
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    points: [Pos2; 2],
    stroke: Stroke,
    from: usize,
    to: usize,
}

#[derive(Clone, Debug)]
struct EdgeChunk {
    block: EdgeBlock,
    range: Range<usize>,
    bounds: Rect,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ViewKey {
    offset: Vec2,
    viewport: Rect,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct EdgeRenderStats {
    /// Full mathematical network, irrespective of the presentation mode.
    pub total_connections: usize,
    /// Actual connections represented by the selected neurons/detail setting.
    pub selected_connections: usize,
    /// Selected connections whose real line intersects the current viewport.
    pub visible_connections: usize,
    pub geometry_rebuilt: bool,
    pub view_rebuilt: bool,
}

#[derive(Clone, Debug, Default)]
pub(super) struct EdgeCache {
    identity: Option<(u64, usize, PpoNetworkMode, usize)>,
    layout: Vec<LayerLayout>,
    // Equality also protects against allocation-address reuse when checkpoints
    // change. This snapshot is copied only at a real cache invalidation.
    parameters: Vec<Vec<f32>>,
    edges: Vec<Edge>,
    chunks: Vec<EdgeChunk>,
    total_connections: usize,
    weight_scale: f32,
    view: Option<ViewKey>,
    meshes: Vec<Arc<Mesh>>,
    visible_connections: usize,
    focused: Option<(usize, usize)>,
    focus_edges: Vec<(usize, usize)>,
}

impl EdgeCache {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn paint(
        &mut self,
        painter: &Painter,
        layout: &NetworkLayout,
        hidden_layers: usize,
        parameters: &[Vec<f32>],
        identity: u64,
        offset: Vec2,
        viewport: Rect,
        mode: PpoNetworkMode,
        condensed_budget: usize,
        focused: Option<(usize, usize)>,
    ) -> EdgeRenderStats {
        let geometry_rebuilt = self.prepare(
            layout,
            hidden_layers,
            parameters,
            identity,
            mode,
            condensed_budget,
        );
        let view = ViewKey { offset, viewport };
        let view_rebuilt = self.view != Some(view);
        if view_rebuilt {
            self.rebuild_view(view);
        }
        let painter = painter.with_clip_rect(viewport);
        for mesh in &self.meshes {
            painter.add(egui::Shape::Mesh(Arc::clone(mesh)));
        }
        if let Some(focused) = focused {
            self.highlight(&painter, view, focused);
        }
        EdgeRenderStats {
            total_connections: self.total_connections,
            selected_connections: self.edges.len(),
            visible_connections: self.visible_connections,
            geometry_rebuilt,
            view_rebuilt,
        }
    }

    fn prepare(
        &mut self,
        layout: &NetworkLayout,
        hidden_layers: usize,
        parameters: &[Vec<f32>],
        identity: u64,
        mode: PpoNetworkMode,
        condensed_budget: usize,
    ) -> bool {
        // Changing a condensed-only slider must not invalidate an expanded view.
        let budget = if mode == PpoNetworkMode::Expanded {
            usize::MAX
        } else {
            condensed_budget
        };
        let key = (identity, hidden_layers, mode, budget);
        if self.identity == Some(key)
            && self.layout == layout.layers
            && self.parameters == parameters
        {
            return false;
        }
        self.identity = Some(key);
        self.layout.clone_from(&layout.layers);
        self.parameters = parameters.to_vec();
        self.edges.clear();
        self.chunks.clear();
        self.view = None;
        self.meshes.clear();
        self.focused = None;
        self.focus_edges.clear();

        let blocks = topology(hidden_layers);
        self.total_connections = blocks
            .iter()
            .map(|block| layout.layers[block.from].width * layout.layers[block.to].width)
            .sum();
        let candidates: Vec<_> = blocks
            .iter()
            .map(|block| {
                layout.layers[block.from].nodes.len() * layout.layers[block.to].nodes.len()
            })
            .collect();
        let budgets = edge_budgets(&candidates, budget);
        self.edges.reserve(budgets.iter().sum());
        let weight_scale = blocks
            .iter()
            .flat_map(|block| parameters[block.parameter].iter())
            .map(|weight| weight.abs())
            .fold(f32::EPSILON, f32::max);
        self.weight_scale = weight_scale;
        for ((block, &count), &budget) in blocks.iter().zip(&candidates).zip(&budgets) {
            let from_layer = &layout.layers[block.from];
            let to_layer = &layout.layers[block.to];
            let dense = count > 4096;
            let start = self.edges.len();
            for selected in 0..budget {
                let index = sampled_edge_index(count, budget, selected);
                let (from, from_position) = from_layer.nodes[index % from_layer.nodes.len()];
                let (to, to_position) = to_layer.nodes[index / from_layer.nodes.len()];
                // Snapshot weights are row-major W[out, in], exactly as forward.
                let weight = parameters[block.parameter][to * from_layer.width + from];
                let mut stroke = weight_stroke(weight, weight_scale);
                // Dense, exact graphs need subtle background lines. Hover draws
                // the incident real connections with full signed-weight colors.
                stroke.color = stroke.color.gamma_multiply(if dense { 0.10 } else { 0.32 });
                stroke.width = stroke.width.min(if dense { 0.65 } else { 1.1 }) * layout.scale;
                self.edges.push(Edge {
                    // Stop at the circle outline, preserving legible node values.
                    points: inset_segment([from_position, to_position], layout.node_radius),
                    stroke,
                    from,
                    to,
                });
            }
            for chunk_start in (start..self.edges.len()).step_by(CHUNK_CONNECTIONS) {
                let range = chunk_start..(chunk_start + CHUNK_CONNECTIONS).min(self.edges.len());
                let bounds = self.edges[range.clone()]
                    .iter()
                    .fold(Rect::NOTHING, |bounds, edge| {
                        bounds.union(Rect::from_two_pos(edge.points[0], edge.points[1]))
                    });
                self.chunks.push(EdgeChunk {
                    block: *block,
                    range,
                    bounds,
                });
            }
        }
        true
    }

    fn rebuild_view(&mut self, view: ViewKey) {
        self.meshes.clear();
        self.visible_connections = 0;
        self.view = Some(view);
        if !view.viewport.is_positive() {
            return;
        }
        // Work in local coordinates; the cached segments never move with scroll.
        let local_viewport = view.viewport.translate(-view.offset).expand(2.0);
        for chunk in &self.chunks {
            if !chunk.bounds.intersects(local_viewport) {
                continue;
            }
            let mut mesh = Mesh::default();
            for edge in &self.edges[chunk.range.clone()] {
                if let Some(points) = clipped_segment(edge.points, local_viewport) {
                    self.visible_connections += 1;
                    add_segment_mesh(
                        &mut mesh,
                        [points[0] + view.offset, points[1] + view.offset],
                        edge.stroke,
                    );
                }
            }
            if !mesh.is_empty() {
                self.meshes.push(Arc::new(mesh));
            }
        }
    }

    fn highlight(&mut self, painter: &Painter, view: ViewKey, focused: (usize, usize)) {
        // Resolve incident indices once per hovered neuron, not on every frame.
        // Store only selected edges; this also preserves condensed line detail.
        if self.focused != Some(focused) {
            self.focused = Some(focused);
            self.focus_edges.clear();
            for (chunk_index, chunk) in self.chunks.iter().enumerate() {
                if chunk.block.from != focused.0 && chunk.block.to != focused.0 {
                    continue;
                }
                for edge_index in chunk.range.clone() {
                    let edge = self.edges[edge_index];
                    if (chunk.block.from == focused.0 && edge.from == focused.1)
                        || (chunk.block.to == focused.0 && edge.to == focused.1)
                    {
                        self.focus_edges.push((chunk_index, edge_index));
                    }
                }
            }
        }
        let local_viewport = view.viewport.translate(-view.offset).expand(3.0);
        for &(chunk_index, edge_index) in &self.focus_edges {
            let chunk = &self.chunks[chunk_index];
            let edge = self.edges[edge_index];
            if let Some(points) = clipped_segment(edge.points, local_viewport) {
                let weight = self.parameters[chunk.block.parameter]
                    [edge.to * self.layout[chunk.block.from].width + edge.from];
                let mut stroke = weight_stroke(weight, self.weight_scale);
                stroke.width = stroke.width.clamp(0.8, 1.8);
                painter.line_segment([points[0] + view.offset, points[1] + view.offset], stroke);
            }
        }
    }
}

/// Preserve both small heads when a user chooses a finite condensed line budget.
fn edge_budgets(candidates: &[usize], maximum: usize) -> Vec<usize> {
    let total = candidates.iter().sum::<usize>();
    if maximum >= total {
        return candidates.to_vec();
    }
    let mut budgets = vec![0; candidates.len()];
    let mut remaining = maximum;
    while remaining > 0 {
        for (budget, &candidate) in budgets.iter_mut().zip(candidates) {
            if *budget < candidate && remaining > 0 {
                *budget += 1;
                remaining -= 1;
            }
        }
    }
    budgets
}

fn sampled_edge_index(count: usize, budget: usize, selected: usize) -> usize {
    if budget <= 1 {
        0
    } else {
        selected * (count - 1) / (budget - 1)
    }
}

fn inset_segment(points: [Pos2; 2], radius: f32) -> [Pos2; 2] {
    let delta = points[1] - points[0];
    let inset = delta.normalized() * radius.min(delta.length() * 0.25);
    [points[0] + inset, points[1] - inset]
}

/// Liang-Barsky clipping includes diagonals crossing the viewport even when
/// neither endpoint is visible. Offscreen lines are never discarded by node ID.
fn clipped_segment(points: [Pos2; 2], clip: Rect) -> Option<[Pos2; 2]> {
    let delta = points[1] - points[0];
    let mut start = 0.0_f32;
    let mut end = 1.0_f32;
    for (p, q) in [
        (-delta.x, points[0].x - clip.left()),
        (delta.x, clip.right() - points[0].x),
        (-delta.y, points[0].y - clip.top()),
        (delta.y, clip.bottom() - points[0].y),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let parameter = q / p;
            if p < 0.0 {
                start = start.max(parameter);
            } else {
                end = end.min(parameter);
            }
            if start > end {
                return None;
            }
        }
    }
    Some([points[0] + delta * start, points[0] + delta * end])
}

fn add_segment_mesh(mesh: &mut Mesh, points: [Pos2; 2], stroke: Stroke) {
    let delta = points[1] - points[0];
    if delta.length_sq() < f32::EPSILON {
        return;
    }
    let normal = Vec2::new(-delta.y, delta.x).normalized() * stroke.width * 0.5;
    let index = mesh.vertices.len() as u32;
    for point in [
        points[0] - normal,
        points[0] + normal,
        points[1] + normal,
        points[1] - normal,
    ] {
        mesh.colored_vertex(point, stroke.color);
    }
    mesh.add_triangle(index, index + 1, index + 2);
    mesh.add_triangle(index, index + 2, index + 3);
}

#[cfg(test)]
mod tests {
    use super::super::layout::build_layout;
    use super::*;

    fn parameters(input: usize, hidden: &[usize]) -> Vec<Vec<f32>> {
        let mut input_width = input;
        let mut parameters = Vec::new();
        for output in hidden.iter().copied().chain([2, 1]) {
            parameters.push(
                (0..input_width * output)
                    .map(|index| (index as f32 + 1.0) / 100.0)
                    .collect(),
            );
            parameters.push(vec![0.0; output]);
            input_width = output;
            if parameters.len() == 2 * (hidden.len() + 1) {
                input_width = hidden.last().copied().unwrap_or(input);
            }
        }
        parameters.push(vec![-0.5; 2]);
        parameters
    }

    fn prepared(
        input: usize,
        hidden: &[usize],
        mode: PpoNetworkMode,
    ) -> (EdgeCache, NetworkLayout) {
        let layout = build_layout(input, hidden, mode, 16, 1.0);
        let mut cache = EdgeCache::default();
        cache.prepare(
            &layout,
            hidden.len(),
            &parameters(input, hidden),
            7,
            mode,
            360,
        );
        (cache, layout)
    }

    #[test]
    fn both_heads_branch_from_shared_trunk_and_globals_have_no_edges() {
        assert_eq!(
            topology(2),
            vec![
                EdgeBlock {
                    from: 0,
                    to: 1,
                    parameter: 0
                },
                EdgeBlock {
                    from: 1,
                    to: 2,
                    parameter: 2
                },
                EdgeBlock {
                    from: 2,
                    to: 3,
                    parameter: 4
                },
                EdgeBlock {
                    from: 2,
                    to: 4,
                    parameter: 6
                },
            ]
        );
        let (cache, _) = prepared(25, &[64, 64], PpoNetworkMode::Expanded);
        assert_eq!(cache.total_connections, 25 * 64 + 64 * 64 + 64 * 3);
        assert_eq!(cache.edges.len(), cache.total_connections);
        assert!(cache.chunks.iter().all(|chunk| chunk.block.to <= 4));
    }

    #[test]
    fn expanded_stress_contains_every_connection_without_budget_or_sampling() {
        let hidden = [256, 256, 256, 256, 128];
        let layout = build_layout(25, &hidden, PpoNetworkMode::Expanded, 6, 1.0);
        let parameters = parameters(25, &hidden);
        let mut cache = EdgeCache::default();
        cache.prepare(
            &layout,
            hidden.len(),
            &parameters,
            1,
            PpoNetworkMode::Expanded,
            0,
        );
        let expected = 25 * 256 + 3 * 256 * 256 + 256 * 128 + 128 * 3;
        assert_eq!(expected, 236_160);
        assert_eq!(cache.edges.len(), expected);
        assert_eq!(cache.total_connections, expected);
        for block in topology(hidden.len()) {
            let edges: Vec<_> = cache
                .chunks
                .iter()
                .filter(|chunk| chunk.block == block)
                .flat_map(|chunk| cache.edges[chunk.range.clone()].iter())
                .collect();
            let width = layout.layers[block.from].width;
            assert_eq!(edges.len(), width * layout.layers[block.to].width);
            for (index, edge) in edges.iter().enumerate() {
                assert_eq!((edge.from, edge.to), (index % width, index / width));
            }
        }
    }

    #[test]
    fn small_network_retains_all_real_connections_in_both_modes() {
        for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
            let (cache, layout) = prepared(3, &[4, 2], mode);
            assert!(
                layout
                    .layers
                    .iter()
                    .all(|layer| layer.nodes.len() == layer.width)
            );
            assert_eq!(cache.edges.len(), 3 * 4 + 4 * 2 + 2 * 3);
            assert!(cache.chunks.iter().all(|chunk| {
                cache.edges[chunk.range.clone()].iter().all(|edge| {
                    edge.from < layout.layers[chunk.block.from].width
                        && edge.to < layout.layers[chunk.block.to].width
                })
            }));
        }
    }

    #[test]
    fn condensed_edges_use_original_indices_and_original_weight_entries() {
        let (cache, layout) = prepared(25, &[64, 64], PpoNetworkMode::Condensed);
        let parameters = parameters(25, &[64, 64]);
        let maximum = topology(2)
            .iter()
            .flat_map(|block| parameters[block.parameter].iter())
            .copied()
            .fold(f32::EPSILON, f32::max);
        for chunk in &cache.chunks {
            for edge in &cache.edges[chunk.range.clone()] {
                assert!(
                    layout.layers[chunk.block.from]
                        .nodes
                        .iter()
                        .any(|node| node.0 == edge.from)
                );
                assert!(
                    layout.layers[chunk.block.to]
                        .nodes
                        .iter()
                        .any(|node| node.0 == edge.to)
                );
                let weight = parameters[chunk.block.parameter]
                    [edge.to * layout.layers[chunk.block.from].width + edge.from];
                let expected = weight_stroke(weight, maximum).color.gamma_multiply(0.32);
                assert_eq!(edge.stroke.color, expected);
            }
        }
        assert!(cache.edges.len() <= 360);
        assert!(cache.edges.iter().any(|edge| edge.from == 63));
    }

    #[test]
    fn unchanged_playback_reuses_geometry_and_visible_mesh_arcs() {
        let (mut cache, layout) = prepared(25, &[64, 64], PpoNetworkMode::Expanded);
        let parameters = parameters(25, &[64, 64]);
        let view = ViewKey {
            offset: Vec2::ZERO,
            viewport: Rect::from_min_size(Pos2::ZERO, layout.canvas_size),
        };
        cache.rebuild_view(view);
        let mesh = Arc::clone(&cache.meshes[0]);
        assert!(!cache.prepare(&layout, 2, &parameters, 7, PpoNetworkMode::Expanded, 0));
        assert!(Arc::ptr_eq(&mesh, &cache.meshes[0]));
        assert_eq!(cache.view, Some(view));
        assert_eq!(cache.visible_connections, cache.total_connections);
    }

    #[test]
    fn checkpoint_parameter_change_invalidates_even_when_identity_is_reused() {
        let (mut cache, layout) = prepared(3, &[4, 2], PpoNetworkMode::Expanded);
        let mut parameters = parameters(3, &[4, 2]);
        parameters[0][0] = -20.0;
        assert!(cache.prepare(&layout, 2, &parameters, 7, PpoNetworkMode::Expanded, 360));
        assert!(cache.view.is_none());
        assert_eq!(cache.parameters[0][0], -20.0);
        assert!(cache.edges[0].stroke.color.r() > cache.edges[0].stroke.color.g());
    }

    #[test]
    fn viewport_culling_preserves_crossing_edges_with_both_endpoints_offscreen() {
        let clip = Rect::from_min_max(egui::pos2(10.0, 10.0), egui::pos2(20.0, 20.0));
        let points = [egui::pos2(0.0, 0.0), egui::pos2(30.0, 30.0)];
        assert!(!clip.contains(points[0]) && !clip.contains(points[1]));
        assert_eq!(clipped_segment(points, clip), Some([clip.min, clip.max]));
        assert!(clipped_segment([egui::pos2(0.0, 0.0), egui::pos2(0.0, 30.0)], clip).is_none());
        assert!(clipped_segment([egui::pos2(0.0, 30.0), egui::pos2(30.0, 30.0)], clip).is_none());
    }

    #[test]
    fn mesh_is_valid_and_only_tessellates_visible_exact_segments() {
        let (mut cache, layout) = prepared(25, &[64, 64], PpoNetworkMode::Expanded);
        let view = ViewKey {
            offset: egui::vec2(-60.0, -100.0),
            viewport: Rect::from_min_size(Pos2::ZERO, egui::vec2(300.0, 150.0)),
        };
        cache.rebuild_view(view);
        assert!(cache.visible_connections > 0);
        assert!(cache.visible_connections < cache.edges.len());
        assert!(cache.meshes.iter().all(|mesh| mesh.is_valid()));
        let vertices: usize = cache.meshes.iter().map(|mesh| mesh.vertices.len()).sum();
        assert_eq!(vertices, cache.visible_connections * 4);
        // Scrolling away does not mutate the architecture or truncate storage.
        cache.rebuild_view(ViewKey {
            offset: Vec2::ZERO,
            viewport: Rect::from_min_size(egui::pos2(20_000.0, 20_000.0), layout.canvas_size),
        });
        assert_eq!(cache.visible_connections, 0);
        assert!(cache.meshes.is_empty());
        assert_eq!(cache.edges.len(), cache.total_connections);
    }

    #[test]
    fn detail_budget_is_explicit_and_never_applies_to_expanded_mode() {
        assert_eq!(edge_budgets(&[10, 10, 2, 1], 0), vec![0, 0, 0, 0]);
        assert_eq!(
            edge_budgets(&[10, 10, 2, 1], usize::MAX),
            vec![10, 10, 2, 1]
        );
        let budgets = edge_budgets(&[1000, 1000, 20, 10], 100);
        assert_eq!(budgets.iter().sum::<usize>(), 100);
        assert!(budgets.iter().all(|&budget| budget > 0));
        let (mut cache, _) = prepared(25, &[64, 64], PpoNetworkMode::Condensed);
        let expanded = build_layout(25, &[64, 64], PpoNetworkMode::Expanded, 6, 1.0);
        cache.prepare(
            &expanded,
            2,
            &parameters(25, &[64, 64]),
            7,
            PpoNetworkMode::Expanded,
            1,
        );
        assert_eq!(cache.edges.len(), cache.total_connections);
    }
}
