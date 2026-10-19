//! Geometry only: every displayed index still identifies one real neuron.
//!
//! The heads occupy the same final column. Global `log_std` parameters share
//! Actor's group, but deliberately do not appear in its neural layer.

use bevy_egui::egui::{self, Pos2, Rect, Vec2};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum PpoNetworkMode {
    #[default]
    Condensed,
    Expanded,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Omission {
    pub position: Pos2,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct LayerLayout {
    pub width: usize,
    /// `(real neuron index, canvas-local position)`, in increasing index order.
    pub nodes: Vec<(usize, Pos2)>,
    pub omission: Option<Omission>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct NetworkLayout {
    /// Input, each shared hidden layer, Actor means, then Critic value.
    pub layers: Vec<LayerLayout>,
    pub log_std: [Pos2; 2],
    pub actor_bounds: Rect,
    pub critic_bounds: Rect,
    pub canvas_size: Vec2,
    pub node_radius: f32,
    pub scale: f32,
}

const COLUMN_SPACING: f32 = 175.0;
const LEFT_MARGIN: f32 = 60.0;
const VERTICAL_MARGIN: f32 = 74.0;
const CONDENSED_SPACING: f32 = 36.0;
const EXPANDED_SPACING: f32 = 44.0;
/// Room for the vertical ellipsis and an optional omitted-count caption.
const OMISSION_GAP: f32 = 92.0;
const HEAD_HEIGHT: f32 = 390.0;

/// Pick consecutive neurons at both ends, rather than inventing aggregate
/// neurons or spacing sampled indices through the hidden middle of a layer.
fn selected_indices(
    width: usize,
    largest: usize,
    mode: PpoNetworkMode,
    detail: usize,
) -> Vec<usize> {
    // Small complete networks are readable without condensation. This also
    // keeps all 25 real inputs visible for e.g. a 25 -> 8 -> 4 architecture.
    if mode == PpoNetworkMode::Expanded || largest <= 32 {
        return (0..width).collect();
    }
    let proportion = (width as f64 / largest as f64).sqrt();
    let shown = ((detail.max(6) as f64 * proportion).ceil() as usize)
        .max(6)
        .min(width);
    if shown == width {
        return (0..width).collect();
    }
    let first = shown.div_ceil(2);
    let last = shown / 2;
    (0..first).chain(width - last..width).collect()
}

fn column_height(nodes: usize, omitted: bool, spacing: f32) -> f32 {
    nodes.saturating_sub(1) as f32 * spacing + if omitted { OMISSION_GAP - spacing } else { 0.0 }
}

/// Build zoomed canvas-local coordinates. Expanded mode has no node budget:
/// canvas height grows by 44 logical pixels per neuron, however wide a layer is.
/// Connections are not expanded into a dense list here; the painter may cull
/// them against its viewport while retaining their exact source/target indices.
pub(super) fn build_layout(
    input_width: usize,
    hidden: &[usize],
    mode: PpoNetworkMode,
    detail: usize,
    zoom: f32,
) -> NetworkLayout {
    let scale = if zoom.is_finite() && zoom > 0.0 {
        zoom
    } else {
        1.0
    };
    let widths: Vec<_> = std::iter::once(input_width)
        .chain(hidden.iter().copied())
        .chain([2, 1])
        .collect();
    let largest = widths.iter().copied().max().unwrap_or(1).max(1);
    let selected: Vec<_> = widths
        .iter()
        .map(|&width| selected_indices(width, largest, mode, detail))
        .collect();
    let spacing = if mode == PpoNetworkMode::Expanded {
        EXPANDED_SPACING
    } else {
        CONDENSED_SPACING
    };
    let body_height = widths
        .iter()
        .zip(&selected)
        .map(|(&width, nodes)| column_height(nodes.len(), nodes.len() < width, spacing))
        .fold(HEAD_HEIGHT, f32::max);
    let center_y = VERTICAL_MARGIN + body_height / 2.0;
    let head_column = hidden.len() + 1;
    // A deep condensed network should remain an overview. Full-size expanded
    // columns retain their wider navigation canvas, as do the compact Base and
    // small architectures.
    let column_spacing = if mode == PpoNetworkMode::Condensed && hidden.len() >= 4 {
        130.0
    } else {
        COLUMN_SPACING
    };
    let head_x = LEFT_MARGIN + head_column as f32 * column_spacing;
    // Both modes share one horizontal axis. Keep the separate Actor/Critic
    // branches, centering their combined envelope around that same axis.
    // Their bounds below extend from -184 to +194 relative to this anchor.
    let head_center = center_y - 5.0;
    let actor_bounds = Rect::from_min_max(
        egui::pos2(head_x - 34.0, head_center - 184.0),
        egui::pos2(head_x + 194.0, head_center + 68.0),
    );
    let critic_bounds = Rect::from_min_max(
        egui::pos2(head_x - 34.0, head_center + 96.0),
        egui::pos2(head_x + 194.0, head_center + 194.0),
    );
    let actor_y = [head_center - 124.0, head_center - 76.0];
    let log_std = [
        egui::pos2(head_x, head_center - 14.0),
        egui::pos2(head_x, head_center + 34.0),
    ];
    let mut layers = Vec::with_capacity(widths.len());
    for (column, (&width, indices)) in widths.iter().zip(selected).enumerate() {
        let omitted = indices.len() < width;
        let height = column_height(indices.len(), omitted, spacing);
        // The omitted middle occupies real layout space too. Center the full
        // visible span, including that gap, without compressing node spacing.
        let top = center_y - height / 2.0;
        let split = indices.len().div_ceil(2);
        let x = LEFT_MARGIN + column.min(head_column) as f32 * column_spacing;
        let nodes = indices
            .into_iter()
            .enumerate()
            .map(|(row, index)| {
                let y = if column == head_column {
                    actor_y[row]
                } else if column == head_column + 1 {
                    head_center + 155.0
                } else {
                    top + row as f32 * spacing
                        + if omitted && row >= split {
                            OMISSION_GAP - spacing
                        } else {
                            0.0
                        }
                };
                (index, egui::pos2(x, y) * scale)
            })
            .collect::<Vec<_>>();
        let omission = omitted.then(|| Omission {
            position: egui::pos2(
                x,
                top + split.saturating_sub(1) as f32 * spacing + OMISSION_GAP / 2.0,
            ) * scale,
            count: width - nodes.len(),
        });
        layers.push(LayerLayout {
            width,
            nodes,
            omission,
        });
    }
    NetworkLayout {
        layers,
        log_std: log_std.map(|position| position * scale),
        actor_bounds: Rect::from_min_max(actor_bounds.min * scale, actor_bounds.max * scale),
        critic_bounds: Rect::from_min_max(critic_bounds.min * scale, critic_bounds.max * scale),
        canvas_size: egui::vec2(head_x + 218.0, body_height + 2.0 * VERTICAL_MARGIN) * scale,
        node_radius: 14.0 * scale,
        scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(hidden: &[usize], mode: PpoNetworkMode) -> NetworkLayout {
        build_layout(25, hidden, mode, 12, 1.0)
    }

    fn assert_exact_ends(layer: &LayerLayout) {
        let indices: Vec<_> = layer.nodes.iter().map(|&(index, _)| index).collect();
        let first = indices.len().div_ceil(2);
        let last = indices.len() / 2;
        assert_eq!(indices[..first], (0..first).collect::<Vec<_>>());
        assert_eq!(
            indices[first..],
            (layer.width - last..layer.width).collect::<Vec<_>>()
        );
        let omission = layer.omission.unwrap();
        assert_eq!(omission.count + indices.len(), layer.width);
        assert!(layer.nodes[first - 1].1.y < omission.position.y);
        assert!(omission.position.y < layer.nodes[first].1.y);
        assert!(omission.position.y - layer.nodes[first - 1].1.y >= 40.0);
        assert!(layer.nodes[first].1.y - omission.position.y >= 40.0);
    }

    #[test]
    fn condensed_base_is_proportional_and_preserves_both_contiguous_ends() {
        let graph = layout(&[64, 64], PpoNetworkMode::Condensed);
        assert_eq!(
            graph
                .layers
                .iter()
                .map(|layer| layer.nodes.len())
                .collect::<Vec<_>>(),
            [8, 12, 12, 2, 1]
        );
        for layer in &graph.layers[..3] {
            assert_exact_ends(layer);
        }
        let input_height = graph.layers[0].nodes.last().unwrap().1.y - graph.layers[0].nodes[0].1.y;
        let hidden_height =
            graph.layers[1].nodes.last().unwrap().1.y - graph.layers[1].nodes[0].1.y;
        assert!(input_height < hidden_height);
    }

    #[test]
    fn stress_layers_have_distinct_proportions_and_real_indices() {
        let graph = layout(&[256, 256, 256, 256, 128], PpoNetworkMode::Condensed);
        assert_eq!(graph.layers[0].nodes.len(), 6);
        assert_eq!(graph.layers[1].nodes.len(), 12);
        assert_eq!(graph.layers[5].nodes.len(), 9);
        for layer in &graph.layers[..6] {
            assert_exact_ends(layer);
            assert!(
                layer
                    .nodes
                    .windows(2)
                    .all(|pair| pair[0].0 < pair[1].0 && pair[0].1.y < pair[1].1.y)
            );
        }
    }

    #[test]
    fn condensed_stress_fits_overview_without_overlapping_neurons_or_heads() {
        let graph = build_layout(
            25,
            &[256, 256, 256, 256, 128],
            PpoNetworkMode::Condensed,
            12,
            0.9,
        );
        let canvas = Rect::from_min_size(Pos2::ZERO, graph.canvas_size);
        assert!(graph.canvas_size.x <= 960.0);
        assert!(canvas.contains_rect(graph.actor_bounds));
        assert!(canvas.contains_rect(graph.critic_bounds));
        let nodes: Vec<_> = graph
            .layers
            .iter()
            .flat_map(|layer| layer.nodes.iter().map(|&(_, position)| position))
            .chain(graph.log_std)
            .collect();
        for (index, &position) in nodes.iter().enumerate() {
            assert!(canvas.contains(position));
            for &other in &nodes[index + 1..] {
                assert!(position.distance(other) > 2.0 * graph.node_radius);
            }
        }
    }

    #[test]
    fn small_networks_keep_every_neuron_without_an_ellipsis() {
        let graph = layout(&[8, 4], PpoNetworkMode::Condensed);
        for layer in &graph.layers {
            assert_eq!(
                layer
                    .nodes
                    .iter()
                    .map(|&(index, _)| index)
                    .collect::<Vec<_>>(),
                (0..layer.width).collect::<Vec<_>>()
            );
            assert!(layer.omission.is_none());
        }
    }

    #[test]
    fn all_input_and_shared_columns_use_the_same_vertical_axis() {
        for (input, hidden) in [
            (25, vec![64, 64]),
            (25, vec![256, 256, 256, 256, 128]),
            (25, vec![8, 4]),
            (3, vec![7, 2]),
        ] {
            for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
                for zoom in [0.1, 0.9, 1.0, 2.5] {
                    let graph = build_layout(input, &hidden, mode, 12, zoom);
                    let center_y = graph.canvas_size.y / 2.0;
                    let spacing = match mode {
                        PpoNetworkMode::Condensed => CONDENSED_SPACING,
                        PpoNetworkMode::Expanded => EXPANDED_SPACING,
                    } * zoom;
                    for layer in &graph.layers[..=hidden.len()] {
                        let top = layer.nodes.first().unwrap().1.y;
                        let bottom = layer.nodes.last().unwrap().1.y;
                        assert!(((top + bottom) / 2.0 - center_y).abs() < 0.01);
                        for pair in layer.nodes.windows(2) {
                            let gap = pair[1].1.y - pair[0].1.y;
                            let expected = if pair[1].0 == pair[0].0 + 1 {
                                spacing
                            } else {
                                OMISSION_GAP * zoom
                            };
                            assert!((gap - expected).abs() < 0.01);
                        }
                    }
                    let branch_center =
                        (graph.actor_bounds.top() + graph.critic_bounds.bottom()) / 2.0;
                    assert!((branch_center - center_y).abs() < 0.01);
                    let canvas = Rect::from_min_size(Pos2::ZERO, graph.canvas_size);
                    assert!(canvas.contains_rect(graph.actor_bounds));
                    assert!(canvas.contains_rect(graph.critic_bounds));
                }
            }
        }
    }

    #[test]
    fn odd_condensed_selection_centers_its_whole_span_including_omission() {
        let graph = layout(&[256, 256, 256, 256, 128], PpoNetworkMode::Condensed);
        let layer = &graph.layers[5];
        assert_eq!(layer.nodes.len(), 9);
        let first_group = layer.nodes.len().div_ceil(2);
        let before = layer.nodes[first_group - 1].1.y;
        let after = layer.nodes[first_group].1.y;
        let omission = layer.omission.unwrap();
        assert!((omission.position.y - (before + after) / 2.0).abs() < 0.01);
        assert_eq!(omission.count, 119);
        let top = layer.nodes.first().unwrap().1.y;
        let bottom = layer.nodes.last().unwrap().1.y;
        assert!(((top + bottom) / 2.0 - graph.canvas_size.y / 2.0).abs() < 0.01);
    }

    #[test]
    fn expanded_has_all_neurons_and_no_artificial_width_cap() {
        for hidden in [
            vec![64, 64],
            vec![256, 256, 256, 256, 128],
            vec![8, 4],
            vec![8192],
        ] {
            let graph = layout(&hidden, PpoNetworkMode::Expanded);
            for layer in &graph.layers {
                assert_eq!(
                    layer
                        .nodes
                        .iter()
                        .map(|&(index, _)| index)
                        .collect::<Vec<_>>(),
                    (0..layer.width).collect::<Vec<_>>()
                );
                assert!(layer.omission.is_none());
            }
            for layer in &graph.layers[..=hidden.len()] {
                assert!(layer.nodes.windows(2).all(|pair| {
                    (pair[1].1.y - pair[0].1.y - EXPANDED_SPACING).abs() < f32::EPSILON
                }));
            }
            assert!(
                graph.canvas_size.y
                    >= (hidden.iter().copied().max().unwrap() - 1) as f32 * EXPANDED_SPACING
            );
        }
    }

    #[test]
    fn actor_contains_its_globals_and_critic_is_an_independent_branch() {
        for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
            let graph = layout(&[64, 64], mode);
            let actor = &graph.layers[3];
            let critic = &graph.layers[4];
            assert_eq!(actor.width, 2);
            assert_eq!(critic.width, 1);
            for &(_, position) in &actor.nodes {
                assert!(graph.actor_bounds.contains(position));
            }
            for position in graph.log_std {
                assert!(graph.actor_bounds.contains(position));
                assert!(
                    actor
                        .nodes
                        .iter()
                        .all(|&(_, neuron)| neuron.distance(position) > graph.node_radius * 2.0)
                );
            }
            assert!(graph.critic_bounds.contains(critic.nodes[0].1));
            assert_eq!(actor.nodes[0].1.x, critic.nodes[0].1.x);
            assert!(graph.actor_bounds.bottom() < graph.critic_bounds.top());
            assert!(
                graph
                    .actor_bounds
                    .intersect(graph.critic_bounds)
                    .is_negative()
            );
        }
    }

    #[test]
    fn zoom_changes_only_geometry_and_retains_true_indices() {
        for mode in [PpoNetworkMode::Condensed, PpoNetworkMode::Expanded] {
            let ordinary = build_layout(25, &[64, 64], mode, 12, 1.0);
            let zoomed = build_layout(25, &[64, 64], mode, 12, 0.5);
            for (original, smaller) in ordinary.layers.iter().zip(&zoomed.layers) {
                for (&(index, position), &(smaller_index, smaller_position)) in
                    original.nodes.iter().zip(&smaller.nodes)
                {
                    assert_eq!(index, smaller_index);
                    assert_eq!(position * 0.5, smaller_position);
                }
                assert_eq!(
                    original.omission.map(|o| o.count),
                    smaller.omission.map(|o| o.count)
                );
            }
            assert_eq!(zoomed.canvas_size, ordinary.canvas_size * 0.5);
            assert_eq!(zoomed.node_radius, ordinary.node_radius * 0.5);
        }
    }
}
