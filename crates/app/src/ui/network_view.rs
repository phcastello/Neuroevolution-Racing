use bevy_egui::egui;

const MIN_CANVAS_HEIGHT: f32 = 230.0;
const CANVAS_PADDING_X: f32 = 28.0;
const CANVAS_PADDING_Y: f32 = 28.0;
const LAYER_SPACING: f32 = 100.0;
const NODE_SPACING: f32 = 24.0;
const NODE_RADIUS: f32 = 10.0;

// Shared by the dense GA view and the branched PPO view. Keep activation and
// signed-weight colors consistent without assuming the networks share topology.
pub(super) fn activation_fill(activation: f32) -> egui::Color32 {
    let intensity = activation.abs().min(1.0);
    if activation >= 0.0 {
        egui::Color32::from_rgb(
            (35.0 + 35.0 * intensity) as u8,
            (85.0 + 150.0 * intensity) as u8,
            (95.0 + 125.0 * intensity) as u8,
        )
    } else {
        egui::Color32::from_rgb(
            (95.0 + 150.0 * intensity) as u8,
            (45.0 + 35.0 * intensity) as u8,
            (55.0 + 35.0 * intensity) as u8,
        )
    }
}

pub(super) fn weight_stroke(weight: f32, scale: f32) -> egui::Stroke {
    let magnitude = (weight.abs() / scale.max(f32::EPSILON)).sqrt();
    let alpha = (55.0 + 200.0 * magnitude) as u8;
    let color = if weight >= 0.0 {
        egui::Color32::from_rgba_unmultiplied(70, 220, 135, alpha)
    } else {
        egui::Color32::from_rgba_unmultiplied(245, 90, 90, alpha)
    };
    egui::Stroke::new(0.35 + 2.4 * magnitude, color)
}

fn expected_parameter_count(layers: &[usize]) -> Option<usize> {
    layers.windows(2).try_fold(0_usize, |total, pair| {
        pair[0]
            .checked_mul(pair[1])?
            .checked_add(pair[1])?
            .checked_add(total)
    })
}

fn canvas_size(layers: &[usize], available_width: f32) -> egui::Vec2 {
    let largest_layer = layers.iter().copied().max().unwrap_or(1) as f32;
    let layer_gaps = layers.len().saturating_sub(1) as f32;
    egui::vec2(
        available_width.max(layer_gaps * LAYER_SPACING + 2.0 * CANVAS_PADDING_X),
        MIN_CANVAS_HEIGHT.max(largest_layer * NODE_SPACING + 2.0 * CANVAS_PADDING_Y),
    )
}

pub fn draw_network(
    ui: &mut egui::Ui,
    layers: &[usize],
    parameters: &[f32],
    activations: &[Vec<f32>],
) {
    if layers.len() < 2
        || layers.contains(&0)
        || expected_parameter_count(layers) != Some(parameters.len())
        || activations.len() != layers.len()
        || activations
            .iter()
            .zip(layers)
            .any(|(values, &size)| values.len() != size)
    {
        ui.colored_label(
            egui::Color32::LIGHT_RED,
            "Network parameters do not match the configured architecture.",
        );
        return;
    }

    ui.small("Edges: green + / red \u{2212} weight \u{2022} Nodes: live activation");
    ui.monospace(
        layers
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" \u{2192} "),
    );

    let available_width = ui.available_width();
    egui::ScrollArea::horizontal()
        .id_salt(("neural_network_canvas", layers))
        .auto_shrink([false, true])
        .show(ui, |ui| {
            draw_network_canvas(ui, layers, parameters, activations, available_width);
        });
}

fn draw_network_canvas(
    ui: &mut egui::Ui,
    layers: &[usize],
    parameters: &[f32],
    activations: &[Vec<f32>],
    available_width: f32,
) {
    let desired = canvas_size(layers, available_width);
    let (response, painter) = ui.allocate_painter(desired, egui::Sense::hover());
    let rect = response
        .rect
        .shrink2(egui::vec2(CANVAS_PADDING_X, CANVAS_PADDING_Y));
    let last_layer = layers.len() - 1;
    let positions: Vec<Vec<egui::Pos2>> = layers
        .iter()
        .enumerate()
        .map(|(layer_index, &count)| {
            let x = egui::lerp(rect.x_range(), layer_index as f32 / last_layer as f32);
            (0..count)
                .map(|node| {
                    let y = egui::lerp(rect.y_range(), (node + 1) as f32 / (count + 1) as f32);
                    egui::pos2(x, y)
                })
                .collect()
        })
        .collect();
    let scale = parameters
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max)
        .max(f32::EPSILON);

    let mut cursor = 0;
    let mut biases: Vec<Vec<f32>> = vec![vec![0.0; layers[0]]];
    for layer_index in 0..last_layer {
        let input_count = layers[layer_index];
        let output_count = layers[layer_index + 1];
        for output in 0..output_count {
            for input in 0..input_count {
                let weight = parameters[cursor + output * input_count + input];
                painter.line_segment(
                    [
                        positions[layer_index][input],
                        positions[layer_index + 1][output],
                    ],
                    weight_stroke(weight, scale),
                );
            }
        }
        cursor += input_count * output_count;
        biases.push(parameters[cursor..cursor + output_count].to_vec());
        cursor += output_count;
    }

    for (layer_index, nodes) in positions.iter().enumerate() {
        for (node_index, &position) in nodes.iter().enumerate() {
            let activation = activations[layer_index][node_index];
            let fill = activation_fill(activation);
            let bias = biases[layer_index][node_index];
            let outline = if layer_index == 0 || bias >= 0.0 {
                egui::Color32::LIGHT_GREEN
            } else {
                egui::Color32::LIGHT_RED
            };
            painter.circle_filled(position, NODE_RADIUS, fill);
            painter.circle_stroke(position, NODE_RADIUS, egui::Stroke::new(1.3, outline));
            painter.text(
                position,
                egui::Align2::CENTER_CENTER,
                format!("{activation:+.1}"),
                egui::FontId::monospace(7.0),
                egui::Color32::WHITE,
            );
        }
    }

    painter.text(
        positions[0][0] + egui::vec2(0.0, -17.0),
        egui::Align2::CENTER_BOTTOM,
        "inputs",
        egui::FontId::proportional(11.0),
        egui::Color32::LIGHT_BLUE,
    );
    painter.text(
        positions[last_layer][0] + egui::vec2(0.0, -17.0),
        egui::Align2::CENTER_BOTTOM,
        "outputs",
        egui::FontId::proportional(11.0),
        egui::Color32::LIGHT_GREEN,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_grows_vertically_to_keep_large_layers_legible() {
        let size = canvas_size(&[6, 32, 32, 2], 360.0);

        assert_eq!(size.x, 360.0);
        assert_eq!(size.y, 32.0 * NODE_SPACING + 2.0 * CANVAS_PADDING_Y);
        assert!(size.y > MIN_CANVAS_HEIGHT);
    }

    #[test]
    fn canvas_grows_horizontally_for_deep_architectures() {
        let layers = [6, 8, 8, 8, 8, 8, 2];
        let size = canvas_size(&layers, 360.0);

        assert_eq!(
            size.x,
            (layers.len() - 1) as f32 * LAYER_SPACING + 2.0 * CANVAS_PADDING_X
        );
        assert!(size.x > 360.0);
    }

    #[test]
    fn parameter_count_supports_any_valid_dense_architecture() {
        assert_eq!(expected_parameter_count(&[6, 8, 2]), Some(74));
        assert_eq!(expected_parameter_count(&[6, 16, 32, 2]), Some(722));
        assert_eq!(expected_parameter_count(&[6, usize::MAX, 2]), None);
    }

    #[test]
    fn shared_colors_preserve_original_dense_network_presentation() {
        assert_eq!(activation_fill(0.0), egui::Color32::from_rgb(35, 85, 95));
        assert_eq!(activation_fill(1.0), egui::Color32::from_rgb(70, 235, 220));
        assert_eq!(activation_fill(-1.0), egui::Color32::from_rgb(245, 80, 90));
        assert_eq!(activation_fill(2.0), activation_fill(1.0));
        assert_eq!(
            weight_stroke(1.0, 1.0),
            egui::Stroke::new(2.75, egui::Color32::from_rgb(70, 220, 135)),
        );
        assert_eq!(
            weight_stroke(-1.0, 1.0),
            egui::Stroke::new(2.75, egui::Color32::from_rgb(245, 90, 90)),
        );
    }
}
