use bevy_egui::egui;

pub fn draw_network(
    ui: &mut egui::Ui,
    layers: &[usize],
    parameters: &[f32],
    activations: &[Vec<f32>],
) {
    let expected_parameters = layers
        .windows(2)
        .map(|pair| pair[0] * pair[1] + pair[1])
        .sum::<usize>();
    if layers.len() < 2
        || parameters.len() != expected_parameters
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

    ui.small("Edges: green + / red − weight • Nodes: live activation");
    let desired = egui::vec2(ui.available_width(), 230.0);
    let (response, painter) = ui.allocate_painter(desired, egui::Sense::hover());
    let rect = response.rect.shrink2(egui::vec2(18.0, 14.0));
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
                let magnitude = (weight.abs() / scale).sqrt();
                let alpha = (55.0 + 200.0 * magnitude) as u8;
                let color = if weight >= 0.0 {
                    egui::Color32::from_rgba_unmultiplied(70, 220, 135, alpha)
                } else {
                    egui::Color32::from_rgba_unmultiplied(245, 90, 90, alpha)
                };
                painter.line_segment(
                    [
                        positions[layer_index][input],
                        positions[layer_index + 1][output],
                    ],
                    egui::Stroke::new(0.35 + 2.4 * magnitude, color),
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
            let intensity = activation.abs().min(1.0);
            let fill = if activation >= 0.0 {
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
            };
            let bias = biases[layer_index][node_index];
            let outline = if layer_index == 0 || bias >= 0.0 {
                egui::Color32::LIGHT_GREEN
            } else {
                egui::Color32::LIGHT_RED
            };
            painter.circle_filled(position, 10.0, fill);
            painter.circle_stroke(position, 10.0, egui::Stroke::new(1.3, outline));
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
