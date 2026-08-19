use bevy_egui::egui;

/// Static architecture preview only. It deliberately has no weights,
/// activations, feed-forward logic, or dependency on the AI crate.
pub fn draw_network_placeholder(ui: &mut egui::Ui) {
    ui.label(egui::RichText::new("Placeholder: 6 → 8 → 2").italics());
    let desired = egui::vec2(ui.available_width(), 210.0);
    let (response, painter) = ui.allocate_painter(desired, egui::Sense::hover());
    let rect = response.rect.shrink(12.0);
    let layers = [6_usize, 8, 2];
    let colors = [
        egui::Color32::from_rgb(70, 180, 230),
        egui::Color32::from_rgb(175, 105, 230),
        egui::Color32::from_rgb(80, 220, 135),
    ];

    let positions: Vec<Vec<egui::Pos2>> = layers
        .iter()
        .enumerate()
        .map(|(layer_index, &count)| {
            let x = egui::lerp(rect.x_range(), layer_index as f32 / 2.0);
            (0..count)
                .map(|node| {
                    let y = egui::lerp(rect.y_range(), (node + 1) as f32 / (count + 1) as f32);
                    egui::pos2(x, y)
                })
                .collect()
        })
        .collect();

    for adjacent in positions.windows(2) {
        for &from in &adjacent[0] {
            for &to in &adjacent[1] {
                painter.line_segment(
                    [from, to],
                    egui::Stroke::new(0.45, egui::Color32::from_white_alpha(35)),
                );
            }
        }
    }
    for (layer_index, nodes) in positions.iter().enumerate() {
        for &position in nodes {
            painter.circle_filled(position, 6.0, colors[layer_index]);
            painter.circle_stroke(position, 6.0, egui::Stroke::new(1.0, egui::Color32::WHITE));
        }
    }
}
