use bevy::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct ControllerInputs {
    pub sensors: [f32; 5],
    pub normalized_speed: f32,
    pub target_bearing: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CarControls {
    pub acceleration: f32,
    pub steering: f32,
}

/// Small seam that will later allow the app to call a controller backed by the
/// manually implemented MLP without coupling that MLP to Bevy.
pub trait CarController {
    fn control(&mut self, inputs: &ControllerInputs) -> CarControls;
}

/// Deliberately simple infrastructure-only controller. It follows the next
/// checkpoint and steers away from nearby walls; it performs no learning.
#[derive(Default)]
pub struct TemporaryController;

impl CarController for TemporaryController {
    fn control(&mut self, inputs: &ControllerInputs) -> CarControls {
        let [left, front_left, front, front_right, right] = inputs.sensors;
        let obstacle_steering = (front_left - front_right) * 0.9 + (left - right) * 0.35;
        let steering = (inputs.target_bearing * 1.65 + obstacle_steering).clamp(-1.0, 1.0);

        let corner_slowdown = inputs.target_bearing.abs().clamp(0.0, 1.0);
        let desired_speed = 0.92 - corner_slowdown * 0.30;
        let acceleration = if front < 0.16 {
            -0.75
        } else if inputs.normalized_speed < desired_speed {
            1.0
        } else {
            -0.12
        };

        CarControls {
            acceleration,
            steering,
        }
    }
}

pub fn signed_angle_to(from_heading: f32, direction: Vec2) -> f32 {
    let target = direction.y.atan2(direction.x);
    wrap_angle(target - from_heading)
}

fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_angle_uses_shortest_turn() {
        let almost_pi = std::f32::consts::PI - 0.1;
        let direction = Vec2::from_angle(-almost_pi);
        let angle = signed_angle_to(almost_pi, direction);
        assert!((angle - 0.2).abs() < 0.001);
    }
}
