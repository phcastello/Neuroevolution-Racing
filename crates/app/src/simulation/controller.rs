use bevy::prelude::*;

#[derive(Component, Clone, Copy, Debug)]
pub struct CarObservation {
    pub sensors: [f32; 5],
    pub normalized_speed: f32,
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct CarControls {
    pub acceleration: f32,
    pub steering: f32,
}

impl CarControls {
    pub const NEUTRAL: Self = Self {
        acceleration: 0.0,
        steering: 0.0,
    };

    pub fn new(acceleration: f32, steering: f32) -> Self {
        Self {
            acceleration: sanitize_control(acceleration),
            steering: sanitize_control(steering),
        }
    }
}

fn sanitize_control(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Small seam that will later allow the app to call a controller backed by the
/// manually implemented MLP without coupling that MLP to Bevy.
pub trait CarController {
    fn control(&mut self, observation: &CarObservation) -> CarControls;
}

/// Privileged centerline information used only by the infrastructure driver.
/// It is deliberately separate from `CarObservation`, which is the future MLP
/// boundary and always contains exactly five sensors plus normalized speed.
#[derive(Clone, Copy, Debug, Default)]
pub struct TemporaryNavigationContext {
    pub target_bearing: f32,
}

/// Deliberately simple infrastructure-only controller. It follows a centerline
/// look-ahead target and steers away from nearby walls; it performs no learning.
#[derive(Default)]
pub struct TemporaryController {
    navigation: TemporaryNavigationContext,
}

impl TemporaryController {
    pub fn set_navigation_context(&mut self, navigation: TemporaryNavigationContext) {
        self.navigation = navigation;
    }
}

impl CarController for TemporaryController {
    fn control(&mut self, observation: &CarObservation) -> CarControls {
        let [left, front_left, front, front_right, right] = observation.sensors;
        let obstacle_steering = (front_left - front_right) * 0.9 + (left - right) * 0.35;
        let steering = (self.navigation.target_bearing * 1.65 + obstacle_steering).clamp(-1.0, 1.0);

        let corner_slowdown = self.navigation.target_bearing.abs().clamp(0.0, 1.0);
        let desired_speed = 0.92 - corner_slowdown * 0.30;
        let acceleration = if front < 0.16 {
            -0.75
        } else if observation.normalized_speed < desired_speed {
            1.0
        } else {
            -0.12
        };

        CarControls::new(acceleration, steering)
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
    fn observation_contains_exactly_six_scalar_inputs() {
        assert_eq!(
            std::mem::size_of::<CarObservation>(),
            6 * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn controls_clamp_to_the_canonical_actuator_range() {
        assert_eq!(CarControls::new(2.5, -3.0), CarControls::new(1.0, -1.0));
        assert_eq!(
            CarControls::new(f32::NAN, f32::INFINITY),
            CarControls::NEUTRAL
        );
    }

    #[test]
    fn signed_angle_uses_shortest_turn() {
        let almost_pi = std::f32::consts::PI - 0.1;
        let direction = Vec2::from_angle(-almost_pi);
        let angle = signed_angle_to(almost_pi, direction);
        assert!((angle - 0.2).abs() < 0.001);
    }
}
