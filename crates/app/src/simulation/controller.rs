use bevy::prelude::*;

/// Six-scalar controller input boundary.
///
/// `sensors` is ordered left +60 degrees, left +30 degrees, front, right -30
/// degrees, right -60 degrees. Normalized speed is always the final input.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct CarObservation {
    pub sensors: [f32; 5],
    pub normalized_speed: f32,
}

impl CarObservation {
    pub const INITIAL: Self = Self {
        sensors: [1.0; 5],
        normalized_speed: 0.0,
    };

    /// Returns the stable future-MLP input order without exposing Bevy to the
    /// pure AI crate.
    pub fn as_inputs(self) -> [f32; 6] {
        let [left_60, left_30, front, right_30, right_60] = self.sensors;
        [
            left_60,
            left_30,
            front,
            right_30,
            right_60,
            self.normalized_speed,
        ]
    }
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

/// Adapts the stable future-network output order `[steering, acceleration]`
/// to the app's existing `CarControls::new(acceleration, steering)` API.
pub fn controls_from_network_outputs(outputs: [f32; 2]) -> CarControls {
    CarControls::new(outputs[1], outputs[0])
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
        let [left_60, left_30, front, right_30, right_60] = observation.sensors;
        let obstacle_steering = (left_30 - right_30) * 0.9 + (left_60 - right_60) * 0.35;
        let steering = (self.navigation.target_bearing * 1.65 + obstacle_steering).clamp(-1.0, 1.0);

        // Do not impose a target-speed ceiling on the temporary training cars.
        // They keep accelerating until an immediate frontal obstacle requires braking.
        let acceleration = if front < 0.16 { -0.75 } else { 1.0 };

        controls_from_network_outputs([steering, acceleration])
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

        let observation = CarObservation {
            sensors: [0.1, 0.2, 0.3, 0.4, 0.5],
            normalized_speed: 0.6,
        };
        assert_eq!(observation.as_inputs(), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
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
    fn network_outputs_map_steering_then_acceleration() {
        let controls = controls_from_network_outputs([0.7, -0.3]);
        assert_eq!(controls.steering, 0.7);
        assert_eq!(controls.acceleration, -0.3);

        let sanitized = controls_from_network_outputs([2.0, f32::NAN]);
        assert_eq!(sanitized.steering, 1.0);
        assert_eq!(sanitized.acceleration, 0.0);
    }

    #[test]
    fn signed_angle_uses_shortest_turn() {
        let almost_pi = std::f32::consts::PI - 0.1;
        let direction = Vec2::from_angle(-almost_pi);
        let angle = signed_angle_to(almost_pi, direction);
        assert!((angle - 0.2).abs() < 0.001);
    }

    #[test]
    fn temporary_controller_keeps_accelerating_at_high_normalized_speed() {
        let mut controller = TemporaryController::default();
        let controls = controller.control(&CarObservation {
            sensors: [1.0; 5],
            normalized_speed: 0.99,
        });

        assert_eq!(controls.acceleration, 1.0);
    }
}
