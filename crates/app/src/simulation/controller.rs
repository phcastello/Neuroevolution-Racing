use bevy::prelude::*;
use neuroevolution::neural::Mlp;

/// Number of scalar values the app supplies to a controller network.
pub const MLP_INPUT_SIZE: usize = 6;
/// Number of scalar values the app expects from a controller network.
pub const MLP_OUTPUT_SIZE: usize = 2;

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
    pub fn as_inputs(self) -> [f32; MLP_INPUT_SIZE] {
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
pub fn controls_from_network_outputs(outputs: [f32; MLP_OUTPUT_SIZE]) -> CarControls {
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

#[derive(Component)]
pub struct MlpController {
    mlp: Mlp,
    parameters: Vec<f32>,
    layer_sizes: Vec<usize>,
    activations: Vec<Vec<f32>>,
}

pub struct NetworkTelemetry<'a> {
    pub layer_sizes: &'a [usize],
    pub parameters: &'a [f32],
    pub activations: &'a [Vec<f32>],
}

impl MlpController {
    pub fn new(mlp: Mlp, parameters: &[f32]) -> Result<Self, &'static str> {
        if mlp.input_size() != MLP_INPUT_SIZE {
            return Err("The MLP input size must match the app controller input contract.");
        }
        if mlp.output_size() != MLP_OUTPUT_SIZE {
            return Err("The MLP output size must match the app controller output contract.");
        }
        if parameters.len() != mlp.parameter_count() {
            return Err("The telemetry parameter count must match the MLP.");
        }

        let layer_sizes = mlp.layer_sizes();
        let activations = layer_sizes.iter().map(|&size| vec![0.0; size]).collect();

        Ok(Self {
            mlp,
            parameters: parameters.to_vec(),
            layer_sizes,
            activations,
        })
    }

    pub fn telemetry(&self) -> NetworkTelemetry<'_> {
        NetworkTelemetry {
            layer_sizes: &self.layer_sizes,
            parameters: &self.parameters,
            activations: &self.activations,
        }
    }

    pub fn control_with_telemetry(&mut self, observation: &CarObservation) -> CarControls {
        let inputs = observation.as_inputs();
        self.activations = self.mlp.forward_with_trace(&inputs).unwrap();
        let outputs = self.activations.last().unwrap();

        controls_from_network_outputs([outputs[0], outputs[1]])
    }
}

impl CarController for MlpController {
    fn control(&mut self, observation: &CarObservation) -> CarControls {
        let inputs = observation.as_inputs();
        let outputs = self.mlp.forward(&inputs).unwrap();

        controls_from_network_outputs([outputs[0], outputs[1]])
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
    use neuroevolution::neural::{Activation, Architecture};

    fn single_layer_mlp(input_size: usize, output_size: usize) -> Mlp {
        let architecture =
            Architecture::new(vec![input_size, output_size], vec![Activation::Linear]).unwrap();
        Mlp::from_parameters(&architecture, &vec![0.0; architecture.parameter_count()]).unwrap()
    }

    #[test]
    fn observation_contains_exactly_six_scalar_inputs() {
        assert_eq!(MLP_INPUT_SIZE, 6);
        assert_eq!(
            std::mem::size_of::<CarObservation>(),
            MLP_INPUT_SIZE * std::mem::size_of::<f32>()
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
        assert_eq!(MLP_OUTPUT_SIZE, 2);
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

    #[test]
    fn mlp_controller_requires_the_app_network_contract() {
        let valid = single_layer_mlp(MLP_INPUT_SIZE, MLP_OUTPUT_SIZE);
        let valid_parameters = vec![0.0; valid.parameter_count()];
        assert!(MlpController::new(valid, &valid_parameters).is_ok());
        let wrong_input = single_layer_mlp(MLP_INPUT_SIZE - 1, MLP_OUTPUT_SIZE);
        let wrong_input_parameters = vec![0.0; wrong_input.parameter_count()];
        assert_eq!(
            MlpController::new(wrong_input, &wrong_input_parameters).err(),
            Some("The MLP input size must match the app controller input contract.")
        );
        let wrong_output = single_layer_mlp(MLP_INPUT_SIZE, MLP_OUTPUT_SIZE - 1);
        let wrong_output_parameters = vec![0.0; wrong_output.parameter_count()];
        assert_eq!(
            MlpController::new(wrong_output, &wrong_output_parameters).err(),
            Some("The MLP output size must match the app controller output contract.")
        );
    }

    #[test]
    fn mlp_controller_exposes_latest_forward_activations() {
        let mlp = single_layer_mlp(MLP_INPUT_SIZE, MLP_OUTPUT_SIZE);
        let parameters = vec![0.0; mlp.parameter_count()];
        let mut controller = MlpController::new(mlp, &parameters).unwrap();
        let observation = CarObservation {
            sensors: [0.1, 0.2, 0.3, 0.4, 0.5],
            normalized_speed: 0.6,
        };

        controller.control_with_telemetry(&observation);
        let telemetry = controller.telemetry();

        assert_eq!(telemetry.layer_sizes, &[MLP_INPUT_SIZE, MLP_OUTPUT_SIZE]);
        assert_eq!(telemetry.parameters, parameters);
        assert_eq!(telemetry.activations[0], observation.as_inputs());
        assert_eq!(telemetry.activations[1], vec![0.0; MLP_OUTPUT_SIZE]);
    }
}
