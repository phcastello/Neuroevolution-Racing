#[derive(Clone, Copy)]
pub enum Activation{
    Tanh,
    Relu,
    Linear,
}

impl Activation {
    pub fn apply(&self, value: f32) -> f32{
        match self {
            Activation::Linear => value,
            Activation::Relu => value.max(0.0),
            Activation::Tanh => value.tanh(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Activation;

    #[test]
    fn tanh_of_zero_is_zero() {
        assert_eq!(Activation::Tanh.apply(0.0), 0.0);
    }

    #[test]
    fn tanh_is_an_odd_function() {
        let value = 0.75;
        let positive = Activation::Tanh.apply(value);
        let negative = Activation::Tanh.apply(-value);

        assert!((negative + positive).abs() < f32::EPSILON);
    }

    #[test]
    fn tanh_output_is_between_negative_one_and_one() {
        let result = Activation::Tanh.apply(2.0);

        assert!((-1.0..=1.0).contains(&result));
    }

    #[test]
    fn relu_clamps_negative_values_to_zero() {
        assert_eq!(Activation::Relu.apply(-2.0), 0.0);
        assert_eq!(Activation::Relu.apply(2.0), 2.0);
    }

    #[test]
    fn linear_preserves_the_value() {
        assert_eq!(Activation::Linear.apply(-2.5), -2.5);
    }
}
