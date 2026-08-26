use crate::neural::activation::Activation;

pub struct Architecture {
    layer_sizes: Vec<usize>,
    activations: Vec<Activation>,
}

impl Architecture {
    pub fn new(layer_sizes: Vec<usize>,activations: Vec<Activation>) -> Result<Self, &'static str> {
        if layer_sizes.iter().any(|&layer_value| layer_value == 0) {
            return Err("The architecture must have at least one neuron in each layer.");
        }
        if layer_sizes.len() < 2 {
            return Err("The architecture must have at least two layers (input and output).");
        }
        if activations.len() != layer_sizes.len() - 1 {
            return Err("The number of activation functions must be equal to the number of layers minus one (input layer does not have an activation function).");
        }

        Ok(Self {
            layer_sizes,
            activations,
        })
    }

    pub fn input_size(&self) -> usize {
        self.layer_sizes.first().copied().unwrap()
    }

    pub fn output_size(&self) -> usize {
        self.layer_sizes.last().copied().unwrap()
    }

    pub fn parameter_count(&self) -> usize {
        self.layer_sizes
            .windows(2)
            .map(|pair| {
                let input_size = pair[0];
                let output_size = pair[1];

                input_size * output_size + output_size
            })
            .sum()
    }

    pub fn layer_sizes(&self) -> &[usize] {
        &self.layer_sizes
    }

    pub fn activations(&self) -> &[Activation] {
        &self.activations
    }
}

#[cfg(test)]
mod tests {
    use super::Architecture;
    use crate::neural::activation::Activation;

    #[test]
    fn new_returns_error_with_fewer_than_two_layers() {
        let architecture = Architecture::new(vec![6], vec![]);

        assert!(architecture.is_err());
    }

    #[test]
    fn new_returns_error_when_any_layer_has_zero_neurons() {
        let architecture =
            Architecture::new(vec![6, 0, 2], vec![Activation::Relu, Activation::Linear]);

        assert!(architecture.is_err());
    }

    #[test]
    fn new_returns_error_with_wrong_number_of_activations() {
        let architecture = Architecture::new(vec![6, 8, 2], vec![Activation::Relu]);

        assert!(architecture.is_err());
    }

    #[test]
    fn input_size_returns_the_first_layer_size() {
        let architecture =
            Architecture::new(vec![6, 8, 2], vec![Activation::Relu, Activation::Linear]).unwrap();

        assert_eq!(architecture.input_size(), 6);
    }

    #[test]
    fn output_size_returns_the_last_layer_size() {
        let architecture =
            Architecture::new(vec![6, 8, 2], vec![Activation::Relu, Activation::Linear]).unwrap();

        assert_eq!(architecture.output_size(), 2);
    }

    #[test]
    fn parameter_count_includes_weights_and_biases_for_each_layer_pair() {
        let architecture =
            Architecture::new(vec![6, 8, 2], vec![Activation::Relu, Activation::Linear]).unwrap();

        assert_eq!(architecture.parameter_count(), 74);
    }

    #[test]
    fn parameter_count_sums_all_layer_pairs() {
        let architecture = Architecture::new(
            vec![6, 8, 8, 2],
            vec![Activation::Relu, Activation::Relu, Activation::Linear],
        )
        .unwrap();

        assert_eq!(architecture.parameter_count(), 146);
    }
}
