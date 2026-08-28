use crate::neural::{architecture::Architecture, layer::DenseLayer};

pub struct Mlp {
    layers: Vec<DenseLayer>,
}

impl Mlp {
    pub fn new(layers: Vec<DenseLayer>) -> Result<Self, &'static str> {
        if layers.is_empty() {
            return Err("The MLP must have at least one layer.");
        }
        if !layers
            .windows(2)
            .all(|pair| pair[0].output_size() == pair[1].input_size())
        {
            return Err(
                "The output size of each layer must match the input size of the next layer.",
            );
        }

        Ok(Self { layers })
    }

    pub fn output_size(&self) -> usize {
        self.layers.last().unwrap().output_size()
    }

    pub fn input_size(&self) -> usize {
        self.layers.first().unwrap().input_size()
    }

    pub fn forward(&self, inputs: &[f32]) -> Result<Vec<f32>, &'static str> {
        let mut trace = self.forward_with_trace(inputs)?;
        Ok(trace.pop().unwrap())
    }

    /// Evaluates the network and returns the input plus every layer output.
    /// This generic introspection boundary can feed diagnostics or visualizers
    /// without making the neural crate depend on either of them.
    pub fn forward_with_trace(&self, inputs: &[f32]) -> Result<Vec<Vec<f32>>, &'static str> {
        let mut current_output: Vec<f32> = inputs.to_vec();
        let mut trace = Vec::with_capacity(self.layers.len() + 1);
        trace.push(current_output.clone());

        for layer in &self.layers {
            current_output = layer.forward(&current_output)?;
            trace.push(current_output.clone());
        }

        Ok(trace)
    }

    pub fn layer_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::with_capacity(self.layers.len() + 1);
        sizes.push(self.input_size());
        sizes.extend(self.layers.iter().map(DenseLayer::output_size));
        sizes
    }

    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.input_size() * layer.output_size() + layer.output_size())
            .sum()
    }

    pub fn from_parameters(
        architecture: &Architecture,
        parameters: &[f32],
    ) -> Result<Self, &'static str> {
        if parameters.len() != architecture.parameter_count() {
            return Err(
                "The number of parameters does not match the architecture's parameter count.",
            );
        }

        let mut layers: Vec<DenseLayer> = Vec::new();
        let mut cursor = 0;

        for i in 0..architecture.layer_sizes().len() - 1 {
            let cols = architecture.layer_sizes()[i];
            let rows = architecture.layer_sizes()[i + 1];

            let mut weights: Vec<Vec<f32>> = Vec::with_capacity(rows);

            for _ in 0..rows {
                let mut row: Vec<f32> = Vec::with_capacity(cols);

                for _ in 0..cols {
                    row.push(parameters[cursor]);
                    cursor += 1;
                }

                weights.push(row);
            }

            let mut biases: Vec<f32> = Vec::with_capacity(rows);

            for _ in 0..rows {
                biases.push(parameters[cursor]);
                cursor += 1;
            }

            let activation = architecture.activations()[i];

            layers.push(DenseLayer::new(weights, biases, activation)?)
        }

        Self::new(layers)
    }
}

#[cfg(test)]
mod test {
    use crate::neural::{
        activation::Activation, architecture::Architecture, layer::DenseLayer, mlp::Mlp,
    };

    #[test]
    fn new_returns_error_when_it_has_no_layers() {
        let result = Mlp::new(vec![]);

        assert_eq!(result.err(), Some("The MLP must have at least one layer."));
    }

    #[test]
    fn new_returns_error_when_consecutive_layers_have_incompatible_sizes() {
        let first_layer = DenseLayer::new(
            vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            vec![0.0, 0.0],
            Activation::Linear,
        )
        .unwrap();
        let second_layer =
            DenseLayer::new(vec![vec![0.5, 0.6, 0.7]], vec![0.0], Activation::Relu).unwrap();

        let result = Mlp::new(vec![first_layer, second_layer]);

        assert_eq!(
            result.err(),
            Some("The output size of each layer must match the input size of the next layer.")
        );
    }

    #[test]
    fn new_creates_network_when_all_layer_sizes_are_compatible() {
        let input_to_hidden = DenseLayer::new(
            vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.5, 0.6]],
            vec![0.0, 0.0, 0.0],
            Activation::Relu,
        )
        .unwrap();
        let hidden_to_output =
            DenseLayer::new(vec![vec![0.7, 0.8, 0.9]], vec![0.0], Activation::Linear).unwrap();

        let result = Mlp::new(vec![input_to_hidden, hidden_to_output]);

        assert!(result.is_ok());
    }

    #[test]
    fn forward_passes_each_layer_output_to_the_next_layer() {
        let first_layer = DenseLayer::new(
            vec![vec![1.0, 0.5], vec![-0.5, 2.0]],
            vec![0.5, -1.0],
            Activation::Linear,
        )
        .unwrap();
        let second_layer =
            DenseLayer::new(vec![vec![3.0, -0.25]], vec![1.0], Activation::Linear).unwrap();
        let mlp = Mlp::new(vec![first_layer, second_layer]).unwrap();

        let output = mlp.forward(&[2.0, -1.0]).unwrap();

        // Primeira camada: [2.0, -4.0]; segunda camada: [8.0].
        assert_eq!(output, vec![8.0]);
    }

    #[test]
    fn forward_trace_contains_inputs_and_each_layer_activation() {
        let first_layer = DenseLayer::new(
            vec![vec![1.0, 0.5], vec![-0.5, 2.0]],
            vec![0.5, -1.0],
            Activation::Linear,
        )
        .unwrap();
        let second_layer =
            DenseLayer::new(vec![vec![3.0, -0.25]], vec![1.0], Activation::Linear).unwrap();
        let mlp = Mlp::new(vec![first_layer, second_layer]).unwrap();

        assert_eq!(
            mlp.forward_with_trace(&[2.0, -1.0]).unwrap(),
            vec![vec![2.0, -1.0], vec![2.0, -4.0], vec![8.0]]
        );
        assert_eq!(mlp.layer_sizes(), vec![2, 2, 1]);
        assert_eq!(mlp.parameter_count(), 9);
    }

    #[test]
    fn input_size_returns_the_first_layer_input_size() {
        let first_layer = DenseLayer::new(
            vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]],
            vec![0.0; 3],
            Activation::Linear,
        )
        .unwrap();
        let second_layer =
            DenseLayer::new(vec![vec![1.0, 1.0, 1.0]], vec![0.0], Activation::Linear).unwrap();
        let mlp = Mlp::new(vec![first_layer, second_layer]).unwrap();

        assert_eq!(mlp.input_size(), 2);
    }

    #[test]
    fn output_size_returns_the_last_layer_output_size() {
        let first_layer = DenseLayer::new(
            vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![1.0, 1.0]],
            vec![0.0; 3],
            Activation::Linear,
        )
        .unwrap();
        let second_layer =
            DenseLayer::new(vec![vec![1.0, 1.0, 1.0]], vec![0.0], Activation::Linear).unwrap();
        let mlp = Mlp::new(vec![first_layer, second_layer]).unwrap();

        assert_eq!(mlp.output_size(), 1);
    }

    #[test]
    fn forward_returns_error_when_input_size_is_incorrect() {
        let layer = DenseLayer::new(vec![vec![1.0, 1.0]], vec![0.0], Activation::Linear).unwrap();
        let mlp = Mlp::new(vec![layer]).unwrap();

        let result = mlp.forward(&[1.0]);

        assert_eq!(
            result.err(),
            Some("The number of inputs must be equal to the number of input weights.")
        );
    }

    #[test]
    fn from_parameters_returns_error_when_parameter_count_does_not_match_architecture() {
        let architecture =
            Architecture::new(vec![2, 2, 1], vec![Activation::Linear, Activation::Linear]).unwrap();

        let result = Mlp::from_parameters(&architecture, &[1.0; 8]);

        assert_eq!(
            result.err(),
            Some("The number of parameters does not match the architecture's parameter count.")
        );
    }

    #[test]
    fn from_parameters_maps_weights_and_biases_layer_by_layer() {
        let architecture =
            Architecture::new(vec![2, 2, 1], vec![Activation::Linear, Activation::Linear]).unwrap();
        let parameters = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let mlp = Mlp::from_parameters(&architecture, &parameters).unwrap();
        let output = mlp.forward(&[1.0, 1.0]).unwrap();

        assert_eq!(output, vec![169.0]);
    }
}
