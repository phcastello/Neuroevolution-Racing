use crate::neural::activation::Activation;

/*
convention: weights[output_neuron][input]
example in a 3 -> 2 layer:
weights = [
    [w00, w01, w02],   ← neurônio 0
    [w10, w11, w12],   ← neurônio 1
]

biases = [
    b0,
    b1,
]
*/

pub struct DenseLayer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    activation: Activation,
}

impl DenseLayer {
    /*
    garantir que a quantidade de pesos em weights seja igual em todas as linhas. Cada linha representa
    um neurônio;
    a quantidade de valores em biases tem que ser igual a quantidade de linhas em weights
    */
    pub fn new(
        weights: Vec<Vec<f32>>,
        biases: Vec<f32>,
        activation: Activation,
    ) -> Result<Self, &'static str> {
        if biases.len() != weights.len() {
            return Err("The number of biases must be equal to the number of neurons.");
        }

        let Some(first_neuron_weights) = weights.first() else {
            return Err("A layer must have at least one neuron.");
        };

        if first_neuron_weights.is_empty() {
            return Err("Each neuron must have at least one input weight.");
        }

        if !weights
            .iter()
            .all(|neuron| neuron.len() == first_neuron_weights.len())
        {
            return Err("All neurons must have the same number of weights.");
        }

        Ok(Self {
            weights,
            biases,
            activation,
        })
    }

    pub fn output_size(&self) -> usize {
        self.weights.len()
    }

    pub fn input_size(&self) -> usize {
        self.weights[0].len()
    }

    pub fn forward(&self, inputs: &[f32]) -> Result<Vec<f32>, &'static str> {
        if inputs.len() != self.input_size() {
            return Err("The number of inputs must be equal to the number of input weights.");
        }

        let neurons_count = self.output_size(); // retorna o tamanho da saida
        let neuron_weights_count = self.input_size(); // retorna o tamanho da entrada
        let mut output: Vec<f32> = Vec::with_capacity(neurons_count);

        for neuron in 0..neurons_count {
            let mut iteration_sum: f32 = 0.0;
            for j in 0..neuron_weights_count {
                iteration_sum += inputs[j] * self.weights[neuron][j];
            }

            output.push(self.activation.apply(iteration_sum + self.biases[neuron]));
        }

        Ok(output)
    }
}

#[cfg(test)]
mod test {
    use crate::neural::{activation::Activation, layer::DenseLayer};

    #[test]
    fn new_returns_error_when_bias_count_differs_from_neuron_count() {
        let result = DenseLayer::new(
            vec![vec![0.5, 1.0], vec![1.0, -0.5]],
            vec![0.25],
            Activation::Linear,
        );

        assert_eq!(
            result.err(),
            Some("The number of biases must be equal to the number of neurons.")
        );
    }

    #[test]
    fn new_returns_error_when_layer_has_no_neurons() {
        let result = DenseLayer::new(vec![], vec![], Activation::Relu);

        assert_eq!(result.err(), Some("A layer must have at least one neuron."));
    }

    #[test]
    fn new_returns_error_when_a_neuron_has_no_input_weights() {
        let result = DenseLayer::new(vec![vec![]], vec![0.0], Activation::Tanh);

        assert_eq!(
            result.err(),
            Some("Each neuron must have at least one input weight.")
        );
    }

    #[test]
    fn new_returns_error_when_neurons_have_different_input_sizes() {
        let result = DenseLayer::new(
            vec![vec![0.5, 1.0], vec![1.0]],
            vec![0.25, -0.5],
            Activation::Linear,
        );

        assert_eq!(
            result.err(),
            Some("All neurons must have the same number of weights.")
        );
    }

    #[test]
    fn forward_works() {
        let weights = vec![vec![0.5, 1.0, -2.0], vec![1.0, -0.5, 0.0]];
        let biases = vec![0.25, -0.5];
        let activation = Activation::Linear;
        let inputs = vec![2.0, -1.0, 0.5];

        let layer = DenseLayer::new(weights, biases, activation).unwrap();
        let output = layer.forward(&inputs).unwrap();

        println!("{:#?}", output);

        assert_eq!(output, vec![-0.75, 2.0])
    }

    #[test]
    fn forward_returns_error_when_input_size_is_incorrect() {
        let layer = DenseLayer::new(vec![vec![0.5, 1.0]], vec![0.25], Activation::Linear).unwrap();

        let result = layer.forward(&[2.0]);

        assert_eq!(
            result.err(),
            Some("The number of inputs must be equal to the number of input weights.")
        );
    }

    #[test]
    fn forward_applies_tanh_to_each_neuron_result() {
        let layer = DenseLayer::new(vec![vec![1.0]], vec![0.0], Activation::Tanh).unwrap();

        let output = layer.forward(&[1.0]).unwrap();

        assert_eq!(output, vec![1.0_f32.tanh()]);
    }

    #[test]
    fn forward_applies_relu_to_each_neuron_result() {
        let layer = DenseLayer::new(
            vec![vec![1.0], vec![1.0]],
            vec![-2.0, 0.5],
            Activation::Relu,
        )
        .unwrap();

        let output = layer.forward(&[1.0]).unwrap();

        assert_eq!(output, vec![0.0, 1.5]);
    }
}
