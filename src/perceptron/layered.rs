use super::activation_functions::*;
use super::perceptron::Perceptron;
use rand::rngs::SmallRng;
use std::vec::Vec;

#[derive(Clone)]
pub struct Layer {
    pub perceptrons: Vec<Perceptron>,
    pub activation: ActivationNames,
}

#[derive(Clone)]
pub struct LayeredNetwork {
    pub layers: Vec<Layer>,
}

impl Layer {
    pub fn new(inputs: usize, size: usize, rng: &mut SmallRng, act: ActivationNames) -> Layer {
        let mut perceptrons = Vec::new();

        for _i in 0..size {
            perceptrons.push(Perceptron::new(inputs, rng));
        }

        Layer {
            perceptrons,
            activation: act,
        }
    }

    pub fn compute(&self, inputs: &[f64]) -> Vec<f64> {
        let mut vec = Vec::new();

        for perceptron in self.perceptrons.clone() {
            vec.push(perceptron.compute(inputs, self.activation));
        }

        vec
    }

    pub fn backpropagate(&self, inputs: &[f64], n_err: &[f64], n_layer: Layer) -> Vec<f64> {
        let mut vec = Vec::new();
        for j in 0..self.perceptrons.len() {
            let mut sum = 0.0;
            for (i, l_err) in n_err.iter().enumerate() {
                sum += l_err * n_layer.perceptrons[i].weights[j];
            }
            vec.push(self.activation.get_deriv()(self.perceptrons[j].weighted_sum(inputs)) * (sum));
        }

        vec
    }
}

impl LayeredNetwork {
    pub fn new(
        rng: &mut SmallRng,
        representation: &[usize],
        activations: &[ActivationNames],
    ) -> LayeredNetwork {
        let mut net = Vec::new();
        let mut input_nb = representation[0];

        for (i, rep) in representation.iter().enumerate() {
            net.push(Layer::new(input_nb, *rep, rng, activations[i]));
            input_nb = *rep;
        }

        LayeredNetwork { layers: net }
    }

    pub fn result(&self, inputs: &[f64]) -> Vec<f64> {
        let outputs = self.compute(inputs);
        let len = outputs.len();
        outputs[len - 1].clone()
    }

    pub fn compute(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut tmp = inputs.to_vec();
        let mut outputs = Vec::new();

        for (i, layer) in self.layers.clone().iter().enumerate() {
            outputs.push(layer.compute(&tmp));
            tmp = outputs[i].clone();
        }

        outputs
    }

    fn get_input_map(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut vec = Vec::new();
        vec.push(inputs.to_vec());
        for layer in self.layers.clone() {
            let out = layer.compute(inputs);
            vec.push(out.clone());
        }

        vec
    }

    fn get_error_map(&self, inputs: &[f64], truth: &[f64]) -> Vec<Vec<f64>> {
        let mut errors = Vec::new();
        let last_layer = self.layers[self.layers.len() - 1].clone();
        let mut exit = Vec::new();
        let input_map = self.get_input_map(inputs);
        for (i, perceptron) in last_layer.perceptrons.iter().enumerate() {
            exit.push(perceptron.error(inputs, truth[i], last_layer.activation));
        }
        errors.push(exit);
        for l in (self.layers.len() - 2)..=0 {
            errors.insert(
                0,
                self.layers[l].backpropagate(
                    (input_map[l]).as_slice(),
                    errors[0].as_slice(),
                    self.layers[l + 1].clone(),
                ),
            );
        }

        errors
    }

    fn get_mean_weights(weights_maps: &[Vec<Vec<Vec<f64>>>]) -> Vec<Vec<Vec<f64>>> {
        let mut mean = weights_maps[0].clone();
        for m in 1..weights_maps.len() {
            for l in 0..weights_maps[m].len() {
                for i in 0..weights_maps[m][l].len() {
                    for j in 0..weights_maps[m][l][i].len() {
                        mean[l][i][j] += weights_maps[m][l][i][j];
                    }
                }
            }
        }

        for l in 0..mean.len() {
            for i in 0..mean[l].len() {
                for j in 0..mean[l][i].len() {
                    mean[l][i][j] = mean[l][i][j] / weights_maps.len() as f64;
                }
            }
        }

        mean
    }

    //input_maps is +1 the size of err_maps, careful with that
    fn get_weights(
        &self,
        err_maps: &[Vec<Vec<f64>>],
        input_maps: &[Vec<Vec<f64>>],
        rate: f64,
    ) -> Vec<Vec<Vec<f64>>> {
        let mut weights_maps = vec![vec![vec![vec![0.0]]]];
        for m in 0..err_maps.len() {
            for l in 0..self.layers.len() {
                for i in 0..self.layers[l].perceptrons.len() {
                    for j in 0..self.layers[l].perceptrons[i].weights.len() {
                        weights_maps[m][l][i][j] = self.layers[l].perceptrons[i].weights[j]
                            - rate * err_maps[m][l][i] * input_maps[m][l][j];
                    }
                }
            }
        }
        LayeredNetwork::get_mean_weights(&weights_maps)
    }

    pub fn backpropagation(&mut self, inputs: &[Vec<f64>], truth: &[Vec<f64>], rate: f64) {
        let mut err_maps = Vec::new();
        let mut input_maps = Vec::new();
        for i in 0..inputs.len() {
            err_maps.push(self.get_error_map(&inputs[i], &truth[i]));
            input_maps.push(self.get_input_map(&inputs[i]));
        }
        let weights = self.get_weights(&err_maps, &input_maps, rate);
        for l in 0..self.layers.len() {
            for i in 0..self.layers[l].perceptrons.len() {
                for j in 0..self.layers[l].perceptrons[i].weights.len() {
                    self.layers[l].perceptrons[i].weights[j] = weights[l][i][j];
                }
            }
        }
    }
}
