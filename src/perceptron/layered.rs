use super::activation_functions::*;
use super::perceptron::Perceptron;
use std::vec::Vec;
use rand::rngs::SmallRng;

#[derive(Clone)]
pub struct Layer {
    pub perceptrons: Vec<Perceptron>,
    pub activation: ActivationNames,
}

#[derive(Clone)]
pub struct Layered_network {
    pub layers: Vec<Layer>,

}

impl Layer {
    pub fn new(inputs: usize, size: usize, rng: &mut SmallRng, act: &ActivationNames) -> Layer {
        let mut perceptrons = Vec::new();

        for i in 0..size {
            perceptrons.push(Perceptron::new(inputs, rng));
        }

        Layer {
            perceptrons,
            activation: *act,
        }
    }

    pub fn compute(&self, inputs: &[f64]) -> Vec<f64> {
        let mut vec = Vec::new();

        for perceptron in self.perceptrons.clone() {
            vec.push(perceptron.compute(inputs, &self.activation));
        }

        vec
    }

    pub fn backpropagate(&self, inputs: &[f64], n_err: &[f64], n_layer: Layer) -> Vec<f64> {
        let mut vec = Vec::new();
        for j in 0..self.perceptrons.len() {
            let mut sum = 0.0;
            for i in 0..n_err.len() {
                sum += n_err[i] * n_layer.perceptrons[i].weights[j];
            }
            vec.push(self.activation.get_deriv()(&self.perceptrons[j].weighted_sum(inputs)) * (sum));
        }

        vec
    }

}

impl Layered_network {
    pub fn new(rng: &mut SmallRng, representation: &[usize], activations: &[ActivationNames]) -> Layered_network {
        let mut net = Vec::new();
        let mut input_nb = representation[0];

        for (i, rep) in representation.iter().enumerate() {
            net.push(Layer::new(input_nb, *rep, rng, &activations[i]));
            input_nb = *rep;
        }

        Layered_network {
            layers: net,
        }
    }

    pub fn result(&self, inputs: &[f64]) -> Vec<f64> {
        let outputs = self.compute(inputs);
        let len = outputs.len();
        outputs[len - 1].clone()
    }

    pub fn compute(&self,inputs: &[f64]) -> Vec<Vec<f64>> {
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
        vec.push(inputs.into_vec());
        for layer in self.layers {
            let out = layer.compute(inputs);
            vec.push(out.clone());
        }

        vec
    }

    fn get_error_map(&self, inputs: &[f64], truth: &[f64]) -> Vec<Vec<f64>> {
        let outputs = self.compute(inputs);
        let mut errors = Vec::new();
        let last_layer = self.layers[self.layers.len() - 1].clone();
        let mut exit = Vec::new()
        let input_map = self.get_input_map(inputs);
        for i in 0..last_layer.perceptrons.len() {
            exit.push(last_layer.perceptrons[i].error(inputs, truth[i], &last_layer.activation));
        }
        errors.push(exit);
        for l in (self.layers.len() - 2)..=0 {
            errors.insert(0, self.layers[l].backpropagate((input_map[l]).as_slice(), errors[0].as_slice(), self.layers[l + 1].clone()));
        }

        errors
    }

    pub fn backpropagation(&mut self, inputs: &[Vec<f64>], truth: &[Vec<f64>]) {

    }
}