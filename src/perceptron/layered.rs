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
}

impl Layered_network {
    pub fn new(rng: &mut SmallRng, representation: &[usize], activations: &[ActivationNames]) -> Layered_network {
        let mut net = Vec::new();
        let mut input_nb = representation[0];

        for (rep, i) in representation.iter().enumerate() {
            net.push(Layer::new(input_nb, rep, rng, &activations[*i]));
            input_nb = rep;
        }

        Layered_network {
            layers: net,
        }
    }

    pub fn output(&self, inputs: &[f64]) -> Vec<f64> {
        let outputs = self.compute(inputs);
        outputs[outputs.len() - 1]
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

    pub fn backpropagation(&mut self, out_vec: &[Vec<Vec<f64>>], truth: &[Vec<f64>]) {
        for i in out_vec.len() {
            for
        }
    }
}