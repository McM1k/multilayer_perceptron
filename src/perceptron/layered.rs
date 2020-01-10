use super::activation_functions::*;
use super::perceptron::Perceptron;
use std::vec::Vec;
use rand::rngs::SmallRng;

pub struct Layer {
    pub perceptrons: Vec<Perceptron>,
    pub activation: ActivationNames,
}

pub struct Layered_network {
    pub layers: Vec<Layer>,

}

impl Layer {
    pub fn new(inputs: usize, size: usize, rng: &SmallRng, act: &ActivationNames) -> Layer {
        let mut perceptrons = Vec::new();

        for i in 0..size {
            perceptrons.push(Perceptron::new(inputs, rng));
        }

        Layer {
            perceptrons,
            activation: *act,
        }
    }

    pub fn get_layer_result(inputs: &[f64], layer: Layer) -> Vec<f64> {
        let mut vec = Vec::new();

        for perceptron in layer.perceptrons {
            vec.push(perceptron.compute(inputs, layer.activation));
        }

        vec
    }
}

impl Layered_network {
    pub fn new(rng: &SmallRng, representation: &[usize], activations: &[ActivationNames]) -> Layered_network {
        let mut net = Vec::new();
        let mut input_nb = representation[0];

        for (rep, i) in representation.iter() {
            net.push(Layer::new(input_nb, rep, rng, activations[i]));
            input_nb = rep;
        }

        Layered_network {
            layers: net,
        }
    }

}