use super::activation_functions::*;
use std::vec::Vec;

pub enum ActivationNames{
    Sigmoid,
    HyperboloidTangent,
    RectifiedLinearUnit,
}

pub struct Perceptron {
    bias: f64,
    weights: Vec<f64>,
    activation: fn(&f64)->f64,
}

impl Perceptron {
    pub fn new(inputs_number: usize, activation: ActivationNames) -> Perceptron {
        Perceptron {
            bias: 1.0,
            weights: vec![0.0; inputs_number],
            activation: match activation {
                ActivationNames::Sigmoid => sigmoid,
                ActivationNames::HyperboloidTangent => tanh,
                ActivationNames::RectifiedLinearUnit => relu,
            }
        }
    }

    fn weighted_sum(&self, inputs: &Vec<f64>) -> f64 {
        let mut sum = self.bias;

        for i in 0..self.weights.len() {
            sum += self.weights[i] * inputs[i];
        }

        sum
    }

    pub fn compute(&self, inputs: &Vec<f64>) -> f64 {
        let sum = self.weighted_sum(inputs);
        (self.activation)(&sum)
    }
}
