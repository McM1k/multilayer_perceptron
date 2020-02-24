use super::activation_functions::*;
use rand::rngs::SmallRng;
use rand::Rng;
use std::vec::Vec;

#[derive(Clone)]
pub struct Perceptron {
    pub bias: f64,
    pub weights: Vec<f64>,
}

impl Perceptron {
    pub fn new(inputs_number: usize, rng: &mut SmallRng) -> Perceptron {
        let mut weights = Vec::new();

        for _i in 0..inputs_number {
            weights.push(rng.gen_range(-1.0, 1.0));
        }

        Perceptron { bias: 1.0, weights }
    }

    pub fn weighted_sum(&self, inputs: &[f64]) -> f64 {
        let mut sum = self.bias;

        for (i, weight) in self.weights.iter().enumerate() {
            sum += weight * inputs[i];
        }

        sum
    }

    pub fn compute(&self, inputs: &[f64], activation: ActivationNames) -> f64 {
        let sum = self.weighted_sum(inputs);
        (activation.get_fn())(sum)
    }

    pub fn error(&self, inputs: &[f64], truth: f64, activation: ActivationNames) -> f64 {
        let sum = self.weighted_sum(inputs);
        (activation.get_deriv())(sum) * (self.compute(inputs, activation) - truth)
    }
}
