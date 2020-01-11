use super::activation_functions::*;
use std::vec::Vec;
use rand::rngs::SmallRng;
use rand::Rng;


#[derive(Clone)]
pub struct Perceptron {
    bias: f64,
    weights: Vec<f64>,
}

impl Perceptron {
    pub fn new(inputs_number: usize, rng: &mut SmallRng) -> Perceptron {
        let mut weights = Vec::new();

        for i in 0..inputs_number {
            weights.push(rng.gen_range(-1.0, 1.0));
        }


        let mut perceptron = Perceptron {
            bias: 1.0,
            weights,
        };

        perceptron
    }

    fn weighted_sum(&self, inputs: &[f64]) -> f64 {
        let mut sum = self.bias;

        for i in 0..self.weights.len() {
            sum += self.weights[i] * inputs[i];
        }

        sum
    }

    pub fn compute(&self, inputs: &[f64], activation: &ActivationNames) -> f64 {
        let sum = self.weighted_sum(inputs);
        (activation.get_fn())(&sum)
    }
}
