use crate::activation_functions::*;

pub enum Activation_names{
    Sigmoid,
    Hyperboloid,
    Tangent,
    Rectified_linear_unit,
}

pub struct Perceptron {
    bias: f64,
    weights: Vec<f64>,
    activation: fn(f64)->f64,
}

impl Perceptron {
    pub fn new(inputs_number: usize, activation: Activation_names) -> Perceptron {
        Perceptron {
            bias: 1.0,
            weights: vec![0.0; usize],
            activation: match activation {
                Activation_names::Sigmoid => sigmoid,
                Activation_names::Hyperboloid => hyperboloid,
                Activation_names::Tangent => tangent,
                Activation_names::Rectified_linear_unit => rectified_linear_unit,
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
        self.activation(sum)
    }
}