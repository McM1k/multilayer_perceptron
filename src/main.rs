mod perceptron;
mod reader;
mod cell;
mod predict;

extern crate rand;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::vec::Vec;
use perceptron::perceptron::Perceptron;
use perceptron::activation_functions::ActivationNames;
use cell::Cell;
use perceptron::layered::Layered_network;

fn main() {
    let data = reader::get_raw_data("resources/data.csv")
        .expect("problem while reading csv");
    let seed: [u8; 16] = [1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8];
    let mut rng: SmallRng = SeedableRng::from_seed(seed);
    println!("{:?}", rng.gen::<f64>());
}

fn epocher(data: &[Cell], network: &mut Layered_network) {
    for cell in data {
        let mut inputs = cell.values.clone();
        let outputs = network.compute(&inputs);


    }
}

fn print_metrics(epoch: usize, epoch_max: usize, loss: f64, val_loss: f64) {
    println!("epoch {}/{} - loss: {} - val_loss: {}", epoch, epoch_max, loss, val_loss);
}