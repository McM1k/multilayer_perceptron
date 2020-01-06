mod perceptron;
mod reader;
mod cell;

extern crate rand;
use crate::rand::Rng;
use rand::rngs::SmallRng;

fn main() {
    let data = reader::get_raw_data("resources/data.csv")
        .expect("problem while reading csv");
    let seed: [u8; 16] = [1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8];
    let mut rng: SmallRng = rand::SeedableRng::from_seed(seed);
    println!("{:?}", rng.gen::<f64>());
}
