mod perceptron;
mod reader;
mod cell;

extern crate serde;

fn main() {
    let data = reader::get_raw_data("resources/data.csv").expect("problem while reading csv");
    println!("{:?}", data);
}
