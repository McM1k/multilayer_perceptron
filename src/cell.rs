use serde::Deserialize;
use std::vec::Vec;

#[derive(Deserialize, Debug)]
pub enum Status {
    M,
    B,
}

#[derive(Deserialize, Debug)]
pub struct Cell {
    id: i32,
    status: Status,
    values: Vec<f64>,
}