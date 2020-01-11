use serde::Deserialize;
use std::vec::Vec;

#[derive(Deserialize, Debug)]
pub enum Status {
    M,
    B,
}

#[derive(Deserialize, Debug)]
pub struct Cell {
    pub id: i32,
    pub status: Status,
    pub values: Vec<f64>,
}