use csv::ReaderBuilder;
use super::cell::Cell;
use std::vec::Vec;
use std::result::Result;
use std::result::Result::*;
use std::string::String;

pub fn get_raw_data(path: &str) -> Result<Vec<Cell>, &str> {
    let mut raw_data: Vec<Cell> = Vec::new();

    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .expect("error while reading csv");
    for result in rdr.deserialize() {
        let record: Cell = result.expect("error in line");
        raw_data.push(record);
    }

    Ok(raw_data)
}