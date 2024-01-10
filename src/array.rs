// Code for the NDarray backend
use ndarray::{Array, Dimension, IxDyn};
use crate::tensor::Data;

pub struct NDArray<T> {
    array: Array<T, IxDyn>,
}

impl<T> Data for NDArray<T> {
    fn new() -> Self {
        todo!()
    }

    fn zeros(shape: Vec<usize>) -> Self {
        todo!()
    }

    fn ones(shape: Vec<usize>) -> Self {
        todo!()
    }

    fn add(&self, rhs: &Self) -> Self {
        todo!()
    }
}