// Code for the NDarray backend
use ndarray::{Array, Dimension, IxDyn};
use num_traits::Zero;
use crate::tensor::Data;

pub struct NDArray
{
    array: Array<f64, IxDyn>,
}

impl Data for NDArray
{
    fn new() -> Self {
        todo!()
    }

    fn zeros(shape: Vec<usize>) -> Self {
        NDArray {
            array: Array::zeros(IxDyn(&shape))
        }
    }

    fn ones(shape: Vec<usize>) -> Self {
        todo!()
    }

    fn add(&self, rhs: &Self) -> Self {
        todo!()
    }
}