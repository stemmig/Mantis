// Code for the NDarray backend
use ndarray::{Array, Dimension, IxDyn};
use num_traits::Zero;
use crate::array::ArrayType::{F32Array, F64Array};
use crate::DType;
use crate::tensor::Data;

pub enum ArrayType {
    F32Array(Array<f32, IxDyn>),
    F64Array(Array<f64, IxDyn>),
}
pub struct NDArray
{
    array: ArrayType,
}

impl Data for NDArray
{
    fn new() -> Self {
        todo!()
    }

    fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        match dtype {
            DType::F32 => NDArray { array: F32Array(Array::zeros(IxDyn(&shape))) },
            DType::F64 => NDArray { array: F64Array(Array::zeros(IxDyn(&shape))) }
        }

    }

    fn ones(shape: Vec<usize>, dtype: DType) -> Self {
        todo!()
    }

    fn add(&self, rhs: &Self) -> Self {
        todo!()
    }
}