use std::ops::Add;
// Code for the NDarray backend
use ndarray::{Array, Dimension, IxDyn};
use crate::array::NDArray::{F32Array, F64Array};
use crate::DType;
use crate::tensor::Data;

pub enum NDArray
{
    F32Array(Array<f32, IxDyn>),
    F64Array(Array<f64, IxDyn>),
}

impl Data for NDArray
{

    fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        match dtype {
            DType::F32 => F32Array(Array::zeros(IxDyn(&shape))),
            DType::F64 => F64Array(Array::zeros(IxDyn(&shape)))
        }
    }

    fn ones(shape: Vec<usize>, dtype: DType) -> Self {
        match dtype {
            DType::F32 => F32Array(Array::ones(IxDyn(&shape))),
            DType::F64 => F64Array(Array::ones(IxDyn(&shape)))
        }
    }

    fn add(&self, rhs: &Self) -> Option<Self> {
        match (self, rhs) {
            (F32Array(ref a), F32Array(ref b)) => {
                Some(F32Array(a.add(b)))
            }
            _ => None
        }
    }
}