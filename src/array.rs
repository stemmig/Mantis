use std::ops::{Add, Div, Mul, Sub};
use ndarray::{Array, Dimension, IxDyn};
use crate::array::CpuArray::{F32Array, F64Array};
use crate::DType;
use crate::tensor::Data;

pub enum CpuArray
{
    F32Array(Array<f32, IxDyn>),
    F64Array(Array<f64, IxDyn>),
}

impl CpuArray
{
    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        match dtype {
            DType::F32 => F32Array(Array::zeros(IxDyn(&shape))),
            DType::F64 => F64Array(Array::zeros(IxDyn(&shape)))
        }
    }

    pub fn ones(shape: Vec<usize>, dtype: DType) -> Self {
        match dtype {
            DType::F32 => F32Array(Array::ones(IxDyn(&shape))),
            DType::F64 => F64Array(Array::ones(IxDyn(&shape)))
        }
    }

    pub fn add(&self, rhs: &Self) -> Option<Self> {
        match (self, rhs) {
            (F32Array(ref a), F32Array(ref b)) => {
                Some(F32Array(a.add(b)))
            }
            _ => None
        }
    }

    pub fn sub(&self, rhs: &Self) -> Option<Self> {
        match (self, rhs) {
            (F32Array(ref a), F32Array(ref b)) => {
                Some(F32Array(a.sub(b)))
            }
            _ => None
        }
    }
    pub fn mul(&self, rhs: &Self) -> Option<Self> {
        match (self, rhs) {
            (F32Array(ref a), F32Array(ref b)) => {
                Some(F32Array(a.mul(b)))
            }
            _ => None
        }
    }
    pub fn div(&self, rhs: &Self) -> Option<Self> {
        match (self, rhs) {
            (F32Array(ref a), F32Array(ref b)) => {
                Some(F32Array(a.div(b)))
            }
            _ => None
        }
    }
}