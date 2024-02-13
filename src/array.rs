use std::ops::{Add, Div, Mul, Sub};
use ndarray::{Array, ArrayD, Dimension, Ix1, Ix2, IxDyn, Dim, ArrayBase, OwnedRepr, Array2, Array1};
use ndarray::linalg::Dot;
use num_traits::{Num, NumCast};
use crate::array::CpuArray::{F32Array, F64Array};
use crate::backend::BackendData;
use crate::DType;

pub enum CpuArray
{
    F32Array(ArrayD<f32>),
    F64Array(ArrayD<f64>),
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

    pub fn fill(value:f64, shape: Vec<usize>, dtype: DType) -> Self {
        match dtype {
            DType::F32 => F32Array(Array::from_elem(IxDyn(&shape), value as f32)),
            DType::F64 => F64Array(Array::from_elem(IxDyn(&shape), value))
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

    pub fn matmul(&self, rhs: &Self) -> Result<Self, String> {
        match (self, rhs) {
            (F32Array(a), F32Array(b)) => {
                let res = a.clone().into_dimensionality::<Ix1>().unwrap().dot(&b.clone().into_dimensionality::<Ix1>().unwrap());
                let wrapped = F32Array(Array1::from_vec(vec![res]).into_dyn());
                Ok(wrapped)
            },
            _ => Err(String::from("Cannot MatMul for the provided data types"))
        }
    }

    pub fn get<T: Num + Copy + NumCast>(&self, index: Vec<usize>) -> Option<T> {
        let val = match self {
            F32Array(arr) => arr.get(IxDyn(&index)).cloned(),
            _ => None
        };
        match val {
          Some(n) => NumCast::from(n),
            _ => None
        }
    }

    pub fn copy_from(&mut self, other: &Self) {
        match (self, other) {
            (F32Array(arr1), F32Array(arr2)) => arr1.assign(arr2),
            _ => ()
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, IxDyn};
    use crate::array::CpuArray::F32Array;

    #[test]
    fn test_matmul_1x1() {
        let cpua1 = F32Array(Array::from_elem(IxDyn(&vec![1]), 5.0f32));
        let cpua2 = F32Array(Array::from_elem(IxDyn(&vec![1]), 3.0f32));
        let cpu_prod = cpua1.matmul(&cpua2);
        assert_eq!(cpu_prod.unwrap().get(vec![0]), Some(15f32))
    }
}