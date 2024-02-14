use std::ops::{Add, Div, Mul, Sub};
use ndarray::{Array, ArrayD, Dimension, Ix1, IxDyn, Array1, Ix2, Array2};
use ndarray::linalg::Dot;
use num_traits::{Num, NumCast};
use crate::array::CpuArray::{F32Array, F64Array};
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
            (F32Array(l), F32Array(r)) if l.ndim() == 1 && l.ndim() == 1 => {
                let res = l.clone().into_dimensionality::<Ix1>().unwrap().dot(&r.clone().into_dimensionality::<Ix1>().unwrap());
                let wrapped = F32Array(Array1::from_vec(vec![res]).into_dyn());
                Ok(wrapped)
            },
            (F32Array(l), F32Array(r)) if l.ndim() == 2 && l.ndim() == 2 => {
                let res = l.clone().into_dimensionality::<Ix2>().unwrap().dot(&r.clone().into_dimensionality::<Ix2>().unwrap());
                Ok(F32Array(res.into_dyn()))
            }
            _ => Err(String::from("Cannot MatMul for the provided data types"))
        }
    }

    pub fn relu(&self) -> Result<Self, String> {
        match self {
            F32Array(ref arr) => Ok(F32Array(arr.mapv(|x| x.max(0.0)))),
            _ => Err(String::from("Cannot ReLU for the provided data types")),
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

    pub fn shape(&self) -> Vec<usize> {
        match self {
            F32Array(arr) => arr.shape().to_vec(),
            F64Array(arr) => arr.shape().to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, IxDyn};
    use crate::array::CpuArray::F32Array;
    use crate::DType::F32;

    #[test]
    fn test_matmul_1x1() {
        let cpua1 = F32Array(Array::from_elem(IxDyn(&vec![1]), 5.0f32));
        let cpua2 = F32Array(Array::from_elem(IxDyn(&vec![1]), 3.0f32));
        let cpu_prod = cpua1.matmul(&cpua2);
        assert_eq!(cpu_prod.unwrap().get(vec![0]), Some(15f32))
    }

    #[test]
    fn test_matmul_2x2() {
        let arr1 = F32Array(Array::from_elem(IxDyn(&vec![2, 3]), 5.0f32));
        let arr2 = F32Array(Array::from_elem(IxDyn(&vec![3, 5]), 3.0f32));
        let cpu_prod = arr1.matmul(&arr2).unwrap();
        assert_eq!(cpu_prod.get(vec![1,1]), Some(45f32));
        assert_eq!(cpu_prod.shape(), vec![2, 5]);
    }

    #[test]
    fn test_relu(){
        let arr_neg = F32Array(Array::from_elem(IxDyn(&vec![2, 3]), -1.0f32));
        let arr_zero = F32Array(Array::zeros(IxDyn(&vec![2, 3])));
        let arr_pos = F32Array(Array::from_elem(IxDyn(&vec![2, 3]), 5.0f32));
        let relu_neg = arr_neg.relu().unwrap().get(vec![1, 1]).unwrap();
        let relu_zero = arr_zero.relu().unwrap().get(vec![1, 1]).unwrap();
        let relu_pos = arr_pos.relu().unwrap().get(vec![1, 1]).unwrap();
        assert_eq!((relu_neg, relu_zero, relu_pos), (0, 0, 5.0));
    }
}