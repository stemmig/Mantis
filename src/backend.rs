use std::ops::Deref;
use std::sync::RwLockReadGuard;
use num_traits::{Num, NumCast};
use crate::array::{CpuArray};
use crate::backend::BackendData::{Metal, Cpu};
use crate::DType;
use crate::tensor::Data;

#[derive(Clone)]
pub enum Backend {
    Cpu,
    Metal
}

pub enum BackendData
{
    Cpu(CpuArray),
    Metal,
}

impl Data for BackendData {
    fn zeros(&self, shape: Vec<usize>, dtype: DType) -> Self {
        match self {
            Cpu(_) => Cpu(CpuArray::zeros(shape, dtype)),
            Metal => Metal
        }
    }

    fn ones(&self, shape: Vec<usize>, dtype: DType) -> Self {
        match self {
            Cpu(_) => Cpu(CpuArray::zeros(shape, dtype)),
            Metal => Metal
        }
    }

    fn add(&self, rhs:&Self) -> Option<Self> {
        let addition = match (self, rhs) {
            (Cpu(lhs), Cpu(rhs) ) => Some(Cpu((*lhs).add(rhs).unwrap())),
            _ => None
        };
        addition
    }

    fn sub(&self, rhs: &Self) -> Option<Self> {
        let addition = match (self, rhs) {
            (Cpu(lhs), Cpu(rhs) ) => Some(Cpu((*lhs).sub(rhs).unwrap())),
            _ => None
        };
        addition
    }

    fn mul(&self, rhs: &Self) -> Option<Self> {
        let addition = match (self, rhs) {
            (Cpu(lhs), Cpu(rhs) ) => Some(Cpu((*lhs).mul(rhs).unwrap())),
            _ => None
        };
        addition
    }

    fn div(&self, rhs: &Self) -> Option<Self> {
        let addition = match (self, rhs) {
            (Cpu(lhs), Cpu(rhs) ) => Some(Cpu((*lhs).div(rhs).unwrap())),
            _ => None
        };
        addition
    }

    fn matmul(&self, rhs: &Self) -> Result<Self, String> {
        match (self, rhs) {
            (Cpu(lhs), Cpu(rhs) ) => Ok(Cpu((*lhs).matmul(rhs)?)),
            _ => Err(String::from("Could not perform MatMul for this backend type."))
        }
    }

    fn relu(&self) -> Option<Self> {
        todo!()
    }

    fn exp(&self) -> Option<Self> {
        todo!()
    }

    fn sum(&self, dims: Vec<usize>) -> Option<Self> {
        todo!()
    }

    fn get<T: Num + Copy + NumCast>(&self, index: Vec<usize>) -> Option<T> {
        let val= match self {
            Cpu(arr) => arr.get(index),
            _ => None
        };
        val
    }

    fn copy_from(&mut self, other: &Self) -> () {
        match (self, other) {
            (Cpu(into), Cpu(from) ) => into.copy_from(from),
            _ => ()
        }
    }
}