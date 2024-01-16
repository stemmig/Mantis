use std::sync::RwLockReadGuard;
use crate::array::{CpuArray};
use crate::backend::BackendData::{Metal, Array, Cpu};
use crate::DType;
use crate::tensor::Data;

#[derive(Clone)]
pub enum Backend {
    Array,
    Cpu,
    Metal
}

pub enum BackendData
{
    Array(CpuArray),
    Cpu,
    Metal,
}

impl Data for BackendData {
    fn zeros(&self, shape: Vec<usize>, dtype: DType) -> Self {
        match self {
            BackendData::Array(_) => BackendData::Array(CpuArray::zeros(shape, dtype)),
            BackendData::Cpu => BackendData::Cpu,
            BackendData::Metal => Metal
        }
    }

    fn ones(&self, shape: Vec<usize>, dtype: DType) -> Self {
        match self {
            BackendData::Array(_) => BackendData::Array(CpuArray::zeros(shape, dtype)),
            BackendData::Cpu => BackendData::Cpu,
            BackendData::Metal => Metal
        }
    }

    fn add(&self, rhs:&Self) -> Option<Self> {
        let addition = match (self, rhs) {
            (Array(lhs), Array(rhs) ) => Some(Array((*lhs).add(rhs).unwrap())),
            _ => None
        };
        addition
    }
}