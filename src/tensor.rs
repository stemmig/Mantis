use std::fmt::{Debug};
use std::sync::Arc;
use crate::array::NDArray;
use crate::operations::{ Op};
use crate::backend::{Backend, BackendData};
use crate::backend::Backend::{Array, Cpu, Metal};
use crate::DType;

pub struct Tensor {
    op: Op,
    data: Arc<BackendData>,
    is_mutable: bool,
    shape: Vec<usize>,
    backend: Backend,
}

pub trait Data where Self: Sized,  {
    fn new() -> Self;

    fn zeros(shape: Vec<usize>, dtype: DType) -> Self;

    fn ones(shape: Vec<usize>, dtype: DType) -> Self;

    fn add(&self, rhs: &Self) -> Self;
}

impl Tensor {
    // Keeping track of compute graph with be handled in Tensor impls,
    // Actually modifying the underlying tensor on the backend will be done as part of data impls
    pub fn zeros(dims: Vec<usize>, backend: Backend, dtype: DType) -> Self {
        let init_data: BackendData = match backend {
            Array => BackendData::Array(NDArray::zeros(dims.clone(), dtype)),
            Cpu => BackendData::Cpu,
            Metal => BackendData::Metal,
        };
        Tensor {
            op: Op::None,
            data: Arc::new(init_data),
            is_mutable: false,
            shape: dims.clone(),
            backend: Array,
        }
    }

    pub fn ones(dims: Vec<usize>, backend: Backend, dtype: DType) -> Self {
        let init_data: BackendData = match backend {
            Array => BackendData::Array(NDArray::ones(dims.clone(), dtype)),
            Cpu => BackendData::Cpu,
            Metal => BackendData::Metal,
        };
        Tensor {
            op: Op::None,
            data: Arc::new(init_data),
            is_mutable: false,
            shape: dims.clone(),
            backend: Array,
        }
    }
}