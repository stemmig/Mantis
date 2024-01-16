use std::fmt::{Debug};
use std::sync::{Arc, RwLock};
use crate::array::{CpuArray};
use crate::operations::{ Op};
use crate::backend::{Backend, BackendData};
use crate::backend::Backend::{Cpu, Metal};
use crate::DType;

pub struct Tensor {
    op: Op,
    data: Arc<RwLock<BackendData>> ,
    is_mutable: bool,
    shape: Vec<usize>,
    backend: Backend,
}

pub trait Data where Self: Sized {

    fn zeros(&self, shape: Vec<usize>, dtype: DType) -> Self;

    fn ones(&self, shape: Vec<usize>, dtype: DType) -> Self;

    fn add(&self, rhs: &Self) -> Option<Self>;
}

impl Tensor {
    // Keeping track of compute graph with be handled in Tensor impls,
    // Actually modifying the underlying tensor on the backend will be done as part of data impls
    pub fn zeros(dims: Vec<usize>, backend: Backend, dtype: DType) -> Self {
        let init_data: BackendData = match backend {
            Cpu => BackendData::Cpu(CpuArray::zeros(dims.clone(), dtype)),
            Metal => BackendData::Metal,
        };
        Tensor {
            op: Op::None,
            data: Arc::new(RwLock::new(init_data)),
            is_mutable: false,
            shape: dims.clone(),
            backend,
        }
    }

    pub fn ones(dims: Vec<usize>, backend: Backend, dtype: DType) -> Self {
        let init_data: BackendData = match backend {
            Cpu => BackendData::Cpu(CpuArray::ones(dims.clone(), dtype)),
            Metal => BackendData::Metal,
        };
        Tensor {
            op: Op::None,
            data: Arc::new(RwLock::new(init_data)),
            is_mutable: false,
            shape: dims.clone(),
            backend,
        }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let lhs_read = self.data.read().unwrap();
        let rhs_read = rhs.data.read().unwrap();
        let added = (*lhs_read).add(&*rhs_read);

        let add_unwrap = match added {
            Some(addition) => addition,
            None => panic!("Could not perform addition!")
        };

        Tensor {
            op: Op::Add,
            data: Arc::new(RwLock::new(add_unwrap)),
            is_mutable: false,
            shape: self.shape.clone(),
            backend: self.backend.clone(),
        }
    }
}