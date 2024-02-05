use std::fmt::{Debug};
use std::sync::{Arc, RwLock};
use crate::array::{CpuArray};
use crate::operations::{ Op};
use crate::backend::{Backend, BackendData};
use crate::backend::Backend::{Cpu, Metal};
use crate::DType;

pub struct Tensor_ {
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

#[derive(Clone)]
// The top-level Tensor struct is actually an Arc to the underlying data.
// Making it cheap to pass around references to the underlying data
pub struct Tensor(Arc<Tensor_>);

impl Tensor {
    // Keeping track of compute graph with be handled in Tensor impls,
    // Actually modifying the underlying tensor on the backend will be done as part of data impls
    pub fn zeros(dims: Vec<usize>, backend: Backend, dtype: DType) -> Self {
        let init_data: BackendData = match backend {
            Cpu => BackendData::Cpu(CpuArray::zeros(dims.clone(), dtype)),
            Metal => BackendData::Metal,
        };
        Tensor(Arc::new(Tensor_ {
            op: Op::None,
            data: Arc::new(RwLock::new(init_data)),
            is_mutable: false,
            shape: dims.clone(),
            backend,
        }))
    }

    pub fn ones(dims: Vec<usize>, backend: Backend, dtype: DType) -> Self {
        let init_data: BackendData = match backend {
            Cpu => BackendData::Cpu(CpuArray::ones(dims.clone(), dtype)),
            Metal => BackendData::Metal,
        };
        Tensor(Arc::new(Tensor_ {
            op: Op::None,
            data: Arc::new(RwLock::new(init_data)),
            is_mutable: false,
            shape: dims.clone(),
            backend,
        }))
    }

    pub fn add(self, rhs: Self) -> Self {
        let lhs_read = self.0.data.read().unwrap();
        let rhs_read = rhs.0.data.read().unwrap();
        let added = (*lhs_read).add(&*rhs_read);

        let add_unwrap = match added {
            Some(addition) => addition,
            None => panic!("Could not perform addition!")
        };

        Tensor(Arc::new(Tensor_ {
            op: Op::Add(self.clone(), rhs.clone()),
            data: Arc::new(RwLock::new(add_unwrap)),
            is_mutable: false,
            shape: self.0.shape.clone(),
            backend: self.0.backend.clone(),
        }))
    }

    fn topo_sort(&self) {

    }

    pub fn backward(&self) {

    }
}

pub struct Parameter(Tensor);