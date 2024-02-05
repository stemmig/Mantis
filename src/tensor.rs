use std::collections::{HashMap, VecDeque};
use std::fmt::{Debug};
use std::sync::{Arc, RwLock};
use uuid::{Uuid};
use crate::array::{CpuArray};
use crate::operations::{ Op};
use crate::backend::{Backend, BackendData};
use crate::backend::Backend::{Cpu, Metal};
use crate::DType;
use crate::grad::Gradients;

pub struct Tensor_ {
    op: Op,
    data: Arc<RwLock<BackendData>> ,
    is_mutable: bool,
    shape: Vec<usize>,
    backend: Backend,
    id: u128,
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

    pub fn uuid() -> u128 {
        let uuid = Uuid::new_v4();
        uuid.as_u128()
    }
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
            id: Self::uuid(),
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
            id: Self::uuid(),
        }))
    }

    pub fn add(&self, rhs: Self) -> Self {
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
            id: Self::uuid(),
        }))
    }

    fn topo_sort(&self) -> Vec<Tensor> {
        let mut queue: VecDeque<Op> = VecDeque::new();
        let mut sorted: Vec<Tensor> = Vec::new();

        queue.push_back(self.0.op.clone());
        sorted.push(self.clone());

        while !queue.is_empty() {
            let curr_op = queue.pop_front().unwrap();
            match curr_op {
                Op::Add(lhs, rhs) => {
                    sorted.push(lhs.clone());
                    sorted.push(rhs.clone());
                    queue.push_back(lhs.0.op.clone());
                    queue.push_back(rhs.0.op.clone());
                },
                Op::None => (),
                _ => ()
            }
        }
        sorted
    }

    pub fn backward(&self) -> Gradients {
        let mut grads = HashMap::new();
        let nodes = self.topo_sort();

        for node in nodes {

        }

        Gradients(grads)
    }
}

pub struct Parameter(Tensor);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let tensor = Tensor::zeros(vec![2, 3], Cpu, DType::F32);
    }

    #[test]
    fn test_ones() {
        let tensor = Tensor::ones(vec![2, 3], Cpu, DType::F32);
    }

    #[test]
    fn test_topo_sort() {
        let tensor_a = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let tensor_b = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let added = tensor_a.add(tensor_b);
        let sorted = added.topo_sort();
        assert_eq!(sorted.len(), 3);
        // todo: add more granular assertions
    }
}