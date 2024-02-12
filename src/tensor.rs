use std::collections::{HashMap, VecDeque};
use std::fmt::{Debug};
use std::ops::Deref;
use std::sync::{Arc, RwLock, RwLockReadGuard};
use num_traits::{Num, NumCast};
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
    dtype: DType,
    id: u128,
}

pub trait Data where Self: Sized {

    fn zeros(&self, shape: Vec<usize>, dtype: DType) -> Self;

    fn ones(&self, shape: Vec<usize>, dtype: DType) -> Self;

    fn add(&self, rhs: &Self) -> Option<Self>;
    fn sub(&self, rhs: &Self) -> Option<Self>;
    fn mul(&self, rhs: &Self) -> Option<Self>;
    fn div(&self, rhs: &Self) -> Option<Self>;
    fn get<T: Num + Copy + NumCast>(&self, index: Vec<usize>) -> Option<T>;

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
            Cpu => BackendData::Cpu(CpuArray::zeros(dims.clone(), dtype.clone())),
            Metal => BackendData::Metal,
        };
        Tensor(Arc::new(Tensor_ {
            op: Op::None,
            data: Arc::new(RwLock::new(init_data)),
            is_mutable: false,
            shape: dims.clone(),
            backend: backend.clone(),
            dtype: dtype.clone(),
            id: Self::uuid(),
        }))
    }

    pub fn ones(dims: Vec<usize>, backend: Backend, dtype: DType) -> Self {
        let init_data: BackendData = match backend {
            Cpu => BackendData::Cpu(CpuArray::ones(dims.clone(), dtype.clone())),
            Metal => BackendData::Metal,
        };
        Tensor(Arc::new(Tensor_ {
            op: Op::None,
            data: Arc::new(RwLock::new(init_data)),
            is_mutable: false,
            shape: dims.clone(),
            backend: backend.clone(),
            dtype: dtype.clone(),
            id: Self::uuid(),
        }))
    }

    pub fn fill(dims: Vec<usize>, value: f64, backend: Backend, dtype: DType) -> Self {
        let init_data: BackendData = match backend {
            Cpu => BackendData::Cpu(CpuArray::fill(value, dims.clone(), dtype.clone())),
            Metal => BackendData::Metal,
        };
        Tensor(Arc::new(Tensor_ {
            op: Op::None,
            data: Arc::new(RwLock::new(init_data)),
            is_mutable: false,
            shape: dims.clone(),
            backend: backend.clone(),
            dtype: dtype.clone(),
            id: Self::uuid(),
        }))
    }

    pub fn zeros_like(&self) -> Self {
        Tensor::zeros(self.shape(), self.backend(), self.dtype())
    }

    pub fn ones_like(&self) -> Self {
        Tensor::ones(self.shape(), self.backend(), self.dtype())
    }

    pub fn add(&self, rhs: &Self) -> Self {
        let lhs_read = self.backend_ref();
        let rhs_read = rhs.backend_ref();
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
            dtype: self.0.dtype.clone(),
            id: Self::uuid(),
        }))
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let lhs_read = self.backend_ref();
        let rhs_read = rhs.backend_ref();
        let data = (*lhs_read).sub(&*rhs_read).expect("Could not perform Sub!");

        Tensor(Arc::new(Tensor_ {
            op: Op::Sub(self.clone(), rhs.clone()),
            data: Arc::new(RwLock::new(data)),
            is_mutable: false,
            shape: self.0.shape.clone(),
            backend: self.0.backend.clone(),
            dtype: self.0.dtype.clone(),
            id: Self::uuid(),
        }))
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let lhs_read = self.backend_ref();
        let rhs_read = rhs.backend_ref();
        let data = (*lhs_read).mul(&*rhs_read).expect("Could not perform Mul!");

        Tensor(Arc::new(Tensor_ {
            op: Op::Mul(self.clone(), rhs.clone()),
            data: Arc::new(RwLock::new(data)),
            is_mutable: false,
            shape: self.0.shape.clone(),
            backend: self.0.backend.clone(),
            dtype: self.0.dtype.clone(),
            id: Self::uuid(),
        }))
    }

    pub fn div(&self, rhs: &Self) -> Self {
        let lhs_read = self.backend_ref();
        let rhs_read = rhs.backend_ref();
        let data = (*lhs_read).div(&*rhs_read).expect("Could not perform Div!");

        Tensor(Arc::new(Tensor_ {
            op: Op::Div(self.clone(), rhs.clone()),
            data: Arc::new(RwLock::new(data)),
            is_mutable: false,
            shape: self.0.shape.clone(),
            backend: self.0.backend.clone(),
            dtype: self.0.dtype.clone(),
            id: Self::uuid(),
        }))
    }

    pub fn get<T: Num + Copy + NumCast>(&self, index: Vec<usize>) -> Option<T> {
        self.backend_ref().get(index)
    }

    fn backend_ref(&self) -> RwLockReadGuard<BackendData> {
        self.0.data.read().unwrap()
    }

    fn topo_sort(&self) -> Vec<Tensor> {
        let mut queue: VecDeque<Tensor> = VecDeque::new();
        let mut sorted: Vec<Tensor> = Vec::new();

        queue.push_back(self.clone());
        sorted.push(self.clone());

        while !queue.is_empty() {
            let curr_tensor = queue.pop_front().unwrap();
            let curr_op = curr_tensor.0.op.clone();
            match curr_op {
                Op::Add(lhs, rhs) | Op::Sub(lhs, rhs)
                | Op::Mul(lhs, rhs) | Op::Div(lhs, rhs)  => {
                    sorted.push(lhs.clone());
                    sorted.push(rhs.clone());
                    queue.push_back(lhs.clone());
                    queue.push_back(rhs.clone());
                },
                Op::None => ()
            }
        }
        sorted
    }

    pub fn backward(&self) -> Gradients {
        let mut grads = Gradients::new(self);
        let nodes = self.topo_sort();

        for node in nodes {
            let grad = grads
                .get(&node)
                .expect("Previous gradient should always be present!")
                .clone();

            match node.0.op.clone() {
                Op::Add(lhs, rhs) => {
                    let lhs_grad = grads.get_or(&lhs);
                    *lhs_grad = lhs_grad.add(&grad);
                    let rhs_grad = grads.get_or(&rhs);
                    *rhs_grad = rhs_grad.add(&grad);
                },
                Op::Sub(lhs, rhs) => {
                    let lhs_grad = grads.get_or(&lhs);
                    *lhs_grad = lhs_grad.add(&grad);
                    let rhs_grad = grads.get_or(&rhs);
                    *rhs_grad = rhs_grad.sub(&grad);
                },
                Op::Mul(lhs, rhs) => {
                    let lhs_grad = grad.mul(&rhs);
                    let lhs_sum_grad = grads.get_or(&lhs);
                    *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad);
                    let rhs_grad = grad.mul(&lhs);
                    let rhs_sum_grad = grads.get_or(&rhs);
                    *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad);
                },
                Op::Div(lhs, rhs) => {

                },
                Op::None => {

                }
            }
        }
        grads
    }

    pub fn id(&self) -> u128 {
        self.0.id.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.shape.clone()
    }

    pub fn backend(&self) -> Backend {
        self.0.backend.clone()
    }

    pub fn dtype(&self) -> DType {
        self.0.dtype.clone()
    }
}

pub struct Parameter(Tensor);

#[cfg(test)]
mod tests {
    use std::ptr::eq;
    use super::*;

    #[test]
    fn test_zeros() {
        let tensor = Tensor::zeros(vec![2, 3], Cpu, DType::F32);
        assert_eq!(tensor.get(vec![1, 1]), Some(0))
    }

    #[test]
    fn test_ones() {
        let tensor = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        assert_eq!(tensor.get(vec![1, 1]), Some(1))
    }

    #[test]
    fn test_fill() {
        let tensor = Tensor::fill(vec![2, 3], 5.0, Cpu, DType::F32);
        assert_eq!(tensor.get(vec![1, 1]), Some(5.0))
    }

    #[test]
    fn test_get() {
        let tensor = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        assert_eq!(tensor.get(vec![1, 1]), Some(1));
        assert_eq!(tensor.get::<f32>(vec![10, 10]), None);
    }

    #[test]
    fn test_addition() {
        let t_a = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let t_b = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let add = t_a.add(&t_b);
        assert_eq!(add.get(vec![1, 1]), Some(2));
    }

    #[test]
    fn test_topo_sort() {
        let tensor_a = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let tensor_b = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let added = tensor_a.add(&tensor_b);
        let sorted = added.topo_sort();
        assert_eq!(sorted.len(), 3);
        // todo: add more granular assertions
    }

    #[test]
    fn test_backward_add() {
        let tensor_a = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let tensor_b = Tensor::fill(vec![2, 3], 5.0, Cpu, DType::F32);
        let added = tensor_a.add(&tensor_b);
        let gradients = added.backward();
        assert_eq!(gradients.get(&tensor_a).unwrap().get(vec![1, 1]), Some(1));
        assert_eq!(gradients.get(&tensor_b).unwrap().get(vec![1, 1]), Some(1));
    }

    #[test]
    fn test_backward_sub() {
        let tensor_a = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let tensor_b = Tensor::fill(vec![2, 3], 5.0, Cpu, DType::F32);
        let subtracted = tensor_a.sub(&tensor_b);
        let gradients = subtracted.backward();
        assert_eq!(gradients.get(&tensor_a).unwrap().get(vec![1, 1]), Some(1));
        assert_eq!(gradients.get(&tensor_b).unwrap().get(vec![1, 1]), Some(-1));
    }

    #[test]
    fn test_backward_mul() {
        let tensor_a = Tensor::fill(vec![2, 3], 5.0, Cpu, DType::F32);
        let tensor_b = Tensor::fill(vec![2, 3], 3.0, Cpu, DType::F32);
        let multiplied = tensor_a.mul(&tensor_b);
        let gradients = multiplied.backward();
        assert_eq!(gradients.get(&tensor_a).unwrap().get(vec![1, 1]), Some(3));
        assert_eq!(gradients.get(&tensor_b).unwrap().get(vec![1, 1]), Some(5));
    }
}