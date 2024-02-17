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

    fn matmul(&self, rhs: &Self) -> Result<Self, String>;

    fn relu(&self) -> Result<Self, String>;

    fn exp(&self) -> Result<Self, String>;

    fn sum(&self, dims: Vec<usize>) -> Result<Self, String>;

    fn transpose(&self) -> Result<Self, String>;

    fn shape(&self) -> Vec<usize>;

    fn get<T: Num + Copy + NumCast>(&self, index: Vec<usize>) -> Option<T>;

    fn copy_from(&mut self, other: &Self) -> ();
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

    pub fn matmul(&self, rhs: &Self) -> Result<Self, String> {
        /*
            (a) @ (b)               => (1)          | a == b
            (a) @ (b, c)            => (c)          | a == b
            (a) @ (b, c, d)         => (b, d)       | a == c
            (a, b) @ (c)            => (a)          | b == c
            (a, b) @ (c, d)         => (a, d)       | b == c
            (a, b) @ (c, d, e)      => (c, a, e)    | b == d
            (a, b, c) @ (d)         => (a, b)       | c == d
            (a, b, c) @ (d, e)      => (a, b, e)    | c == d
            (a, b, c) @ (d, e, f)   => (a, b, f)    | c == e && a == d
         */
        let err_message = format!("Expected tensor shapes did not match {:?} @ {:?}", self.shape(), rhs.shape());

        let shape: Vec<usize> = match (self.shape().as_slice(), rhs.shape().as_slice()) {
            (&[a], &[b])                                          if a == b => vec![1],
            (&[a], &[b, c])                                if a == b => vec![c],
            (&[a], &[b, c, d])                      if a == c => vec![b, d],
            (&[a, b], &[c])                                if b == c => vec![a],
            (&[a, b], &[c, d])                      if b == c => vec![a, d],
            (&[a, b], &[c, d, e])            if b == d => vec![c, a, e],
            (&[a, b, c], &[d])                      if c == d => vec![a, b],
            (&[a, b, c], &[d, e])            if c == d => vec![a, b, e],
            (&[a, b, c], &[d, e, f])  if c == d => vec![a, b, f],
            _ => return Err(err_message)
        };

        let data = (*self.backend_ref()).matmul(&*rhs.backend_ref()).expect("Could not perform MatMul operation!");

        Ok(Tensor(Arc::new(Tensor_ {
            op: Op::MatMul(self.clone(), rhs.clone()),
            data: Arc::new(RwLock::new(data)),
            is_mutable: false,
            shape: shape.clone(),
            backend: self.0.backend.clone(),
            dtype: self.0.dtype.clone(),
            id: Self::uuid(),
        })))
    }

    pub fn relu(&self) -> Result<Self, String> {
        let data = (*self.backend_ref()).relu().expect("Could not perform ReLU operation!");

        Ok(Tensor(Arc::new(Tensor_ {
            op: Op::ReLU(self.clone()),
            data: Arc::new(RwLock::new(data)),
            is_mutable: false,
            shape: self.0.shape.clone(),
            backend: self.0.backend.clone(),
            dtype: self.0.dtype.clone(),
            id: Self::uuid(),
        })))
    }

    pub fn exp(&self) -> Result<Self, String> {
        let data = (*self.backend_ref()).exp().expect("Could not perform EXP operation!");

        Ok(Tensor(Arc::new(Tensor_ {
            op: Op::Exp(self.clone()),
            data: Arc::new(RwLock::new(data)),
            is_mutable: false,
            shape: self.0.shape.clone(),
            backend: self.0.backend.clone(),
            dtype: self.0.dtype.clone(),
            id: Self::uuid(),
        })))
    }

    pub fn sum(&self, dims: Vec<usize>) -> Result<Self, String> {
        let sum = (*self.backend_ref()).sum(dims.clone()).expect("Could not perform sum operation!");

        Ok(Tensor(Arc::new(Tensor_ {
            op: Op::Sum(self.clone(), dims.clone()),
            data: Arc::new(RwLock::new(sum)),
            is_mutable: false,
            shape: self.0.shape.clone(),
            backend: self.0.backend.clone(),
            dtype: self.0.dtype.clone(),
            id: Self::uuid(),
        })))
    }

    // Alias for summing all elements in the Tensor
    pub fn sum_all(&self) -> Result<Self, String> {
        Self::sum(&self, vec![])
    }

    pub fn transpose(&self) -> Result<Self, String> {
        let transpose = (*self.backend_ref()).transpose().expect("Could not perform transpose operation!");
        let t_shaped = transpose.shape().clone();

        Ok(Tensor(Arc::new(Tensor_ {
            op: Op::Transpose(self.clone()),
            data: Arc::new(RwLock::new(transpose)),
            is_mutable: false,
            shape: t_shaped,
            backend: self.0.backend.clone(),
            dtype: self.0.dtype.clone(),
            id: Self::uuid(),
        })))
    }

    pub fn get<T: Num + Copy + NumCast>(&self, index: Vec<usize>) -> Option<T> {
        self.backend_ref().get(index)
    }

    fn backend_ref(&self) -> RwLockReadGuard<BackendData> {
        self.0.data.read().unwrap()
    }

    pub fn copy_from(&self, other: &Self) -> () {

        self.0.data.write().unwrap().copy_from(other.backend_ref().deref());
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
                | Op::Mul(lhs, rhs) | Op::Div(lhs, rhs)
                | Op::MatMul(lhs, rhs) => {
                    sorted.push(lhs.clone());
                    sorted.push(rhs.clone());
                    queue.push_back(lhs.clone());
                    queue.push_back(rhs.clone());
                },
                Op::Transpose(_) => {},
                Op::None => (),
                Op::ReLU(_) => {},
                Op::Exp(_) => {},
                Op::Sum(_, _) => {}
            }
        }
        sorted
    }

    pub fn backward(&self) -> Result<Gradients, String> {
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
                Op::MatMul(lhs, rhs) => {
                    let lhs_grad = grad.matmul(&rhs.transpose().unwrap()).unwrap();
                    let lhs_sum_grad = grads.get_or(&lhs);
                    *lhs_sum_grad = lhs_sum_grad.add(&lhs_grad);

                    let rhs_grad = grad.transpose().unwrap().matmul(&lhs).unwrap();
                    let rhs_sum_grad = grads.get_or(&rhs);
                    *rhs_sum_grad = rhs_sum_grad.add(&rhs_grad.transpose().unwrap());
                }
                Op::Transpose(tensor) => {}
                Op::ReLU(tensor) => {}
                Op::Exp(tensor) => {}
                Op::Sum(tensor, dims) => {}
            }
        }
        Ok(grads)
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

#[cfg(test)]
mod tests {

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
        let tensor_a = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let tensor_b = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let add = tensor_a.add(&tensor_b);
        assert_eq!(add.get(vec![1, 1]), Some(2));
    }


    #[test]
    fn test_matmul() {
        let tensor_a = Tensor::fill(vec![2, 3], 5.0, Cpu, DType::F32);
        let tensor_b = Tensor::fill(vec![3, 4], 3.0, Cpu, DType::F32);
        let matmul = tensor_a.matmul(&tensor_b).unwrap();
        assert_eq!(matmul.shape(), vec![2, 4]);
        assert_eq!(matmul.get(vec![1, 1]), Some(45.0));
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
        let gradients = added.backward().unwrap();
        assert_eq!(gradients.get(&tensor_a).unwrap().get(vec![1, 1]), Some(1));
        assert_eq!(gradients.get(&tensor_b).unwrap().get(vec![1, 1]), Some(1));
    }

    #[test]
    fn test_backward_sub() {
        let tensor_a = Tensor::ones(vec![2, 3], Cpu, DType::F32);
        let tensor_b = Tensor::fill(vec![2, 3], 5.0, Cpu, DType::F32);
        let subtracted = tensor_a.sub(&tensor_b);
        let gradients = subtracted.backward().unwrap();
        assert_eq!(gradients.get(&tensor_a).unwrap().get(vec![1, 1]), Some(1));
        assert_eq!(gradients.get(&tensor_b).unwrap().get(vec![1, 1]), Some(-1));
    }

    #[test]
    fn test_backward_mul() {
        let tensor_a = Tensor::fill(vec![2, 3], 5.0, Cpu, DType::F32);
        let tensor_b = Tensor::fill(vec![2, 3], 3.0, Cpu, DType::F32);
        let multiplied = tensor_a.mul(&tensor_b);
        let gradients = multiplied.backward().unwrap();
        assert_eq!(gradients.get(&tensor_a).unwrap().get(vec![1, 1]), Some(3));
        assert_eq!(gradients.get(&tensor_b).unwrap().get(vec![1, 1]), Some(5));
    }

    #[test]
    fn test_backward_matmul() {
        let tensor_a = Tensor::fill(vec![2, 3], 5.0, Cpu, DType::F32);
        let tensor_b = Tensor::fill(vec![3, 5], 3.0, Cpu, DType::F32);
        let matmul = tensor_a.matmul(&tensor_b).unwrap();
        let grads = matmul.backward().unwrap();
        assert_eq!(grads.get(&tensor_a).unwrap().get(vec![1, 1]), Some(15));
        assert_eq!(grads.get(&tensor_b).unwrap().get(vec![1, 1]), Some(10));
    }
}