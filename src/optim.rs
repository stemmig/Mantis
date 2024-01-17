use std::sync::{Arc, RwLock};
use crate::Tensor;
use crate::grad::{Gradients};

pub trait Optimizer {

    fn new(tensors: Vec<Arc<RwLock<Tensor>>>) -> Self;
    fn step(&self, grads: &Gradients);
    fn update_weights(&self, loss: &Tensor);
}
pub struct SGD {
    tensors: Vec<Arc<RwLock<Tensor>>>,
    eps: f64
}

impl Optimizer for SGD {

    fn new(tensors: Vec<Arc<RwLock<Tensor>>>) -> Self {
        SGD {
            tensors,
            eps: 0.001,
        }
    }
    fn step(&self, grads: &Gradients) {
        todo!()
    }

    fn update_weights(&self, loss: &Tensor) {
        todo!()
    }
}

impl SGD {
    fn set_epsilon(&mut self, epsilon: f64) -> () {
        self.eps = epsilon;
    }
}