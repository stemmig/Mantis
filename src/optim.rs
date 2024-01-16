use crate::Tensor;
use crate::grad::{Gradients};

trait Optimizer {
    fn step(&self, grads: &Gradients);

    fn update_weights(&self, loss: &Tensor);
}
struct SGD {
    tensors: Vec<Tensor>,
    eps: f64
}

impl Optimizer for SGD {
    fn step(&self, grads: &Gradients) {
        todo!()
    }

    fn update_weights(&self, loss: &Tensor) {
        todo!()
    }
}