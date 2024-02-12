use std::sync::{Arc, RwLock};
use crate::Tensor;
use crate::grad::{Gradients};

pub trait Optimizer {

    fn new(tensors: Vec<Tensor>) -> Self;
    fn step(&self, grads: &Gradients);
    fn zero_grad(&self);
}
pub struct SGD {
    params: Vec<Tensor>,
    eps: f64
}

impl Optimizer for SGD {

    fn new(params: Vec<Tensor>) -> Self {
        SGD {
            params,
            eps: 0.001,
        }
    }
    fn step(&self, grads: &Gradients) {
        for param in &self.params {
            let param_grad = grads.get(&param).expect("Could not find gradients!");
            let lr_tensor = Tensor::fill(param_grad.shape(), self.eps, param_grad.backend(), param_grad.dtype());
            let scaled_grad = param_grad.mul(&lr_tensor);
            let updated_tensor = param.add(&scaled_grad);
            param.copy_from(&updated_tensor)
        }
    }

    fn zero_grad(&self) {
        todo!()
    }
}

impl SGD {
    fn set_epsilon(&mut self, epsilon: f64) -> () {
        self.eps = epsilon;
    }
}

#[cfg(test)]
mod tests {
    use crate::Backend::Cpu;
    use crate::DType;
    use super::*;

    #[test]
    fn test_simple_sgd() {
        let tensor_a = Tensor::fill(vec![2, 3], 5.0, Cpu, DType::F32);
        let tensor_b = Tensor::fill(vec![2, 3], 3.0, Cpu, DType::F32);
        let mut sgd = SGD::new(vec![tensor_a.clone(), tensor_b.clone()]);
        sgd.set_epsilon(0.0001);
        let multiplied = tensor_a.mul(&tensor_b);
        let gradients = multiplied.backward();
        sgd.step(&gradients);
        assert_eq!(tensor_a.get(vec![1,1]), Some(5.00029993f32));
        assert_eq!(tensor_b.get(vec![1,1]), Some(3.00049996f32));
    }
}