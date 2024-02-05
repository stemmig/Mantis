use std::collections::HashMap;
use crate::{Tensor, tensor};
use crate::tensor::Tensor_;

pub struct Gradients(pub HashMap<u128, Tensor>);

impl Gradients {
    pub fn new() -> Self {
        Gradients(HashMap::new())
    }

    pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
        self.0.get(&tensor.id())
    }

    pub fn get_or(&mut self, tensor: &Tensor) -> &mut Tensor {
        if !self.0.contains_key(&tensor.id()) {
            self.0.insert(tensor.id(), tensor.zeros_like());
        }
        self.0.get_mut(&tensor.id()).unwrap()
    }
}