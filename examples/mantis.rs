extern crate mantis;

use mantis::{Tensor, Backend, DType};

fn main() {
    let tensor_a = Tensor::ones(vec![3, 4], Backend::Array, DType::F32);
    let tensor_b = Tensor::ones(vec![3, 4], Backend::Array, DType::F32);

    let tensor_c = tensor_a.add(&tensor_b);
    println!("Hello!");
}