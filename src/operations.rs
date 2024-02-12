use crate::Tensor;

#[derive(Clone)]
pub enum Op {
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),
    // MatMul(Tensor, Tensor),
    // ReLU(Tensor),
    None
}