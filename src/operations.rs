use crate::Tensor;

#[derive(Clone)]
pub enum Op {
    // Arithmetic
    Add(Tensor, Tensor),
    Sub(Tensor, Tensor),
    Mul(Tensor, Tensor),
    Div(Tensor, Tensor),

    // Tensor
    MatMul(Tensor, Tensor),

    // Unary
    ReLU(Tensor),
    Exp(Tensor),

    // Reductions
    Sum(Tensor, Vec<usize>),

    // Leaf
    None
}