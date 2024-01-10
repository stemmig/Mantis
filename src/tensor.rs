use std::sync::Arc;
use crate::operations::{Backend, BackendData, Op};

pub struct Tensor {
    op: Op,
    data: Arc<BackendData>,
    is_mutable: bool,
    shape: Vec<usize>,
    backend: Backend,
}

pub trait Data where Self: Sized {
    fn new() -> Self;

    fn zeros(shape: Vec<usize>) -> Self;

    fn ones(shape: Vec<usize>) -> Self;

    fn add(&self, rhs: &Self) -> Self;

}

impl Tensor {
    // Keeping track of compute graph with be handled in Tensor impls,
    // Actually modifying the underlying tensor on the backend will be done as part of data impls
    // pub fn new(op: Op, ) -> Self {
    //     Tensor {
    //
    //     }
    // }
    //
    // pub fn zeros(dims: Vec<usize>) {
    //     Tensor {
    //         None,
    //
    //     }
    // }
}