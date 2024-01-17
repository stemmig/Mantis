use std::fs::File;
use crate::optim::Optimizer;
use crate::Tensor;

pub trait Model {
    fn forward(&self, input: &Tensor) -> Tensor;

    fn new() -> Self;

    fn training_step(&self);

    fn predict_step(&self);

    fn validation_step(&self);

    fn configure_optimizers(&self);

    fn topo_sort(&self) -> Vec<&Tensor>;

    fn parameters(&self) -> Vec<&Tensor>;

    fn load_from_file(weights_path: File) {
        todo!();
    }
}