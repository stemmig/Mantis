pub mod operations;
mod tensor;
mod array;
mod backend;
mod dtype;
mod grad;
mod optim;
mod model;
mod trainer;
mod data;

pub use tensor::Tensor;
pub use backend::Backend;
pub use dtype::DType;
pub use optim::{SGD};
pub use model::Model;
pub use trainer::Trainer;