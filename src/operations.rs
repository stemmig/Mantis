use crate::array::NDArray;

pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    None
}

pub enum Backend {
    Array,
    Cpu,
    Metal
}

pub enum BackendData {
    Array(NDArray<f32>),
    Cpu,
    Metal,
}