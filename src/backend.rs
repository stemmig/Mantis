use num_traits::Zero;
use crate::array::NDArray;

pub enum Backend {
    Array,
    Cpu,
    Metal
}

pub enum BackendData
{
    Array(NDArray),
    Cpu,
    Metal,
}