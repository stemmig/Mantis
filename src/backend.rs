use crate::array::{CpuArray};

pub enum Backend {
    Array,
    Cpu,
    Metal
}

pub enum BackendData
{
    Array(CpuArray),
    Cpu,
    Metal,
}