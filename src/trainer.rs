use crate::data::{DataLoader, DataSet};
use crate::Model;

pub trait TrainingManager<M, D> where
        M: Model,
        D: DataLoader,
{

    fn new() -> Self;
    fn fit(&self, model: M, data_set: D);
    fn predict(&self, model: M, data_set: D);

    // TODO:
    // fn validate(&self, model: M, data_set: D);
    // fn test(&self, model: M, data_set: D);
}

pub struct Trainer<M, D> where
    M: Model,
    D: DataLoader
{
    model: M,
    dataset: D
}

impl<M, D> TrainingManager<M, D> for Trainer<M, D>
    where
        M: Model,
        D: DataLoader,
{
    fn new() -> Self {
        todo!()
    }

    fn fit(&self, model: M, data_set: D) {
        todo!()
    }

    fn predict(&self, model: M, data_set: D) {

    }
}