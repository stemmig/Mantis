use crate::Tensor;

pub trait DataLoader {

}

pub struct DataSet {

}

impl DataLoader for DataSet {

}

impl DataSet {
    fn from_tensor(train:&Tensor, val:&Tensor, test:&Tensor, batch_size:i32){
        todo!()
    }
}
