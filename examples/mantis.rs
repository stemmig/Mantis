extern crate mantis;

use mantis::{Tensor, Backend, DType, SGD, Model, Trainer};
use mantis::Layer::{linear, conv2d, softmax};

struct MLP {
    // Linear is a wrapper around a weights Tensor w and a bias b
    fc1: Linear,
    fc2: Linear,
    loss_fn: MSE
}

impl Model for MLP {
    fn forward(&self, input: &Tensor) -> Tensor {
        let l1 = self.fc1.invoke(input);
        let l2 = self.fc2.invoke(l1);
        softmax(l2)
    }

    fn new() -> MLP {
        MLP {
            fc1: Linear::new(3, 2), // 2 In -> 3 Out
            fc2: Linear::new(2, 3), // 2 In -> 2 Out
            loss: MSE::new(),
        }
    }
}

fn main() {
    // Low level tensor operations
    let tensor_a = Tensor::ones(vec![3, 4], Backend::Cpu, DType::F32);
    let tensor_b = Tensor::ones(vec![3, 4], Backend::Cpu, DType::F32);

    let loss = tensor_a.add(&tensor_b);

    // Option 1: PyTorch Lightning style Trainer

    let dataset = DataSet::from_tensor(tensor_a, 2);

    let mlp: MLP = MLP::new();
    let trainer: Trainer = Trainer::new();

    trainer.fit(&mlp, &dataset);

    // Option 2: Manual implementation of train / val / test loops

    let optimizer = SGD::new();

    for epoch in 0..10 {
        // For now use a zero tensor and a single batch per epoch
        let target = Tensor::zeros(vec![2, 1], Backend::Cpu, DType::F32);
        let batch = Tensor::zeros(vec![2, 5], Backend::Cpu, DType::F32);
        let out = mlp.forward(&batch);
        let loss = mlp.loss(&out, &target);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
    println!("Hello!");
}