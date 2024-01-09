# Mantis Design
We will need to use Reverse Mode Autodiff 

### Classes
1. Tensor
   1. Each Tensor should have a list of children (i.e. the previous tensors)
   2. Tensors are connected to their child by an Operation
   3. Tensors that contain weights will need to be able to accumulate gradients.

### Enums
1. Operation
2. Device
   1. CPU
   2. Metal

### Resources
1. Autodiff lecture https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf
2. Original Oldschool Autograd: https://github.com/HIPS/autograd
3. Simplified Autodiff: https://github.com/mattjj/autodidact