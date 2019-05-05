# cublas-inverse-pytorch
cuBLAS-based batch inverse for PyTorch.

References:
[1] [KinglittleQ/torch-batch-svd](https://github.com/KinglittleQ/torch-batch-svd)
[2] [ShigekiKarita/pytorch-cusolver](https://github.com/ShigekiKarita/pytorch-cusolver) 

## Installation
``` shell
python3 setup.py install
```

## Usage
``` python
from cuda_inverse import cuda_inverse
A = torch.rand(1000,30,30)
A_inv = cuda_inverse(A)
```
## Dependencies 
Tested with
* CUDA 10
* gcc 7.3.0
* Pytorch 1.1
