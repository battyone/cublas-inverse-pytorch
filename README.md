# cublas-inverse-pytorch
cuBLAS-based batch inverse for PyTorch

## Installation
``` shell
python setup.py install
```

## Usage
``` python
from cuda_inverse import cuda_inverse
A = torch.rand(1000,30,30)
A_inv = cuda_inverse(A)
```
