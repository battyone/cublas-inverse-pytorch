import torch
import time
from cuda_inverse import cuda_inverse

A = torch.rand(50000,64,64).cuda()


torch.inverse(A)
cuda_inverse(A)
start2 = time.time()
print(torch.inverse(A))
end2 = time.time()
start1 = time.time()
print(cuda_inverse(A))
end1 = time.time()

print("cuBLAS:")
print(end1-start1)
print("MAGMA:")
print(end2-start2)
