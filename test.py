import torch
import time
import cuda_batch_inverse

A = torch.rand(50000,64,64).cuda()


torch.inverse(A)
cuda_batch_inverse.forward(A)
start2 = time.time()
print(torch.inverse(A))
end2 = time.time()
start1 = time.time()
print(cuda_batch_inverse.forward(A))
end1 = time.time()

print("cuBLAS:")
print(end1-start1)
print("MAGMA:")
print(end2-start2)
