import torch
import time
import cuda_batch_inverse

A = torch.rand(10000,64,64).cuda()

#start2 = time.time()
#print(torch.inverse(A))
#end2 = time.time()
start1 = time.time()
print(cuda_batch_inverse.forward(A))
end1 = time.time()


print(end1-start1)
#print(end2-start2)
