import torch
import cuda_batch_inverse

class cuda_inverse_op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = cuda_batch_inverse.forward(input)
        ctx.save_for_backward(output);

        return output;
    
    def backward(ctx, grad_output):

def cuda_inverse(input):
    return cuda_inverse_op.apply(input)
