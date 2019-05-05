import torch
import cuda_batch_inverse

class cuda_inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = cuda_batch_inverse.forward(input)
        ctx.save_for_backward(output);
        return output;

    @staticmethod
    def backward(ctx, grad_output):
        output = cuda_batch_inverse.backward(ctx.saved_variables[0], grad_output)
        return output
