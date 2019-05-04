#include <torch/extension.h>
#include <THC/THC.h>
//#undef NDEBUG

#include <cstdio>
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>


cublasHandle_t getCurrentCUDABlasHandle() {
    return THCState_getCurrentBlasHandle(at::globalContext().getTHCState());
}

namespace torch_cublas_inverse{
    at::Tensor batch_inverse_forward(at::Tensor input){
        AT_CHECK(input.is_cuda(), "Only CUDA tensors supported.");
        AT_CHECK(input.dtype() == at::kFloat, "Only float is supported.");

        const auto batch_size = input.size(0);
        const auto n = input.size(1);
        const auto m = input.size(2);

        AT_CHECK(m <= 32, "matrix row should be <= 32");
        AT_CHECK(n <= 32, "matrix col should be <= 32");
        AT_CHECK(n == m, "matrix not square.");

        const auto output = input.contiguous().clone();

        //need to mediate between float** and pytorch output (float*).
        float** batched_input;
        float** batched_input_host = new float*[batch_size];

        float** batched_output;
        float** batched_output_host = new float*[batch_size];

        cudaMalloc((void**)&batched_input, batch_size*sizeof(float*));
        cudaMalloc((void**)&batched_output, batch_size*sizeof(float*));

        //iterate over batches, assign data.
        for(int i = 0; i < batch_size; i++){
            batched_input_host[i] = input[i].data<float>();
            batched_output_host[i] = output[i].data<float>();
        }

        // copy batch pointers to device
        cudaMemcpy(batched_input, batched_input_host, batch_size*sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(batched_output, batched_output_host, batch_size*sizeof(float*), cudaMemcpyHostToDevice);
        
        
        int* INFO;
        cudaMalloc((void**)&INFO, batch_size*sizeof(int));

        auto cuBlasHandle = getCurrentCUDABlasHandle();
        

        
        //kernel call
        cublasSmatinvBatched(cuBlasHandle, 
                n, 
                (const float**)batched_input, 
                n, 
                batched_output,
                n, 
                INFO, 
                batch_size);
    
        //free allocated
        cudaFree(batched_input);
        cudaFree(batched_output);
        cudaFree(INFO);
        delete [] batched_input_host;
        delete [] batched_output_host;

        return output;
    }
    
    at::Tensor batch_inverse_backward(at::Tensor grad_input, at::Tensor output){
        return at::matmul(output.transpose(-2,-1), at::matmul(grad_input, output.transpose(-2,-1)));
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("forward", &torch_cublas_inverse::batch_inverse_forward, "cuBLAS inverse forward");
    m.def("backward", &torch_cublas_inverse::batch_inverse_backward, "cuBLAS inverse backward");
}
