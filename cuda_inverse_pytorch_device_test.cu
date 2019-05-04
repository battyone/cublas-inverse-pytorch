#include <stdio.h>
#include "cuda_inverse_pytorch_device_test.h"

__global__ void printDevice(float** input){
    printf("%f \n", input[99][99]);
    return;
}

void printDeviceHost(float** input){
    printDevice<<<1,1>>>(input);
}
