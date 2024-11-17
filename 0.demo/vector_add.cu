#include <stdio.h>

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA Kernel function to add the elements of an array
__global__ void arraySumKernel(float *array, float *sum, int numElements)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < numElements)
    {
        atomicAdd(sum, array[index]);
    }
}

int main(void)
{
    int numElements = 1<<20;
    size_t size = numElements * sizeof(float);
    float *h_array = (float *)malloc(size);
    float *d_array = NULL;
    float *d_sum = NULL;
    float h_sum = 0.0f;

    // Initialize the host input array
    for (int i = 0; i < numElements; ++i)
    {
        h_array[i] = 1.0f;
    }

    // Allocate the device input array
    CUDA_CALL(cudaMalloc((void **)&d_array, size));
    CUDA_CALL(cudaMalloc((void **)&d_sum, sizeof(float)));

    // Copy the host input array h_array to the device input array d_array
    CUDA_CALL(cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMemset(d_sum, 0, sizeof(float)));

    // Launch the Array Sum CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    arraySumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_array, d_sum, numElements);
    cudaDeviceSynchronize();

    // Check for any errors launching the kernel
    CUDA_CALL(cudaGetLastError());

    // Copy the device result array in device memory to the host result array
    CUDA_CALL(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Sum = %f\n", h_sum);

    // Free device global memory
    CUDA_CALL(cudaFree(d_array));
    CUDA_CALL(cudaFree(d_sum));

    // Free host memory
    free(h_array);

    return 0;
}
