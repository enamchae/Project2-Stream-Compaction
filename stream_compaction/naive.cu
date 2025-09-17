#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        int *dev_arr1;
        int *dev_arr2;

        __global__ void add(int n, int skip, int* out, int* in) {
            unsigned long long int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;

            out[index] = in[index];

            if (index >= skip) {
                out[index] += in[index - skip];
            }
        }

        __global__ void insert0(int n, int* out, int* in) {
            unsigned long long int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) return;

            out[index] = index == 0 ? 0 : in[index - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            cudaMalloc((void**)&dev_arr1, n * sizeof(int));
            cudaMalloc((void**)&dev_arr2, n * sizeof(int));

            cudaMemcpy(dev_arr1, idata, n * sizeof(int), cudaMemcpyHostToDevice);


            timer().startGpuTimer();


			int blockSize = 256;
			int nBlocks = (n + blockSize - 1) / blockSize;


            int *curIn = dev_arr1;
            int *curOut = dev_arr2;

            insert0<<<nBlocks, blockSize>>>(n, curOut, curIn);

            for (int skip = 1; skip < n; skip <<= 1) {
                int* tmp = curIn;
                curIn = curOut;
                curOut = tmp;

                add<<<nBlocks, blockSize>>>(n, skip, curOut, curIn);
            }
            

            
            timer().endGpuTimer();
            
            
            cudaMemcpy(odata, curOut, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_arr1);
            cudaFree(dev_arr2);
        }
    }
}
