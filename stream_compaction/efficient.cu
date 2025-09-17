#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        int* dev_arr;
        int *dev_arr2;
        int* dev_bools;

        __global__ void upsweepStep(int nc, int step, int* arr) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            int indexRight = (index + 1) * step - 1;
            if (indexRight >= nc) return;

            if (indexRight == nc - 1 && step == nc) {
                arr[indexRight] = 0;
                return;
            }

            int indexLeft = indexRight - step / 2;

            arr[indexRight] = arr[indexLeft] + arr[indexRight];
        }

        __global__ void downsweepStep(int nc, int step, int* arr) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            int indexRight = (index + 1) * step - 1;
            if (indexRight >= nc) return;

            int indexLeft = indexRight - step / 2;

            int right = arr[indexRight];
            arr[indexRight] += arr[indexLeft];
            arr[indexLeft] = right;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int nc = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_arr, nc * sizeof(int));


            timer().startGpuTimer();
            // TODO

            
			cudaMemcpy(dev_arr, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(dev_arr + n, 0, (nc - n) * sizeof(int));


			int blockSize = 256;


            for (int step = 2; step <= nc; step <<= 1) {
                int nBlocks = (nc / step + blockSize - 1) / blockSize;
                upsweepStep<<<nBlocks, blockSize>>>(nc, step, dev_arr);
            }

            for (int step = nc; step >= 2; step >>= 1) {
                int nBlocks = (nc / step + blockSize - 1) / blockSize;
                downsweepStep<<<nBlocks, blockSize>>>(nc, step, dev_arr);
            }
            

			cudaMemcpy(odata, dev_arr, n * sizeof(int), cudaMemcpyDeviceToHost);


            timer().endGpuTimer();


            cudaFree(dev_arr);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int nc = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_arr, nc * sizeof(int));
            cudaMalloc((void**)&dev_arr2, nc * sizeof(int));
            cudaMalloc((void**)&dev_bools, nc * sizeof(int));


            timer().startGpuTimer();

            if (n == 0) {
                timer().endGpuTimer();
                return 0;
            }


            int blockSize = 256;
            int nBlocksOuter = (nc + blockSize - 1) / blockSize;


            cudaMemcpy(dev_arr, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemset(dev_arr + n, 0, (nc - n) * sizeof(int));

            Common::kernMapToBoolean<<<nBlocksOuter, blockSize>>>(nc, dev_bools, dev_arr);

            for (int step = 2; step <= nc; step <<= 1) {
                int nBlocks = (nc / step + blockSize - 1) / blockSize;
                upsweepStep<<<nBlocks, blockSize>>>(nc, step, dev_bools);
            }

            for (int step = nc; step >= 2; step >>= 1) {
                int nBlocks = (nc / step + blockSize - 1) / blockSize;
                downsweepStep<<<nBlocks, blockSize>>>(nc, step, dev_bools);
            }

            Common::kernScatter<<<nBlocksOuter, blockSize>>>(n, dev_arr2, dev_arr, NULL, dev_bools);

            int nFinal;
            cudaMemcpy(&nFinal, dev_bools + nc - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaMemcpy(odata, dev_arr2, nFinal * sizeof(int), cudaMemcpyDeviceToHost);


            timer().endGpuTimer();


            cudaFree(dev_arr);
            cudaFree(dev_arr2);
            cudaFree(dev_bools);

            return nFinal;
        }
    }
}
