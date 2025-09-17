CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Daniel Chen
  * https://www.linkedin.com/in/daniel-c-a02ba2229/
* Tested on: Windows 11, AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (3.80 GHz), RTX 4070 notebook


# Writeup
This project contains several CPU/GPU prefix sum and stream compaction algorithms to compare their performance.

## Algorithm comparison
![algorithm comparison graphs](/img/graphs.png)

The above log-log plots show the running times of each of the implemented scan algorithms, at power-of-2 array sizes (left) and 3 less than each of those power-of-2 array sizes (right). Initial memory setup using `cudaMalloc`, `cudaMemset`, and `thrust::{host_vector, copy}` is excluded from the timings as per Part 7 of [INSTRUCTION.md](INSTRUCTION.md).

For small arrays, CPU scan was the fastest, followed by naive on average, then work-efficient, then thrust. However, throughout array sizes in the low millions, the order reverses: thrust is the fastest, followed by work-efficient, then naive, and then CPU. As the array sizes grow large, they all seem to increase by the same power (about linear). The reversal of the order would likely be due to overhead of the kernel launches at the smaller array sizes and/or possible block size misalignment with the number of threads. As the arrays grow large, however, the benefits of parallelization and GPU memory access optimization become more pronounced the more times the kernels are launched and the more blocks that are instantiated, even if the long-term complexity of all the algorithms remains $O(n)$ (not $O(\log(n))$ for the GPU implementations as the number of concurrent threads will be capped to be some fixed number based on the GPU hardware).

## Extra credit
As requested in Part 5 of [INSTRUCTION.md](INSTRUCTION.md), the work-efficient GPU implementation is over 4 times faster than the CPU implementation for array sizes greater than ~1 million even without further optimization.

## Test output
```

****************
** SCAN TESTS **
****************
    [  29  22  44  14  19  35  38  23  43  43  35  30   6 ...  20   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0012ms    (std::chrono Measured)
    [   0  29  51  95 109 128 163 201 224 267 310 345 375 ... 6278 6298 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0006ms    (std::chrono Measured)
    [   0  29  51  95 109 128 163 201 224 267 310 345 375 ... 6216 6219 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 1.81264ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.605056ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.677728ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.334272ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 2.58243ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.525216ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   2   1   2   0   2   3   0   3   1   1   2   1 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0014ms    (std::chrono Measured)
    [   2   2   1   2   2   3   3   1   1   2   1   1   3 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0011ms    (std::chrono Measured)
    [   2   2   1   2   2   3   3   1   1   2   1   1   3 ...   2   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0057ms    (std::chrono Measured)
    [   2   2   1   2   2   3   3   1   1   2   1   1   3 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.727648ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 1.61846ms    (CUDA Measured)
    passed
Press any key to continue . . .
```