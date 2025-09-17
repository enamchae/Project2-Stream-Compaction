CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Daniel Chen
  * https://www.linkedin.com/in/daniel-c-a02ba2229/
* Tested on: Windows 11, AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (3.80 GHz), RTX 4070 notebook


# Writeup

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