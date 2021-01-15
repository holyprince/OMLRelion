### 2019.11.5
1. #define TIMEICT to test the all module time with all process
2. time version1

### 2019.11.6
1. time version add iter

### 2019.11.26
1. #define TIMING  (the flag in file : acc_ml_optimiser_impl.h)  and TIMING_FILES( the flag in file : CUDA_BENCHMARK_UTILS_H_)  to test all module time for expectation

### 2020.3.4
backup 2019 from 51 server
makefile path : /GPUFS/ict_zyliu_2/code/relion-nsight/testDebug/build.sh


### 2020.3.4
version for mixed-presion
1.profile expectation again
2.check tensor core avail

### 2020.3.20
1.MPI块数确定
2.debug pack 去掉，只保留send之前的一句输出 ml.cpp 1978
3.before combine data
write data at 1,10,25


### 2020.9.13
including debug info for twonodes FFT

### 2020.9.22
0922 maximazation multicard ok debug prepare for projection compression
add define info
FMDEBUG RANDOM

### 2020.9.28
prepare for one slice backprojection

### 2020.11.09
use one month to code slice backprojection
and two weeks to improve the performance
new macro including :

//#define TIMING  
//#define COMGPU  
#define FMDEBUG      using this to convergence immediately  
//#define RANDOM     using this to test 750 accuracy  
#define BACKSLICE    using this to calc slice backprojection  
//#define PINMEM     using this to test pin memory  
#define TIMBPS       using this to calcuate three module time  
//#define MAXCPU     using cpu to calc maximazation  



### 2020.11.20
finish filterslice backprojection with sort and reduction using thrust but the performance is not very well
so next try to improve the throughput at the same time.

