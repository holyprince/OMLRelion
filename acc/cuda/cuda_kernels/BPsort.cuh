
#ifndef CUDA_BPS_KERNELS_CU_
#define CUDA_BPS_KERNELS_CU_

#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include "../../acc_projector.h"
#include "../../acc_backprojector.h"
#include "../cuda_settings.h"
#include "../cuda_kernels/cuda_device_utils.cuh"



#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>




void cuda_kernel_backproject3D_sortbykey(int *d_index,XFLOAT *d_real,XFLOAT *d_imag,XFLOAT *d_weight,int *d_count,int numElements,int &countnum);
#endif
