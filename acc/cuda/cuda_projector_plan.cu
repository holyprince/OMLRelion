#include "../acc_projector_plan.h"
#include "../../time.h"
#include <cuda_runtime.h>

#ifdef CUDA
//#include <cuda_runtime.h>
#ifdef CUDA_FORCESTL
#include "cuda_utils_stl.cuh"
#else
#include "cuda_utils_cub.cuh"
#endif
#endif

#include "../utilities.h"

#include "../acc_projector_plan_impl.h"
