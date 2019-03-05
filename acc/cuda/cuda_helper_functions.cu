#undef ALTCPU
#include <cuda_runtime.h>
#include "../../ml_optimiser.h"
#include "../acc_ptr.h"
#include "../acc_projector.h"
#include "../acc_projector_plan.h"
#include "../acc_backprojector.h"
#include "cuda_settings.h"
#include "cuda_fft.h"
#include "cuda_kernels/cuda_device_utils.cuh"

#ifdef CUDA_FORCESTL
#include "cuda_utils_stl.cuh"
#else
#include "cuda_utils_cub.cuh"
#endif

#include "../utilities.h"
#include "../acc_helper_functions.h"
#include "../cuda/cuda_kernels/BP.cuh"
#include "../../macros.h"
#include "../../error.h"

#include "../acc_ml_optimiser.h"
#include "cuda_ml_optimiser.h"
#include "../acc_helper_functions.h"


#include "../acc_helper_functions_impl.h"
