#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "complex.h"
#include <stdio.h>
#include <stdlib.h>
#include "multidim_array.h"
#include "mpi.h"
/*
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}*/


typedef struct
{
    //Host-side input data
    size_t datasize; // full size and the real size is half

    int devicenum;

    int selfoffset;
    int realsize;

    cufftComplex *h_Data;

    //Device buffers
    cufftComplex *d_Data;
    int selfZ;

} MultiGPUplan;

void initgpu();
void initgpu_mpi(int ranknum);
double * gpusetdata_double(double *d_data,int N ,double *c_data);
float * gpusetdata_float(float *d_data,int N ,float *c_data);
void vector_Multi(double *data1, float *data2, cufftComplex *res,int numElements);

void vector_Multi_layout(double *data1, float *data2, cufftComplex *res, int numElements,int dimx,int paddim);
cufftComplex * gpumallocdata(cufftComplex *d_outData,int N);
void cpugetdata(tComplex<float> *c_outData, cufftComplex *d_outData,int N);
void printdatatofile(Complex *data,int N,int dimx,int rank,int iter,int flag);
//void printdatatofile(double *data,int N,int dimx,int flag);
//void printdatatofile(cufftComplex *data,int N,int dimx,int flag);
void printdatatofile(float *data,int N,int dimx,int rank,int iter,int flag);
void volume_Multi(float *data1, double *data2, int numElements, int xdim, double sampling , \
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob);


void volume_Multi_float(cufftComplex *data1, float *data2, int numElements, int xdim, double sampling , \
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob);

void volume_Multi_float_mpi(cufftComplex *data1, float *data2, int numElements, int tabxdim, double sampling ,
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob,int ydim,int offset);

void vector_Normlize(cufftComplex *data1, long int normsize, long int numElements);
void fft_Divide(cufftComplex *data1, double *Fnewweight, long int numElements,int xysize,int xsize,int ysize,int zsize,int halfxsize,int max_r2);

void layoutchange(cufftComplex *data,int dimx,int dimy,int dimz, int padx, cufftComplex *newdata);
void layoutchangecomp(Complex *data,int dimx,int dimy,int dimz, int padx, cufftComplex *newdata);
void windowFourier(cufftComplex *d_Fconv,cufftComplex *d_Fconv_window,int rawdim, int newdim);



//=======================================================================backproject_3d_fft.cu
void datainit(cufftComplex *data,int NXYZ);

void multi_enable_access(MultiGPUplan *plan,int GPU_N);
void multi_memcpy_data(MultiGPUplan *plan, cufftComplex *f,int GPU_N,int dimx,int dimy);
void multi_plan_init(MultiGPUplan *plan, int GPU_N, size_t fullsize, int dimx,int dimy,int dimz);
void multi_memcpy_data_gpu(MultiGPUplan *plan,int GPU_N,int dimx,int dimy );

void multi_memcpy_databack(MultiGPUplan *plan, cufftComplex *f,int GPU_N,int dimx,int dimy);
void mulit_alltoall_one(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ);
void mulit_alltoall_two(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ);
void mulit_alltoall_all1to0(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ);
void mulit_datacopy_0to1(MultiGPUplan *plan, int dimx,int dimy,int *offsetZ);
void multi_sync(MultiGPUplan *plan,int GPU_N);


//==============================================================mpi version
void multi_plan_init_mpi(MultiGPUplan *plan, size_t fullsize,size_t realznum,size_t offsetz,int cardnum,int dimx,int dimy);
void gpu_to_cpu(MultiGPUplan *plan,cufftComplex *cpu_data);
void gpu_to_cpu_inverse(MultiGPUplan *plan,cufftComplex *cpu_data);
void cpu_alltoall_inverse(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize);
void cpu_alltoall(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize);
void cpu_alltoalltozero(cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize);
void cpu_allcombine(cufftComplex *cpu_data,int ranknum, int *numberZ, int *offsetZ,int padsize);
void printres(cufftComplex *cpu_data,  int *numberZ ,int *offsetZ,int pad_size,int ranknum);
void printwhole(double *cpu_data,  int fullszie ,int ranknum);
void printwhole(cufftComplex *cpu_data,  int fullszie ,int ranknum);
//void multi_plan_init_mpi(MultiGPUplan *plan, int GPU_N, size_t fullsize, int dimx,int dimy,int dimz,int ranknum);
