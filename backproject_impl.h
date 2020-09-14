#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "complex.h"
#include <stdio.h>
#include <stdlib.h>
#include "multidim_array.h"
#include "mpi.h"
#include "fftw3.h"
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
    size_t selfZ;
    size_t selfoffset;
    size_t tempydim;
    size_t realsize;
    size_t tempsize; // need pad to max size

    cufftComplex *h_Data;
    //Device buffers
    cufftComplex *d_Data;
    cufftComplex *temp_Data;


} MultiGPUplan;

void initgpu(int GPU_N);
void initgpu_mpi(int ranknum);
double * gpusetdata_double(double *d_data,int N ,double *c_data);
float * gpusetdata_float(float *d_data,int N ,float *c_data);
float * gpusetdata_float(float *d_data,int N ,double *c_data);

void vector_Multi(double *data1, float *data2, cufftComplex *res,int numElements);

void vector_Multi_layout(double *data1, RFLOAT *data2, cufftComplex *res, int numElements,int dimx,int paddim);
void vector_Multi_layout_mpi(double *data1, float *data2, cufftComplex *res,size_t numElements);

cufftComplex * gpumallocdata(cufftComplex *d_outData,int N);
void cpugetdata(tComplex<float> *c_outData, cufftComplex *d_outData,int N);
void printdatatofile(Complex *data,int N,int dimx,int rank,int iter,int flag);
//void printdatatofile(double *data,int N,int dimx,int flag);
//void printdatatofile(cufftComplex *data,int N,int dimx,int flag);
void printdatatofile(float *data,int N,int dimx,int rank,int iter,int flag);


void volume_Multi(float *data1, double *data2, int numElements, int xdim, double sampling , \
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob);


void volume_Multi_float(cufftComplex *data1, RFLOAT *data2, int numElements, int xdim, double sampling , \
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob);

void volume_Multi_float_mpi(cufftComplex *data1, float *data2, int numElements, int tabxdim, double sampling ,
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob,int ydim,int offset);

void vector_Normlize(cufftComplex *data1, long int normsize, size_t numElements);
void fft_Divide(cufftComplex *data1, double *Fnewweight, long int numElements,int xysize,int xsize,int ysize,int zsize,int halfxsize,int max_r2);
void fft_Divide_mpi(cufftComplex *data1, double *Fnewweight, size_t numElements,int xysize,
		int xsize,int ysize,int zsize, int halfxsize,int max_r2,int zoffset);


void layoutchange(cufftComplex *data,size_t dimx,size_t dimy,size_t dimz, size_t padx, cufftComplex *newdata);
void layoutchange(double *data,size_t dimx,size_t dimy,size_t dimz, size_t padx, double *newdata);
void layoutchange(float *data,size_t dimx,size_t dimy,size_t dimz, size_t padx, float *newdata);
void layoutchange(double *data,size_t dimx,size_t dimy,size_t dimz, size_t padx, float *newdata);
void layoutchangecomp(Complex *data,size_t dimx,size_t dimy,size_t dimz, size_t padx, cufftComplex *newdata);
void layoutchangeback(double *newdata,size_t dimx,size_t dimy,size_t dimz, size_t padx, double *data);
void windowFourier(cufftComplex *d_Fconv,cufftComplex *d_Fconv_window,int rawdim, int newdim);
void validateconj(cufftComplex *data,int dimx,int dimy,int dimz, int padx);
void validateconj(fftwf_complex *data,int dimx,int dimy,int dimz, int padx);


//=======================================================================backproject_3d_fft.cu
void datainit(cufftComplex *data,int NXYZ);

void multi_enable_access(MultiGPUplan *plan,int GPU_N);
void multi_memcpy_data(MultiGPUplan *plan, cufftComplex *f,int GPU_N,int dimx,int dimy);
void multi_plan_init(MultiGPUplan *plan, int GPU_N, size_t fullsize, int *numberZ, int *offsetZ, int pad_size);
void multi_plan_init(MultiGPUplan *plan, int GPU_N, size_t fullsize, int dimx,int dimy,int dimz);
void multi_memcpy_data_gpu(MultiGPUplan *plan,int GPU_N,int dimx,int dimy );

void multi_memcpy_databack(MultiGPUplan *plan, cufftComplex *f,int GPU_N,int dimx,int dimy);
void mulit_alltoall_one(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ);
void gpu_alltoall_multinode(MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ);
void gpu_alltoall_multinode_inverse(MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ);
void mulit_alltoall_two(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ);
void mulit_alltoall_all1to0(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ);
void mulit_datacopy_0to1(MultiGPUplan *plan, int dimx,int dimy,int *offsetZ);
void multi_sync(MultiGPUplan *plan,int GPU_N);


//==============================================================mpi version
void multi_plan_init_mpi(MultiGPUplan *plan, size_t fullsize,size_t realznum,size_t offsetz,int cardnum,int dimx,int dimy);
void gpu_to_cpu(MultiGPUplan *plan,cufftComplex *cpu_data);
void gpu_to_cpu_1dfft(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int *offsetZ,int padsize,int ranknum);
void gpu_to_cpu_inverse(MultiGPUplan *plan,cufftComplex *cpu_data);
void validatealltoall(cufftComplex *cpu_data,int *numberZ,int *offsetZ, int ranknum,int padsize);
void cpu_alltoall_inverse(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize);
void cpu_alltoall_inverse_multinode(MultiGPUplan *plan,cufftComplex *cpu_data,
		int *numberZ,int *offsetZ,int ranknum,int padsize,int ranksize,int *realrankarray);
void cpu_alltoall(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize);
void cpu_alltoall_multinode(MultiGPUplan *plan,cufftComplex *cpu_data,
		int *numberZ,int *offsetZ,int ranknum,int padsize,int ranksize,int *realrankarray);
void cpu_alltoalltozero(cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize);
void cpu_alltoalltozero_multi(cufftComplex *cpu_data,int *numberZ,int *offsetZ,
		int ranknum,int padsize,int ranksize,int *realrankarray);
void cpu_allcombine(cufftComplex *cpu_data,int ranknum, int *numberZ, int *offsetZ,int padsize);
void cpu_allcombine_multi(cufftComplex *cpu_data,int ranknum, int *numberZ, int *offsetZ,
		int padsize,int ranksize,int *realrankarray);
void printres(cufftComplex *cpu_data,  int *numberZ ,int *offsetZ,int pad_size,int ranknum);
void printgpures(cufftComplex *cpu_data, int fullsize,int ranknum);
void printgpures(double *cpu_data, int fullsize,int ranknum);
void printgpures(float *cpu_data, int fullsize,int ranknum);
void printwhole(double *cpu_data,  int fullszie ,int ranknum);
void printwhole(float *cpu_data,  size_t fullszie ,int ranknum);
void printwhole(cufftComplex *cpu_data,  int fullszie ,int ranknum);
//void multi_plan_init_mpi(MultiGPUplan *plan, int GPU_N, size_t fullsize, int dimx,int dimy,int dimz,int ranknum);


//==========================transpose version
void dividetask(int *numberZ, int *offsetZ,int pad_size,int ranksize);
void multi_plan_init_transpose(MultiGPUplan *plan, int GPU_N, int *numberZ, int *offsetZ, int pad_size);
void multi_plan_init_transpose_multi(MultiGPUplan *plan, int GPU_N, int *numberZ, int *offsetZ, size_t pad_size,int ranknum,int ranksize);
void transpose_exchange(MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ,int *offsettmpZ);

void transpose_exchange_intra(MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ ,int *numberZ,
		int *offsettmpZ,int ranknum);
void data_exchange_gputocpu(MultiGPUplan *plan, cufftComplex *cpu_data, int GPU_N,size_t pad_size,int *offsetZ ,int ranknum);

void data_exchange_cputogpu(MultiGPUplan *plan, cufftComplex *cpu_data, int GPU_N,int pad_size,int *offsetZ ,
		int *numberZ, int *offsettmpZ, int *numbertmpZ, int ranknum,int sumranknum, int padsizetmp);


void yzlocal_transpose(MultiGPUplan *plan,int GPU_N,int pad_size,int *numberZ,int *offsetZ,int *offsettmpZ);
void volume_Multi_float_transone(cufftComplex *data1, RFLOAT *data2, int numElements, int tabxdim, double sampling ,
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob,int ydim,int offset);

void yzlocal_transpose_multicard(MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ,int *numberZ,int *offsettmpZ,int ranknum,int sumrank);

void cpu_alltoalltrans_multinode(MultiGPUplan *plan,cufftComplex *cpu_data,size_t padsize, int *numberZ,
		int *offsetZ,int ranknum,int padtmpsize,int ranksize,int *realrankarray);
void cpu_alltozero_multinode(cufftComplex *pad_data,size_t padsize,
		int *processnumberZ, int *processoffsetZ, int ranknum,size_t padtmpsize,int ranksize,int *realrankarray);
void cpu_datarearrange_multinode(cufftComplex *pad_data,cufftComplex *real_data,size_t padsize, int *numberZ,int *offsetZ,
		int *numbertmpZ,int *offsettmpZ, int ranknum,size_t padtmpsize,int sumranknum);
