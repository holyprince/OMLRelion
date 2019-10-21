#include "cuda_runtime.h"
#include "cufft.h"
#include "complex.h"
#include <stdio.h>
#include <stdlib.h>


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}


void initgpu();
double * gpusetdata_double(double *d_data,int N ,double *c_data);
float * gpusetdata_float(float *d_data,int N ,float *c_data);
void vector_Multi(double *data1, float *data2, cufftComplex *res,int numElements);
cufftComplex * gpumallocdata(cufftComplex *d_outData,int N);
void cpugetdata(tComplex<float> *c_outData, cufftComplex *d_outData,int N);
void printdatatofile(Complex *data,int N);

void volume_Multi(float *data1, double *data2, int numElements, int xdim, double sampling , \
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob);


void volume_Multi_float(float *data1, float *data2, int numElements, int xdim, double sampling , \
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob);
void vector_Normlize(cufftComplex *data1, long int normsize, long int numElements);
void fft_Divide(cufftComplex *data1, double *Fnewweight, long int numElements,int xysize,int xsize,int zsize,int ysize,int max_r2);
