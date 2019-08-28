#include "cuda_runtime.h"
#include "cufft.h"
#include "complex.h"
#include <stdio.h>
#include <stdlib.h>

void initgpu();
void gpusetdata_double(double *d_data,int N ,double *c_data);
void vector_Multi(double *data1, double *data2, cufftDoubleComplex *res,int numElements);
void gpumallocdata(cufftDoubleComplex *d_outData,int N);
void cpugetdata(cufftDoubleComplex *d_outData, cufftDoubleComplex *c_outData,int N);
void printdatatofile(Complex *data,int N);
