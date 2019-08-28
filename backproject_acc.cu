

#include "backproject_impl.h"


__global__ void vectorMulti(double *A, double *B, cufftDoubleComplex *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        //C[i].x = A[i] * B[i];
        C[i].x = 100;
    }

    printf("%f from gpu \n",C[i].x);

}

void initgpu()
{
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("GPU num for max %d \n",devCount);
	cudaSetDevice(0);
}



void gpusetdata_double(double *d_data,int N ,double *c_data)
{
	cudaMalloc((void**) &d_data, N * sizeof(double));
	cudaMemcpy(d_data, c_data, N * sizeof(double),cudaMemcpyHostToDevice);
}
void gpusetdata_float(float *d_data,int N ,float *c_data)
{
	cudaMalloc((void**) &d_data, N * sizeof(float));
	cudaMemcpy(d_data, c_data, N * sizeof(float),cudaMemcpyHostToDevice);
}

void vector_Multi(double *data1, double *data2, cufftDoubleComplex *res,int numElements)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	vectorMulti<<<blocksPerGrid, threadsPerBlock>>>(data1, data2, res, numElements);
}

void cpugetdata(cufftDoubleComplex *d_outData, cufftDoubleComplex *c_outData,int N)
{
	cudaMemcpy(d_outData, c_outData, N * sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost);
}
void gpumallocdata(cufftDoubleComplex *d_outData,int N)
{
	cudaMalloc((void**) &d_outData,  N * sizeof(cufftDoubleComplex));
}

void printdatatofile(Complex *data,int N)
{
	FILE *fp= fopen("data1.out","w+");
	for(int i=0;i< N ;i++)
	{
		fprintf(fp,"%f %f |",data[i].real,data[i].imag);
		if(i%100==0)
			fprintf(fp,"\n");
	}
	fclose(fp);
}
