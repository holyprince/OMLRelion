

#include "backproject_impl.h"

__device__ double tab_ftblobgetvalue(double *tabulatedValues, double val,double sampling,int xdim)
{

	int idx = (int)( ABS(val) / sampling);
	if (idx >= xdim)
		return 0.;
	else
		return tabulatedValues[idx];
}
__global__ void vectorMulti(double *A, double *B, cufftComplex *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i].x = A[i] * B[i];
    }
}

__global__ void volumeMulti(double *Mconv, double *tabdata, int numElements, int xdim, double sampling , int padhdim, int pad_size, int ori_size, double padding_factor, float normftblob, int zslice)
{


    int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < numElements) {

		int k = index / zslice;
		int xyslice = index % (zslice);
		int i = xyslice / 723;
		int j = xyslice % 723;

		int kp = (k < padhdim) ? k : k - pad_size;
		int ip = (i < padhdim) ? i : i - pad_size;
		int jp = (j < padhdim) ? j : j - pad_size;
		double rval = sqrt((double) (kp * kp + ip * ip + jp * jp)) / (ori_size * padding_factor);

		Mconv[index] *= (tab_ftblobgetvalue(tabdata, rval, sampling, xdim) / normftblob);

		if ( index ==0)
		{
			printf("From GPU : %f %f %f \n",Mconv[index],tab_ftblobgetvalue(tabdata, rval, sampling, xdim),rval);
		}
	}
}

void initgpu()
{
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("GPU num for max %d \n",devCount);
	cudaSetDevice(0);
}



double * gpusetdata_double(double *d_data,int N ,double *c_data)
{
	cudaMalloc((void**) &d_data, N * sizeof(double));
	cudaMemcpy(d_data, c_data, N * sizeof(double),cudaMemcpyHostToDevice);
	return d_data;
}
float * gpusetdata_float(float *d_data,int N ,float *c_data)
{
	cudaMalloc((void**) &d_data, N * sizeof(float));
	cudaMemcpy(d_data, c_data, N * sizeof(float),cudaMemcpyHostToDevice);
	return d_data;
}

void vector_Multi(double *data1, double *data2, cufftComplex *res, int numElements)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	vectorMulti<<<blocksPerGrid, threadsPerBlock>>>(data1, data2, res, numElements);
}

void cpugetdata(tComplex<float> *c_outData, cufftComplex *d_outData,int N)
{
	cudaMemcpy(c_outData, d_outData, N * sizeof(cufftComplex),cudaMemcpyDeviceToHost);
}
cufftComplex* gpumallocdata(cufftComplex *d_outData,int N)
{
	cudaMalloc((void**) &d_outData,  N * sizeof(cufftComplex));
	return d_outData;
}

void volume_Multi(double *data1, double *data2, int numElements, int xdim, double sampling , int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    int zslice= pad_size*pad_size ;
    volumeMulti<<<blocksPerGrid, threadsPerBlock>>>(data1, data2,numElements, xdim, sampling,padhdim,pad_size,ori_size,padding_factor,normftblob,zslice);
}

void printdatatofile(Complex *data,int N)
{
	FILE *fp= fopen("data1.out","w+");
	for(int i=0;i< 300 ;i++)
	{
		fprintf(fp,"%f %f |",data[i].real,data[i].imag);
		if(i%100==0)
			fprintf(fp,"\n");
	}
	fclose(fp);
}

