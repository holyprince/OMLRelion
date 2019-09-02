

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

__global__ void volumeMulti(float *Mconv, double *tabdata, int numElements, int xdim, double sampling , int padhdim, int pad_size, int ori_size, double padding_factor, float normftblob, int zslice)
{


    int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < numElements) {

		int k = index / zslice;
		int xyslice = index % (zslice);
		int i = xyslice / pad_size;
		int j = xyslice % pad_size;

		int kp = (k < padhdim) ? k : k - pad_size;
		int ip = (i < padhdim) ? i : i - pad_size;
		int jp = (j < padhdim) ? j : j - pad_size;
		double rval = sqrt((double) (kp * kp + ip * ip + jp * jp)) / (ori_size * padding_factor);

		Mconv[index] *= (tab_ftblobgetvalue(tabdata, rval, sampling, xdim) / normftblob);

	}
}
__global__ void vectorNormlize(cufftComplex *A, long int size , long int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        A[i].x = A[i].x / size;
        A[i].y = A[i].y / size;
    }
}
__device__ float absfftcomplex(cufftComplex A)
{
	return sqrt(A.x*A.x+A.y*A.y);
}

__global__ void fftDivide(cufftComplex *A, double *Fnewweight, long int numElements,int xysize,int xsize,int zsize,int ysize,int max_r2)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < numElements)
    {
    	double w;
		int k = index / xysize;
		int xyslicenum = index % (xysize);
		int i = xyslicenum / xsize;
		int j = xyslicenum % xsize;
		int kp,ip,jp;
		kp = (k < xsize) ? k : k - zsize;
		ip = (i < xsize) ? i : i - ysize;
		jp=j;
		if (kp * kp + ip * ip + jp * jp < max_r2)
		{

			w = XMIPP_MAX(1e-6, absfftcomplex(A[index]));
			Fnewweight[index] = Fnewweight[index] / w;
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

void cpusenddata(cufftComplex *d_outData, tComplex<float> *c_outData,int N)
{
	cudaMemcpy(d_outData, c_outData, N * sizeof(cufftComplex),cudaMemcpyHostToDevice);
}
cufftComplex* gpumallocdata(cufftComplex *d_outData,int N)
{
	cudaMalloc((void**) &d_outData,  N * sizeof(cufftComplex));
	return d_outData;
}

void volume_Multi(float *data1, double *data2, int numElements, int xdim, double sampling , int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    int zslice= pad_size*pad_size ;
    volumeMulti<<<blocksPerGrid, threadsPerBlock>>>(data1, data2,numElements, xdim, sampling,padhdim,pad_size,ori_size,padding_factor,normftblob,zslice);
}


void vector_Normlize(cufftComplex *data1, long int normsize, long int numElements)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorNormlize<<<blocksPerGrid, threadsPerBlock>>>(data1, normsize, numElements);
}

void fft_Divide(cufftComplex *data1, double *Fnewweight, long int numElements,int xysize,int xsize,int zsize,int ysize,int max_r2)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	fftDivide<<<blocksPerGrid, threadsPerBlock>>>(data1, Fnewweight, numElements, xysize,xsize,zsize,ysize, max_r2);
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

