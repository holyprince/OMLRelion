#include "backproject_impl.h"



__device__ double tab_ftblobgetvalue(double *tabulatedValues, double val,double sampling,int xdim)
{

	int idx = (int)( ABS(val) / sampling);
	if (idx >= xdim)
		return 0.;
	else
		return tabulatedValues[idx];
}
__device__ float tab_ftblobgetvalue(float *tabulatedValues, float val,float sampling,int tabxdim)
{

	int idx = (int)( ABS(val) / sampling);
	if (idx >= tabxdim)
		return 0.;
	else
		return tabulatedValues[idx];
}
__global__ void vectorMulti(double *A, float *B, cufftComplex *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i].x = A[i] * B[i];
        C[i].y = 0;
    }
}

__global__ void vectorMulti_layout(double *A, RFLOAT *B, cufftComplex *C, int numElements,int dimx,int paddim)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
    	int zsclie= paddim*paddim;
    	//coordinate change
		int zindex = i / zsclie ;
		int xyslice = i % (zsclie);
		int yindex = xyslice / paddim;
		int xindex = xyslice % paddim;
		int dimy = paddim;
		int dimz = paddim;

		if(xindex < dimx)
		{
			int inputindex = zindex * dimy * dimx + yindex * dimx + xindex;
			C[i].x = A[inputindex] * B[inputindex];
			C[i].y = 0;
		}
		else{
			int desy, desz, desx;
			desx = paddim - xindex;
			if (yindex == 0)
				desy = 0;
			else
				desy = dimy - yindex;
			if (zindex == 0)
				desz = 0;
			else
				desz = dimz - zindex;
			int inputindex = desz * dimy * dimx + desy * dimx + desx;
			C[i].x = A[inputindex] * B[inputindex];
			C[i].y = 0;
		}

    }
}
//d_blockone,d_Fweight,dataplan[0].d_Data+dataplan[0].selfoffset,dataplan[0].realsize,Fconv.xdim,pad_size
__global__ void vectorMulti_layout_mpi(double *A, float *B, cufftComplex *C, size_t numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
		C[i].x = A[i] * B[i];
		C[i].y = 0;
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
__global__ void volumeMulti_float(cufftComplex *Mconv, RFLOAT *tabdata, int numElements, int xdim, double sampling , int padhdim, int pad_size, int ori_size, double padding_factor, float normftblob, int zslice)
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
/*
    	if (do_mask && rval > 1./(2. * padding_factor))
    		DIRECT_A3D_ELEM(Mconv, k, i, j) = 0.;*/

		Mconv[index].x *= (tab_ftblobgetvalue(tabdata, rval, sampling, xdim) / normftblob);
		Mconv[index].y =0;

	}
}
//zslice changed to ydim* padsize
__global__ void volumeMulti_float_mpi(cufftComplex *Mconv, float *tabdata, int numElements, int tabxdim, double sampling , int padhdim, int pad_size,
		int ori_size, double padding_factor, float normftblob, int zslice, int ydim, int offset)
{


    int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < numElements) {

		int k = index / zslice;
		int xyslice = index % (zslice);
		int i = xyslice / pad_size;
		int j = xyslice % pad_size;

		i=i+offset;
		// real j , i , k ;

		int kp = (k < padhdim) ? k : k - pad_size;
		int ip = (i < padhdim) ? i : i - pad_size;
		int jp = (j < padhdim) ? j : j - pad_size;
		double rval = sqrt((double) (kp * kp + ip * ip + jp * jp)) / (ori_size * padding_factor);
/*
    	if (do_mask && rval > 1./(2. * padding_factor))
    		DIRECT_A3D_ELEM(Mconv, k, i, j) = 0.;*/

		int realindex= k*(pad_size*pad_size)+ i*pad_size + j;
		Mconv[realindex].x *= (tab_ftblobgetvalue(tabdata, rval, sampling, tabxdim) / normftblob);
		Mconv[realindex].y =0;

	}
}

__global__ void volumeMulti_float_transone(cufftComplex *Mconv, RFLOAT *tabdata, int numElements, int tabxdim, double sampling , int padhdim, int pad_size,
		int ori_size, double padding_factor, float normftblob, int zslice, int ydim, int offset)
{


	size_t index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < numElements) {

		int k = index / zslice;
		int xyslice = index % (zslice);
		int i = xyslice / pad_size;
		int j = xyslice % pad_size;

		k=k+offset;
		// real j , i , k ;

		size_t kp = (k < padhdim) ? k : k - pad_size;
		size_t ip = (i < padhdim) ? i : i - pad_size;
		size_t jp = (j < padhdim) ? j : j - pad_size;
		double rval = sqrt((double) (kp * kp + ip * ip + jp * jp)) / (ori_size * padding_factor);
/*
    	if (do_mask && rval > 1./(2. * padding_factor))
    		DIRECT_A3D_ELEM(Mconv, k, i, j) = 0.;*/

		//int realindex= k*(pad_size*pad_size)+ i*pad_size + j;
		Mconv[index].x *= (tab_ftblobgetvalue(tabdata, rval, sampling, tabxdim) / normftblob);
		Mconv[index].y =0;

	}
}
__global__ void volumeMulti_float_transonetest(cufftComplex *Mconv, RFLOAT *tabdata, int numElements, int tabxdim, double sampling , int padhdim, int pad_size,
		int ori_size, double padding_factor, float normftblob, int zslice, int ydim, int offset)
{


    int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < 2) {

		int k = index / zslice;
		int xyslice = index % (zslice);
		int i = xyslice / pad_size;
		int j = xyslice % pad_size;

		k=k+offset;
		// real j , i , k ;

		int kp = (k < padhdim) ? k : k - pad_size;
		int ip = (i < padhdim) ? i : i - pad_size;
		int jp = (j < padhdim) ? j : j - pad_size;
		double rval = sqrt((double) (kp * kp + ip * ip + jp * jp)) / (ori_size * padding_factor);
/*
    	if (do_mask && rval > 1./(2. * padding_factor))
    		DIRECT_A3D_ELEM(Mconv, k, i, j) = 0.;*/

		//int realindex= k*(pad_size*pad_size)+ i*pad_size + j;
		//Mconv[index].x *= (tab_ftblobgetvalue(tabdata, rval, sampling, tabxdim) / normftblob);
		Mconv[index].y =0;
		int res;
		int idx = (int)( ABS(rval) / sampling);
		if (idx >= tabxdim)
			res = 0.;
		else
			res= tabdata[idx];
		printf("\n res in kernel %f \n",res);

		Mconv[index+1].x= rval;
		Mconv[index+2].x= normftblob;
		Mconv[index+3].x= sampling;
		Mconv[index+4].x= tabxdim;
        Mconv[index+5].x= tabdata[0];

	}
}

__global__ void vectorNormlize(cufftComplex *A, long int size , size_t numElements)
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

__global__ void fftDivide(cufftComplex *A, double *Fnewweight, long int numElements,int xysize,int xsize,int ysize,int zsize,int xhalfsize,int max_r2)
{


    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int index2;
    if (index < numElements)
    {

    	double w;
		int k = index / xysize;
		int xyslicenum = index % (xysize);
		int i = xyslicenum / xsize;
		int j = xyslicenum % xsize;

		if (j < xhalfsize) {
			int kp, ip, jp;
			kp = (k < xhalfsize) ? k : k - zsize;
			ip = (i < xhalfsize) ? i : i - ysize;
			jp = (j < xhalfsize) ? j : j - xsize;
			index2 = j + i * xhalfsize + k * xhalfsize * ysize;
			if (kp * kp + ip * ip + jp * jp < max_r2) {

				w = XMIPP_MAX(1e-6, absfftcomplex(A[index]));
				Fnewweight[index2] = Fnewweight[index2] / w;
			}
		}

    }
}
__global__ void fftDivide_mpi(cufftComplex *A, double *Fnewweight, size_t numElements,int xysize,int xsize,int ysize,int zsize,int xhalfsize,int max_r2,int zoffset)
{


	size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < numElements)
    {

    	double w;
		int k = index / xysize;
		int xyslicenum = index % (xysize);
		int i = xyslicenum / xsize;
		int j = xyslicenum % xsize;

		k=k+zoffset;
		if (j < xhalfsize) {
			size_t kp, ip, jp;
			kp = (k < xhalfsize) ? k : k - zsize;
			ip = (i < xhalfsize) ? i : i - ysize;
			jp = (j < xhalfsize) ? j : j - xsize;
//			index2 = j + i * xhalfsize + k * xhalfsize * ysize;
			if (kp * kp + ip * ip + jp * jp < max_r2) {

				w = XMIPP_MAX(1e-6, absfftcomplex(A[index]));
				Fnewweight[index] = Fnewweight[index] / w;
			}
		}
		else //j >= xhalfsize
		{// find raw xyz
			//now x,y, z = j,i,k

			int rawy, rawz, rawx;
			rawx = xsize - j;
			if (i == 0)
				rawy = 0;
			else
				rawy = ysize - i;
			if (k == 0)
				rawz = 0;
			else
				rawz = zsize - k;

			int kp, ip, jp;
			kp = (rawz < xhalfsize) ? rawz : rawz - zsize;
			ip = (rawy < xhalfsize) ? rawy : rawy - ysize;
			jp = (rawx < xhalfsize) ? rawx : rawx - xsize;
			//index2 = j + i * xhalfsize + k * xhalfsize * ysize;
			if (kp * kp + ip * ip + jp * jp < max_r2) {

				w = XMIPP_MAX(1e-6, absfftcomplex(A[index]));
				Fnewweight[index] = Fnewweight[index] / w;
			}

		}

    }
}


__global__ void transpositionyz(cufftComplex *src_data, cufftComplex *des_data, size_t dimy,size_t dimz, size_t offset,size_t tmpoffset,size_t tempydim,size_t pad_size)
{


	size_t xindex = threadIdx.x + blockIdx.x * blockDim.x;
	size_t yindex = threadIdx.y + blockIdx.y * blockDim.y;
	size_t zindex = blockIdx.z * blockDim.z;

    int NX=pad_size;
    int NY=pad_size;
    if(xindex < NX && yindex < dimy && zindex < dimz )
    {
    	des_data[yindex * NX*tempydim + (zindex + tmpoffset) *NX+ xindex ]  = src_data[zindex * NX* NY + (yindex+offset)*NX+ xindex ] ;
    }
}


void initgpu(int GPU_N)
{
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("GPU num for max %d \n",devCount);
	if(devCount < GPU_N)
		printf("Error\n");


	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(i);
		size_t freedata1,total1;
		cudaMemGetInfo( &freedata1, &total1 );
		printf("before alloccation  : %ld   %ld and gpu num %d \n",freedata1,total1,i);
	}
}

void initgpu_mpi(int ranknum)
{
	int devCount;
	cudaGetDeviceCount(&devCount);
	//printf("GPU num for max %d \n",devCount);

	cudaSetDevice(ranknum);  //need change
	size_t freedata1,total1;
	cudaMemGetInfo( &freedata1, &total1 );
	//printf("before alloccation  : %ld   %ld and gpu num is %d \n",freedata1,total1,ranknum);

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

float * gpusetdata_float(float *d_data,int N ,double *c_data)
{
	cudaMalloc((void**) &d_data, N * sizeof(float));
	float *tempdata= (float *)malloc(sizeof(float)*N);
	for(int i=0;i<N;i++)
		tempdata[i]=c_data[i];
	cudaMemcpy(d_data, tempdata, N * sizeof(float),cudaMemcpyHostToDevice);
	return d_data;
}



void vector_Multi(double *data1, float *data2, cufftComplex *res, int numElements)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	vectorMulti<<<blocksPerGrid, threadsPerBlock>>>(data1, data2, res, numElements);
}

void vector_Multi_layout(double *data1, RFLOAT *data2, cufftComplex *res, int numElements,int dimx,int paddim)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorMulti_layout<<<blocksPerGrid, threadsPerBlock>>>(data1, data2, res, numElements,dimx,paddim);
	cudaDeviceSynchronize();
}
void vector_Multi_layout_mpi(double *data1, float *data2, cufftComplex *res, size_t numElements)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorMulti_layout_mpi<<<blocksPerGrid, threadsPerBlock>>>(data1, data2, res, numElements);
	cudaDeviceSynchronize();
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

void volume_Multi_float(cufftComplex *data1, RFLOAT *data2, int numElements, int xdim, double sampling , int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    int zslice= pad_size*pad_size ;
    volumeMulti_float<<<blocksPerGrid, threadsPerBlock>>>(data1, data2,numElements, xdim, sampling,padhdim,pad_size,ori_size,padding_factor,normftblob,zslice);
}

//int offset meaning is distance to (0,0,0) (0,y,0)
void volume_Multi_float_mpi(cufftComplex *data1, float *data2, int numElements, int tabxdim, double sampling ,
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob,int ydim,int offset)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    int zslice= ydim*pad_size ;
    volumeMulti_float_mpi<<<blocksPerGrid, threadsPerBlock>>>(data1, data2,numElements, tabxdim, sampling,padhdim,
    		pad_size,ori_size,padding_factor,normftblob,zslice,ydim,offset);
}

void volume_Multi_float_transone(cufftComplex *data1, RFLOAT *data2, int numElements, int tabxdim, double sampling ,
		int padhdim, int pad_size, int ori_size, float padding_factor, double normftblob,int ydim,int offset)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    int zslice= ydim*pad_size ;
    volumeMulti_float_transone<<<blocksPerGrid, threadsPerBlock>>>(data1, data2,numElements, tabxdim, sampling,padhdim,
    		pad_size,ori_size,padding_factor,normftblob,zslice,ydim,offset);
    //volumeMulti_float_transonetest<<<blocksPerGrid, threadsPerBlock>>>(data1, data2,numElements, tabxdim, sampling,padhdim,
   // 	   	pad_size,ori_size,padding_factor,normftblob,zslice,ydim,offset);
}


void vector_Normlize(cufftComplex *data1, long int normsize, size_t numElements)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorNormlize<<<blocksPerGrid, threadsPerBlock>>>(data1, normsize, numElements);
	cudaDeviceSynchronize();
}

void fft_Divide(cufftComplex *data1, double *Fnewweight, long int numElements,int xysize,int xsize,int ysize,int zsize, int halfxsize,int max_r2)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	fftDivide<<<blocksPerGrid, threadsPerBlock>>>(data1, Fnewweight, numElements, xysize,xsize,ysize,zsize,halfxsize, max_r2);
	cudaDeviceSynchronize();
}
void fft_Divide_mpi(cufftComplex *data1, double *Fnewweight, size_t numElements,int xysize,int xsize,int ysize,int zsize, int halfxsize,int max_r2,int zoffset)
{
    int threadsPerBlock = 512;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
	fftDivide_mpi<<<blocksPerGrid, threadsPerBlock>>>(data1, Fnewweight, numElements, xysize,xsize,ysize,zsize,halfxsize, max_r2,zoffset);
	cudaDeviceSynchronize();
}

void layoutchange(cufftComplex *data,size_t dimx,size_t dimy,size_t dimz, size_t padx, cufftComplex *newdata)
{
	for (int z = 0; z < dimz; z++)
		for (int y = 0; y < dimy; y++) {
			memcpy(newdata + z * dimy * padx + y * padx, data + z * dimy * dimx + y * dimx, dimx * sizeof(cufftComplex));
		}

	for(int z=0;z< dimz;z++)
	for (int y = 0; y < dimy; y++)
		for (int x = dimx; x < padx; x++) {
			int desx,desy,desz;
			if (y == 0)
				desy = 0;
			else
				desy = dimy - y;
			if(z==0)
				desz =0;
			else
				desz = dimz-z;

			desx=padx - x;

			newdata[z*padx*dimy+y * padx + x].x= newdata[desz*padx*dimy+desy * padx + desx].x;
			newdata[z*padx*dimy+y * padx + x].y= - newdata[desz*padx*dimy+desy * padx + desx].y;
		}

}

void layoutchange(double *data,size_t dimx,size_t dimy,size_t dimz, size_t padx, double *newdata)
{
	for (int z = 0; z < dimz; z++)
		for (int y = 0; y < dimy; y++) {
			memcpy(newdata + z * dimy * padx + y * padx, data + z * dimy * dimx + y * dimx, dimx * sizeof(double));
		}

	for(int z=0;z< dimz;z++)
	for (int y = 0; y < dimy; y++)
		for (int x = dimx; x < padx; x++) {
			int desx,desy,desz;
			if (y == 0)
				desy = 0;
			else
				desy = dimy - y;
			if(z==0)
				desz =0;
			else
				desz = dimz-z;

			desx=padx - x;
			newdata[z*padx*dimy+y * padx + x]= newdata[desz*padx*dimy+desy * padx + desx];
		}

}
void layoutchange(float *data,size_t dimx,size_t dimy,size_t dimz, size_t padx, float *newdata)
{
	for (int z = 0; z < dimz; z++)
		for (int y = 0; y < dimy; y++) {
			memcpy(newdata + z * dimy * padx + y * padx, data + z * dimy * dimx + y * dimx, dimx * sizeof(float));
		}

	for(int z=0;z< dimz;z++)
	for (int y = 0; y < dimy; y++)
		for (int x = dimx; x < padx; x++) {
			int desx,desy,desz;
			if (y == 0)
				desy = 0;
			else
				desy = dimy - y;
			if(z==0)
				desz =0;
			else
				desz = dimz-z;

			desx=padx - x;
			newdata[z*padx*dimy+y * padx + x]= newdata[desz*padx*dimy+desy * padx + desx];
		}
}
void layoutchange(double *data,size_t dimx,size_t dimy,size_t dimz, size_t padx, float *newdata)
{
	for (int z = 0; z < dimz; z++)
		for (int y = 0; y < dimy; y++) {
			for(int x=0;x<dimx;x++)
				newdata[z * dimy * padx + y * padx+x] =  data[z * dimy * dimx + y * dimx+x];
			//memcpy(newdata + z * dimy * padx + y * padx, data + z * dimy * dimx + y * dimx, dimx * sizeof(float));
		}

	for(int z=0;z< dimz;z++)
	for (int y = 0; y < dimy; y++)
		for (int x = dimx; x < padx; x++) {
			int desx,desy,desz;
			if (y == 0)
				desy = 0;
			else
				desy = dimy - y;
			if(z==0)
				desz =0;
			else
				desz = dimz-z;

			desx=padx - x;
			newdata[z*padx*dimy+y * padx + x]= newdata[desz*padx*dimy+desy * padx + desx];
		}
}

void validateconj(fftwf_complex *data,int dimx,int dimy,int dimz, int padx)
{

	int count1=0;
	int count2=0;
	int desx,desy,desz;
	int x,y,z;
	for(z=0;z< dimz;z++)
	for (y = 0; y < dimy; y++)
		for (x = dimx; x < padx; x++) {

			if (y == 0)
				desy = 0;
			else
				desy = dimy - y;
			if(z==0)
				desz =0;
			else
				desz = dimz-z;

			desx=padx - x;


			if(data[z*padx*dimy+y * padx + x][0] != data[desz*padx*dimy+desy * padx + desx][0])
			{
				count1++;
				if(count1<10)
				{
					printf("%d %d %d %d %d %d and value %f %f\n",x,y,z,desx,desy,desz,
							data[z*padx*dimy+y * padx + x][0],data[desz*padx*dimy+desy * padx + desx][0]);
				}
			}
			if(data[z*padx*dimy+y * padx + x][1] != -data[desz*padx*dimy+desy * padx + desx][1])
			{
				count2++;
				if(count2<10)
				{
					printf("%d %d %d %d %d %d and value %f %f\n",x,y,z,desx,desy,desz,
							data[z*padx*dimy+y * padx + x][1],-data[desz*padx*dimy+desy * padx + desx][1]);
				}
			}
		}
	printf("Final different x %d\n",count1);
	printf("Final different y %d\n",count2);
	x=362;y=0;z=0;
	desx=361;desy=0;desz=0;
	printf("%f %f \n",data[z*padx*dimy+y * padx + x][0],data[desz*padx*dimy+desy * padx + desx][0]);
}
void validateconj(cufftComplex *data,int dimx,int dimy,int dimz, int padx)
{

	int count1=0;
	int count2=0;
	for(int z=0;z< dimz;z++)
	for (int y = 0; y < dimy; y++)
		for (int x = dimx; x < padx; x++) {
			int desx,desy,desz;
			if (y == 0)
				desy = 0;
			else
				desy = dimy - y;
			if(z==0)
				desz =0;
			else
				desz = dimz-z;

			desx=padx - x;

			if(data[z*padx*dimy+y * padx + x].x != data[desz*padx*dimy+desy * padx + desx].x)
			{
				count1++;
				if(count1<10)
				{
					printf("%d %d %d %d %d %d and value %f %f\n",x,y,z,desx,desy,desz,
							data[z*padx*dimy+y * padx + x].x,data[desz*padx*dimy+desy * padx + desx].x);
				}
			}
			if(data[z*padx*dimy+y * padx + x].y != -data[desz*padx*dimy+desy * padx + desx].y)
			{
				count2++;
				if(count2<10)
				{
					printf("%d %d %d %d %d %d and value %f %f\n",x,y,z,desx,desy,desz,
							data[z*padx*dimy+y * padx + x].y,-data[desz*padx*dimy+desy * padx + desx].y);
				}
			}
		}
	printf("Final different x %d\n",count1);
	printf("Final different y %d\n",count2);
}
void layoutchangeback(double *newdata,size_t dimx,size_t dimy,size_t dimz, size_t padx, double *data)
{
	for (int z = 0; z < dimz; z++)
		for (int y = 0; y < dimy; y++) {
			memcpy(data + z * dimy * dimx + y * dimx, newdata + z * dimy * padx + y * padx, dimx * sizeof(double));
		}
}

void layoutchangecomp(Complex *data,size_t dimx,size_t dimy,size_t dimz, size_t padx, cufftComplex *newdata)
{

	for(int z=0;z< dimz;z++)
		for (int y = 0; y < dimy; y++)
			for (int x = 0; x < dimx; x++) {
				newdata[z*dimy*padx+y*padx+x].x=data[z*dimy*dimx+y*dimx+x].real;
				newdata[z*dimy*padx+y*padx+x].y=data[z*dimy*dimx+y*dimx+x].imag;
			}

	for(int z=0;z< dimz;z++)
	for (int y = 0; y < dimy; y++)
		for (int x = dimx; x < padx; x++) {
			int desx,desy,desz;
			if (y == 0)
				desy = 0;
			else
				desy = dimy - y;
			if(z==0)
				desz =0;
			else
				desz = dimz-z;

			desx=padx - x;

			newdata[z*padx*dimy+y * padx + x].x= newdata[desz*padx*dimy+desy * padx + desx].x;
			newdata[z*padx*dimy+y * padx + x].y= - newdata[desz*padx*dimy+desy * padx + desx].y;
		}

}
void windowFourier(cufftComplex *d_Fconv,cufftComplex *d_Fconv_window,int rawdim, int newdim)
{
	int winkp,winip,winjp;
	int rawkp,rawip,rawjp;
	int newdimx=newdim/2+1;
	int rawdimx=rawdim/2+1;
    for (long int k = 0, kp = 0; k<newdim; k++, kp = (k < newdimx) ? k : k - newdim) \
    	for (long int i = 0, ip = 0 ; i<newdim; i++, ip = (i < newdimx) ? i : i - newdim) \
    		for (long int j = 0, jp = 0; j<newdim; j++, jp = (j < newdimx) ? j : j - newdim)
    		{

    			winkp=(kp < 0) ? (kp + newdim) : (kp);
    			winip=(ip < 0) ? (ip + newdim) : (ip);
    			winjp = (jp < 0) ? (jp + newdim) : (jp);
    			int index1=winkp * newdimx *newdim+ winip *newdimx + winjp ;
    			rawkp=(kp < 0) ? (kp + rawdim) : (kp);
    			rawip=(ip < 0) ? (ip + rawdim) : (ip);
    			rawjp = (jp < 0) ? (jp + rawdim) : (jp);
    			int index2=rawkp * rawdimx *rawdim+ rawip *rawdimx + rawjp ;
    			d_Fconv_window[index1].x=d_Fconv[index2].x;
    			d_Fconv_window[index1].y=d_Fconv[index2].y;
    		}
}

void printdatatofile(Complex *data,int N,int dimx,int rank,int iter,int flag)
{
	FILE *fp1;
	FILE *fp2;
	char filename[100];
	memset(filename,0,100*sizeof(char));
	if(flag==0)
		sprintf(filename,"%d_%d_before_float_real.out",iter,rank);
	else
		sprintf(filename,"%d_%d_after_float_real.out",iter,rank);


	char filename2[100];
	memset(filename2,0,100*sizeof(char));
	if(flag==0)
		sprintf(filename2,"%d_%d_before_float_imag.out",iter,rank);
	else
		sprintf(filename2,"%d_%d_after_float_imag.out",iter,rank);

	fp1= fopen(filename,"w+");
	fp2= fopen(filename2,"w+");

	for(int i=0;i< N ;i++)
	{
		//fprintf(fp,"%f %f ",data[i].real,data[i].imag);
		fprintf(fp1,"%f ",data[i].real);
		if(i%dimx==0 && i!=0)
			fprintf(fp1,"\n");
		fprintf(fp2,"%f ",data[i].imag);
		if(i%dimx==0 && i!=0)
			fprintf(fp2,"\n");
	}
	fclose(fp1);
	fclose(fp2);
}
/*
void printdatatofile(double *data,int N,int dimx,int flag)
{
	FILE *fp;
	if(flag == 0)
	{
		fp= fopen("double_gpu.out","w+");
	}
	else
	{
		fp= fopen("double_cpu.out","w+");
	}
	for(int i=0;i< N ;i++)
	{
		fprintf(fp,"%f ",data[i]);
		if(i%dimx==0 && i!=0)
			fprintf(fp,"\n");
	}
	fclose(fp);
}*/
void printdatatofile(float *data,int N,int dimx,int rank,int iter,int flag)
{
	FILE *fp;

	char filename[100];
	memset(filename,0,100*sizeof(char));


	if(flag==0)
		sprintf(filename,"%d_%d_before_float_weight.out",iter,rank);
	else
		sprintf(filename,"%d_%d_after_float_weight.out",iter,rank);

	fp= fopen(filename,"w+");

	for(int i=0;i< N ;i++)
	{
		fprintf(fp,"%f ",data[i]);
		if(i%dimx==0 && i!=0)
			fprintf(fp,"\n");
	}
	fclose(fp);
}
/*
void printdatatofile(cufftComplex *data,int N,int dimx,int flag)
{
	FILE *fp;
	if(flag == 0)
	{
		fp= fopen("cufftcomplex_gpu.out","w+");
	}
	else
	{
		fp= fopen("cufftcomplex_cpu.out","w+");
	}
	for(int i=0;i< N ;i++)
	{
		//fprintf(fp,"%f %f ",data[i].x,data[i].y);
		fprintf(fp,"%f ",data[i].x);
		if(i%dimx==0 && i!=0)
			fprintf(fp,"\n");
	}
	fclose(fp);
}*/


//MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ,int *numberZ,int *offsettmpZ,int ranknum,int sumrank
void yzlocal_transpose(MultiGPUplan *plan,int GPU_N,int pad_size,int *numberZ,int *offsetZ,int *offsettmpZ)
{
	int K =32;
	int NX=pad_size;
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(plan[i].devicenum);
		for(int desnum=0;desnum<GPU_N;desnum++)
		{
			dim3 threads(K,K);
			dim3 blocks((NX + threads.x - 1)/ threads.x, (numberZ[desnum] + threads.y - 1) / threads.y, numberZ[i]);
			size_t offset = offsetZ[desnum];
			size_t offsettmp = offsettmpZ[desnum];
			transpositionyz<<<blocks, threads>>>(plan[i].d_Data, plan[i].temp_Data,numberZ[desnum],
					numberZ[i],offset,offsettmp,plan[i].tempydim,pad_size);
		}
		//copy data to de

		cudaDeviceSynchronize();
	}
}




void yzlocal_transpose_multicard(MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ,int *numberZ,int *offsettmpZ,int ranknum,int sumrank)
{
	int K =32;
	int NX=pad_size;
	for (int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(plan[i].devicenum);
		int reali= i+ranknum*GPU_N;  //
		for(int desnum=0;desnum<sumrank;desnum++)
		{
			dim3 threads(K,K);
			dim3 blocks((NX + threads.x - 1)/ threads.x, (numberZ[desnum] + threads.y - 1) / threads.y, numberZ[reali]);
			size_t offset = offsetZ[desnum];
			size_t tmpoffset = offsettmpZ[desnum];
			transpositionyz<<<blocks, threads>>>(plan[i].d_Data, plan[i].temp_Data,numberZ[desnum],
					numberZ[reali],offset,tmpoffset,plan[i].tempydim, pad_size);   //int dimy,int dimz
		}
		//copy data to de

		cudaDeviceSynchronize();
	}
}

