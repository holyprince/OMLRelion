//#include <signal.h>
//#include <cuda_runtime.h>
//#include "src/acc/settings.h"
//#include "src/acc/acc_backprojector.h"
//#include "src/acc/cuda/cuda_kernels/cuda_device_utils.cuh"
//#include "src/acc/acc_projector.h"
#include <stdio.h>
#include <sys/time.h>
size_t AccBackprojector::setMdlDim(
			int xdim, int ydim, int zdim,
			int inity, int initz,
			int max_r, int paddingFactor
#ifdef FILTERSLICE
			,int fxsize,int fysize,int fzsize, int startx,int starty,int startz,int endx,int endy,int endz
#endif
			)
{
	if (xdim != mdlX ||
		ydim != mdlY ||
		zdim != mdlZ ||
		inity != mdlInitY ||
		initz != mdlInitZ ||
		max_r != maxR ||
		paddingFactor != padding_factor)
	{
		clear();

		mdlX = xdim;
		mdlY = ydim;
		mdlZ = zdim;
		if (mdlZ < 1) mdlZ = 1;
		mdlXYZ = (size_t)xdim*(size_t)ydim*(size_t)zdim;
		mdlInitY = inity;
		mdlInitZ = initz;
		maxR = max_r;
		maxR2 = max_r*max_r;
		padding_factor = paddingFactor;


		//Allocate space for model
#ifdef CUDA

#ifdef BACKSLICE
		int imgx = mdlX /2;
		int imgy= (imgx-1)*2;
		mdlXYZ = imgx*imgy*8;

		cudaMalloc((void **)&d_index,sizeof(int)*mdlXYZ);
		cudaMalloc((void **)&d_real,sizeof(XFLOAT)*mdlXYZ);
		cudaMalloc((void **)&d_imag,sizeof(XFLOAT)*mdlXYZ);
		cudaMalloc((void **)&d_weight,sizeof(XFLOAT)*mdlXYZ);
		cudaMalloc((void **)&d_count,sizeof(int)*mdlXYZ);


#else

#ifdef  NOCOMGPUTIME

	    struct timeval tv1,tv2;
	    struct timezone tz;
	    long int time_use;
	    gettimeofday (&tv1, &tz);
#endif
		HANDLE_ERROR(cudaMalloc( (void**) &d_mdlReal,   mdlXYZ * sizeof(XFLOAT)));
		HANDLE_ERROR(cudaMalloc( (void**) &d_mdlImag,   mdlXYZ * sizeof(XFLOAT)));
		HANDLE_ERROR(cudaMalloc( (void**) &d_mdlWeight, mdlXYZ * sizeof(XFLOAT)));
#endif



#ifdef  NOCOMGPUTIME
		gettimeofday (&tv2, &tz);
		time_use=(tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec);
		 printf("1. GPU malloc data time : %d  \n ",time_use);
#endif
#else
		if (posix_memalign((void **)&d_mdlReal,   MEM_ALIGN, mdlXYZ * sizeof(XFLOAT))) CRITICAL(RAMERR);
		if (posix_memalign((void **)&d_mdlImag,   MEM_ALIGN, mdlXYZ * sizeof(XFLOAT))) CRITICAL(RAMERR);
		if (posix_memalign((void **)&d_mdlWeight, MEM_ALIGN, mdlXYZ * sizeof(XFLOAT))) CRITICAL(RAMERR);

		mutexes = new tbb::spin_mutex[mdlZ*mdlY];

#endif

#ifdef FILTERSLICE
		this->fxsize = fxsize;
		this->fysize = fysize;
		this->fzsize = fzsize;
		mdlfxyzsize = fxsize * fysize * fzsize;
		fstartx = startx;
		fstarty = starty;
		fstartz = startz;

		fendx = endx;
		fendy = endy;
		fendz = endz;


		printf("FILTERSLICE: %d %d %d %d\n",mdlfxyzsize,fxsize,fysize,fzsize);
		fflush(stdout);

		cudaMalloc((void **)&d_filterreal,sizeof(XFLOAT)*mdlfxyzsize);
		cudaMalloc((void **)&d_filterimag,sizeof(XFLOAT)*mdlfxyzsize);
		cudaMalloc((void **)&d_filterweight,sizeof(XFLOAT)*mdlfxyzsize);
#endif

#ifdef BACKSLICE
		allocaton_size = mdlXYZ * sizeof(XFLOAT) * 4;
#else
		allocaton_size = mdlXYZ * sizeof(XFLOAT) * 3;
#endif

#ifdef FILTERSLICE
		allocaton_size  = mdlfxyzsize * sizeof(XFLOAT) * 3 +  mdlXYZ * sizeof(XFLOAT) * 5;
		printf("FILTERSLICE: allocaton_size %d\n",allocaton_size);
		fflush(stdout);
#endif
	}

	return allocaton_size;
}
size_t AccBackprojector::setcompressMdlDim(
			int xdim, int ydim, int zdim,
			int inity, int initz,
			int max_r, int paddingFactor,int padsize,size_t sumdata,size_t *yoffsetdata)
{
	if (xdim != mdlX ||
		ydim != mdlY ||
		zdim != mdlZ ||
		inity != mdlInitY ||
		initz != mdlInitZ ||
		max_r != maxR ||
		paddingFactor != padding_factor)
	{
		clear();

		mdlX = xdim;
		mdlY = ydim;
		mdlZ = zdim;
		if (mdlZ < 1) mdlZ = 1;
		mdlXYZ = (size_t)xdim*(size_t)ydim*(size_t)zdim;
		mdlInitY = inity;
		mdlInitZ = initz;
		maxR = max_r;
		maxR2 = max_r*max_r;
		padding_factor = paddingFactor;
		sumalldata = sumdata;
		pad_size = padsize;
		//Allocate space for model

#ifdef FMDEBUG
		printf("%ld %ld %ld %ld\n",sumalldata,xdim,ydim,zdim);
	    size_t avail;
	    size_t total;
	    cudaMemGetInfo( &avail, &total );
	    printf("setcompressMdlDim: %ld %ld \n",avail,total);

#endif

#ifdef CUDA
#ifndef BACKSLICE

#ifdef  COMGPUTIME

	    struct timeval tv1,tv2;
	    struct timezone tz;
	    long int time_use;
	    gettimeofday (&tv1, &tz);
#endif
		HANDLE_ERROR(cudaMalloc( (void**) &d_mdlReal,   sumalldata * sizeof(XFLOAT)));
		HANDLE_ERROR(cudaMalloc( (void**) &d_mdlImag,   sumalldata * sizeof(XFLOAT)));
		HANDLE_ERROR(cudaMalloc( (void**) &d_mdlWeight, sumalldata * sizeof(XFLOAT)));
#ifdef  COMGPUTIME
		gettimeofday (&tv2, &tz);
		time_use=(tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec);
		 printf("1. GPU malloc data time : %d  \n ",time_use);
		 fflush(stdout);
#endif

#ifdef  COMGPUTIME

	    gettimeofday (&tv1, &tz);
#endif

		HANDLE_ERROR(cudaMalloc( (void**) &d_yoffsetdata, pad_size*pad_size * sizeof(size_t)));
		DEBUG_HANDLE_ERROR(cudaMemcpy(d_yoffsetdata, yoffsetdata, pad_size*pad_size * sizeof(size_t),cudaMemcpyHostToDevice));

#ifdef  COMGPUTIME
		gettimeofday (&tv2, &tz);
		time_use=(tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec);
		 printf("2. GPU malloc table and send table time : %d \n",time_use);
		 fflush(stdout);
#endif
#endif
#else
		if (posix_memalign((void **)&d_mdlReal,   MEM_ALIGN, mdlXYZ * sizeof(XFLOAT))) CRITICAL(RAMERR);
		if (posix_memalign((void **)&d_mdlImag,   MEM_ALIGN, mdlXYZ * sizeof(XFLOAT))) CRITICAL(RAMERR);
		if (posix_memalign((void **)&d_mdlWeight, MEM_ALIGN, mdlXYZ * sizeof(XFLOAT))) CRITICAL(RAMERR);

		mutexes = new tbb::spin_mutex[mdlZ*mdlY];

#endif

		allocaton_size = sumalldata * sizeof(XFLOAT) * 3;
	}

	return allocaton_size;
}

void AccBackprojector::initMdl()
{
#ifdef DEBUG_CUDA
	if (mdlXYZ == 0)
	{
        printf("Model dimensions must be set with setMdlDim before call to initMdl.");
        CRITICAL(ERR_MDLDIM);
	}
	if (voxelCount != 0)
	{
        printf("DEBUG_ERROR: Duplicated call to model setup");
        CRITICAL(ERR_MDLSET);
	}
#endif

	//Initiate model with zeros
#ifdef CUDA
#ifdef BACKSLICE
	DEBUG_HANDLE_ERROR(cudaMemset( d_index,   0, mdlXYZ * sizeof(int)));
	DEBUG_HANDLE_ERROR(cudaMemset( d_real,   0, mdlXYZ * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(cudaMemset( d_imag, 0, mdlXYZ * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(cudaMemset( d_weight, 0, mdlXYZ * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(cudaMemset( d_count, 0, mdlXYZ * sizeof(int)));

#else
	DEBUG_HANDLE_ERROR(cudaMemset( d_mdlReal,   0, mdlXYZ * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(cudaMemset( d_mdlImag,   0, mdlXYZ * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(cudaMemset( d_mdlWeight, 0, mdlXYZ * sizeof(XFLOAT)));
#endif
#else
	memset(d_mdlReal,     0, mdlXYZ * sizeof(XFLOAT));
	memset(d_mdlImag,     0, mdlXYZ * sizeof(XFLOAT));
	memset(d_mdlWeight,   0, mdlXYZ * sizeof(XFLOAT));
#endif


	voxelCount =  mdlXYZ ;
#ifdef FILTERSLICE

	DEBUG_HANDLE_ERROR(cudaMemset( d_filterreal,   0, mdlfxyzsize * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(cudaMemset( d_filterimag,   0, mdlfxyzsize * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(cudaMemset( d_filterweight, 0, mdlfxyzsize * sizeof(XFLOAT)));
    voxelCount = mdlXYZ + mdlfxyzsize;
#endif


}
void AccBackprojector::initcompressMdl()
{
#ifdef DEBUG_CUDA
	if (mdlXYZ == 0)
	{
        printf("Model dimensions must be set with setMdlDim before call to initMdl.");
        CRITICAL(ERR_MDLDIM);
	}
	if (voxelCount != 0)
	{
        printf("DEBUG_ERROR: Duplicated call to model setup");
        CRITICAL(ERR_MDLSET);
	}
#endif

	//Initiate model with zeros
#ifdef CUDA
#ifndef BACKSLICE
	DEBUG_HANDLE_ERROR(cudaMemset( d_mdlReal,   0, sumalldata * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(cudaMemset( d_mdlImag,   0, sumalldata * sizeof(XFLOAT)));
	DEBUG_HANDLE_ERROR(cudaMemset( d_mdlWeight, 0, sumalldata * sizeof(XFLOAT)));
#endif
#else
	memset(d_mdlReal,     0, mdlXYZ * sizeof(XFLOAT));
	memset(d_mdlImag,     0, mdlXYZ * sizeof(XFLOAT));
	memset(d_mdlWeight,   0, mdlXYZ * sizeof(XFLOAT));
#endif

    voxelCount = sumalldata;
}

void AccBackprojector::getcompressMdlData(XFLOAT *r, XFLOAT *i, XFLOAT * w)
{
#ifdef CUDA
#ifdef FILTERSLICE
	DEBUG_HANDLE_ERROR(cudaMemcpy( r, d_filterreal,   mdlfxyzsize * sizeof(XFLOAT), cudaMemcpyDeviceToHost));
	DEBUG_HANDLE_ERROR(cudaMemcpy( i, d_filterimag,   mdlfxyzsize * sizeof(XFLOAT), cudaMemcpyDeviceToHost));
	DEBUG_HANDLE_ERROR(cudaMemcpy( w, d_filterweight, mdlfxyzsize * sizeof(XFLOAT), cudaMemcpyDeviceToHost));
#endif

#ifndef BACKSLICE
	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream)); //Make sure to wait for remaining kernel executions

	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( r, d_mdlReal,   sumalldata * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( i, d_mdlImag,   sumalldata * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( w, d_mdlWeight, sumalldata * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));

	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream)); //Wait for copy
#endif

#else
	memcpy(r, d_mdlReal,   mdlXYZ * sizeof(XFLOAT));
	memcpy(i, d_mdlImag,   mdlXYZ * sizeof(XFLOAT));
	memcpy(w, d_mdlWeight, mdlXYZ * sizeof(XFLOAT));
#endif
}


void AccBackprojector::getMdlData(XFLOAT *r, XFLOAT *i, XFLOAT * w)
{
#ifdef CUDA

#ifndef BACKSLICE

	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream)); //Make sure to wait for remaining kernel executions

	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( r, d_mdlReal,   mdlXYZ * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( i, d_mdlImag,   mdlXYZ * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));
	DEBUG_HANDLE_ERROR(cudaMemcpyAsync( w, d_mdlWeight, mdlXYZ * sizeof(XFLOAT), cudaMemcpyDeviceToHost, stream));

	DEBUG_HANDLE_ERROR(cudaStreamSynchronize(stream)); //Wait for copy
#endif
#else
	memcpy(r, d_mdlReal,   mdlXYZ * sizeof(XFLOAT));
	memcpy(i, d_mdlImag,   mdlXYZ * sizeof(XFLOAT));
	memcpy(w, d_mdlWeight, mdlXYZ * sizeof(XFLOAT));
#endif
}

void AccBackprojector::getMdlDataPtrs(XFLOAT *& r, XFLOAT *& i, XFLOAT *& w)
{
#ifndef CUDA
	r = d_mdlReal;
	i = d_mdlImag;
	w = d_mdlWeight;
#endif
}

void AccBackprojector::clear()
{
	mdlX = 0;
	mdlY = 0;
	mdlZ = 0;
	mdlXYZ = 0;
	mdlInitY = 0;
	mdlInitZ = 0;
	maxR = 0;
	maxR2 = 0;
	padding_factor = 0;
	allocaton_size = 0;
#ifdef BACKSLICE
	if (d_index != NULL)
	{
#else
	if (d_mdlReal != NULL)
	{
#endif
#ifdef CUDA
#ifdef BACKSLICE

		DEBUG_HANDLE_ERROR(cudaFree(d_index));
		DEBUG_HANDLE_ERROR(cudaFree(d_real));
		DEBUG_HANDLE_ERROR(cudaFree(d_imag));
		DEBUG_HANDLE_ERROR(cudaFree(d_weight));
		DEBUG_HANDLE_ERROR(cudaFree(d_count));


		d_real = d_imag = d_weight = NULL;
		d_index = NULL;
		d_count = NULL;


#ifdef FILTERSLICE
		DEBUG_HANDLE_ERROR(cudaFree(d_filterreal));
		DEBUG_HANDLE_ERROR(cudaFree(d_filterimag));
		DEBUG_HANDLE_ERROR(cudaFree(d_filterweight));

		d_filterreal= NULL;
		d_filterimag = NULL;
		d_filterweight=NULL;
#endif
#else
		DEBUG_HANDLE_ERROR(cudaFree(d_mdlReal));
		DEBUG_HANDLE_ERROR(cudaFree(d_mdlImag));
		DEBUG_HANDLE_ERROR(cudaFree(d_mdlWeight));
		d_mdlReal = d_mdlImag = d_mdlWeight = NULL;
#endif
#else
		free(d_mdlReal);
		free(d_mdlImag);
		free(d_mdlWeight);
		delete [] mutexes;
#endif 


	}
}
void AccBackprojector::compressclear()
{
	mdlX = 0;
	mdlY = 0;
	mdlZ = 0;
	mdlXYZ = 0;
	mdlInitY = 0;
	mdlInitZ = 0;
	maxR = 0;
	maxR2 = 0;
	padding_factor = 0;
	allocaton_size = 0;
	sumalldata =0;
	pad_size = 0;
#ifndef BACKSLICE
	if (d_mdlReal != NULL)
	{

#endif
#ifdef CUDA
#ifndef BACKSLICE
		DEBUG_HANDLE_ERROR(cudaFree(d_mdlReal));
		DEBUG_HANDLE_ERROR(cudaFree(d_mdlImag));
		DEBUG_HANDLE_ERROR(cudaFree(d_mdlWeight));
		DEBUG_HANDLE_ERROR(cudaFree(d_yoffsetdata));
#endif
#else
		free(d_mdlReal);
		free(d_mdlImag);
		free(d_mdlWeight);
		delete [] mutexes;
#endif
#ifndef BACKSLICE
		d_mdlReal = d_mdlImag = d_mdlWeight = NULL;
		d_yoffsetdata=NULL;

	}
#endif
}

AccBackprojector::~AccBackprojector()
{
	clear();
}
