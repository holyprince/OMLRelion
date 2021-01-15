#ifndef ACC_BACKPROJECTOR_H_
#define ACC_BACKPROJECTOR_H_

#ifdef CUDA
#  include <cuda_runtime.h>
#endif
#include "../complex.h"
#include "settings.h"
#include "acc_ptr.h"

#ifndef CUDA
#  include <tbb/spin_mutex.h>
#endif

class AccBackprojector
{

public:
	int mdlX, mdlY, mdlZ,
	    mdlInitY, mdlInitZ,
	    padding_factor,pad_size;
	size_t mdlXYZ,sumalldata, maxR, maxR2;

#ifndef CUDA
	tbb::spin_mutex *mutexes;
#endif

	size_t allocaton_size;
	size_t voxelCount;


#ifdef BACKSLICE
	int *d_index;
	XFLOAT *d_real,*d_imag,*d_weight;
	int *d_count;
#ifdef FILTERSLICE

	int fxsize,fysize,fzsize;
	int fstartx,fstarty,fstartz;
	int fendx, fendy,fendz;
	size_t mdlfxyzsize;

	XFLOAT *d_filterreal,*d_filterimag,*d_filterweight;

#endif
#else
	XFLOAT *d_mdlReal, *d_mdlImag, *d_mdlWeight;
	size_t *d_yoffsetdata;
#endif



	cudaStream_t stream;

public:

	AccBackprojector():
				mdlX(0), mdlY(0), mdlZ(0), mdlXYZ(0),
				mdlInitY(0), mdlInitZ(0),
				maxR(0), maxR2(0),
				padding_factor(0),
				allocaton_size(0), voxelCount(0),
#ifdef BACKSLICE
				d_index(NULL), d_real(NULL), d_imag(NULL),d_weight(NULL),
#else
				d_mdlReal(NULL), d_mdlImag(NULL), d_mdlWeight(NULL),
#endif
				stream(0)
#ifndef CUDA
				, mutexes(0)
#endif
	{}

	size_t setMdlDim(
			int xdim, int ydim, int zdim,
			int inity, int initz,
			int max_r, int paddingFactor
#ifdef FILTERSLICE
			,int fxsize,int fysize,int fzsize,int startx,int starty,int startz,int endx,int endy,int endz
#endif
	);
	size_t setcompressMdlDim(
			int xdim, int ydim, int zdim,
			int inity, int initz,
			int max_r, int paddingFactor,int padsize, size_t sumdata,size_t *yoffsetdata);
	void initMdl();
    void initcompressMdl();

	void backproject(
			XFLOAT *d_imgs_nomask_real,
			XFLOAT *d_imgs_nomask_imag,
			XFLOAT *trans_x,
			XFLOAT *trans_y,
			XFLOAT *trans_z,
			XFLOAT* d_weights,
			XFLOAT* d_Minvsigma2s,
			XFLOAT* d_ctfs,
			unsigned long translation_num,
			XFLOAT significant_weight,
			XFLOAT weight_norm,
			XFLOAT *d_eulers,
			int imgX,
			int imgY,
			int imgZ,
			unsigned long imageCount,
			bool data_is_3D,
			cudaStream_t optStream);

	void getMdlData(XFLOAT *real, XFLOAT *imag, XFLOAT * weights);
	void getcompressMdlData(XFLOAT *real, XFLOAT *imag, XFLOAT * weights);
	void getMdlDataPtrs(XFLOAT *& real, XFLOAT *& imag, XFLOAT *& weights);

	void setStream(cudaStream_t s) { stream = s; }
	cudaStream_t getStream() { return stream; }

	void clear();
	void compressclear();
	~AccBackprojector();
};

#endif
