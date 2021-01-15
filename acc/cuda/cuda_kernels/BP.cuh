#ifndef CUDA_BP_KERNELS_CUH_
#define CUDA_BP_KERNELS_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "../../acc_projector.h"
#include "../../acc_backprojector.h"
#include "../cuda_settings.h"
#include "../cuda_kernels/cuda_device_utils.cuh"




/*
 *   	BP KERNELS
 */

__global__ void cuda_kernel_backproject2D(
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT* g_weights,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *g_eulers,
		XFLOAT *g_model_real,
		XFLOAT *g_model_imag,
		XFLOAT *g_model_weight,
		int max_r,
		int max_r2,
		XFLOAT padding_factor,
		unsigned img_x,
		unsigned img_y,
		unsigned img_xy,
		unsigned mdl_x,
		int mdl_inity)
{
	unsigned tid = threadIdx.x;
	unsigned img = blockIdx.x;

	__shared__ XFLOAT s_eulers[4];

	XFLOAT minvsigma2, ctf, img_real, img_imag, Fweight, real, imag, weight;

	if (tid == 0)
		s_eulers[0] = g_eulers[img*9+0] * padding_factor;
	else if (tid == 1)
		s_eulers[1] = g_eulers[img*9+1] * padding_factor;
	else if (tid == 2)
		s_eulers[2] = g_eulers[img*9+3] * padding_factor;
	else if (tid == 3)
		s_eulers[3] = g_eulers[img*9+4] * padding_factor;

	__syncthreads();

	int pixel_pass_num(ceilf((float)img_xy/(float)BP_2D_BLOCK_SIZE));

	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
    {
		unsigned pixel = (pass * BP_2D_BLOCK_SIZE) + tid;

		if (pixel >= img_xy)
			continue;

		int x = pixel % img_x;
		int y = (int)floorf( (float)pixel / (float)img_x);

		// Don't search beyond square with side max_r
		if (y > max_r)
		{
			if (y >= img_y - max_r)
				y -= img_y;
			else
				continue;
		}

		if (x * x + y * y > max_r2)
			continue;

		//WAVG
		// __ldg
		minvsigma2 =ldg(&g_Minvsigma2s[pixel]);
		ctf = ldg(&g_ctfs[pixel]);
		img_real = ldg(&g_img_real[pixel]);
		img_imag = ldg(&g_img_imag[pixel]);
		Fweight = (XFLOAT) 0.0;
		real = (XFLOAT) 0.0;
		imag = (XFLOAT) 0.0;

		XFLOAT temp_real, temp_imag;

		for (unsigned long itrans = 0; itrans < translation_num; itrans++)
		{
			weight = g_weights[img * translation_num + itrans];

			if (weight >= significant_weight)
			{
				weight = (weight / weight_norm) * ctf * minvsigma2;
				Fweight += weight * ctf;

				translatePixel(x, y, g_trans_x[itrans], g_trans_y[itrans], img_real, img_imag, temp_real, temp_imag);

				real += temp_real * weight;
				imag += temp_imag * weight;

			}
		}

		if (Fweight > (XFLOAT) 0.0)
		{

			// Get logical coordinates in the 3D map
			XFLOAT xp = (s_eulers[0] * x + s_eulers[1] * y );
			XFLOAT yp = (s_eulers[2] * x + s_eulers[3] * y );

			// Only asymmetric half is stored
			if (xp < 0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				imag = -imag;
			}

			int x0 = floorf(xp);
			XFLOAT fx = xp - x0;
			int x1 = x0 + 1;

			int y0 = floorf(yp);
			XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			int y1 = y0 + 1;

			XFLOAT mfx = (XFLOAT) 1.0 - fx;
			XFLOAT mfy = (XFLOAT) 1.0 - fy;

			XFLOAT dd00 = mfy * mfx;
			XFLOAT dd01 = mfy *  fx;
			XFLOAT dd10 =  fy * mfx;
			XFLOAT dd11 =  fy *  fx;

			cuda_atomic_add(&g_model_real  [y0 * mdl_x + x0], dd00 * real);
			cuda_atomic_add(&g_model_imag  [y0 * mdl_x + x0], dd00 * imag);
			cuda_atomic_add(&g_model_weight[y0 * mdl_x + x0], dd00 * Fweight);

			cuda_atomic_add(&g_model_real  [y0 * mdl_x + x1], dd01 * real);
			cuda_atomic_add(&g_model_imag  [y0 * mdl_x + x1], dd01 * imag);
			cuda_atomic_add(&g_model_weight[y0 * mdl_x + x1], dd01 * Fweight);

			cuda_atomic_add(&g_model_real  [y1 * mdl_x + x0], dd10 * real);
			cuda_atomic_add(&g_model_imag  [y1 * mdl_x + x0], dd10 * imag);
			cuda_atomic_add(&g_model_weight[y1 * mdl_x + x0], dd10 * Fweight);

			cuda_atomic_add(&g_model_real  [y1 * mdl_x + x1], dd11 * real);
			cuda_atomic_add(&g_model_imag  [y1 * mdl_x + x1], dd11 * imag);
			cuda_atomic_add(&g_model_weight[y1 * mdl_x + x1], dd11 * Fweight);
		}
	}
}

template < bool DATA3D >
__global__ void cuda_kernel_backproject3D(
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *g_eulers,
		XFLOAT *g_model_real,
		XFLOAT *g_model_imag,
		XFLOAT *g_model_weight,
		int max_r,
		int max_r2,
		XFLOAT padding_factor,
		unsigned img_x,
		unsigned img_y,
		unsigned img_z,
		unsigned img_xyz,
		unsigned mdl_x,
		unsigned mdl_y,
		int mdl_inity,
		int mdl_initz)
{
	unsigned tid = threadIdx.x;
	unsigned img = blockIdx.x;

	__shared__ XFLOAT s_eulers[9];
	XFLOAT minvsigma2, ctf, img_real, img_imag, Fweight, real, imag, weight;

	if (tid < 9)
		s_eulers[tid] = g_eulers[img*9+tid];

	__syncthreads();

	int pixel_pass_num(0);
	if(DATA3D)
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_DATA3D_BLOCK_SIZE));
	else
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_REF3D_BLOCK_SIZE));

	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
    {
		unsigned pixel(0);
		if(DATA3D)
			pixel = (pass * BP_DATA3D_BLOCK_SIZE) + tid;
		else
			pixel = (pass * BP_REF3D_BLOCK_SIZE) + tid;

		if (pixel >= img_xyz)
			continue;

		int x,y,z,xy;

		if(DATA3D)
		{
			z =  floorfracf(pixel, img_x*img_y);
			xy = pixel % (img_x*img_y);
			x =             xy  % img_x;
			y = floorfracf( xy,   img_x);
			if (z > max_r)
			{
				if (z >= img_z - max_r)
					z = z - img_z;
				else
					continue;

				if(x==0)
					continue;
			}
		}
		else
		{
			x =             pixel % img_x;
			y = floorfracf( pixel , img_x);
		}
		if (y > max_r)
		{
			if (y >= img_y - max_r)
				y = y - img_y;
			else
				continue;
		}

		if(DATA3D)
			if ( ( x * x + y * y  + z * z ) > max_r2)
				continue;
		else
			if ( ( x * x + y * y ) > max_r2)
				continue;

		//WAVG
		minvsigma2 = ldg(&g_Minvsigma2s[pixel]);
		ctf = ldg(&g_ctfs[pixel]);
		img_real = ldg(&g_img_real[pixel]);
		img_imag = ldg(&g_img_imag[pixel]);
		Fweight = (XFLOAT) 0.0;
		real = (XFLOAT) 0.0;
		imag = (XFLOAT) 0.0;

		XFLOAT temp_real, temp_imag;

		for (unsigned long itrans = 0; itrans < translation_num; itrans++)
		{
			weight = g_weights[img * translation_num + itrans];

			if (weight >= significant_weight)
			{
				weight = (weight / weight_norm) * ctf * minvsigma2;
				Fweight += weight * ctf;

				if(DATA3D)
					translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, temp_real, temp_imag);
				else
					translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, temp_real, temp_imag);

				real += temp_real * weight;
				imag += temp_imag * weight;
			}
		}

		//BP
		if (Fweight > (XFLOAT) 0.0)
		{
			// Get logical coordinates in the 3D map

			XFLOAT xp,yp,zp;
			if(DATA3D)
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y + s_eulers[2] * z) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y + s_eulers[5] * z) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y + s_eulers[8] * z) * padding_factor;
			}
			else
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y ) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y ) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y ) * padding_factor;
			}
			// Only asymmetric half is stored
			if (xp < (XFLOAT) 0.0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;
				imag = -imag;
			}

			int x0 = floorf(xp);
			XFLOAT fx = xp - x0;
			int x1 = x0 + 1;

			int y0 = floorf(yp);
			XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			int y1 = y0 + 1;

			int z0 = floorf(zp);
			XFLOAT fz = zp - z0;
			z0 -= mdl_initz;
			int z1 = z0 + 1;

			XFLOAT mfx = (XFLOAT)1.0 - fx;
			XFLOAT mfy = (XFLOAT)1.0 - fy;
			XFLOAT mfz = (XFLOAT)1.0 - fz;

			XFLOAT dd000 = mfz * mfy * mfx;


			cuda_atomic_add(&g_model_real  [z0 * mdl_x * mdl_y + y0 * mdl_x + x0], dd000 * real);
			cuda_atomic_add(&g_model_imag  [z0 * mdl_x * mdl_y + y0 * mdl_x + x0], dd000 * imag);
			cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y0 * mdl_x + x0], dd000 * Fweight);

			XFLOAT dd001 = mfz * mfy *  fx;

			cuda_atomic_add(&g_model_real  [z0 * mdl_x * mdl_y + y0 * mdl_x + x1], dd001 * real);
			cuda_atomic_add(&g_model_imag  [z0 * mdl_x * mdl_y + y0 * mdl_x + x1], dd001 * imag);
			cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y0 * mdl_x + x1], dd001 * Fweight);
			XFLOAT dd010 = mfz *  fy * mfx;

			cuda_atomic_add(&g_model_real  [z0 * mdl_x * mdl_y + y1 * mdl_x + x0], dd010 * real);
			cuda_atomic_add(&g_model_imag  [z0 * mdl_x * mdl_y + y1 * mdl_x + x0], dd010 * imag);
			cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y1 * mdl_x + x0], dd010 * Fweight);
			XFLOAT dd011 = mfz *  fy *  fx;

			cuda_atomic_add(&g_model_real  [z0 * mdl_x * mdl_y + y1 * mdl_x + x1], dd011 * real);
			cuda_atomic_add(&g_model_imag  [z0 * mdl_x * mdl_y + y1 * mdl_x + x1], dd011 * imag);
			cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y1 * mdl_x + x1], dd011 * Fweight);
			XFLOAT dd100 =  fz * mfy * mfx;

			cuda_atomic_add(&g_model_real  [z1 * mdl_x * mdl_y + y0 * mdl_x + x0], dd100 * real);
			cuda_atomic_add(&g_model_imag  [z1 * mdl_x * mdl_y + y0 * mdl_x + x0], dd100 * imag);
			cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y0 * mdl_x + x0], dd100 * Fweight);
			XFLOAT dd101 =  fz * mfy *  fx;

			cuda_atomic_add(&g_model_real  [z1 * mdl_x * mdl_y + y0 * mdl_x + x1], dd101 * real);
			cuda_atomic_add(&g_model_imag  [z1 * mdl_x * mdl_y + y0 * mdl_x + x1], dd101 * imag);
			cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y0 * mdl_x + x1], dd101 * Fweight);

			XFLOAT dd110 =  fz *  fy * mfx;

			cuda_atomic_add(&g_model_real  [z1 * mdl_x * mdl_y + y1 * mdl_x + x0], dd110 * real);
			cuda_atomic_add(&g_model_imag  [z1 * mdl_x * mdl_y + y1 * mdl_x + x0], dd110 * imag);
			cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y1 * mdl_x + x0], dd110 * Fweight);
			XFLOAT dd111 =  fz *  fy *  fx;

			cuda_atomic_add(&g_model_real  [z1 * mdl_x * mdl_y + y1 * mdl_x + x1], dd111 * real);
			cuda_atomic_add(&g_model_imag  [z1 * mdl_x * mdl_y + y1 * mdl_x + x1], dd111 * imag);
			cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y1 * mdl_x + x1], dd111 * Fweight);
		}
	}
}


template < bool DATA3D >
__global__ void cuda_kernel_compressbackproject3D(
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *g_eulers,
		XFLOAT *g_model_real,
		XFLOAT *g_model_imag,
		XFLOAT *g_model_weight,
		size_t max_r,
		size_t max_r2,
		XFLOAT padding_factor,
		int pad_size,
		size_t *g_yoffsetdata,
		unsigned img_x,
		unsigned img_y,
		unsigned img_z,
		unsigned img_xyz,
		unsigned mdl_x,
		unsigned mdl_y,
		int mdl_inity,
		int mdl_initz)
{
	unsigned tid = threadIdx.x;
	unsigned img = blockIdx.x;

	__shared__ XFLOAT s_eulers[9];
	XFLOAT minvsigma2, ctf, img_real, img_imag, Fweight, real, imag, weight;

	if (tid < 9)
		s_eulers[tid] = g_eulers[img*9+tid];

	__syncthreads();

	int pixel_pass_num(0);
	if(DATA3D)
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_DATA3D_BLOCK_SIZE));
	else
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_REF3D_BLOCK_SIZE));

	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
    {
		unsigned pixel(0);
		if(DATA3D)
			pixel = (pass * BP_DATA3D_BLOCK_SIZE) + tid;
		else
			pixel = (pass * BP_REF3D_BLOCK_SIZE) + tid;

		if (pixel >= img_xyz)
			continue;

		int x,y,z,xy;

		if(DATA3D)
		{
			z =  floorfracf(pixel, img_x*img_y);
			xy = pixel % (img_x*img_y);
			x =             xy  % img_x;
			y = floorfracf( xy,   img_x);
			if (z > max_r)
			{
				if (z >= img_z - max_r)
					z = z - img_z;
				else
					continue;

				if(x==0)
					continue;
			}
		}
		else
		{
			x =             pixel % img_x;
			y = floorfracf( pixel , img_x);
		}
		if (y > max_r)
		{
			if (y >= img_y - max_r)
				y = y - img_y;
			else
				continue;
		}

		if(DATA3D)
			if ( ( x * x + y * y  + z * z ) > max_r2)
				continue;
		else
			if ( ( x * x + y * y ) > max_r2)
				continue;

		//WAVG
		minvsigma2 = ldg(&g_Minvsigma2s[pixel]);
		ctf = ldg(&g_ctfs[pixel]);
		img_real = ldg(&g_img_real[pixel]);
		img_imag = ldg(&g_img_imag[pixel]);
		Fweight = (XFLOAT) 0.0;
		real = (XFLOAT) 0.0;
		imag = (XFLOAT) 0.0;

		XFLOAT temp_real, temp_imag;

		for (unsigned long itrans = 0; itrans < translation_num; itrans++)
		{
			weight = g_weights[img * translation_num + itrans];

			if (weight >= significant_weight)
			{
				weight = (weight / weight_norm) * ctf * minvsigma2;
				Fweight += weight * ctf;

				if(DATA3D)
					translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, temp_real, temp_imag);
				else
					translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, temp_real, temp_imag);

				real += temp_real * weight;
				imag += temp_imag * weight;
			}
		}

		//BP
		if (Fweight > (XFLOAT) 0.0)
		{
			// Get logical coordinates in the 3D map

			XFLOAT xp,yp,zp;
			if(DATA3D)
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y + s_eulers[2] * z) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y + s_eulers[5] * z) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y + s_eulers[8] * z) * padding_factor;
			}
			else
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y ) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y ) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y ) * padding_factor;
			}
			// Only asymmetric half is stored
			if (xp < (XFLOAT) 0.0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;
				imag = -imag;
			}

			int x0 = floorf(xp);
			XFLOAT fx = xp - x0;
			int x1 = x0 + 1;

			int y0 = floorf(yp);
			XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			int y1 = y0 + 1;

			int z0 = floorf(zp);
			XFLOAT fz = zp - z0;
			z0 -= mdl_initz;
			int z1 = z0 + 1;

			XFLOAT mfx = (XFLOAT)1.0 - fx;
			XFLOAT mfy = (XFLOAT)1.0 - fy;
			XFLOAT mfz = (XFLOAT)1.0 - fz;

			size_t datarange= (max_r+2)*(max_r+2)*4;
			XFLOAT dd000 = mfz * mfy * mfx;
			size_t datacur;
			if(x0 * x0 + (y0+mdl_inity) * (y0+mdl_inity) + (z0+mdl_initz) * (z0+mdl_initz) < datarange )
			{
				datacur=g_yoffsetdata[(z0)*pad_size+(y0)]+(x0);
				cuda_atomic_add(&g_model_real  [datacur], dd000 * real);
				cuda_atomic_add(&g_model_imag  [datacur], dd000 * imag);
				cuda_atomic_add(&g_model_weight[datacur], dd000 * Fweight);
			}

			XFLOAT dd001 = mfz * mfy *  fx;
			if(x1 * x1 + (y0+mdl_inity) * (y0+mdl_inity) + (z0+mdl_initz) * (z0+mdl_initz) < datarange )
			{
				datacur=g_yoffsetdata[(z0)*pad_size+(y0)]+(x1);
				cuda_atomic_add(&g_model_real  [datacur], dd001 * real);
				cuda_atomic_add(&g_model_imag  [datacur], dd001 * imag);
				cuda_atomic_add(&g_model_weight[datacur], dd001 * Fweight);
			}
			XFLOAT dd010 = mfz *  fy * mfx;
			if(x0 * x0 + (y1+mdl_inity) * (y1+mdl_inity) + (z0+mdl_initz) * (z0+mdl_initz) < datarange )
			{
				datacur=g_yoffsetdata[(z0)*pad_size+(y1)]+(x0);
				cuda_atomic_add(&g_model_real  [datacur], dd010 * real);
				cuda_atomic_add(&g_model_imag  [datacur], dd010 * imag);
				cuda_atomic_add(&g_model_weight[datacur], dd010 * Fweight);
			}

			XFLOAT dd011 = mfz *  fy *  fx;
			if(x1 * x1 + (y1+mdl_inity) * (y1+mdl_inity) + (z0+mdl_initz) * (z0+mdl_initz) < datarange )
			{
				datacur=g_yoffsetdata[(z0)*pad_size+(y1)]+(x1);
			cuda_atomic_add(&g_model_real  [datacur], dd011 * real);
			cuda_atomic_add(&g_model_imag  [datacur], dd011 * imag);
			cuda_atomic_add(&g_model_weight[datacur], dd011 * Fweight);
			}

			XFLOAT dd100 =  fz * mfy * mfx;
			if(x0 * x0 + (y0+mdl_inity) * (y0+mdl_inity) + (z1+mdl_initz) * (z1+mdl_initz) < datarange )
			{
				datacur=g_yoffsetdata[(z1)*pad_size+(y0)]+(x0);
				cuda_atomic_add(&g_model_real  [datacur], dd100 * real);
				cuda_atomic_add(&g_model_imag  [datacur], dd100 * imag);
				cuda_atomic_add(&g_model_weight[datacur], dd100 * Fweight);

			}
			XFLOAT dd101 =  fz * mfy *  fx;
			if(x1 * x1 + (y0+mdl_inity) * (y0+mdl_inity) + (z1+mdl_initz) * (z1+mdl_initz) < datarange )
			{
				datacur=g_yoffsetdata[(z1)*pad_size+(y0)]+(x1);
				cuda_atomic_add(&g_model_real  [datacur], dd101 * real);
				cuda_atomic_add(&g_model_imag  [datacur], dd101 * imag);
				cuda_atomic_add(&g_model_weight[datacur], dd101 * Fweight);
			}


			XFLOAT dd110 =  fz *  fy * mfx;
			if(x0 * x0 + (y1+mdl_inity) * (y1+mdl_inity) + (z1+mdl_initz) * (z1+mdl_initz) < datarange )
			{
				datacur=g_yoffsetdata[(z1)*pad_size+(y1)]+(x0);
				cuda_atomic_add(&g_model_real  [datacur], dd110 * real);
				cuda_atomic_add(&g_model_imag  [datacur], dd110 * imag);
				cuda_atomic_add(&g_model_weight[datacur], dd110 * Fweight);
			}


			XFLOAT dd111 =  fz *  fy *  fx;
			if(x1 * x1 + (y1+mdl_inity) * (y1+mdl_inity) + (z1+mdl_initz) * (z1+mdl_initz) < datarange )
			{
				datacur=g_yoffsetdata[(z1)*pad_size+(y1)]+(x1);
				cuda_atomic_add(&g_model_real  [datacur], dd111 * real);
				cuda_atomic_add(&g_model_imag  [datacur], dd111 * imag);
				cuda_atomic_add(&g_model_weight[datacur], dd111 * Fweight);
			}

		}
	}
}



template < bool DATA3D >
__global__ void cuda_kernel_backproject3D_filterslice(
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *g_eulers,
		int *g_model_index,
		XFLOAT *g_model_real,
		XFLOAT *g_model_imag,
		XFLOAT *g_model_weight,
		int *g_model_count,
		XFLOAT *g_filterreal,
		XFLOAT *g_filterimag,
		XFLOAT *g_filterweight,
		int max_r,
		int max_r2,
		XFLOAT padding_factor,
		unsigned img_x,
		unsigned img_y,
		unsigned img_z,
		unsigned img_xyz,
		unsigned mdl_x,
		unsigned mdl_y,
		int mdl_inity,
		int mdl_initz,
		int imagenum,
		int *flagptr,int mdl_fx,int mdl_fy,int mdl_fz,int fstartx,int fstarty,int fstartz,int fendx,int fendy,int fendz)
{
	unsigned tid = threadIdx.x;
	unsigned img = imagenum;

	__shared__ XFLOAT s_eulers[9];
	XFLOAT minvsigma2, ctf, img_real, img_imag, Fweight, real, imag, weight;

	if (tid < 9)
		s_eulers[tid] = g_eulers[img*9+tid];

	__syncthreads();

	int pixel_pass_num(0);
	if(DATA3D)
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_DATA3D_BLOCK_SIZE));
	else
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_REF3D_BLOCK_SIZE));

	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
    {
		unsigned pixel(0);
		if(DATA3D)
			pixel = (pass * BP_DATA3D_BLOCK_SIZE) + tid;
		else
			pixel = (pass * BP_REF3D_BLOCK_SIZE) + tid;

		if (pixel >= img_xyz)
			continue;

		int x,y,z,xy;

		if(DATA3D)
		{
			z =  floorfracf(pixel, img_x*img_y);
			xy = pixel % (img_x*img_y);
			x =             xy  % img_x;
			y = floorfracf( xy,   img_x);
			if (z > max_r)
			{
				if (z >= img_z - max_r)
					z = z - img_z;
				else
					continue;

				if(x==0)
					continue;
			}
		}
		else
		{
			x =             pixel % img_x;
			y = floorfracf( pixel , img_x);
		}
		if (y > max_r)
		{
			if (y >= img_y - max_r)
				y = y - img_y;
			else
				continue;
		}

		if(DATA3D)
			if ( ( x * x + y * y  + z * z ) > max_r2)
				continue;
		else
			if ( ( x * x + y * y ) > max_r2)
				continue;

		//WAVG
		minvsigma2 = ldg(&g_Minvsigma2s[pixel]);
		ctf = ldg(&g_ctfs[pixel]);
		img_real = ldg(&g_img_real[pixel]);
		img_imag = ldg(&g_img_imag[pixel]);
		Fweight = (XFLOAT) 0.0;
		real = (XFLOAT) 0.0;
		imag = (XFLOAT) 0.0;

		XFLOAT temp_real, temp_imag;

		for (unsigned long itrans = 0; itrans < translation_num; itrans++)
		{
			weight = g_weights[img * translation_num + itrans];

			if (weight >= significant_weight)
			{
				weight = (weight / weight_norm) * ctf * minvsigma2;
				Fweight += weight * ctf;

				if(DATA3D)
					translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, temp_real, temp_imag);
				else
					translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, temp_real, temp_imag);

				real += temp_real * weight;
				imag += temp_imag * weight;
			}
		}

		//BP
		if (Fweight > (XFLOAT) 0.0)
		{
			// Get logical coordinates in the 3D map
			flagptr[0] = 1;
			XFLOAT xp,yp,zp;
			if(DATA3D)
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y + s_eulers[2] * z) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y + s_eulers[5] * z) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y + s_eulers[8] * z) * padding_factor;
			}
			else
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y ) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y ) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y ) * padding_factor;
			}
			// Only asymmetric half is stored
			if (xp < (XFLOAT) 0.0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;
				imag = -imag;
			}

			int x0 = floorf(xp);
			XFLOAT fx = xp - x0;
			int x1 = x0 + 1;

			int y0 = floorf(yp);
			XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			int y1 = y0 + 1;

			int z0 = floorf(zp);
			XFLOAT fz = zp - z0;
			z0 -= mdl_initz;
			int z1 = z0 + 1;

			XFLOAT mfx = (XFLOAT)1.0 - fx;
			XFLOAT mfy = (XFLOAT)1.0 - fy;
			XFLOAT mfz = (XFLOAT)1.0 - fz;


			int fxindex=x0-fstartx;
			int fyindex=y0-fstarty;
			int fzindex=z0-fstartz;

			int startpixel = pixel * 8;
			size_t startindex = z0 * mdl_x * mdl_y + y0 * mdl_x + x0;
			size_t curindex=startindex;
			XFLOAT dd000 = mfz * mfy * mfx;
//z0 * mdl_x * mdl_y + y0 * mdl_x + x0

			if(x0>=fstartx && x0 <=fendx && y0 >=fstarty &&  y0 <= fendy && z0 >=fstartz &&  z0 <= fendz )
			{

				cuda_atomic_add(&g_filterreal  [fzindex * mdl_fx * mdl_fy + fyindex * mdl_fx + fxindex], dd000 * real);
				cuda_atomic_add(&g_filterimag  [fzindex * mdl_fx * mdl_fy + fyindex * mdl_fx + fxindex], dd000 * imag);
				cuda_atomic_add(&g_filterweight[fzindex * mdl_fx * mdl_fy + fyindex * mdl_fx + fxindex], dd000 * Fweight);
			}
			else
			{
				g_model_index[startpixel]=curindex;
				g_model_real[startpixel]=dd000 * real;
				g_model_imag[startpixel]=dd000 * imag;
				g_model_weight[startpixel]=dd000 * Fweight;
		//		g_model_count[startpixel]=1;
				//cuda_atomic_add(&(flagptr[0]),1);
			}

//z0 * mdl_x * mdl_y + y0 * mdl_x + x0+1

			XFLOAT dd001 = mfz * mfy *  fx;

			if(x1>=fstartx && x1 <=fendx && y0 >=fstarty &&  y0 <= fendy && z0 >=fstartz &&  z0 <= fendz )
			{
				cuda_atomic_add(&g_filterreal  [fzindex * mdl_fx * mdl_fy + fyindex * mdl_fx + (fxindex+1)], dd001 * real);
				cuda_atomic_add(&g_filterimag  [fzindex * mdl_fx * mdl_fy + fyindex * mdl_fx + (fxindex+1)], dd001 * imag);
				cuda_atomic_add(&g_filterweight[fzindex * mdl_fx * mdl_fy + fyindex * mdl_fx + (fxindex+1)], dd001 * Fweight);
			}
			else
			{
				g_model_index[startpixel+1]=curindex+1;
				g_model_real[startpixel+1]=dd001 * real;
				g_model_imag[startpixel+1]=dd001 * imag;
				g_model_weight[startpixel+1]=dd001 * Fweight;
		//		g_model_count[startpixel+1]=1;
				//cuda_atomic_add(&(flagptr[0]),1);
			}


//z0 * mdl_x * mdl_y + (y0+1) * mdl_x +x0

			XFLOAT dd010 = mfz *  fy * mfx;
			curindex = curindex+mdl_x ;

			if(x0 >=fstartx && x0 <=fendx && y1 >=fstarty &&  y1 <= fendy && z0 >=fstartz &&  z0 <= fendz )
			{
				cuda_atomic_add(&g_filterreal  [fzindex * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex)], dd010 * real);
				cuda_atomic_add(&g_filterimag  [fzindex * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex)], dd010 * imag);
				cuda_atomic_add(&g_filterweight[fzindex * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex)], dd010 * Fweight);
			}
			else
			{
				g_model_index[startpixel+2]=curindex;
				g_model_real[startpixel+2]=dd010 * real;
				g_model_imag[startpixel+2]=dd010 * imag;
				g_model_weight[startpixel+2]=dd010 * Fweight;
		//		g_model_count[startpixel+2]=1;
				//cuda_atomic_add(&(flagptr[0]),1);
			}

			//z0 * mdl_x * mdl_y + (y0+1) * mdl_x +x0 +1

			XFLOAT dd011 = mfz *  fy *  fx;

			if(x1 >=fstartx && x1 <=fendx && y1 >=fstarty &&  y1 <= fendy && z0 >=fstartz &&  z0 <= fendz )
			{
				cuda_atomic_add(&g_filterreal  [fzindex * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex+1)], dd011 * real);
				cuda_atomic_add(&g_filterimag  [fzindex * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex+1)], dd011 * imag);
				cuda_atomic_add(&g_filterweight[fzindex * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex+1)], dd011 * Fweight);
			}
			else
			{
				g_model_index[startpixel+3]=curindex+1;
				g_model_real[startpixel+3]=dd011 * real;
				g_model_imag[startpixel+3]=dd011 * imag;
				g_model_weight[startpixel+3]=dd011 * Fweight;
		//		g_model_count[startpixel+3]=1;
				//cuda_atomic_add(&(flagptr[0]),1);
			}



			XFLOAT dd100 =  fz * mfy * mfx;
			//(z0+1) * mdl_x * mdl_y + y0 * mdl_x + x0
			curindex = startindex+mdl_x * mdl_y;
			if( x0 >=fstartx && x0 <=fendx && y0 >=fstarty &&  y0 <= fendy && z1 >=fstartz &&  z1 <= fendz )
			{
				cuda_atomic_add(&g_filterreal  [(fzindex+1) * mdl_fx * mdl_fy + (fyindex) * mdl_fx + (fxindex)], dd100 * real);
				cuda_atomic_add(&g_filterimag  [(fzindex+1) * mdl_fx * mdl_fy + (fyindex) * mdl_fx + (fxindex)], dd100 * imag);
				cuda_atomic_add(&g_filterweight[(fzindex+1) * mdl_fx * mdl_fy + (fyindex) * mdl_fx + (fxindex)], dd100 * Fweight);
			}
			else
			{
				g_model_index[startpixel+4]=curindex;
				g_model_real[startpixel+4]=dd100 * real;
				g_model_imag[startpixel+4]=dd100 * imag;
				g_model_weight[startpixel+4]=dd100 * Fweight;
		//		g_model_count[startpixel+4]=1;
				//cuda_atomic_add(&(flagptr[0]),1);
			}



			XFLOAT dd101 =  fz * mfy *  fx;
			//(z0+1) * mdl_x * mdl_y + y0 * mdl_x + x0 +1
			if( x1 >=fstartx && x1 <=fendx && y0 >=fstarty &&  y0 <= fendy && z1 >=fstartz &&  z1 <= fendz )
			{
				cuda_atomic_add(&g_filterreal  [(fzindex+1) * mdl_fx * mdl_fy + (fyindex) * mdl_fx + (fxindex+1)], dd101 * real);
				cuda_atomic_add(&g_filterimag  [(fzindex+1) * mdl_fx * mdl_fy + (fyindex) * mdl_fx + (fxindex+1)], dd101 * imag);
				cuda_atomic_add(&g_filterweight[(fzindex+1) * mdl_fx * mdl_fy + (fyindex) * mdl_fx + (fxindex+1)], dd101 * Fweight);
			}
			else
			{
				g_model_index[startpixel+5]=curindex+1;
				g_model_real[startpixel+5]=dd101 * real;
				g_model_imag[startpixel+5]=dd101 * imag;
				g_model_weight[startpixel+5]=dd101 * Fweight;
			//	g_model_count[startpixel+5]=1;
				//cuda_atomic_add(&(flagptr[0]),1);
			}



			XFLOAT dd110 =  fz *  fy * mfx;
			curindex = curindex+mdl_x;
			//(z0+1) * mdl_x * mdl_y + (y0+1) * mdl_x + x0
			if( x0 >=fstartx && x0 <=fendx && y1 >=fstarty &&  y1 <= fendy && z1 >=fstartz &&  z1 <= fendz )
			{
				cuda_atomic_add(&g_filterreal  [(fzindex+1) * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex)], dd110 * real);
				cuda_atomic_add(&g_filterimag  [(fzindex+1) * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex)], dd110 * imag);
				cuda_atomic_add(&g_filterweight[(fzindex+1) * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex)], dd110 * Fweight);
			}
			else
			{
				g_model_index[startpixel+6]=curindex;
				g_model_real[startpixel+6]=dd110 * real;
				g_model_imag[startpixel+6]=dd110 * imag;
				g_model_weight[startpixel+6]=dd110 * Fweight;
		//		g_model_count[startpixel+6]=1;
				//cuda_atomic_add(&(flagptr[0]),1);
			}



			XFLOAT dd111 =  fz *  fy *  fx;
			if( x1 >=fstartx && x1 <=fendx && y1 >=fstarty &&  y1 <= fendy && z1 >=fstartz &&  z1 <= fendz )
			{
				cuda_atomic_add(&g_filterreal  [(fzindex+1) * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex+1)], dd111 * real);
				cuda_atomic_add(&g_filterimag  [(fzindex+1) * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex+1)], dd111 * imag);
				cuda_atomic_add(&g_filterweight[(fzindex+1) * mdl_fx * mdl_fy + (fyindex+1) * mdl_fx + (fxindex+1)], dd111 * Fweight);
			}
			else
			{
				g_model_index[startpixel+7]=curindex+1;
				g_model_real[startpixel+7]=dd111 * real;
				g_model_imag[startpixel+7]=dd111 * imag;
				g_model_weight[startpixel+7]=dd111 * Fweight;
		//		g_model_count[startpixel+7]=1;
				//cuda_atomic_add(&(flagptr[0]),1);
			}
			//(z0+1) * mdl_x * mdl_y + (y0+1) * mdl_x + x0+1

		}
	}
}

template < bool DATA3D >
__global__ void cuda_kernel_backproject3D_slice(
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *g_eulers,
		int *g_model_index,
		XFLOAT *g_model_real,
		XFLOAT *g_model_imag,
		XFLOAT *g_model_weight,
		int max_r,
		int max_r2,
		XFLOAT padding_factor,
		unsigned img_x,
		unsigned img_y,
		unsigned img_z,
		unsigned img_xyz,
		unsigned mdl_x,
		unsigned mdl_y,
		int mdl_inity,
		int mdl_initz,
		int imagenum,
		int *flagptr)
{
	unsigned tid = threadIdx.x;
	unsigned img = imagenum;

	__shared__ XFLOAT s_eulers[9];
	XFLOAT minvsigma2, ctf, img_real, img_imag, Fweight, real, imag, weight;

	if (tid < 9)
		s_eulers[tid] = g_eulers[img*9+tid];

	__syncthreads();

	int pixel_pass_num(0);
	if(DATA3D)
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_DATA3D_BLOCK_SIZE));
	else
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_REF3D_BLOCK_SIZE));

	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
    {
		unsigned pixel(0);
		if(DATA3D)
			pixel = (pass * BP_DATA3D_BLOCK_SIZE) + tid;
		else
			pixel = (pass * BP_REF3D_BLOCK_SIZE) + tid;

		if (pixel >= img_xyz)
			continue;

		int x,y,z,xy;

		if(DATA3D)
		{
			z =  floorfracf(pixel, img_x*img_y);
			xy = pixel % (img_x*img_y);
			x =             xy  % img_x;
			y = floorfracf( xy,   img_x);
			if (z > max_r)
			{
				if (z >= img_z - max_r)
					z = z - img_z;
				else
					continue;

				if(x==0)
					continue;
			}
		}
		else
		{
			x =             pixel % img_x;
			y = floorfracf( pixel , img_x);
		}
		if (y > max_r)
		{
			if (y >= img_y - max_r)
				y = y - img_y;
			else
				continue;
		}

		if(DATA3D)
			if ( ( x * x + y * y  + z * z ) > max_r2)
				continue;
		else
			if ( ( x * x + y * y ) > max_r2)
				continue;

		//WAVG
		minvsigma2 = ldg(&g_Minvsigma2s[pixel]);
		ctf = ldg(&g_ctfs[pixel]);
		img_real = ldg(&g_img_real[pixel]);
		img_imag = ldg(&g_img_imag[pixel]);
		Fweight = (XFLOAT) 0.0;
		real = (XFLOAT) 0.0;
		imag = (XFLOAT) 0.0;

		XFLOAT temp_real, temp_imag;

		for (unsigned long itrans = 0; itrans < translation_num; itrans++)
		{
			weight = g_weights[img * translation_num + itrans];

			if (weight >= significant_weight)
			{
				weight = (weight / weight_norm) * ctf * minvsigma2;
				Fweight += weight * ctf;

				if(DATA3D)
					translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, temp_real, temp_imag);
				else
					translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, temp_real, temp_imag);

				real += temp_real * weight;
				imag += temp_imag * weight;
			}
		}

		//BP
		if (Fweight > (XFLOAT) 0.0)
		{
			// Get logical coordinates in the 3D map
			flagptr[0]=1;
			XFLOAT xp,yp,zp;
			if(DATA3D)
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y + s_eulers[2] * z) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y + s_eulers[5] * z) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y + s_eulers[8] * z) * padding_factor;
			}
			else
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y ) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y ) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y ) * padding_factor;
			}
			// Only asymmetric half is stored
			if (xp < (XFLOAT) 0.0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;
				imag = -imag;
			}

			int x0 = floorf(xp);
			XFLOAT fx = xp - x0;
			int x1 = x0 + 1;

			int y0 = floorf(yp);
			XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			int y1 = y0 + 1;

			int z0 = floorf(zp);
			XFLOAT fz = zp - z0;
			z0 -= mdl_initz;
			int z1 = z0 + 1;

			XFLOAT mfx = (XFLOAT)1.0 - fx;
			XFLOAT mfy = (XFLOAT)1.0 - fy;
			XFLOAT mfz = (XFLOAT)1.0 - fz;


			int startpixel = pixel * 8;
			size_t startindex = z0 * mdl_x * mdl_y + y0 * mdl_x + x0;
			size_t curindex=startindex;
			XFLOAT dd000 = mfz * mfy * mfx;
//z0 * mdl_x * mdl_y + y0 * mdl_x + x0


			g_model_index[startpixel]=curindex;
			g_model_real[startpixel]=dd000 * real;
			g_model_imag[startpixel]=dd000 * imag;
			g_model_weight[startpixel]=dd000 * Fweight;

//z0 * mdl_x * mdl_y + y0 * mdl_x + x0+1

			XFLOAT dd001 = mfz * mfy *  fx;
			g_model_index[startpixel+1]=curindex+1;
			g_model_real[startpixel+1]=dd001 * real;
			g_model_imag[startpixel+1]=dd001 * imag;
			g_model_weight[startpixel+1]=dd001 * Fweight;


//z0 * mdl_x * mdl_y + (y0+1) * mdl_x +x0

			XFLOAT dd010 = mfz *  fy * mfx;
			curindex = curindex+mdl_x ;
			g_model_index[startpixel+2]=curindex;
			g_model_real[startpixel+2]=dd010 * real;
			g_model_imag[startpixel+2]=dd010 * imag;
			g_model_weight[startpixel+2]=dd010 * Fweight;

			//z0 * mdl_x * mdl_y + (y0+1) * mdl_x +x0 +1

			XFLOAT dd011 = mfz *  fy *  fx;
			g_model_index[startpixel+3]=curindex+1;
			g_model_real[startpixel+3]=dd011 * real;
			g_model_imag[startpixel+3]=dd011 * imag;
			g_model_weight[startpixel+3]=dd011 * Fweight;

			XFLOAT dd100 =  fz * mfy * mfx;
			//(z0+1) * mdl_x * mdl_y + y0 * mdl_x + x0

			curindex = startindex+mdl_x * mdl_y;
			g_model_index[startpixel+4]=curindex;
			g_model_real[startpixel+4]=dd100 * real;
			g_model_imag[startpixel+4]=dd100 * imag;
			g_model_weight[startpixel+4]=dd100 * Fweight;




			XFLOAT dd101 =  fz * mfy *  fx;
			//(z0+1) * mdl_x * mdl_y + y0 * mdl_x + x0 +1


			g_model_index[startpixel+5]=curindex+1;
			g_model_real[startpixel+5]=dd101 * real;
			g_model_imag[startpixel+5]=dd101 * imag;
			g_model_weight[startpixel+5]=dd101 * Fweight;


			XFLOAT dd110 =  fz *  fy * mfx;
			//(z0+1) * mdl_x * mdl_y + (y0+1) * mdl_x + x0

			curindex = curindex+mdl_x;
			g_model_index[startpixel+6]=curindex;
			g_model_real[startpixel+6]=dd110 * real;
			g_model_imag[startpixel+6]=dd110 * imag;
			g_model_weight[startpixel+6]=dd110 * Fweight;



			XFLOAT dd111 =  fz *  fy *  fx;

			//(z0+1) * mdl_x * mdl_y + (y0+1) * mdl_x + x0+1
			g_model_index[startpixel+7]=curindex+1;
			g_model_real[startpixel+7]=dd111 * real;
			g_model_imag[startpixel+7]=dd111 * imag;
			g_model_weight[startpixel+7]=dd111 * Fweight;

		}
	}
}

template < bool DATA3D >
__global__ void cuda_kernel_backprojectSGD(
		AccProjectorKernel projector,
		XFLOAT *g_img_real,
		XFLOAT *g_img_imag,
		XFLOAT *g_trans_x,
		XFLOAT *g_trans_y,
		XFLOAT *g_trans_z,
		XFLOAT* g_weights,
		XFLOAT* g_Minvsigma2s,
		XFLOAT* g_ctfs,
		unsigned long translation_num,
		XFLOAT significant_weight,
		XFLOAT weight_norm,
		XFLOAT *g_eulers,
		XFLOAT *g_model_real,
		XFLOAT *g_model_imag,
		XFLOAT *g_model_weight,
		int max_r,
		int max_r2,
		XFLOAT padding_factor,
		unsigned img_x,
		unsigned img_y,
		unsigned img_z,
		unsigned img_xyz,
		unsigned mdl_x,
		unsigned mdl_y,
		int mdl_inity,
		int mdl_initz)
{
	unsigned tid = threadIdx.x;
	unsigned img = blockIdx.x;

	__shared__ XFLOAT s_eulers[9];
	XFLOAT minvsigma2, ctf, img_real, img_imag, Fweight, real, imag, weight;

	if (tid < 9)
		s_eulers[tid] = g_eulers[img*9+tid];

	__syncthreads();

	int pixel_pass_num(0);
	if(DATA3D)
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_DATA3D_BLOCK_SIZE));
	else
		pixel_pass_num = (ceilf((float)img_xyz/(float)BP_REF3D_BLOCK_SIZE));

	for (unsigned pass = 0; pass < pixel_pass_num; pass++)
    {
		unsigned pixel(0);
		if(DATA3D)
			pixel = (pass * BP_DATA3D_BLOCK_SIZE) + tid;
		else
			pixel = (pass * BP_REF3D_BLOCK_SIZE) + tid;

		if (pixel >= img_xyz)
			continue;

		int x,y,z,xy;

		if(DATA3D)
		{
			z =  floorfracf(pixel, img_x*img_y);
			xy = pixel % (img_x*img_y);
			x =             xy  % img_x;
			y = floorfracf( xy,   img_x);
			if (z > max_r)
			{
				if (z >= img_z - max_r)
					z = z - img_z;
				else
					continue;

				if(x==0)
					continue;
			}
		}
		else
		{
			x =             pixel % img_x;
			y = floorfracf( pixel , img_x);
		}
		if (y > max_r)
		{
			if (y >= img_y - max_r)
				y = y - img_y;
			else
				continue;
		}

		if(DATA3D)
			if ( ( x * x + y * y  + z * z ) > max_r2)
				continue;
		else
			if ( ( x * x + y * y ) > max_r2)
				continue;

		XFLOAT ref_real = (XFLOAT) 0.0;
		XFLOAT ref_imag = (XFLOAT) 0.0;

		if(DATA3D)
			projector.project3Dmodel(
				x,y,z,
				s_eulers[0], s_eulers[1], s_eulers[2],
				s_eulers[3], s_eulers[4], s_eulers[5],
				s_eulers[6], s_eulers[7], s_eulers[8],
				ref_real, ref_imag);
		else
			projector.project3Dmodel(
				x,y,
				s_eulers[0], s_eulers[1],
				s_eulers[3], s_eulers[4],
				s_eulers[6], s_eulers[7],
				ref_real, ref_imag);

		//WAVG
		minvsigma2 = ldg(&g_Minvsigma2s[pixel]);
		ctf = ldg(&g_ctfs[pixel]);
		img_real = ldg(&g_img_real[pixel]);
		img_imag = ldg(&g_img_imag[pixel]);
		Fweight = (XFLOAT) 0.0;
		real = (XFLOAT) 0.0;
		imag = (XFLOAT) 0.0;
		ref_real *= ctf;
		ref_imag *= ctf;

		XFLOAT temp_real, temp_imag;

		for (unsigned long itrans = 0; itrans < translation_num; itrans++)
		{
			weight = g_weights[img * translation_num + itrans];

			if (weight >= significant_weight)
			{
				weight = (weight / weight_norm) * ctf * minvsigma2;
				Fweight += weight * ctf;

				if(DATA3D)
					translatePixel(x, y, z, g_trans_x[itrans], g_trans_y[itrans], g_trans_z[itrans], img_real, img_imag, temp_real, temp_imag);
				else
					translatePixel(x, y,    g_trans_x[itrans], g_trans_y[itrans],                    img_real, img_imag, temp_real, temp_imag);

				real += (temp_real-ref_real) * weight;
				imag += (temp_imag-ref_imag) * weight;
			}
		}

		//BP
		if (Fweight > (XFLOAT) 0.0)
		{
			// Get logical coordinates in the 3D map

			XFLOAT xp,yp,zp;
			if(DATA3D)
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y + s_eulers[2] * z) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y + s_eulers[5] * z) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y + s_eulers[8] * z) * padding_factor;
			}
			else
			{
				xp = (s_eulers[0] * x + s_eulers[1] * y ) * padding_factor;
				yp = (s_eulers[3] * x + s_eulers[4] * y ) * padding_factor;
				zp = (s_eulers[6] * x + s_eulers[7] * y ) * padding_factor;
			}
			// Only asymmetric half is stored
			if (xp < (XFLOAT) 0.0)
			{
				// Get complex conjugated hermitian symmetry pair
				xp = -xp;
				yp = -yp;
				zp = -zp;
				imag = -imag;
			}

			int x0 = floorf(xp);
			XFLOAT fx = xp - x0;
			int x1 = x0 + 1;

			int y0 = floorf(yp);
			XFLOAT fy = yp - y0;
			y0 -= mdl_inity;
			int y1 = y0 + 1;

			int z0 = floorf(zp);
			XFLOAT fz = zp - z0;
			z0 -= mdl_initz;
			int z1 = z0 + 1;

			XFLOAT mfx = (XFLOAT)1.0 - fx;
			XFLOAT mfy = (XFLOAT)1.0 - fy;
			XFLOAT mfz = (XFLOAT)1.0 - fz;

			XFLOAT dd000 = mfz * mfy * mfx;

			cuda_atomic_add(&g_model_real  [z0 * mdl_x * mdl_y + y0 * mdl_x + x0], dd000 * real);
			cuda_atomic_add(&g_model_imag  [z0 * mdl_x * mdl_y + y0 * mdl_x + x0], dd000 * imag);
			cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y0 * mdl_x + x0], dd000 * Fweight);

			XFLOAT dd001 = mfz * mfy *  fx;

			cuda_atomic_add(&g_model_real  [z0 * mdl_x * mdl_y + y0 * mdl_x + x1], dd001 * real);
			cuda_atomic_add(&g_model_imag  [z0 * mdl_x * mdl_y + y0 * mdl_x + x1], dd001 * imag);
			cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y0 * mdl_x + x1], dd001 * Fweight);

			XFLOAT dd010 = mfz *  fy * mfx;

			cuda_atomic_add(&g_model_real  [z0 * mdl_x * mdl_y + y1 * mdl_x + x0], dd010 * real);
			cuda_atomic_add(&g_model_imag  [z0 * mdl_x * mdl_y + y1 * mdl_x + x0], dd010 * imag);
			cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y1 * mdl_x + x0], dd010 * Fweight);

			XFLOAT dd011 = mfz *  fy *  fx;

			cuda_atomic_add(&g_model_real  [z0 * mdl_x * mdl_y + y1 * mdl_x + x1], dd011 * real);
			cuda_atomic_add(&g_model_imag  [z0 * mdl_x * mdl_y + y1 * mdl_x + x1], dd011 * imag);
			cuda_atomic_add(&g_model_weight[z0 * mdl_x * mdl_y + y1 * mdl_x + x1], dd011 * Fweight);

			XFLOAT dd100 =  fz * mfy * mfx;

			cuda_atomic_add(&g_model_real  [z1 * mdl_x * mdl_y + y0 * mdl_x + x0], dd100 * real);
			cuda_atomic_add(&g_model_imag  [z1 * mdl_x * mdl_y + y0 * mdl_x + x0], dd100 * imag);
			cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y0 * mdl_x + x0], dd100 * Fweight);

			XFLOAT dd101 =  fz * mfy *  fx;

			cuda_atomic_add(&g_model_real  [z1 * mdl_x * mdl_y + y0 * mdl_x + x1], dd101 * real);
			cuda_atomic_add(&g_model_imag  [z1 * mdl_x * mdl_y + y0 * mdl_x + x1], dd101 * imag);
			cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y0 * mdl_x + x1], dd101 * Fweight);

			XFLOAT dd110 =  fz *  fy * mfx;

			cuda_atomic_add(&g_model_real  [z1 * mdl_x * mdl_y + y1 * mdl_x + x0], dd110 * real);
			cuda_atomic_add(&g_model_imag  [z1 * mdl_x * mdl_y + y1 * mdl_x + x0], dd110 * imag);
			cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y1 * mdl_x + x0], dd110 * Fweight);

			XFLOAT dd111 =  fz *  fy *  fx;

			cuda_atomic_add(&g_model_real  [z1 * mdl_x * mdl_y + y1 * mdl_x + x1], dd111 * real);
			cuda_atomic_add(&g_model_imag  [z1 * mdl_x * mdl_y + y1 * mdl_x + x1], dd111 * imag);
			cuda_atomic_add(&g_model_weight[z1 * mdl_x * mdl_y + y1 * mdl_x + x1], dd111 * Fweight);

		}
	}
}






#endif /* CUDA_PB_KERNELS_CUH_ */
