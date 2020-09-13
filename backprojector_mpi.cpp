/***************************************************************************
 *
 * Author: "Zihao Wang"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
/*
 * backprojector_mpi.cpp
 *
 *  Created on: 23 July 2020
 *      Author: Zihao Wang
 */

#include "mpi.h"
#include "backprojector.h"
#include "time.h"
#include "backproject_impl.h"
#include "fftw3.h"



#define TIMINGREC
#ifdef TIMINGREC
	#define RCTICREC(timer,label) (timer.tic(label))
    #define RCTOCREC(timer,label) (timer.toc(label))
	#define RCTIC(timer,label)
    #define RCTOC(timer,label)
#else
	#define RCTICREC(timer,label)
	#define RCTOCREC(timer,label)
	#define RCTIC(timer,label)
    #define RCTOC(timer,label)
#endif


void BackProjector::reconstruct_gpustd(MultidimArray<RFLOAT> &vol_out,
                                int max_iter_preweight,
                                bool do_map,
                                RFLOAT tau2_fudge,
                                MultidimArray<RFLOAT> &tau2_io, // can be input/output
                                MultidimArray<RFLOAT> &sigma2_out,
                                MultidimArray<RFLOAT> &data_vs_prior_out,
                                MultidimArray<RFLOAT> &fourier_coverage_out,
                                const MultidimArray<RFLOAT> &fsc, // only input
                                RFLOAT normalise,
                                bool update_tau2_with_fsc,
                                bool is_whole_instead_of_half,
                                int nr_threads,
                                int minres_map,
                                bool printTimes,
								bool do_fsc0999,
								int realranknum,
								int ranksize)
{

#ifdef TIMINGREC
	Timer ReconTimer;
	printTimes=1;
	int ReconS_1 = ReconTimer.setNew(" RcS1_Init ");
	int ReconS_2 = ReconTimer.setNew(" RcS2_Shape&Noise ");
	int ReconS_2_5 = ReconTimer.setNew(" RcS2.5_Regularize ");
	int ReconS_3 = ReconTimer.setNew(" RcS3_skipGridding ");
	int ReconS_4 = ReconTimer.setNew(" RcS4_doGridding_norm ");
	int ReconS_5 = ReconTimer.setNew(" RcS5_doGridding_init ");
	int ReconS_6 = ReconTimer.setNew(" RcS6_doGridding_iter ");
	int ReconS_7 = ReconTimer.setNew(" RcS7_doGridding_apply ");
	int ReconS_8 = ReconTimer.setNew(" RcS8_blobConvolute ");
	int ReconS_9 = ReconTimer.setNew(" RcS9_blobResize ");
	int ReconS_10 = ReconTimer.setNew(" RcS10_blobSetReal ");
	int ReconS_11 = ReconTimer.setNew(" RcS11_blobSetTemp ");
	int ReconS_12 = ReconTimer.setNew(" RcS12_blobTransform ");
	int ReconS_13 = ReconTimer.setNew(" RcS13_blobCenterFFT ");
	int ReconS_14 = ReconTimer.setNew(" RcS14_blobNorm1 ");
	int ReconS_15 = ReconTimer.setNew(" RcS15_blobSoftMask ");
	int ReconS_16 = ReconTimer.setNew(" RcS16_blobNorm2 ");
	int ReconS_17 = ReconTimer.setNew(" RcS17_WindowReal ");
	int ReconS_18 = ReconTimer.setNew(" RcS18_GriddingCorrect ");
	int ReconS_19 = ReconTimer.setNew(" RcS19_tauInit ");
	int ReconS_20 = ReconTimer.setNew(" RcS20_tausetReal ");
	int ReconS_21 = ReconTimer.setNew(" RcS21_tauTransform ");
	int ReconS_22 = ReconTimer.setNew(" RcS22_tautauRest ");
	int ReconS_23 = ReconTimer.setNew(" RcS23_tauShrinkToFit ");
	int ReconS_24 = ReconTimer.setNew(" RcS24_extra ");
#endif

    // never rely on references (handed to you from the outside) for computation:
    // they could be the same (i.e. reconstruct(..., dummy, dummy, dummy, dummy, ...); )

	// process change
	int ranknum;
	int realrankarray[5];
	long int  rawvolx,rawvoly,rawvolz,rawvoln;
    //mpi table
	//for(int i=0;i<ranksize;i++)
	{
		int flag=0;
		if(realranknum == 1)
			ranknum =0;
		if(realranknum == 3)
			ranknum =1;


		if(realranknum ==2 )
		{
			ranknum =0;flag=1;
		}
		if(realranknum ==4 )
		{
			ranknum =1;flag=1;
		}
		if(flag ==0)
		{
			realrankarray[0]=1;
			realrankarray[1]=3;
		}
		if(flag==1)
		{
			realrankarray[0]=2;
			realrankarray[1]=4;
		}
		rawvolx=vol_out.xdim;
		rawvoly=vol_out.ydim;
		rawvolz=vol_out.zdim;
		rawvoln=vol_out.ndim;

	}
	printf("ranknum is %d and ranksize is %d\n",ranknum,ranksize);
	//end rank change
    MultidimArray<RFLOAT> sigma2, data_vs_prior, fourier_coverage;
	MultidimArray<RFLOAT> tau2 = tau2_io;
    FourierTransformer transformer;
	MultidimArray<RFLOAT> Fweight;
	int max_r2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor);
    MultidimArray<Complex>& Fconv = transformer.getFourierReference();
	// Fnewweight can become too large for a float: always keep this one in double-precision
	MultidimArray<double> Fnewweight;


	RCTICREC(ReconTimer,ReconS_1);


//#define DEBUG_RECONSTRUCT
#ifdef DEBUG_RECONSTRUCT
	Image<RFLOAT> ttt;
	FileName fnttt;
	ttt()=weight;
	ttt.write("reconstruct_initial_weight.spi");
	std::cerr << " pad_size= " << pad_size << " padding_factor= " << padding_factor << " max_r2= " << max_r2 << std::endl;
#endif

    // Set Fweight, Fnewweight and Fconv to the right size
    if (ref_dim == 2)
        vol_out.setDimensions(pad_size, pad_size, 1, 1);
    else
        // Too costly to actually allocate the space
        // Trick transformer with the right dimensions
        vol_out.setDimensions(pad_size, pad_size, pad_size, 1);

    transformer.setReal(vol_out); // Fake set real. 1. Allocate space for Fconv 2. calculate plans.
    vol_out.clear(); // Reset dimensions to 0

    RCTOCREC(ReconTimer,ReconS_1);
    RCTICREC(ReconTimer,ReconS_2);

    Fweight.reshape(Fconv);
    if (!skip_gridding)
    	Fnewweight.reshape(Fconv);

	// Go from projector-centered to FFTW-uncentered
	decenter(weight, Fweight, max_r2);

	// Take oversampling into account
	RFLOAT oversampling_correction = (ref_dim == 3) ? (padding_factor * padding_factor * padding_factor) : (padding_factor * padding_factor);
	MultidimArray<RFLOAT> counter;

	// First calculate the radial average of the (inverse of the) power of the noise in the reconstruction
	// This is the left-hand side term in the nominator of the Wiener-filter-like update formula
	// and it is stored inside the weight vector
	// Then, if (do_map) add the inverse of tau2-spectrum values to the weight
	sigma2.initZeros(ori_size/2 + 1);
	counter.initZeros(ori_size/2 + 1);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
	{
		int r2 = kp * kp + ip * ip + jp * jp;
		if (r2 < max_r2)
		{
			int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
			RFLOAT invw = oversampling_correction * DIRECT_A3D_ELEM(Fweight, k, i, j);
			DIRECT_A1D_ELEM(sigma2, ires) += invw;
			DIRECT_A1D_ELEM(counter, ires) += 1.;
		}
    }

	// Average (inverse of) sigma2 in reconstruction
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
	{
        if (DIRECT_A1D_ELEM(sigma2, i) > 1e-10)
            DIRECT_A1D_ELEM(sigma2, i) = DIRECT_A1D_ELEM(counter, i) / DIRECT_A1D_ELEM(sigma2, i);
        else if (DIRECT_A1D_ELEM(sigma2, i) == 0)
            DIRECT_A1D_ELEM(sigma2, i) = 0.;
		else
		{
			std::cerr << " DIRECT_A1D_ELEM(sigma2, i)= " << DIRECT_A1D_ELEM(sigma2, i) << std::endl;
			REPORT_ERROR("BackProjector::reconstruct: ERROR: unexpectedly small, yet non-zero sigma2 value, this should not happen...a");
        }
    }

	if (update_tau2_with_fsc)
    {
        tau2.reshape(ori_size/2 + 1);
        data_vs_prior.initZeros(ori_size/2 + 1);
		// Then calculate new tau2 values, based on the FSC
		if (!fsc.sameShape(sigma2) || !fsc.sameShape(tau2))
		{
			fsc.printShape(std::cerr);
			tau2.printShape(std::cerr);
			sigma2.printShape(std::cerr);
			REPORT_ERROR("ERROR BackProjector::reconstruct: sigma2, tau2 and fsc have different sizes");
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
        {
			// FSC cannot be negative or zero for conversion into tau2
			RFLOAT myfsc = XMIPP_MAX(0.001, DIRECT_A1D_ELEM(fsc, i));
			if (is_whole_instead_of_half)
			{
				// Factor two because of twice as many particles
				// Sqrt-term to get 60-degree phase errors....
				myfsc = sqrt(2. * myfsc / (myfsc + 1.));
			}
			myfsc = XMIPP_MIN(0.999, myfsc);
			RFLOAT myssnr = myfsc / (1. - myfsc);
			// Sjors 29nov2017 try tau2_fudge for pulling harder on Refine3D runs...
            myssnr *= tau2_fudge;
			RFLOAT fsc_based_tau = myssnr * DIRECT_A1D_ELEM(sigma2, i);
			DIRECT_A1D_ELEM(tau2, i) = fsc_based_tau;
			// data_vs_prior is merely for reporting: it is not used for anything in the reconstruction
			DIRECT_A1D_ELEM(data_vs_prior, i) = myssnr;
		}
	}
    RCTOCREC(ReconTimer,ReconS_2);
    RCTICREC(ReconTimer,ReconS_2_5);
	// Apply MAP-additional term to the Fnewweight array
	// This will regularise the actual reconstruction
    if (do_map)
	{

    	// Then, add the inverse of tau2-spectrum values to the weight
		// and also calculate spherical average of data_vs_prior ratios
		if (!update_tau2_with_fsc)
			data_vs_prior.initZeros(ori_size/2 + 1);
		fourier_coverage.initZeros(ori_size/2 + 1);
		counter.initZeros(ori_size/2 + 1);
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
 		{
			int r2 = kp * kp + ip * ip + jp * jp;
			if (r2 < max_r2)
			{
				int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
				RFLOAT invw = DIRECT_A3D_ELEM(Fweight, k, i, j);

				RFLOAT invtau2;
				if (DIRECT_A1D_ELEM(tau2, ires) > 0.)
				{
					// Calculate inverse of tau2
					invtau2 = 1. / (oversampling_correction * tau2_fudge * DIRECT_A1D_ELEM(tau2, ires));
				}
				else if (DIRECT_A1D_ELEM(tau2, ires) == 0.)
				{
					// If tau2 is zero, use small value instead
					invtau2 = 1./ ( 0.001 * invw);
				}
				else
				{
					std::cerr << " sigma2= " << sigma2 << std::endl;
					std::cerr << " fsc= " << fsc << std::endl;
					std::cerr << " tau2= " << tau2 << std::endl;
					REPORT_ERROR("ERROR BackProjector::reconstruct: Negative or zero values encountered for tau2 spectrum!");
				}

				// Keep track of spectral evidence-to-prior ratio and remaining noise in the reconstruction
				if (!update_tau2_with_fsc)
					DIRECT_A1D_ELEM(data_vs_prior, ires) += invw / invtau2;

				// Keep track of the coverage in Fourier space
				if (invw / invtau2 >= 1.)
					DIRECT_A1D_ELEM(fourier_coverage, ires) += 1.;

				DIRECT_A1D_ELEM(counter, ires) += 1.;

				// Only for (ires >= minres_map) add Wiener-filter like term
				if (ires >= minres_map)
				{
					// Now add the inverse-of-tau2_class term
					invw += invtau2;
					// Store the new weight again in Fweight
					DIRECT_A3D_ELEM(Fweight, k, i, j) = invw;
				}
			}
		}

		// Average data_vs_prior
		if (!update_tau2_with_fsc)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(data_vs_prior)
			{
				if (i > r_max)
					DIRECT_A1D_ELEM(data_vs_prior, i) = 0.;
				else if (DIRECT_A1D_ELEM(counter, i) < 0.001)
					DIRECT_A1D_ELEM(data_vs_prior, i) = 999.;
				else
					DIRECT_A1D_ELEM(data_vs_prior, i) /= DIRECT_A1D_ELEM(counter, i);
			}
		}

		// Calculate Fourier coverage in each shell
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fourier_coverage)
		{
			if (DIRECT_A1D_ELEM(counter, i) > 0.)
				DIRECT_A1D_ELEM(fourier_coverage, i) /= DIRECT_A1D_ELEM(counter, i);
		}

	} //end if do_map
    else if (do_fsc0999)
    {

     	// Sjors 9may2018: avoid numerical instabilities with unregularised reconstructions....
        FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
        {
            int r2 = kp * kp + ip * ip + jp * jp;
            if (r2 < max_r2)
            {
                int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
                if (ires >= minres_map)
                {
                    // add 1/1000th of the radially averaged sigma2 to the Fweight, to avoid having zeros there...
                	DIRECT_A3D_ELEM(Fweight, k, i, j) += 1./(999. * DIRECT_A1D_ELEM(sigma2, ires));
                }
            }
        }

    }


	//==============================================================================add multi- GPU version

	int Ndim[3];
	int GPU_N=ranksize;
	Ndim[0]=pad_size;
	Ndim[1]=pad_size;
	Ndim[2]=pad_size;
	size_t fullsize= pad_size*pad_size*pad_size;
	initgpu_mpi(ranknum);

	//divide task
	//process divide data
	int *numberZ;
	int *offsetZ;
	numberZ = (int *)malloc(sizeof(int)*ranksize);
	offsetZ = (int *)malloc(sizeof(int)*ranksize);
	int baseznum= pad_size / ranksize;
	for(int i=0; i<ranksize;i++)
	{
		if(i==0)
		{
			offsetZ[0]=0;
			numberZ[0]=baseznum;
		}
		else
		{
			offsetZ[i] = offsetZ[i-1]+numberZ[i-1];
			numberZ[i]= baseznum;
		}
	}
    numberZ[ranksize-1]= pad_size - baseznum*(ranksize-1);
	if(ranknum==0)
	{
		for(int i=0;i<ranksize;i++)
			printf(" rank :%d : num %d and offset %d \n",i,numberZ[i],offsetZ[i]);
	}
	//int extraz = offsetZ[1] - offsetZ[0];
	//int extraz = numberZ[1] - numberZ[0];
	//numberZ

	RFLOAT *d_Fweight;
	double *d_Fnewweight;
	int offset;

	MultiGPUplan *dataplan;
	dataplan = (MultiGPUplan *)malloc(sizeof(MultiGPUplan)*1);
	multi_plan_init_mpi(dataplan,fullsize, numberZ[ranknum], offsetZ[ranknum],ranknum,pad_size,pad_size);
	cufftComplex *cpu_data,*c_Fconv2;

	cudaMalloc((void**) &(dataplan[0].d_Data),sizeof(cufftComplex) * dataplan[0].datasize);

	//=================================
	int xyN[2];
	xyN[0] = Ndim[0];
	xyN[1] = Ndim[1];
	cufftHandle xyplan;
	cufftHandle zplan;


    RCTOCREC(ReconTimer,ReconS_2_5);
	if (skip_gridding)
	{


	    RCTICREC(ReconTimer,ReconS_3);
		std::cerr << "Skipping gridding!" << std::endl;
		Fconv.initZeros(); // to remove any stuff from the input volume
		decenter(data, Fconv, max_r2);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			if (DIRECT_MULTIDIM_ELEM(Fweight, n) > 0.)
				DIRECT_MULTIDIM_ELEM(Fconv, n) /= DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		RCTOCREC(ReconTimer,ReconS_3);
#ifdef DEBUG_RECONSTRUCT
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			DIRECT_MULTIDIM_ELEM(ttt(), n) = DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		ttt.write("reconstruct_skipgridding_correction_term.spi");
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			if (DIRECT_MULTIDIM_ELEM(Fweight, n) > 0.)
				DIRECT_MULTIDIM_ELEM(ttt(), n) = 1./DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		ttt.write("reconstruct_skipgridding_correction_term_inverse.spi");
#endif

	}
	else
	{

		RCTICREC(ReconTimer,ReconS_4);
		// Divide both data and Fweight by normalisation factor to prevent FFT's with very large values....
	#ifdef DEBUG_RECONSTRUCT
		std::cerr << " normalise= " << normalise << std::endl;
	#endif
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fweight)
		{
			DIRECT_MULTIDIM_ELEM(Fweight, n) /= normalise;
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data)
		{
			DIRECT_MULTIDIM_ELEM(data, n) /= normalise;
		}
		RCTOCREC(ReconTimer,ReconS_4);
		RCTICREC(ReconTimer,ReconS_5);
        // Initialise Fnewweight with 1's and 0's. (also see comments below)
		FOR_ALL_ELEMENTS_IN_ARRAY3D(weight)
		{
			if (k * k + i * i + j * j < max_r2)
				A3D_ELEM(weight, k, i, j) = 1.;
			else
				A3D_ELEM(weight, k, i, j) = 0.;
		}



		decenter(weight, Fnewweight, max_r2);
		RCTOCREC(ReconTimer,ReconS_5);
		// Iterative algorithm as in  Eq. [14] in Pipe & Menon (1999)
		// or Eq. (4) in Matej (2001)



		long int Fconvnum=Fconv.nzyxdim;


		long int normsize= Ndim[0]*Ndim[1]*Ndim[2];
		int halfxdim=Fconv.xdim;

		printf("%ld %ld %ld \n",Fconv.xdim,Fconv.ydim,Fconv.zdim);
		printf("Fnewweight: %ld %ld %ld \n",Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim);




		cpu_data= (cufftComplex *)malloc(fullsize * sizeof(cufftComplex));
		//c_Fconv2 = (cufftComplex *)malloc(fullsize * sizeof(cufftComplex));
		cudaMallocHost((void **) &c_Fconv2, sizeof(cufftComplex) * fullsize);
#ifdef RELION_SINGLE_PRECISION
		d_Fweight = gpusetdata_float(d_Fweight,Fweight.nzyxdim,Fweight.data);
#else
		d_Fweight = gpusetdata_double(d_Fweight,Fweight.nzyxdim,Fweight.data);
#endif
		d_Fnewweight = gpusetdata_double(d_Fnewweight,Fnewweight.nzyxdim,Fnewweight.data);


		printf("d_Fweight num : %d ,d_Fnewweight num :%d \n",Fweight.nzyxdim,Fnewweight.nzyxdim);


         //==================2d fft plan


		cufftPlanMany(&xyplan, 2, xyN, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, numberZ[ranknum]);


	    //==================1d fft plan
		int Ncol[1];
		Ncol[0] = pad_size;
		int inembed[3], extraembed[3];
		inembed[0] = Ndim[0];inembed[1] = numberZ[ranknum];inembed[2] = Ndim[2];
		//extraembed[0] = Ndim[0];extraembed[1] = numberZ[1];extraembed[2] = Ndim[2];
		cufftPlanMany(&zplan, 1, Ncol, inembed, pad_size * pad_size, 1, inembed, pad_size * pad_size, 1, CUFFT_C2C, Ndim[0] * (numberZ[ranknum]));


		double *tempdata= (double *)malloc(sizeof(double)*pad_size*pad_size*pad_size);
		float *fweightdata= (float *)malloc(sizeof(float)*pad_size*pad_size*pad_size);
		// every thread map two block to used
		layoutchange(Fnewweight.data,Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim,pad_size,tempdata);
		layoutchange(Fweight.data,Fweight.xdim,Fweight.ydim,Fweight.zdim,pad_size,fweightdata);
		double *d_blockone;
		float *d_blocktwo;
		cudaMalloc((void**) &(d_blockone),sizeof(double) * dataplan[0].realsize);
		cudaMalloc((void**) &(d_blocktwo),sizeof(float) * dataplan[0].realsize);
		//cudaMalloc((void**) &(d_blocktwo),sizeof(double) * dataplan[0].realsize);

		cudaMemcpy(d_blockone,tempdata+dataplan[0].selfoffset,dataplan[0].realsize*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_blocktwo,fweightdata+dataplan[0].selfoffset,dataplan[0].realsize*sizeof(float),cudaMemcpyHostToDevice);



		for (int iter = 0; iter < max_iter_preweight; iter++)
		{

            //std::cout << "    iteration " << (iter+1) << "/" << max_iter_preweight << "\n";
			RCTICREC(ReconTimer,ReconS_6);

			cudaMemcpy(tempdata+dataplan[0].selfoffset,d_blockone,dataplan[0].realsize*sizeof(double),cudaMemcpyDeviceToHost);
			printf("Iteration %d and Step0.0: \n",iter);
			printgpures(tempdata + dataplan[0].selfoffset,dataplan[0].realsize,ranknum);
			cudaMemcpy(fweightdata+dataplan[0].selfoffset,d_blocktwo,dataplan[0].realsize*sizeof(float),cudaMemcpyDeviceToHost);
			printf("Iteration %d and Step0.1: \n",iter);
			printgpures(fweightdata + dataplan[0].selfoffset,dataplan[0].realsize,ranknum);

			vector_Multi_layout(d_Fnewweight,d_Fweight,dataplan[0].d_Data,fullsize,Fconv.xdim,pad_size);

			printf("Iteration %d and Step1: \n",iter);
			gpu_to_cpu(dataplan,cpu_data);

			cufftExecC2C(xyplan, dataplan[0].d_Data + dataplan[0].selfoffset,dataplan[0].d_Data + dataplan[0].selfoffset, CUFFT_INVERSE);
			cudaDeviceSynchronize();
			printf("Iteration %d and Step2: \n",iter);
			gpu_to_cpu(dataplan,cpu_data);

			cpu_alltoall_multinode(dataplan,cpu_data,numberZ,offsetZ,ranknum,pad_size,ranksize,realrankarray);

			cufftExecC2C(zplan, dataplan[0].d_Data + (offsetZ[ranknum]*pad_size),dataplan[0].d_Data + (offsetZ[ranknum]*pad_size), CUFFT_INVERSE);
			cudaDeviceSynchronize();

//          //send to 0 process
			//if(ranknum ==1) //all copy
			{
				int sliceoffset=pad_size*offsetZ[ranknum];
				int slicesize=pad_size*numberZ[ranknum];
				for(int i=0;i<pad_size;i++)
				{
					cudaMemcpy(cpu_data+sliceoffset,dataplan[0].d_Data + sliceoffset,slicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
					sliceoffset+=pad_size*pad_size;
				}
			}

			cpu_alltoalltozero_multi(cpu_data,numberZ,offsetZ,ranknum,pad_size,ranksize,realrankarray);
			RFLOAT normftblob = tab_ftblob(0.);
			float *d_tab_ftblob;
			int tabxdim=tab_ftblob.tabulatedValues.xdim;
			d_tab_ftblob=gpusetdata_float(d_tab_ftblob,tab_ftblob.tabulatedValues.xdim,tab_ftblob.tabulatedValues.data);
			//gpu_kernel2
			volume_Multi_float_mpi(dataplan[0].d_Data,d_tab_ftblob, Ndim[0]*numberZ[ranknum]*Ndim[2],
					tabxdim, tab_ftblob.sampling , pad_size/2, pad_size, ori_size, padding_factor, normftblob,
					numberZ[ranknum],offsetZ[ranknum]);
			cudaDeviceSynchronize();

//=========================EXE CUFFT_FORWARD

			// First do z fft forward
			cufftExecC2C(zplan, dataplan[0].d_Data + (offsetZ[ranknum]*pad_size),dataplan[0].d_Data + (offsetZ[ranknum]*pad_size), CUFFT_FORWARD);
			cudaDeviceSynchronize();

			cpu_alltoall_inverse_multinode(dataplan,cpu_data,numberZ,offsetZ,ranknum,pad_size,ranksize,realrankarray);

			cufftExecC2C(xyplan, dataplan[0].d_Data + dataplan[0].selfoffset,dataplan[0].d_Data + dataplan[0].selfoffset, CUFFT_FORWARD);
			cudaDeviceSynchronize();

			vector_Normlize(dataplan[0].d_Data, normsize ,Ndim[0]*Ndim[1]*Ndim[2]);

			//gpudata->cpu
			gpu_to_cpu(dataplan,cpu_data);

			cpu_allcombine_multi(cpu_data,ranknum,numberZ,offsetZ,pad_size,ranksize,realrankarray);

			if(ranknum==0)
			{
				//validate the complex conj
				validateconj(cpu_data,halfxdim,pad_size,pad_size,pad_size);

			}
			//cpu to gpu

			//fft_Divide_mpi(dataplan[0].d_Data+dataplan[0].selfoffset,d_blockone,dataplan[0].realsize,
			//		pad_size*pad_size,pad_size,pad_size,pad_size,Fconv.xdim,max_r2,offsetZ[ranknum]);

			if(ranknum == 0)
			{
				cudaMemcpy(dataplan[0].d_Data,cpu_data,fullsize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				fft_Divide(dataplan[0].d_Data,d_Fnewweight,fullsize,pad_size*pad_size,pad_size,pad_size,pad_size,halfxdim,max_r2);
				cudaMemcpy(Fnewweight.data,d_Fnewweight,Fnewweight.nzyxdim*sizeof(double),cudaMemcpyDeviceToHost);
				for(int i=1;i<ranksize;i++)
					MPI_Send(Fnewweight.data,Fnewweight.nzyxdim,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
				layoutchange(Fnewweight.data,Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim,pad_size,tempdata);
				printf("Focus:\n");
				printwhole(tempdata, fullsize,ranknum);
			}
			else
			{
				MPI_Recv(Fnewweight.data,Fnewweight.nzyxdim,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				cudaMemcpy(d_Fnewweight,Fnewweight.data,Fnewweight.nzyxdim*sizeof(double),cudaMemcpyHostToDevice);
			}

			//gpu_kernel3

			//gpudata->cpu             //gpudata->cpu          //redesign d_Fnewweight
			//cpuallcombine 		   //cpuallcombine
			//cpu to gpu               // cpu calc
			//gpu kernel               // cpu fenfa
			//gpu to cpu               // cpu copy
			//cpu to gpu [1-N]


			RCTOCREC(ReconTimer,ReconS_6);

	#ifdef DEBUG_RECONSTRUCT
			std::cerr << " PREWEIGHTING ITERATION: "<< iter + 1 << " OF " << max_iter_preweight << std::endl;
			// report of maximum and minimum values of current conv_weight
			std::cerr << " corr_avg= " << corr_avg / corr_nn << std::endl;
			std::cerr << " corr_min= " << corr_min << std::endl;
			std::cerr << " corr_max= " << corr_max << std::endl;
	#endif

		}
/*
		cudaMemcpy(tempdata+dataplan[0].selfoffset,d_blockone,dataplan[0].realsize*sizeof(double),cudaMemcpyDeviceToHost);

		if(ranknum!=0)
		{
			MPI_Send(tempdata+dataplan[0].selfoffset,dataplan[0].realsize,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
		}
		else
		{
			for(int i=1;i<ranksize;i++)
				MPI_Recv(tempdata+(offsetZ[i]*pad_size*pad_size),pad_size*pad_size*numberZ[i],MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			layoutchangeback(tempdata,Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim,pad_size,Fnewweight.data);

		}

		//printf("Focus : \n");
		if(ranknum==0)
		{
			for(int i=1;i<ranksize;i++)
				MPI_Send(Fnewweight.data,Fnewweight.nzyxdim,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
		}
		else
		{
			MPI_Recv(Fnewweight.data,Fnewweight.nzyxdim,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}*/
	//	if(ranknum==0)
	//	printwhole(Fnewweight.data, Fnewweight.nzyxdim,ranknum);

		cufftDestroy(xyplan);
		cufftDestroy(zplan);

		cudaFree(d_Fnewweight);
		cudaFree(d_Fweight);
		cudaFree(d_blockone);
		RCTICREC(ReconTimer,ReconS_7);


	#ifdef DEBUG_RECONSTRUCT
		Image<double> tttt;
		tttt()=Fnewweight;
		tttt.write("reconstruct_gridding_weight.spi");
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			DIRECT_MULTIDIM_ELEM(ttt(), n) = abs(DIRECT_MULTIDIM_ELEM(Fconv, n));
		}
		ttt.write("reconstruct_gridding_correction_term.spi");
	#endif


		// Clear memory
		Fweight.clear();

		// Note that Fnewweight now holds the approximation of the inverse of the weights on a regular grid

		// Now do the actual reconstruction with the data array
		// Apply the iteratively determined weight
		Fconv.initZeros(); // to remove any stuff from the input volume
		decenter(data, Fconv, max_r2);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
#ifdef  RELION_SINGLE_PRECISION
			// Prevent numerical instabilities in single-precision reconstruction with very unevenly sampled orientations
			if (DIRECT_MULTIDIM_ELEM(Fnewweight, n) > 1e20)
				DIRECT_MULTIDIM_ELEM(Fnewweight, n) = 1e20;
#endif
			DIRECT_MULTIDIM_ELEM(Fconv, n) *= DIRECT_MULTIDIM_ELEM(Fnewweight, n);
		}

		// Clear memory
		Fnewweight.clear();
		RCTOCREC(ReconTimer,ReconS_7);
	} // end if skip_gridding


// Gridding theory says one now has to interpolate the fine grid onto the coarse one using a blob kernel
// and then do the inverse transform and divide by the FT of the blob (i.e. do the gridding correction)
// In practice, this gives all types of artefacts (perhaps I never found the right implementation?!)
// Therefore, window the Fourier transform and then do the inverse transform
//#define RECONSTRUCT_CONVOLUTE_BLOB
#ifdef RECONSTRUCT_CONVOLUTE_BLOB

	// Apply the same blob-convolution as above to the data array
	// Mask real-space map beyond its original size to prevent aliasing in the downsampling step below
	RCTICREC(ReconTimer,ReconS_8);
	convoluteBlobRealSpace(transformer, true);
	RCTOCREC(ReconTimer,ReconS_8);
	RCTICREC(ReconTimer,ReconS_9);
	// Now just pick every 3rd pixel in Fourier-space (i.e. down-sample)
	// and do a final inverse FT
	if (ref_dim == 2)
		vol_out.resize(ori_size, ori_size);
	else
		vol_out.resize(ori_size, ori_size, ori_size);
	RCTOCREC(ReconTimer,ReconS_9);
	RCTICREC(ReconTimer,ReconS_10);
	FourierTransformer transformer2;
	MultidimArray<Complex > Ftmp;
	transformer2.setReal(vol_out); // cannot use the first transformer because Fconv is inside there!!
	transformer2.getFourierAlias(Ftmp);
	RCTOCREC(ReconTimer,ReconS_10);
	RCTICREC(ReconTimer,ReconS_11);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Ftmp)
	{
		if (kp * kp + ip * ip + jp * jp < r_max * r_max)
		{
			DIRECT_A3D_ELEM(Ftmp, k, i, j) = FFTW_ELEM(Fconv, kp * padding_factor, ip * padding_factor, jp * padding_factor);
		}
		else
		{
			DIRECT_A3D_ELEM(Ftmp, k, i, j) = 0.;
		}
	}
	RCTOCREC(ReconTimer,ReconS_11);
	RCTICREC(ReconTimer,ReconS_12);
	// inverse FFT leaves result in vol_out
	transformer2.inverseFourierTransform();
	RCTOCREC(ReconTimer,ReconS_12);
	RCTICREC(ReconTimer,ReconS_13);
	// Shift the map back to its origin
	CenterFFT(vol_out, false);
	RCTOCREC(ReconTimer,ReconS_13);
	RCTICREC(ReconTimer,ReconS_14);
	// Un-normalize FFTW (because original FFTs were done with the size of 2D FFTs)
	if (ref_dim==3)
		vol_out /= ori_size;
	RCTOCREC(ReconTimer,ReconS_14);
	RCTICREC(ReconTimer,ReconS_15);
	// Mask out corners to prevent aliasing artefacts
	softMaskOutsideMap(vol_out);
	RCTOCREC(ReconTimer,ReconS_15);
	RCTICREC(ReconTimer,ReconS_16);
	// Gridding correction for the blob
	RFLOAT normftblob = tab_ftblob(0.);
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol_out)
	{

		RFLOAT r = sqrt((RFLOAT)(k*k+i*i+j*j));
		RFLOAT rval = r / (ori_size * padding_factor);
		A3D_ELEM(vol_out, k, i, j) /= tab_ftblob(rval) / normftblob;
		//if (k==0 && i==0)
		//	std::cerr << " j= " << j << " rval= " << rval << " tab_ftblob(rval) / normftblob= " << tab_ftblob(rval) / normftblob << std::endl;
	}
	RCTOCREC(ReconTimer,ReconS_16);

#else

	// rather than doing the blob-convolution to downsample the data array, do a windowing operation:
	// This is the same as convolution with a SINC. It seems to give better maps.
	// Then just make the blob look as much as a SINC as possible....
	// The "standard" r1.9, m2 and a15 blob looks quite like a sinc until the first zero (perhaps that's why it is standard?)
	//for (RFLOAT r = 0.1; r < 10.; r+=0.01)
	//{
	//	RFLOAT sinc = sin(PI * r / padding_factor ) / ( PI * r / padding_factor);
	//	std::cout << " r= " << r << " sinc= " << sinc << " blob= " << blob_val(r, blob) << std::endl;
	//}

	// Now do inverse FFT and window to original size in real-space
	// Pass the transformer to prevent making and clearing a new one before clearing the one declared above....
	// The latter may give memory problems as detected by electric fence....
	RCTICREC(ReconTimer,ReconS_17);

	// Size of padded real-space volume
	int padoridim = ROUND(padding_factor * ori_size);
	// make sure padoridim is even
	padoridim += padoridim%2;
    vol_out.reshape(padoridim, padoridim, padoridim);
    vol_out.setXmippOrigin();
	fullsize = padoridim *padoridim*padoridim;

	baseznum= padoridim / ranksize;
	for(int i=0; i<ranksize;i++)
	{
		if(i==0)
		{
			offsetZ[0]=0;
			numberZ[0]=baseznum;
		}
		else
		{
			offsetZ[i] = offsetZ[i-1]+numberZ[i-1];
			numberZ[i]= baseznum;
		}
	}
    numberZ[ranksize-1]= padoridim - baseznum*(ranksize-1);

    if(ranknum==0)
    {
    	for(int i=0;i<ranksize;i++)
    		printf("task divide : %d %d \n",numberZ[i],offsetZ[i]);
    }

	multi_plan_init_mpi(dataplan,fullsize, numberZ[ranknum],offsetZ[ranknum],ranknum,padoridim,padoridim);
	if(padoridim > pad_size)
	{

		cudaFree((dataplan[0].d_Data));
		cudaMalloc((void**) &(dataplan[0].d_Data),sizeof(cufftComplex) * dataplan[0].datasize);
		cudaFreeHost(c_Fconv2);
		cudaMallocHost((void **) &c_Fconv2, sizeof(cufftComplex) * fullsize);
	}
	windowFourierTransform(Fconv, padoridim);

//	printf("layoutchangecomp : %ld %ld %ld %d\n",Fconv.xdim,Fconv.ydim,Fconv.zdim,padoridim);
	layoutchangecomp(Fconv.data,Fconv.xdim,Fconv.ydim,Fconv.zdim,padoridim,c_Fconv2);
//	printf("layoutchangecomp : %ld %ld %ld %d\n",Fconv.xdim,Fconv.ydim,Fconv.zdim,padoridim);
//init plan and for even data



	xyN[0] = padoridim;
	xyN[1] = padoridim;
    //==================2d fft plan
	cufftPlanMany(&xyplan, 2, xyN, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, numberZ[ranknum]);
   //==================1d fft plan
	int Ncol[1];
	Ncol[0] = padoridim;
	int inembed[3];
	inembed[0] = padoridim;inembed[1] = dataplan[0].selfZ;inembed[2] = padoridim;
	cufftPlanMany(&zplan, 1, Ncol, inembed, padoridim * padoridim, 1,inembed, padoridim * padoridim, 1, CUFFT_C2C, padoridim * (dataplan[0].selfZ));
	cudaMemcpy(dataplan[0].d_Data + dataplan[0].selfoffset, c_Fconv2 + dataplan[0].selfoffset,
			(dataplan[0].selfZ * padoridim * padoridim) * sizeof(cufftComplex),cudaMemcpyHostToDevice);

	cufftExecC2C(xyplan, dataplan[0].d_Data + dataplan[0].selfoffset,dataplan[0].d_Data + dataplan[0].selfoffset, CUFFT_INVERSE);
		//offset += padoridim * padoridim * (padoridim / 2);

	gpu_to_cpu(dataplan,cpu_data); //cpu_data - > c_Fconv2
	//cpu all to all

//	int starindex=(0);
//	int endindex=(fullsize/2);
//	int nonzeronum=0;
//	for(int i=starindex;i<endindex;i++)
//	{
//		if(cpu_data[i].x !=0)
//			nonzeronum++;
//		if(i<10)
//		printf("%f ",cpu_data[i].x);
//	}
//	printf("nonzeronum block1  : %d from rank %d\n",nonzeronum,ranknum);
//
//	nonzeronum=0;
//	starindex=(fullsize/2);
//	endindex = fullsize;
//	for(int i=starindex;i<endindex;i++)
//	{
//		if(cpu_data[i].x !=0)
//			nonzeronum++;
//	}
//	printf("nonzeronum block2 : %d from rank %d\n",nonzeronum,ranknum);


	cpu_alltoall_multinode(dataplan,cpu_data,numberZ,offsetZ,ranknum,padoridim,ranksize,realrankarray);


	//mulit_alltoall_one(dataplan,padoridim,padoridim,padoridim,0,offsetz);

	cufftExecC2C(zplan, dataplan[0].d_Data + (offsetZ[ranknum]*padoridim),dataplan[0].d_Data + (offsetZ[ranknum]*padoridim), CUFFT_INVERSE);
	//gpudata->cpu

	{
		int sliceoffset=padoridim*offsetZ[ranknum];
		int slicesize=padoridim*numberZ[ranknum];
		for(int i=0;i<padoridim;i++)
		{
			cudaMemcpy(cpu_data+sliceoffset,dataplan[0].d_Data + sliceoffset,slicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
			sliceoffset+=padoridim*padoridim;
		}
	}

	cpu_alltoalltozero_multi(cpu_data,numberZ,offsetZ,ranknum,padoridim,ranksize,realrankarray);
	cufftDestroy(zplan);
	cufftDestroy(xyplan);
	cudaFree(dataplan[0].d_Data);

	size_t freedata1,total1;
	cudaMemGetInfo( &freedata1, &total1 );
	//printf("After alloccation  : %ld   %ld and gpu num %d \n",freedata1,total1,ranknum);

	printf("MPI final  \n");
	if(ranknum == 0)
	{
	//copy data
	//	memcpy(vol_out.data,cpu_data,sizeof(float)*padoridim *padoridim*padoridim);
    for(int i=0;i<padoridim *padoridim*padoridim;i++)
    	vol_out.data[i]=cpu_data[i].x;


	printwhole(vol_out.data, padoridim *padoridim*padoridim ,ranknum);

    CenterFFT(vol_out,true);

	// Window in real-space
	if (ref_dim==2)
	{
		vol_out.window(FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size),
				       LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size));
	}
	else
	{
		vol_out.window(FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size),
				       LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size));
	}
	vol_out.setXmippOrigin();

	// Normalisation factor of FFTW
	// The Fourier Transforms are all "normalised" for 2D transforms of size = ori_size x ori_size
	float normfft = (RFLOAT)(padding_factor * padding_factor * padding_factor * ori_size);
	vol_out /= normfft;
	// Mask out corners to prevent aliasing artefacts
	softMaskOutsideMap(vol_out);
	//windowToOridimRealSpace(transformer, vol_out, nr_threads, printTimes);
	RCTOCREC(ReconTimer,ReconS_17);
	}
    free(cpu_data);
    cudaFreeHost(c_Fconv2); // all process need
#endif

#ifdef DEBUG_RECONSTRUCT
	ttt()=vol_out;
	ttt.write("reconstruct_before_gridding_correction.spi");
#endif

	if(ranknum == 0)
	{
	// Correct for the linear/nearest-neighbour interpolation that led to the data array
	RCTICREC(ReconTimer,ReconS_18);
	griddingCorrect(vol_out);
	RCTOCREC(ReconTimer,ReconS_18);
	// If the tau-values were calculated based on the FSC, then now re-calculate the power spectrum of the actual reconstruction
	if (update_tau2_with_fsc)
	{

		// New tau2 will be the power spectrum of the new map
		MultidimArray<RFLOAT> spectrum, count;

		// Calculate this map's power spectrum
		// Don't call getSpectrum() because we want to use the same transformer object to prevent memory trouble....
		RCTICREC(ReconTimer,ReconS_19);
		spectrum.initZeros(XSIZE(vol_out));
	    count.initZeros(XSIZE(vol_out));
		RCTOCREC(ReconTimer,ReconS_19);
		RCTICREC(ReconTimer,ReconS_20);
	    // recycle the same transformer for all images
        transformer.setReal(vol_out);
		RCTOCREC(ReconTimer,ReconS_20);
		RCTICREC(ReconTimer,ReconS_21);
        transformer.FourierTransform();
		RCTOCREC(ReconTimer,ReconS_21);
		RCTICREC(ReconTimer,ReconS_22);
	    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
	    {
	    	long int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
	    	spectrum(idx) += norm(dAkij(Fconv, k, i, j));
	        count(idx) += 1.;
	    }
	    spectrum /= count;

		// Factor two because of two-dimensionality of the complex plane
		// (just like sigma2_noise estimates, the power spectra should be divided by 2)
		RFLOAT normfft = (ref_dim == 3 && data_dim == 2) ? (RFLOAT)(ori_size * ori_size) : 1.;
		spectrum *= normfft / 2.;

		// New SNR^MAP will be power spectrum divided by the noise in the reconstruction (i.e. sigma2)
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data_vs_prior)
		{
			DIRECT_MULTIDIM_ELEM(tau2, n) =  tau2_fudge * DIRECT_MULTIDIM_ELEM(spectrum, n);
		}
		RCTOCREC(ReconTimer,ReconS_22);
	}
	RCTICREC(ReconTimer,ReconS_23);
	// Completely empty the transformer object
	transformer.cleanup();
    // Now can use extra mem to move data into smaller array space
    vol_out.shrinkToFit();

	RCTOCREC(ReconTimer,ReconS_23);
#ifdef TIMINGREC
    if(printTimes)
    	ReconTimer.printTimes(true);
#endif


#ifdef DEBUG_RECONSTRUCT
    std::cerr<<"done with reconstruct"<<std::endl;
#endif

	tau2_io = tau2;
    sigma2_out = sigma2;
    data_vs_prior_out = data_vs_prior;
    fourier_coverage_out = fourier_coverage;
	}
	else
	{
		vol_out.clear();
		vol_out.resize(rawvoln,rawvolz,rawvoly,rawvolx);
		//vol_out.reshape(rawvoln,rawvolz,rawvoly,rawvolx);
	}

}
void BackProjector::reconstruct_gpumpi(MultidimArray<RFLOAT> &vol_out,
                                int max_iter_preweight,
                                bool do_map,
                                RFLOAT tau2_fudge,
                                MultidimArray<RFLOAT> &tau2_io, // can be input/output
                                MultidimArray<RFLOAT> &sigma2_out,
                                MultidimArray<RFLOAT> &data_vs_prior_out,
                                MultidimArray<RFLOAT> &fourier_coverage_out,
                                const MultidimArray<RFLOAT> &fsc, // only input
                                RFLOAT normalise,
                                bool update_tau2_with_fsc,
                                bool is_whole_instead_of_half,
                                int nr_threads,
                                int minres_map,
                                bool printTimes,
								bool do_fsc0999,
								int realranknum,
								int ranksize)
{

#ifdef TIMINGREC
	Timer ReconTimer;
	printTimes=1;
	int ReconS_1 = ReconTimer.setNew(" RcS1_Init ");
	int ReconS_2 = ReconTimer.setNew(" RcS2_Shape&Noise ");
	int ReconS_2_5 = ReconTimer.setNew(" RcS2.5_Regularize ");
	int ReconS_3 = ReconTimer.setNew(" RcS3_skipGridding ");
	int ReconS_4 = ReconTimer.setNew(" RcS4_doGridding_norm ");
	int ReconS_5 = ReconTimer.setNew(" RcS5_doGridding_init ");
	int ReconS_6 = ReconTimer.setNew(" RcS6_doGridding_iter ");
	int ReconS_7 = ReconTimer.setNew(" RcS7_doGridding_apply ");
	int ReconS_8 = ReconTimer.setNew(" RcS8_blobConvolute ");
	int ReconS_9 = ReconTimer.setNew(" RcS9_blobResize ");
	int ReconS_10 = ReconTimer.setNew(" RcS10_blobSetReal ");
	int ReconS_11 = ReconTimer.setNew(" RcS11_blobSetTemp ");
	int ReconS_12 = ReconTimer.setNew(" RcS12_blobTransform ");
	int ReconS_13 = ReconTimer.setNew(" RcS13_blobCenterFFT ");
	int ReconS_14 = ReconTimer.setNew(" RcS14_blobNorm1 ");
	int ReconS_15 = ReconTimer.setNew(" RcS15_blobSoftMask ");
	int ReconS_16 = ReconTimer.setNew(" RcS16_blobNorm2 ");
	int ReconS_17 = ReconTimer.setNew(" RcS17_WindowReal ");
	int ReconS_18 = ReconTimer.setNew(" RcS18_GriddingCorrect ");
	int ReconS_19 = ReconTimer.setNew(" RcS19_tauInit ");
	int ReconS_20 = ReconTimer.setNew(" RcS20_tausetReal ");
	int ReconS_21 = ReconTimer.setNew(" RcS21_tauTransform ");
	int ReconS_22 = ReconTimer.setNew(" RcS22_tautauRest ");
	int ReconS_23 = ReconTimer.setNew(" RcS23_tauShrinkToFit ");
	int ReconS_24 = ReconTimer.setNew(" RcS24_extra ");
#endif

    // never rely on references (handed to you from the outside) for computation:
    // they could be the same (i.e. reconstruct(..., dummy, dummy, dummy, dummy, ...); )

	// process change
	int ranknum;
	int realrankarray[5];
	long int  rawvolx,rawvoly,rawvolz,rawvoln;
    //mpi table
	//for(int i=0;i<ranksize;i++)
	{
		int flag=0;
		if(realranknum == 1)
			ranknum =0;
		if(realranknum == 3)
			ranknum =1;


		if(realranknum ==2 )
		{
			ranknum =0;flag=1;
		}
		if(realranknum ==4 )
		{
			ranknum =1;flag=1;
		}
		if(flag ==0)
		{
			realrankarray[0]=1;
			realrankarray[1]=3;
		}
		if(flag==1)
		{
			realrankarray[0]=2;
			realrankarray[1]=4;
		}
		rawvolx=vol_out.xdim;
		rawvoly=vol_out.ydim;
		rawvolz=vol_out.zdim;
		rawvoln=vol_out.ndim;
	}
	printf("ranknum is %d and ranksize is %d\n",ranknum,ranksize);
	//end rank change
    MultidimArray<RFLOAT> sigma2, data_vs_prior, fourier_coverage;
	MultidimArray<RFLOAT> tau2 = tau2_io;
    FourierTransformer transformer;
	MultidimArray<RFLOAT> Fweight;
	int max_r2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor);
    MultidimArray<Complex>& Fconv = transformer.getFourierReference();
	// Fnewweight can become too large for a float: always keep this one in double-precision
	MultidimArray<double> Fnewweight;


	RCTICREC(ReconTimer,ReconS_1);


//#define DEBUG_RECONSTRUCT
#ifdef DEBUG_RECONSTRUCT
	Image<RFLOAT> ttt;
	FileName fnttt;
	ttt()=weight;
	ttt.write("reconstruct_initial_weight.spi");
	std::cerr << " pad_size= " << pad_size << " padding_factor= " << padding_factor << " max_r2= " << max_r2 << std::endl;
#endif

    // Set Fweight, Fnewweight and Fconv to the right size
    if (ref_dim == 2)
        vol_out.setDimensions(pad_size, pad_size, 1, 1);
    else
        // Too costly to actually allocate the space
        // Trick transformer with the right dimensions
        vol_out.setDimensions(pad_size, pad_size, pad_size, 1);

    transformer.setReal(vol_out); // Fake set real. 1. Allocate space for Fconv 2. calculate plans.
    vol_out.clear(); // Reset dimensions to 0

    RCTOCREC(ReconTimer,ReconS_1);
    RCTICREC(ReconTimer,ReconS_2);

    Fweight.reshape(Fconv);
    if (!skip_gridding)
    	Fnewweight.reshape(Fconv);

	// Go from projector-centered to FFTW-uncentered
	decenter(weight, Fweight, max_r2);

	// Take oversampling into account
	RFLOAT oversampling_correction = (ref_dim == 3) ? (padding_factor * padding_factor * padding_factor) : (padding_factor * padding_factor);
	MultidimArray<RFLOAT> counter;

	// First calculate the radial average of the (inverse of the) power of the noise in the reconstruction
	// This is the left-hand side term in the nominator of the Wiener-filter-like update formula
	// and it is stored inside the weight vector
	// Then, if (do_map) add the inverse of tau2-spectrum values to the weight
	sigma2.initZeros(ori_size/2 + 1);
	counter.initZeros(ori_size/2 + 1);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
	{
		int r2 = kp * kp + ip * ip + jp * jp;
		if (r2 < max_r2)
		{
			int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
			RFLOAT invw = oversampling_correction * DIRECT_A3D_ELEM(Fweight, k, i, j);
			DIRECT_A1D_ELEM(sigma2, ires) += invw;
			DIRECT_A1D_ELEM(counter, ires) += 1.;
		}
    }

	// Average (inverse of) sigma2 in reconstruction
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
	{
        if (DIRECT_A1D_ELEM(sigma2, i) > 1e-10)
            DIRECT_A1D_ELEM(sigma2, i) = DIRECT_A1D_ELEM(counter, i) / DIRECT_A1D_ELEM(sigma2, i);
        else if (DIRECT_A1D_ELEM(sigma2, i) == 0)
            DIRECT_A1D_ELEM(sigma2, i) = 0.;
		else
		{
			std::cerr << " DIRECT_A1D_ELEM(sigma2, i)= " << DIRECT_A1D_ELEM(sigma2, i) << std::endl;
			REPORT_ERROR("BackProjector::reconstruct: ERROR: unexpectedly small, yet non-zero sigma2 value, this should not happen...a");
        }
    }

	if (update_tau2_with_fsc)
    {
        tau2.reshape(ori_size/2 + 1);
        data_vs_prior.initZeros(ori_size/2 + 1);
		// Then calculate new tau2 values, based on the FSC
		if (!fsc.sameShape(sigma2) || !fsc.sameShape(tau2))
		{
			fsc.printShape(std::cerr);
			tau2.printShape(std::cerr);
			sigma2.printShape(std::cerr);
			REPORT_ERROR("ERROR BackProjector::reconstruct: sigma2, tau2 and fsc have different sizes");
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
        {
			// FSC cannot be negative or zero for conversion into tau2
			RFLOAT myfsc = XMIPP_MAX(0.001, DIRECT_A1D_ELEM(fsc, i));
			if (is_whole_instead_of_half)
			{
				// Factor two because of twice as many particles
				// Sqrt-term to get 60-degree phase errors....
				myfsc = sqrt(2. * myfsc / (myfsc + 1.));
			}
			myfsc = XMIPP_MIN(0.999, myfsc);
			RFLOAT myssnr = myfsc / (1. - myfsc);
			// Sjors 29nov2017 try tau2_fudge for pulling harder on Refine3D runs...
            myssnr *= tau2_fudge;
			RFLOAT fsc_based_tau = myssnr * DIRECT_A1D_ELEM(sigma2, i);
			DIRECT_A1D_ELEM(tau2, i) = fsc_based_tau;
			// data_vs_prior is merely for reporting: it is not used for anything in the reconstruction
			DIRECT_A1D_ELEM(data_vs_prior, i) = myssnr;
		}
	}
    RCTOCREC(ReconTimer,ReconS_2);
    RCTICREC(ReconTimer,ReconS_2_5);
	// Apply MAP-additional term to the Fnewweight array
	// This will regularise the actual reconstruction
    if (do_map)
	{

    	// Then, add the inverse of tau2-spectrum values to the weight
		// and also calculate spherical average of data_vs_prior ratios
		if (!update_tau2_with_fsc)
			data_vs_prior.initZeros(ori_size/2 + 1);
		fourier_coverage.initZeros(ori_size/2 + 1);
		counter.initZeros(ori_size/2 + 1);
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
 		{
			int r2 = kp * kp + ip * ip + jp * jp;
			if (r2 < max_r2)
			{
				int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
				RFLOAT invw = DIRECT_A3D_ELEM(Fweight, k, i, j);

				RFLOAT invtau2;
				if (DIRECT_A1D_ELEM(tau2, ires) > 0.)
				{
					// Calculate inverse of tau2
					invtau2 = 1. / (oversampling_correction * tau2_fudge * DIRECT_A1D_ELEM(tau2, ires));
				}
				else if (DIRECT_A1D_ELEM(tau2, ires) == 0.)
				{
					// If tau2 is zero, use small value instead
					invtau2 = 1./ ( 0.001 * invw);
				}
				else
				{
					std::cerr << " sigma2= " << sigma2 << std::endl;
					std::cerr << " fsc= " << fsc << std::endl;
					std::cerr << " tau2= " << tau2 << std::endl;
					REPORT_ERROR("ERROR BackProjector::reconstruct: Negative or zero values encountered for tau2 spectrum!");
				}

				// Keep track of spectral evidence-to-prior ratio and remaining noise in the reconstruction
				if (!update_tau2_with_fsc)
					DIRECT_A1D_ELEM(data_vs_prior, ires) += invw / invtau2;

				// Keep track of the coverage in Fourier space
				if (invw / invtau2 >= 1.)
					DIRECT_A1D_ELEM(fourier_coverage, ires) += 1.;

				DIRECT_A1D_ELEM(counter, ires) += 1.;

				// Only for (ires >= minres_map) add Wiener-filter like term
				if (ires >= minres_map)
				{
					// Now add the inverse-of-tau2_class term
					invw += invtau2;
					// Store the new weight again in Fweight
					DIRECT_A3D_ELEM(Fweight, k, i, j) = invw;
				}
			}
		}

		// Average data_vs_prior
		if (!update_tau2_with_fsc)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(data_vs_prior)
			{
				if (i > r_max)
					DIRECT_A1D_ELEM(data_vs_prior, i) = 0.;
				else if (DIRECT_A1D_ELEM(counter, i) < 0.001)
					DIRECT_A1D_ELEM(data_vs_prior, i) = 999.;
				else
					DIRECT_A1D_ELEM(data_vs_prior, i) /= DIRECT_A1D_ELEM(counter, i);
			}
		}

		// Calculate Fourier coverage in each shell
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fourier_coverage)
		{
			if (DIRECT_A1D_ELEM(counter, i) > 0.)
				DIRECT_A1D_ELEM(fourier_coverage, i) /= DIRECT_A1D_ELEM(counter, i);
		}

	} //end if do_map
    else if (do_fsc0999)
    {

     	// Sjors 9may2018: avoid numerical instabilities with unregularised reconstructions....
        FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
        {
            int r2 = kp * kp + ip * ip + jp * jp;
            if (r2 < max_r2)
            {
                int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
                if (ires >= minres_map)
                {
                    // add 1/1000th of the radially averaged sigma2 to the Fweight, to avoid having zeros there...
                	DIRECT_A3D_ELEM(Fweight, k, i, j) += 1./(999. * DIRECT_A1D_ELEM(sigma2, ires));
                }
            }
        }

    }


	//==============================================================================add multi- GPU version

	int Ndim[3];
	int GPU_N=ranksize;
	Ndim[0]=pad_size;
	Ndim[1]=pad_size;
	Ndim[2]=pad_size;
	size_t fullsize= pad_size*pad_size*pad_size;
	initgpu_mpi(ranknum);

	//divide task
	//process divide data
	int *numberZ;
	int *offsetZ;
	numberZ = (int *)malloc(sizeof(int)*ranksize);
	offsetZ = (int *)malloc(sizeof(int)*ranksize);
	int baseznum= pad_size / ranksize;
	for(int i=0; i<ranksize;i++)
	{
		if(i==0)
		{
			offsetZ[0]=0;
			numberZ[0]=baseznum;
		}
		else
		{
			offsetZ[i] = offsetZ[i-1]+numberZ[i-1];
			numberZ[i]= baseznum;
		}
	}
    numberZ[ranksize-1]= pad_size - baseznum*(ranksize-1);
	if(ranknum==0)
	{
		for(int i=0;i<ranksize;i++)
			printf(" rank :%d : num %d and offset %d \n",i,numberZ[i],offsetZ[i]);
	}
	//int extraz = offsetZ[1] - offsetZ[0];
	//int extraz = numberZ[1] - numberZ[0];
	//numberZ

	RFLOAT *d_Fweight;
	double *d_Fnewweight;
	int offset;

	MultiGPUplan *dataplan;
	dataplan = (MultiGPUplan *)malloc(sizeof(MultiGPUplan)*1);
	multi_plan_init_mpi(dataplan,fullsize, numberZ[ranknum], offsetZ[ranknum],ranknum,pad_size,pad_size);
	cufftComplex *cpu_data,*c_Fconv2;

	cudaMalloc((void**) &(dataplan[0].d_Data),sizeof(cufftComplex) * dataplan[0].datasize);

	//=================================
	int xyN[2];
	xyN[0] = Ndim[0];
	xyN[1] = Ndim[1];
	cufftHandle xyplan;
	cufftHandle zplan;


    RCTOCREC(ReconTimer,ReconS_2_5);
	if (skip_gridding)
	{


	    RCTICREC(ReconTimer,ReconS_3);
		std::cerr << "Skipping gridding!" << std::endl;
		Fconv.initZeros(); // to remove any stuff from the input volume
		decenter(data, Fconv, max_r2);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			if (DIRECT_MULTIDIM_ELEM(Fweight, n) > 0.)
				DIRECT_MULTIDIM_ELEM(Fconv, n) /= DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		RCTOCREC(ReconTimer,ReconS_3);
#ifdef DEBUG_RECONSTRUCT
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			DIRECT_MULTIDIM_ELEM(ttt(), n) = DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		ttt.write("reconstruct_skipgridding_correction_term.spi");
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			if (DIRECT_MULTIDIM_ELEM(Fweight, n) > 0.)
				DIRECT_MULTIDIM_ELEM(ttt(), n) = 1./DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		ttt.write("reconstruct_skipgridding_correction_term_inverse.spi");
#endif

	}
	else
	{

		RCTICREC(ReconTimer,ReconS_4);
		// Divide both data and Fweight by normalisation factor to prevent FFT's with very large values....
	#ifdef DEBUG_RECONSTRUCT
		std::cerr << " normalise= " << normalise << std::endl;
	#endif
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fweight)
		{
			DIRECT_MULTIDIM_ELEM(Fweight, n) /= normalise;
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data)
		{
			DIRECT_MULTIDIM_ELEM(data, n) /= normalise;
		}
		RCTOCREC(ReconTimer,ReconS_4);
		RCTICREC(ReconTimer,ReconS_5);
        // Initialise Fnewweight with 1's and 0's. (also see comments below)
		FOR_ALL_ELEMENTS_IN_ARRAY3D(weight)
		{
			if (k * k + i * i + j * j < max_r2)
				A3D_ELEM(weight, k, i, j) = 1.;
			else
				A3D_ELEM(weight, k, i, j) = 0.;
		}



		decenter(weight, Fnewweight, max_r2);
		RCTOCREC(ReconTimer,ReconS_5);
		// Iterative algorithm as in  Eq. [14] in Pipe & Menon (1999)
		// or Eq. (4) in Matej (2001)



		long int Fconvnum=Fconv.nzyxdim;


		long int normsize= Ndim[0]*Ndim[1]*Ndim[2];
		int halfxdim=Fconv.xdim;

		printf("%ld %ld %ld \n",Fconv.xdim,Fconv.ydim,Fconv.zdim);
		printf("Fnewweight: %ld %ld %ld \n",Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim);




		cpu_data= (cufftComplex *)malloc(fullsize * sizeof(cufftComplex));
		//c_Fconv2 = (cufftComplex *)malloc(fullsize * sizeof(cufftComplex));
		cudaMallocHost((void **) &c_Fconv2, sizeof(cufftComplex) * fullsize);
#ifdef RELION_SINGLE_PRECISION
		d_Fweight = gpusetdata_float(d_Fweight,Fweight.nzyxdim,Fweight.data);
#else
		d_Fweight = gpusetdata_double(d_Fweight,Fweight.nzyxdim,Fweight.data);
#endif
		d_Fnewweight = gpusetdata_double(d_Fnewweight,Fnewweight.nzyxdim,Fnewweight.data);


		printf("d_Fweight num : %d ,d_Fnewweight num :%d \n",Fweight.nzyxdim,Fnewweight.nzyxdim);


         //==================2d fft plan


		cufftPlanMany(&xyplan, 2, xyN, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, numberZ[ranknum]);


	    //==================1d fft plan
		int Ncol[1];
		Ncol[0] = pad_size;
		int inembed[3], extraembed[3];
		inembed[0] = Ndim[0];inembed[1] = numberZ[ranknum];inembed[2] = Ndim[2];
		//extraembed[0] = Ndim[0];extraembed[1] = numberZ[1];extraembed[2] = Ndim[2];
		cufftPlanMany(&zplan, 1, Ncol, inembed, pad_size * pad_size, 1, inembed, pad_size * pad_size, 1, CUFFT_C2C, Ndim[0] * (numberZ[ranknum]));


		double *tempdata= (double *)malloc(sizeof(double)*pad_size*pad_size*pad_size);
		float *fweightdata= (float *)malloc(sizeof(float)*pad_size*pad_size*pad_size);
		// every thread map two block to used
		layoutchange(Fnewweight.data,Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim,pad_size,tempdata);
		layoutchange(Fweight.data,Fweight.xdim,Fweight.ydim,Fweight.zdim,pad_size,fweightdata);
		double *d_blockone;
		float *d_blocktwo;
		cudaMalloc((void**) &(d_blockone),sizeof(double) * dataplan[0].realsize);
		cudaMalloc((void**) &(d_blocktwo),sizeof(float) * dataplan[0].realsize);
		//cudaMalloc((void**) &(d_blocktwo),sizeof(double) * dataplan[0].realsize);

		cudaMemcpy(d_blockone,tempdata+dataplan[0].selfoffset,dataplan[0].realsize*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_blocktwo,fweightdata+dataplan[0].selfoffset,dataplan[0].realsize*sizeof(float),cudaMemcpyHostToDevice);
		free(fweightdata);


		for (int iter = 0; iter < max_iter_preweight; iter++)
		{

			RCTICREC(ReconTimer,ReconS_6);

			vector_Multi_layout_mpi(d_blockone,d_blocktwo,dataplan[0].d_Data+dataplan[0].selfoffset,dataplan[0].realsize);

			cufftExecC2C(xyplan, dataplan[0].d_Data + dataplan[0].selfoffset,dataplan[0].d_Data + dataplan[0].selfoffset, CUFFT_INVERSE);
			cudaDeviceSynchronize();
			//printf("Iteration %d and Step2: \n",iter);
			gpu_to_cpu(dataplan,cpu_data);

			cpu_alltoall_multinode(dataplan,cpu_data,numberZ,offsetZ,ranknum,pad_size,ranksize,realrankarray);

			cufftExecC2C(zplan, dataplan[0].d_Data + (offsetZ[ranknum]*pad_size),dataplan[0].d_Data + (offsetZ[ranknum]*pad_size), CUFFT_INVERSE);
			cudaDeviceSynchronize();

			RFLOAT normftblob = tab_ftblob(0.);
			float *d_tab_ftblob;
			int tabxdim=tab_ftblob.tabulatedValues.xdim;
			d_tab_ftblob=gpusetdata_float(d_tab_ftblob,tab_ftblob.tabulatedValues.xdim,tab_ftblob.tabulatedValues.data);
			//gpu_kernel2
			volume_Multi_float_mpi(dataplan[0].d_Data,d_tab_ftblob, Ndim[0]*numberZ[ranknum]*Ndim[2],
					tabxdim, tab_ftblob.sampling , pad_size/2, pad_size, ori_size, padding_factor, normftblob,
					numberZ[ranknum],offsetZ[ranknum]);
			cudaDeviceSynchronize();
			cufftExecC2C(zplan, dataplan[0].d_Data + (offsetZ[ranknum]*pad_size),dataplan[0].d_Data + (offsetZ[ranknum]*pad_size), CUFFT_FORWARD);
			cudaDeviceSynchronize();

			cpu_alltoall_inverse_multinode(dataplan,cpu_data,numberZ,offsetZ,ranknum,pad_size,ranksize,realrankarray);

			cufftExecC2C(xyplan, dataplan[0].d_Data + dataplan[0].selfoffset,dataplan[0].d_Data + dataplan[0].selfoffset, CUFFT_FORWARD);
			cudaDeviceSynchronize();

			vector_Normlize(dataplan[0].d_Data, normsize ,Ndim[0]*Ndim[1]*Ndim[2]);


			fft_Divide_mpi(dataplan[0].d_Data+dataplan[0].selfoffset,d_blockone,dataplan[0].realsize,
					pad_size*pad_size,pad_size,pad_size,pad_size,Fconv.xdim,max_r2,offsetZ[ranknum]);

			RCTOCREC(ReconTimer,ReconS_6);

	#ifdef DEBUG_RECONSTRUCT
			std::cerr << " PREWEIGHTING ITERATION: "<< iter + 1 << " OF " << max_iter_preweight << std::endl;
			// report of maximum and minimum values of current conv_weight
			std::cerr << " corr_avg= " << corr_avg / corr_nn << std::endl;
			std::cerr << " corr_min= " << corr_min << std::endl;
			std::cerr << " corr_max= " << corr_max << std::endl;
	#endif

		}

		cudaMemcpy(tempdata+dataplan[0].selfoffset,d_blockone,dataplan[0].realsize*sizeof(double),cudaMemcpyDeviceToHost);

		if(ranknum!=0)
		{
			MPI_Send(tempdata+dataplan[0].selfoffset,dataplan[0].realsize,MPI_DOUBLE,realrankarray[0],0,MPI_COMM_WORLD);
		}
		else
		{
			for(int i=1;i<ranksize;i++)
				MPI_Recv(tempdata+(offsetZ[i]*pad_size*pad_size),pad_size*pad_size*numberZ[i],MPI_DOUBLE,realrankarray[i],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			layoutchangeback(tempdata,Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim,pad_size,Fnewweight.data);

		}

		if(ranknum==0)
		{
			for(int i=1;i<ranksize;i++)
				MPI_Send(Fnewweight.data,Fnewweight.nzyxdim,MPI_DOUBLE,realrankarray[i],0,MPI_COMM_WORLD);
		}
		else
		{
			MPI_Recv(Fnewweight.data,Fnewweight.nzyxdim,MPI_DOUBLE,realrankarray[0],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}

		cufftDestroy(xyplan);
		cufftDestroy(zplan);
		free(tempdata);

		cudaFree(d_Fnewweight);
		cudaFree(d_Fweight);
		cudaFree(d_blockone);
		RCTICREC(ReconTimer,ReconS_7);


	#ifdef DEBUG_RECONSTRUCT
		Image<double> tttt;
		tttt()=Fnewweight;
		tttt.write("reconstruct_gridding_weight.spi");
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			DIRECT_MULTIDIM_ELEM(ttt(), n) = abs(DIRECT_MULTIDIM_ELEM(Fconv, n));
		}
		ttt.write("reconstruct_gridding_correction_term.spi");
	#endif


		// Clear memory
		Fweight.clear();

		// Note that Fnewweight now holds the approximation of the inverse of the weights on a regular grid

		// Now do the actual reconstruction with the data array
		// Apply the iteratively determined weight
		Fconv.initZeros(); // to remove any stuff from the input volume
		decenter(data, Fconv, max_r2);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
#ifdef  RELION_SINGLE_PRECISION
			// Prevent numerical instabilities in single-precision reconstruction with very unevenly sampled orientations
			if (DIRECT_MULTIDIM_ELEM(Fnewweight, n) > 1e20)
				DIRECT_MULTIDIM_ELEM(Fnewweight, n) = 1e20;
#endif
			DIRECT_MULTIDIM_ELEM(Fconv, n) *= DIRECT_MULTIDIM_ELEM(Fnewweight, n);
		}

		// Clear memory
		Fnewweight.clear();
		RCTOCREC(ReconTimer,ReconS_7);
	} // end if skip_gridding


// Gridding theory says one now has to interpolate the fine grid onto the coarse one using a blob kernel
// and then do the inverse transform and divide by the FT of the blob (i.e. do the gridding correction)
// In practice, this gives all types of artefacts (perhaps I never found the right implementation?!)
// Therefore, window the Fourier transform and then do the inverse transform
//#define RECONSTRUCT_CONVOLUTE_BLOB
#ifdef RECONSTRUCT_CONVOLUTE_BLOB

	// Apply the same blob-convolution as above to the data array
	// Mask real-space map beyond its original size to prevent aliasing in the downsampling step below
	RCTICREC(ReconTimer,ReconS_8);
	convoluteBlobRealSpace(transformer, true);
	RCTOCREC(ReconTimer,ReconS_8);
	RCTICREC(ReconTimer,ReconS_9);
	// Now just pick every 3rd pixel in Fourier-space (i.e. down-sample)
	// and do a final inverse FT
	if (ref_dim == 2)
		vol_out.resize(ori_size, ori_size);
	else
		vol_out.resize(ori_size, ori_size, ori_size);
	RCTOCREC(ReconTimer,ReconS_9);
	RCTICREC(ReconTimer,ReconS_10);
	FourierTransformer transformer2;
	MultidimArray<Complex > Ftmp;
	transformer2.setReal(vol_out); // cannot use the first transformer because Fconv is inside there!!
	transformer2.getFourierAlias(Ftmp);
	RCTOCREC(ReconTimer,ReconS_10);
	RCTICREC(ReconTimer,ReconS_11);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Ftmp)
	{
		if (kp * kp + ip * ip + jp * jp < r_max * r_max)
		{
			DIRECT_A3D_ELEM(Ftmp, k, i, j) = FFTW_ELEM(Fconv, kp * padding_factor, ip * padding_factor, jp * padding_factor);
		}
		else
		{
			DIRECT_A3D_ELEM(Ftmp, k, i, j) = 0.;
		}
	}
	RCTOCREC(ReconTimer,ReconS_11);
	RCTICREC(ReconTimer,ReconS_12);
	// inverse FFT leaves result in vol_out
	transformer2.inverseFourierTransform();
	RCTOCREC(ReconTimer,ReconS_12);
	RCTICREC(ReconTimer,ReconS_13);
	// Shift the map back to its origin
	CenterFFT(vol_out, false);
	RCTOCREC(ReconTimer,ReconS_13);
	RCTICREC(ReconTimer,ReconS_14);
	// Un-normalize FFTW (because original FFTs were done with the size of 2D FFTs)
	if (ref_dim==3)
		vol_out /= ori_size;
	RCTOCREC(ReconTimer,ReconS_14);
	RCTICREC(ReconTimer,ReconS_15);
	// Mask out corners to prevent aliasing artefacts
	softMaskOutsideMap(vol_out);
	RCTOCREC(ReconTimer,ReconS_15);
	RCTICREC(ReconTimer,ReconS_16);
	// Gridding correction for the blob
	RFLOAT normftblob = tab_ftblob(0.);
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol_out)
	{

		RFLOAT r = sqrt((RFLOAT)(k*k+i*i+j*j));
		RFLOAT rval = r / (ori_size * padding_factor);
		A3D_ELEM(vol_out, k, i, j) /= tab_ftblob(rval) / normftblob;
		//if (k==0 && i==0)
		//	std::cerr << " j= " << j << " rval= " << rval << " tab_ftblob(rval) / normftblob= " << tab_ftblob(rval) / normftblob << std::endl;
	}
	RCTOCREC(ReconTimer,ReconS_16);

#else

	// rather than doing the blob-convolution to downsample the data array, do a windowing operation:
	// This is the same as convolution with a SINC. It seems to give better maps.
	// Then just make the blob look as much as a SINC as possible....
	// The "standard" r1.9, m2 and a15 blob looks quite like a sinc until the first zero (perhaps that's why it is standard?)
	//for (RFLOAT r = 0.1; r < 10.; r+=0.01)
	//{
	//	RFLOAT sinc = sin(PI * r / padding_factor ) / ( PI * r / padding_factor);
	//	std::cout << " r= " << r << " sinc= " << sinc << " blob= " << blob_val(r, blob) << std::endl;
	//}

	// Now do inverse FFT and window to original size in real-space
	// Pass the transformer to prevent making and clearing a new one before clearing the one declared above....
	// The latter may give memory problems as detected by electric fence....
	RCTICREC(ReconTimer,ReconS_17);


	// Size of padded real-space volume
	int padoridim = ROUND(padding_factor * ori_size);
	// make sure padoridim is even
	padoridim += padoridim%2;
    vol_out.reshape(padoridim, padoridim, padoridim);
    vol_out.setXmippOrigin();
	fullsize = padoridim *padoridim*padoridim;
	baseznum= padoridim / ranksize;
	for(int i=0; i<ranksize;i++)
	{
		if(i==0)
		{
			offsetZ[0]=0;
			numberZ[0]=baseznum;
		}
		else
		{
			offsetZ[i] = offsetZ[i-1]+numberZ[i-1];
			numberZ[i]= baseznum;
		}
	}
    numberZ[ranksize-1]= padoridim - baseznum*(ranksize-1);

    if(ranknum==0)
    {
    	for(int i=0;i<ranksize;i++)
    		printf("task divide : %d %d \n",numberZ[i],offsetZ[i]);
    }

	multi_plan_init_mpi(dataplan,fullsize, numberZ[ranknum],offsetZ[ranknum],ranknum,padoridim,padoridim);
	if(padoridim > pad_size)
	{

		cudaFree((dataplan[0].d_Data));
		cudaMalloc((void**) &(dataplan[0].d_Data),sizeof(cufftComplex) * dataplan[0].datasize);
		cudaFreeHost(c_Fconv2);
		cudaMallocHost((void **) &c_Fconv2, sizeof(cufftComplex) * fullsize);
		free(cpu_data);
		cpu_data= (cufftComplex *)malloc(fullsize * sizeof(cufftComplex));
	}
	windowFourierTransform(Fconv, padoridim);

	layoutchangecomp(Fconv.data,Fconv.xdim,Fconv.ydim,Fconv.zdim,padoridim,c_Fconv2);
//init plan and for even data


	xyN[0] = padoridim;
	xyN[1] = padoridim;
    //==================2d fft plan
	cufftPlanMany(&xyplan, 2, xyN, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, numberZ[ranknum]);
   //==================1d fft plan
	int Ncol[1];
	Ncol[0] = padoridim;
	int inembed[3];
	inembed[0] = padoridim;inembed[1] = dataplan[0].selfZ;inembed[2] = padoridim;
	cufftPlanMany(&zplan, 1, Ncol, inembed, padoridim * padoridim, 1,inembed, padoridim * padoridim, 1, CUFFT_C2C, padoridim * (dataplan[0].selfZ));
	cudaMemcpy(dataplan[0].d_Data + dataplan[0].selfoffset, c_Fconv2 + dataplan[0].selfoffset,
			(dataplan[0].selfZ * padoridim * padoridim) * sizeof(cufftComplex),cudaMemcpyHostToDevice);

	cufftExecC2C(xyplan, dataplan[0].d_Data + dataplan[0].selfoffset,dataplan[0].d_Data + dataplan[0].selfoffset, CUFFT_INVERSE);
		//offset += padoridim * padoridim * (padoridim / 2);
	gpu_to_cpu(dataplan,cpu_data); //cpu_data - > c_Fconv2
	cpu_alltoall_multinode(dataplan,cpu_data,numberZ,offsetZ,ranknum,padoridim,ranksize,realrankarray);
	cufftExecC2C(zplan, dataplan[0].d_Data + (offsetZ[ranknum]*padoridim),dataplan[0].d_Data + (offsetZ[ranknum]*padoridim), CUFFT_INVERSE);
	//gpudata->cpu
	{
		int sliceoffset=padoridim*offsetZ[ranknum];
		int slicesize=padoridim*numberZ[ranknum];
		for(int i=0;i<padoridim;i++)
		{
			cudaMemcpy(cpu_data+sliceoffset,dataplan[0].d_Data + sliceoffset,slicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
			sliceoffset+=padoridim*padoridim;
		}
	}
	cpu_alltoalltozero_multi(cpu_data,numberZ,offsetZ,ranknum,padoridim,ranksize,realrankarray);
	cufftDestroy(zplan);
	cufftDestroy(xyplan);
	cudaFree(dataplan[0].d_Data);


	if(ranknum == 0)
	{
	//copy data
	//	memcpy(vol_out.data,cpu_data,sizeof(float)*padoridim *padoridim*padoridim);
    for(int i=0;i<padoridim *padoridim*padoridim;i++)
    	vol_out.data[i]=cpu_data[i].x;


	printwhole(vol_out.data, padoridim *padoridim*padoridim ,ranknum);

    CenterFFT(vol_out,true);

	// Window in real-space
	if (ref_dim==2)
	{
		vol_out.window(FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size),
				       LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size));
	}
	else
	{
		vol_out.window(FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size),
				       LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size));
	}
	vol_out.setXmippOrigin();

	// Normalisation factor of FFTW
	// The Fourier Transforms are all "normalised" for 2D transforms of size = ori_size x ori_size
	float normfft = (RFLOAT)(padding_factor * padding_factor * padding_factor * ori_size);
	vol_out /= normfft;
	// Mask out corners to prevent aliasing artefacts
	softMaskOutsideMap(vol_out);
	//windowToOridimRealSpace(transformer, vol_out, nr_threads, printTimes);
	RCTOCREC(ReconTimer,ReconS_17);
	}
    free(cpu_data);
    cudaFreeHost(c_Fconv2); // all process need
#endif

#ifdef DEBUG_RECONSTRUCT
	ttt()=vol_out;
	ttt.write("reconstruct_before_gridding_correction.spi");
#endif

	if(ranknum == 0)
	{
	// Correct for the linear/nearest-neighbour interpolation that led to the data array
	RCTICREC(ReconTimer,ReconS_18);
	griddingCorrect(vol_out);
	RCTOCREC(ReconTimer,ReconS_18);
	// If the tau-values were calculated based on the FSC, then now re-calculate the power spectrum of the actual reconstruction
	if (update_tau2_with_fsc)
	{

		// New tau2 will be the power spectrum of the new map
		MultidimArray<RFLOAT> spectrum, count;

		// Calculate this map's power spectrum
		// Don't call getSpectrum() because we want to use the same transformer object to prevent memory trouble....
		RCTICREC(ReconTimer,ReconS_19);
		spectrum.initZeros(XSIZE(vol_out));
	    count.initZeros(XSIZE(vol_out));
		RCTOCREC(ReconTimer,ReconS_19);
		RCTICREC(ReconTimer,ReconS_20);
	    // recycle the same transformer for all images
        transformer.setReal(vol_out);
		RCTOCREC(ReconTimer,ReconS_20);
		RCTICREC(ReconTimer,ReconS_21);
        transformer.FourierTransform();
		RCTOCREC(ReconTimer,ReconS_21);
		RCTICREC(ReconTimer,ReconS_22);
	    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
	    {
	    	long int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
	    	spectrum(idx) += norm(dAkij(Fconv, k, i, j));
	        count(idx) += 1.;
	    }
	    spectrum /= count;

		// Factor two because of two-dimensionality of the complex plane
		// (just like sigma2_noise estimates, the power spectra should be divided by 2)
		RFLOAT normfft = (ref_dim == 3 && data_dim == 2) ? (RFLOAT)(ori_size * ori_size) : 1.;
		spectrum *= normfft / 2.;

		// New SNR^MAP will be power spectrum divided by the noise in the reconstruction (i.e. sigma2)
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data_vs_prior)
		{
			DIRECT_MULTIDIM_ELEM(tau2, n) =  tau2_fudge * DIRECT_MULTIDIM_ELEM(spectrum, n);
		}
		RCTOCREC(ReconTimer,ReconS_22);
	}
	RCTICREC(ReconTimer,ReconS_23);
	// Completely empty the transformer object
	transformer.cleanup();
    // Now can use extra mem to move data into smaller array space
    vol_out.shrinkToFit();

	RCTOCREC(ReconTimer,ReconS_23);
#ifdef TIMINGREC
    if(printTimes)
    	ReconTimer.printTimes(true);
#endif


#ifdef DEBUG_RECONSTRUCT
    std::cerr<<"done with reconstruct"<<std::endl;
#endif

	tau2_io = tau2;
    sigma2_out = sigma2;
    data_vs_prior_out = data_vs_prior;
    fourier_coverage_out = fourier_coverage;
	}
	else
	{
		vol_out.clear();
		vol_out.resize(rawvoln,rawvolz,rawvoly,rawvolx);
		//vol_out.reshape(rawvoln,rawvolz,rawvoly,rawvolx);
	}
}
/*
			if(ranknum==0)
			{
				cudaMemcpy(dataplan[0].d_Data,cpu_data,fullsize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				RFLOAT normftblob = tab_ftblob(0.);
				float *d_tab_ftblob;
				int tabxdim=tab_ftblob.tabulatedValues.xdim;
				d_tab_ftblob=gpusetdata_float(d_tab_ftblob,tab_ftblob.tabulatedValues.xdim,tab_ftblob.tabulatedValues.data);
				//gpu_kernel2
				volume_Multi_float_mpi(dataplan[0].d_Data,d_tab_ftblob, Ndim[0]*Ndim[1]*Ndim[2],
						tabxdim, tab_ftblob.sampling , pad_size/2, pad_size, ori_size, padding_factor, normftblob,
						Ndim[1],0);
				cudaDeviceSynchronize();
				cudaMemcpy(cpu_data,dataplan[0].d_Data,fullsize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
				for(int i=1;i<ranksize;i++)
					MPI_Send(cpu_data,fullsize*2,MPI_FLOAT,i,0,MPI_COMM_WORLD);
			}
			else
			{
				MPI_Recv(cpu_data,fullsize*2,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				cudaMemcpy(dataplan[0].d_Data,cpu_data,fullsize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
			}
*/

void BackProjector::reconstruct_gpumpicard(MultidimArray<RFLOAT> &vol_out,
                                int max_iter_preweight,
                                bool do_map,
                                RFLOAT tau2_fudge,
                                MultidimArray<RFLOAT> &tau2_io, // can be input/output
                                MultidimArray<RFLOAT> &sigma2_out,
                                MultidimArray<RFLOAT> &data_vs_prior_out,
                                MultidimArray<RFLOAT> &fourier_coverage_out,
                                const MultidimArray<RFLOAT> &fsc, // only input
                                RFLOAT normalise,
                                bool update_tau2_with_fsc,
                                bool is_whole_instead_of_half,
                                int nr_threads,
                                int minres_map,
                                bool printTimes,
								bool do_fsc0999,
								int realranknum,
								int ranksize)
{

#ifdef TIMINGREC
	Timer ReconTimer;
	printTimes=1;
	int ReconS_1 = ReconTimer.setNew(" RcS1_Init ");
	int ReconS_2 = ReconTimer.setNew(" RcS2_Shape&Noise ");
	int ReconS_2_5 = ReconTimer.setNew(" RcS2.5_Regularize ");
	int ReconS_3 = ReconTimer.setNew(" RcS3_skipGridding ");
	int ReconS_4 = ReconTimer.setNew(" RcS4_doGridding_norm ");
	int ReconS_5 = ReconTimer.setNew(" RcS5_doGridding_init ");
	int ReconS_6 = ReconTimer.setNew(" RcS6_doGridding_iter ");
	int ReconS_7 = ReconTimer.setNew(" RcS7_doGridding_apply ");
	int ReconS_8 = ReconTimer.setNew(" RcS8_blobConvolute ");
	int ReconS_9 = ReconTimer.setNew(" RcS9_blobResize ");
	int ReconS_10 = ReconTimer.setNew(" RcS10_blobSetReal ");
	int ReconS_11 = ReconTimer.setNew(" RcS11_blobSetTemp ");
	int ReconS_12 = ReconTimer.setNew(" RcS12_blobTransform ");
	int ReconS_13 = ReconTimer.setNew(" RcS13_blobCenterFFT ");
	int ReconS_14 = ReconTimer.setNew(" RcS14_blobNorm1 ");
	int ReconS_15 = ReconTimer.setNew(" RcS15_blobSoftMask ");
	int ReconS_16 = ReconTimer.setNew(" RcS16_blobNorm2 ");
	int ReconS_17 = ReconTimer.setNew(" RcS17_WindowReal ");
	int ReconS_18 = ReconTimer.setNew(" RcS18_GriddingCorrect ");
	int ReconS_19 = ReconTimer.setNew(" RcS19_tauInit ");
	int ReconS_20 = ReconTimer.setNew(" RcS20_tausetReal ");
	int ReconS_21 = ReconTimer.setNew(" RcS21_tauTransform ");
	int ReconS_22 = ReconTimer.setNew(" RcS22_tautauRest ");
	int ReconS_23 = ReconTimer.setNew(" RcS23_tauShrinkToFit ");
	int ReconS_24 = ReconTimer.setNew(" RcS24_extra ");
#endif

    // never rely on references (handed to you from the outside) for computation:
    // they could be the same (i.e. reconstruct(..., dummy, dummy, dummy, dummy, ...); )

	// process change
	int ranknum;
	int realrankarray[5];
	long int  rawvolx,rawvoly,rawvolz,rawvoln;
    //mpi table
	//for(int i=0;i<ranksize;i++)
	{
		int flag=0;
		if(realranknum == 1)
			ranknum =0;
		if(realranknum == 5)
			ranknum =1;


		if(realranknum ==2 )
		{
			ranknum =0;flag=1;
		}
		if(realranknum ==6 )
		{
			ranknum =1;flag=1;
		}
		if(flag ==0)
		{
			realrankarray[0]=1;
			realrankarray[1]=5;
		}
		if(flag==1)
		{
			realrankarray[0]=2;
			realrankarray[1]=6;
		}
		rawvolx=vol_out.xdim;
		rawvoly=vol_out.ydim;
		rawvolz=vol_out.zdim;
		rawvoln=vol_out.ndim;
	}
	printf("ranknum is %d and ranksize is %d\n",ranknum,ranksize);
	size_t Ndim[3];
	int GPU_N;
	//cudaGetDeviceCount(&GPU_N);
	GPU_N =4;
	Ndim[0]=pad_size;
	Ndim[1]=pad_size;
	Ndim[2]=pad_size;
	size_t fullsize= (size_t)pad_size*(size_t)pad_size*(size_t)pad_size;
	int sumranknum = ranksize*GPU_N ;
	int *numberZ;
	int *offsetZ;
	numberZ = (int *)malloc(sizeof(int)*sumranknum);
	offsetZ = (int *)malloc(sizeof(int)*sumranknum);

	dividetask(numberZ,offsetZ,pad_size,sumranknum);
	size_t padtrans_size = numberZ[0]*sumranknum;
	  char cmd[100];
	    memset(cmd,0,sizeof(cmd));

	int iPid = (int)getpid();
	sprintf(cmd, "cat /proc/%d/status | grep -E 'VmSize|VmData'",iPid);


	if(ranknum==0)
	{
		printf("end part1 \n");
		system(cmd);
	}
  	//end part1


	cufftComplex *cpu_data;
	cpu_data= (cufftComplex *)malloc(pad_size*padtrans_size*padtrans_size* sizeof(cufftComplex));
	double *tempdata= (double *)malloc(sizeof(double)*pad_size*pad_size*pad_size);
/*
	for(int i=0;i<fullsize;i++)
	{
		tempdata[i]=100;
	}*/


	sleep(20);
	if(ranknum==0)
	{
		printf("end part1 \n");	system(cmd);
	}

    //end part2
	//end rank change
    MultidimArray<RFLOAT> sigma2, data_vs_prior, fourier_coverage;
	MultidimArray<RFLOAT> tau2 = tau2_io;
    FourierTransformer transformer;
	MultidimArray<RFLOAT> Fweight;
	int max_r2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor);
    MultidimArray<Complex>& Fconv = transformer.getFourierReference();
	// Fnewweight can become too large for a float: always keep this one in double-precision
	MultidimArray<double> Fnewweight;


	RCTICREC(ReconTimer,ReconS_1);


//#define DEBUG_RECONSTRUCT
#ifdef DEBUG_RECONSTRUCT
	Image<RFLOAT> ttt;
	FileName fnttt;
	ttt()=weight;
	ttt.write("reconstruct_initial_weight.spi");
	std::cerr << " pad_size= " << pad_size << " padding_factor= " << padding_factor << " max_r2= " << max_r2 << std::endl;
#endif

    // Set Fweight, Fnewweight and Fconv to the right size
    if (ref_dim == 2)
        vol_out.setDimensions(pad_size, pad_size, 1, 1);
    else
        // Too costly to actually allocate the space
        // Trick transformer with the right dimensions
        vol_out.setDimensions(pad_size, pad_size, pad_size, 1);

    transformer.setReal(vol_out); // Fake set real. 1. Allocate space for Fconv 2. calculate plans.
    vol_out.clear(); // Reset dimensions to 0

    RCTOCREC(ReconTimer,ReconS_1);
    RCTICREC(ReconTimer,ReconS_2);

    Fweight.reshape(Fconv);
    if (!skip_gridding)
    	Fnewweight.reshape(Fconv);

	// Go from projector-centered to FFTW-uncentered
	decenter(weight, Fweight, max_r2);

	// Take oversampling into account
	RFLOAT oversampling_correction = (ref_dim == 3) ? (padding_factor * padding_factor * padding_factor) : (padding_factor * padding_factor);
	MultidimArray<RFLOAT> counter;

	// First calculate the radial average of the (inverse of the) power of the noise in the reconstruction
	// This is the left-hand side term in the nominator of the Wiener-filter-like update formula
	// and it is stored inside the weight vector
	// Then, if (do_map) add the inverse of tau2-spectrum values to the weight
	sigma2.initZeros(ori_size/2 + 1);
	counter.initZeros(ori_size/2 + 1);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
	{
		int r2 = kp * kp + ip * ip + jp * jp;
		if (r2 < max_r2)
		{
			int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
			RFLOAT invw = oversampling_correction * DIRECT_A3D_ELEM(Fweight, k, i, j);
			DIRECT_A1D_ELEM(sigma2, ires) += invw;
			DIRECT_A1D_ELEM(counter, ires) += 1.;
		}
    }

	// Average (inverse of) sigma2 in reconstruction
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
	{
        if (DIRECT_A1D_ELEM(sigma2, i) > 1e-10)
            DIRECT_A1D_ELEM(sigma2, i) = DIRECT_A1D_ELEM(counter, i) / DIRECT_A1D_ELEM(sigma2, i);
        else if (DIRECT_A1D_ELEM(sigma2, i) == 0)
            DIRECT_A1D_ELEM(sigma2, i) = 0.;
		else
		{
			std::cerr << " DIRECT_A1D_ELEM(sigma2, i)= " << DIRECT_A1D_ELEM(sigma2, i) << std::endl;
			REPORT_ERROR("BackProjector::reconstruct: ERROR: unexpectedly small, yet non-zero sigma2 value, this should not happen...a");
        }
    }

	if (update_tau2_with_fsc)
    {
        tau2.reshape(ori_size/2 + 1);
        data_vs_prior.initZeros(ori_size/2 + 1);
		// Then calculate new tau2 values, based on the FSC
		if (!fsc.sameShape(sigma2) || !fsc.sameShape(tau2))
		{
			fsc.printShape(std::cerr);
			tau2.printShape(std::cerr);
			sigma2.printShape(std::cerr);
			REPORT_ERROR("ERROR BackProjector::reconstruct: sigma2, tau2 and fsc have different sizes");
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
        {
			// FSC cannot be negative or zero for conversion into tau2
			RFLOAT myfsc = XMIPP_MAX(0.001, DIRECT_A1D_ELEM(fsc, i));
			if (is_whole_instead_of_half)
			{
				// Factor two because of twice as many particles
				// Sqrt-term to get 60-degree phase errors....
				myfsc = sqrt(2. * myfsc / (myfsc + 1.));
			}
			myfsc = XMIPP_MIN(0.999, myfsc);
			RFLOAT myssnr = myfsc / (1. - myfsc);
			// Sjors 29nov2017 try tau2_fudge for pulling harder on Refine3D runs...
            myssnr *= tau2_fudge;
			RFLOAT fsc_based_tau = myssnr * DIRECT_A1D_ELEM(sigma2, i);
			DIRECT_A1D_ELEM(tau2, i) = fsc_based_tau;
			// data_vs_prior is merely for reporting: it is not used for anything in the reconstruction
			DIRECT_A1D_ELEM(data_vs_prior, i) = myssnr;
		}
	}
    RCTOCREC(ReconTimer,ReconS_2);
    RCTICREC(ReconTimer,ReconS_2_5);
	// Apply MAP-additional term to the Fnewweight array
	// This will regularise the actual reconstruction
    if (do_map)
	{

    	// Then, add the inverse of tau2-spectrum values to the weight
		// and also calculate spherical average of data_vs_prior ratios
		if (!update_tau2_with_fsc)
			data_vs_prior.initZeros(ori_size/2 + 1);
		fourier_coverage.initZeros(ori_size/2 + 1);
		counter.initZeros(ori_size/2 + 1);
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
 		{
			int r2 = kp * kp + ip * ip + jp * jp;
			if (r2 < max_r2)
			{
				int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
				RFLOAT invw = DIRECT_A3D_ELEM(Fweight, k, i, j);

				RFLOAT invtau2;
				if (DIRECT_A1D_ELEM(tau2, ires) > 0.)
				{
					// Calculate inverse of tau2
					invtau2 = 1. / (oversampling_correction * tau2_fudge * DIRECT_A1D_ELEM(tau2, ires));
				}
				else if (DIRECT_A1D_ELEM(tau2, ires) == 0.)
				{
					// If tau2 is zero, use small value instead
					invtau2 = 1./ ( 0.001 * invw);
				}
				else
				{
					std::cerr << " sigma2= " << sigma2 << std::endl;
					std::cerr << " fsc= " << fsc << std::endl;
					std::cerr << " tau2= " << tau2 << std::endl;
					REPORT_ERROR("ERROR BackProjector::reconstruct: Negative or zero values encountered for tau2 spectrum!");
				}

				// Keep track of spectral evidence-to-prior ratio and remaining noise in the reconstruction
				if (!update_tau2_with_fsc)
					DIRECT_A1D_ELEM(data_vs_prior, ires) += invw / invtau2;

				// Keep track of the coverage in Fourier space
				if (invw / invtau2 >= 1.)
					DIRECT_A1D_ELEM(fourier_coverage, ires) += 1.;

				DIRECT_A1D_ELEM(counter, ires) += 1.;

				// Only for (ires >= minres_map) add Wiener-filter like term
				if (ires >= minres_map)
				{
					// Now add the inverse-of-tau2_class term
					invw += invtau2;
					// Store the new weight again in Fweight
					DIRECT_A3D_ELEM(Fweight, k, i, j) = invw;
				}
			}
		}

		// Average data_vs_prior
		if (!update_tau2_with_fsc)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(data_vs_prior)
			{
				if (i > r_max)
					DIRECT_A1D_ELEM(data_vs_prior, i) = 0.;
				else if (DIRECT_A1D_ELEM(counter, i) < 0.001)
					DIRECT_A1D_ELEM(data_vs_prior, i) = 999.;
				else
					DIRECT_A1D_ELEM(data_vs_prior, i) /= DIRECT_A1D_ELEM(counter, i);
			}
		}

		// Calculate Fourier coverage in each shell
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fourier_coverage)
		{
			if (DIRECT_A1D_ELEM(counter, i) > 0.)
				DIRECT_A1D_ELEM(fourier_coverage, i) /= DIRECT_A1D_ELEM(counter, i);
		}

	} //end if do_map
    else if (do_fsc0999)
    {

     	// Sjors 9may2018: avoid numerical instabilities with unregularised reconstructions....
        FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
        {
            int r2 = kp * kp + ip * ip + jp * jp;
            if (r2 < max_r2)
            {
                int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
                if (ires >= minres_map)
                {
                    // add 1/1000th of the radially averaged sigma2 to the Fweight, to avoid having zeros there...
                	DIRECT_A3D_ELEM(Fweight, k, i, j) += 1./(999. * DIRECT_A1D_ELEM(sigma2, ires));
                }
            }
        }

    }
	sleep(20);
	if(ranknum==0)
	{
		printf("end part3 \n");
		system(cmd);
	}
	//end part 3
	//==============================================================================add multi- GPU version




	//initgpu_mpi(GPU_N);

	//divide task
	//process divide data


	printf("numberZ[0]: %d padtrans_size %d and pad_size %d \n",numberZ[0],padtrans_size,pad_size);
	int *processnumberZ;
	int *processoffsetZ;
	processnumberZ = (int *)malloc(sizeof(int)*ranksize);
	processoffsetZ = (int *)malloc(sizeof(int)*ranksize);
	memset(processnumberZ,0,sizeof(int)*ranksize);
	memset(processoffsetZ,0,sizeof(int)*ranksize);

	dividetask(processnumberZ,processoffsetZ,padtrans_size,ranksize);

	int *prodatanumZ;
	int *prodataoffZ;
	prodatanumZ = (int *)malloc(sizeof(int)*ranksize);
	prodataoffZ = (int *)malloc(sizeof(int)*ranksize);
	memset(prodatanumZ,0,sizeof(int)*ranksize);
	memset(prodataoffZ,0,sizeof(int)*ranksize);

	dividetask(prodatanumZ,prodataoffZ,pad_size,ranksize);


	int *numbertmpZ;
	int *offsettmpZ;
	numbertmpZ = (int *)malloc(sizeof(int)*sumranknum);
	offsettmpZ = (int *)malloc(sizeof(int)*sumranknum);
	dividetask(numbertmpZ,offsettmpZ,padtrans_size,sumranknum);


	MultiGPUplan *plan;
	plan = (MultiGPUplan *)malloc(sizeof(MultiGPUplan)*GPU_N);
	multi_plan_init_transpose_multi(plan,GPU_N,numberZ,offsetZ,pad_size,ranknum,ranksize);


	for (int i = 0; i < GPU_N; ++i) {
		cudaSetDevice(plan[i].devicenum);
		cudaMalloc((void**) & (plan[i].d_Data),sizeof(cufftComplex) * plan[i].datasize);
		cudaMalloc((void**) & (plan[i].temp_Data), sizeof(cufftComplex) * plan[i].tempsize);
	}

	int xyN[2];
	xyN[0] = Ndim[0];
	xyN[1] = Ndim[1];
	cufftHandle *xyplan;
	xyplan = (cufftHandle*) malloc(sizeof(cufftHandle)*GPU_N);
	cufftHandle *zplan;
	zplan = (cufftHandle*) malloc(sizeof(cufftHandle)*GPU_N);






    RCTOCREC(ReconTimer,ReconS_2_5);
	if (skip_gridding)
	{


	    RCTICREC(ReconTimer,ReconS_3);
		std::cerr << "Skipping gridding!" << std::endl;
		Fconv.initZeros(); // to remove any stuff from the input volume
		decenter(data, Fconv, max_r2);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			if (DIRECT_MULTIDIM_ELEM(Fweight, n) > 0.)
				DIRECT_MULTIDIM_ELEM(Fconv, n) /= DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		RCTOCREC(ReconTimer,ReconS_3);
#ifdef DEBUG_RECONSTRUCT
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			DIRECT_MULTIDIM_ELEM(ttt(), n) = DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		ttt.write("reconstruct_skipgridding_correction_term.spi");
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			if (DIRECT_MULTIDIM_ELEM(Fweight, n) > 0.)
				DIRECT_MULTIDIM_ELEM(ttt(), n) = 1./DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		ttt.write("reconstruct_skipgridding_correction_term_inverse.spi");
#endif

	}
	else
	{

		RCTICREC(ReconTimer,ReconS_4);
		// Divide both data and Fweight by normalisation factor to prevent FFT's with very large values....
	#ifdef DEBUG_RECONSTRUCT
		std::cerr << " normalise= " << normalise << std::endl;
	#endif
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fweight)
		{
			DIRECT_MULTIDIM_ELEM(Fweight, n) /= normalise;
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data)
		{
			DIRECT_MULTIDIM_ELEM(data, n) /= normalise;
		}
		RCTOCREC(ReconTimer,ReconS_4);
		RCTICREC(ReconTimer,ReconS_5);
        // Initialise Fnewweight with 1's and 0's. (also see comments below)
		FOR_ALL_ELEMENTS_IN_ARRAY3D(weight)
		{
			if (k * k + i * i + j * j < max_r2)
				A3D_ELEM(weight, k, i, j) = 1.;
			else
				A3D_ELEM(weight, k, i, j) = 0.;
		}



		decenter(weight, Fnewweight, max_r2);
		RCTOCREC(ReconTimer,ReconS_5);
		// Iterative algorithm as in  Eq. [14] in Pipe & Menon (1999)
		// or Eq. (4) in Matej (2001)

		if(ranknum==0)
		{
			printf("2463 \n");
			system(cmd);
		}

		long int Fconvnum=Fconv.nzyxdim;
		sleep(20);

		long int normsize= Ndim[0]*Ndim[1]*Ndim[2];
		int halfxdim=Fconv.xdim;

		printf("%ld %ld %ld \n",Fconv.xdim,Fconv.ydim,Fconv.zdim);
		printf("Fnewweight: %ld %ld %ld \n",Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim);

		multi_enable_access(plan,GPU_N);

		for (int i = 0; i < GPU_N; i++) {
				cudaSetDevice(plan[i].devicenum);
				int reali= i+ranknum*GPU_N;
				cufftPlanMany(&xyplan[i], 2, xyN, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, numberZ[reali]);
			}

		for (int i = 0; i < GPU_N; i++) {
		    cudaSetDevice(plan[i].devicenum);
			int rank=1;
			int nrank[1];
			nrank[0]=Ndim[1];
			int inembed[2];
			inembed[0]=Ndim[0];
			inembed[1]=Ndim[1];
			cufftPlanMany(&zplan[i], rank, nrank, inembed, Ndim[0] , 1, inembed, Ndim[0], 1, CUFFT_C2C, Ndim[0]);
		}
		if(ranknum==0)
		{
			printf("After plan  \n");
			system(cmd);
		}


		float *fweightdata= (float *)malloc(sizeof(float)*pad_size*pad_size*pad_size);
		//printf("Fweight datadim: %d %d %d %d\n",Fweight.xdim,Fweight.ydim,Fweight.zdim,pad_size);
		layoutchange(Fweight.data,Fweight.xdim,Fweight.ydim,Fweight.zdim,pad_size,fweightdata);
		float **d_blocktwo;
		d_blocktwo =(float **)malloc(GPU_N*sizeof(float*));
		for(int i=0;i< GPU_N;i++)
		{
			cudaSetDevice(plan[i].devicenum);
			cudaMalloc((void**) &(d_blocktwo[i]),sizeof(float) * plan[i].realsize);
			cudaMemcpy(d_blocktwo[i],fweightdata+plan[i].selfoffset,plan[i].realsize*sizeof(float),cudaMemcpyHostToDevice);
		}
		multi_sync(plan,GPU_N);
		free(fweightdata);
		printf("skip_gridding : %d \n",skip_gridding);


		printf("Fnewweight datadim  : %d %d %d %d\n",Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim,pad_size);
		// every thread map two block to used
		layoutchange(Fnewweight.data,Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim,pad_size,tempdata);

		double **d_blockone;
		d_blockone = (double **)malloc(GPU_N*sizeof(double*));

		for (int i = 0; i < GPU_N; i++) {
			cudaSetDevice(plan[i].devicenum);
			cudaMalloc((void**) &(d_blockone[i]),sizeof(double) * plan[i].realsize);
			cudaMemcpy(d_blockone[i],tempdata+plan[i].selfoffset,plan[i].realsize*sizeof(double),cudaMemcpyHostToDevice);
		}
		multi_sync(plan,GPU_N);
		if(ranknum==0)
		{
			printf("end  \n");
			system(cmd);
		}
       //end part 4



		printf("Start calc \n ");
		for (int iter = 0; iter < max_iter_preweight; iter++)
		{

			RCTICREC(ReconTimer,ReconS_6);

			for (int i = 0; i < GPU_N; i++) {
				cudaSetDevice(plan[i].devicenum);
				vector_Multi_layout_mpi(d_blockone[i],d_blocktwo[i],plan[i].d_Data, plan[i].realsize);
				cufftExecC2C(xyplan[i], plan[i].d_Data,plan[i].d_Data , CUFFT_INVERSE);
				cudaDeviceSynchronize();
			}
			if(ranknum==1)
			{
				cudaMemcpy(cpu_data,plan[0].d_Data,pad_size*pad_size*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

				printf("1\n");
				for(int i=0;i<pad_size;i++)
					//for(int j=0;j<pad_size;j++)
					printf("%f ",cpu_data[i*pad_size].x);
				printf("\n");
			}

			yzlocal_transpose_multicard(plan,GPU_N,pad_size,offsetZ,numberZ,offsettmpZ,ranknum,sumranknum);
			transpose_exchange_intra(plan,GPU_N,pad_size,offsetZ,numberZ,offsettmpZ,ranknum);
			memset(cpu_data,0,pad_size*padtrans_size*padtrans_size* sizeof(cufftComplex));
			data_exchange_gputocpu(plan,cpu_data,GPU_N,pad_size,offsettmpZ,ranknum);
			cpu_alltoalltrans_multinode(plan,cpu_data,pad_size,processnumberZ,processoffsetZ,ranknum,padtrans_size,ranksize,realrankarray);
			data_exchange_cputogpu(plan, cpu_data, GPU_N, pad_size,offsetZ,numberZ, offsettmpZ, numbertmpZ, ranknum,sumranknum,padtrans_size);


			if(ranknum==0)
			{
				cudaMemcpy(cpu_data,plan[0].d_Data,pad_size*pad_size*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

				printf("2\n");
				for(int i=0;i<pad_size;i++)
					//for(int j=0;j<pad_size;j++)
					printf("%f ",cpu_data[i*pad_size].x);
				printf("\n");
			}

			for (int i = 0; i < GPU_N; i++) {
				cudaSetDevice(plan[i].devicenum);
				int reali= i+ranknum*GPU_N;
				size_t offset = 0;
				for(int zslice=0;zslice<numberZ[reali];zslice++)
				{
					cufftExecC2C(zplan[i], plan[i].d_Data+offset, plan[i].d_Data+offset, CUFFT_INVERSE);
					offset+= pad_size*pad_size;
				}
			}
			cudaDeviceSynchronize();


			if(ranknum==0)
			{
				cudaMemcpy(cpu_data,plan[0].d_Data,pad_size*pad_size*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
				printf("3\n");
					for(int i=0;i<pad_size;i++)
						//for(int j=0;j<pad_size;j++)
						printf("%f ",cpu_data[i*pad_size].x);
					printf("\n");
			}


			yzlocal_transpose_multicard(plan,GPU_N,pad_size,offsetZ,numberZ,offsettmpZ,ranknum,sumranknum);
			transpose_exchange_intra(plan,GPU_N,pad_size,offsetZ,numberZ,offsettmpZ,ranknum);
			memset(cpu_data,0,pad_size*padtrans_size*padtrans_size* sizeof(cufftComplex));
			data_exchange_gputocpu(plan,cpu_data,GPU_N,pad_size,offsettmpZ,ranknum);
			cpu_alltoalltrans_multinode(plan,cpu_data,pad_size,processnumberZ,processoffsetZ,ranknum,padtrans_size,ranksize,realrankarray);
			data_exchange_cputogpu(plan, cpu_data, GPU_N, pad_size,offsetZ,numberZ, offsettmpZ, numbertmpZ, ranknum,sumranknum,padtrans_size);




			printf("Norm blob \n ");
			RFLOAT normftblob = tab_ftblob(0.);
			float *d_tab_ftblob;
			int tabxdim=tab_ftblob.tabulatedValues.xdim;



			for(int i = 0; i < GPU_N; i++)
			{
				cudaSetDevice(plan[i].devicenum);
				d_tab_ftblob=gpusetdata_float(d_tab_ftblob,tab_ftblob.tabulatedValues.xdim,tab_ftblob.tabulatedValues.data);
				int reali= i+ranknum*GPU_N;
				volume_Multi_float_transone(plan[i].d_Data,d_tab_ftblob, Ndim[0]*Ndim[1]*numberZ[reali],
									tabxdim, tab_ftblob.sampling , pad_size/2, pad_size, ori_size, padding_factor, normftblob,
									Ndim[1],offsetZ[reali]);
			}
			cudaDeviceSynchronize();

			if(ranknum==0)
			{
				cudaMemcpy(cpu_data,plan[0].d_Data,pad_size*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
				for(int i=0;i<10;i++)
					printf("%f ",cpu_data[i].x);
			}

			printf("\n");

			for (int i = 0; i < GPU_N; i++) {
				cudaSetDevice(plan[i].devicenum);
				cufftExecC2C(xyplan[i], plan[i].d_Data,plan[i].d_Data , CUFFT_FORWARD);
				cudaDeviceSynchronize();
			}
			yzlocal_transpose_multicard(plan,GPU_N,pad_size,offsetZ,numberZ,offsettmpZ,ranknum,sumranknum);
			transpose_exchange_intra(plan,GPU_N,pad_size,offsetZ,numberZ,offsettmpZ,ranknum);
			memset(cpu_data,0,pad_size*padtrans_size*padtrans_size* sizeof(cufftComplex));
			data_exchange_gputocpu(plan,cpu_data,GPU_N,pad_size,offsettmpZ,ranknum);
			cpu_alltoalltrans_multinode(plan,cpu_data,pad_size,processnumberZ,processoffsetZ,ranknum,padtrans_size,ranksize,realrankarray);
			data_exchange_cputogpu(plan, cpu_data, GPU_N, pad_size,offsetZ,numberZ, offsettmpZ, numbertmpZ, ranknum,sumranknum,padtrans_size);

//			if(ranknum==0)
//			{
//				cudaMemcpy(cpu_data,plan[0].d_Data,pad_size*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
//				for(int i=0;i<10;i++)
//					printf("%f ",cpu_data[i].x);
//			}
//			printf("\n");
			for (int i = 0; i < GPU_N; i++) {
				cudaSetDevice(plan[i].devicenum);
				int reali= i+ranknum*GPU_N;
				size_t offset = 0;
				for(int zslice=0;zslice<numberZ[reali];zslice++)
				{
					cufftExecC2C(zplan[i], plan[i].d_Data+offset, plan[i].d_Data+offset, CUFFT_FORWARD);
					offset+= pad_size*pad_size;
				}
				cudaDeviceSynchronize();
			}
			yzlocal_transpose_multicard(plan,GPU_N,pad_size,offsetZ,numberZ,offsettmpZ,ranknum,sumranknum);
			transpose_exchange_intra(plan,GPU_N,pad_size,offsetZ,numberZ,offsettmpZ,ranknum);
			memset(cpu_data,0,pad_size*padtrans_size*padtrans_size* sizeof(cufftComplex));
			data_exchange_gputocpu(plan,cpu_data,GPU_N,pad_size,offsettmpZ,ranknum);
			cpu_alltoalltrans_multinode(plan,cpu_data,pad_size,processnumberZ,processoffsetZ,ranknum,padtrans_size,ranksize,realrankarray);
			data_exchange_cputogpu(plan, cpu_data, GPU_N, pad_size,offsetZ,numberZ, offsettmpZ, numbertmpZ, ranknum,sumranknum,padtrans_size);

//			if(ranknum==0)
//			{
//				cudaMemcpy(cpu_data,plan[0].d_Data,pad_size*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
//				for(int i=0;i<10;i++)
//					printf("%f ",cpu_data[i].x);
//			}
//			printf("\n");
//			printf("fft_Divide_mpi\n ");

			for (int i = 0; i < GPU_N; i++) {
				cudaSetDevice(plan[i].devicenum);
				int reali= i+ranknum*GPU_N;
				vector_Normlize(plan[i].d_Data, normsize ,Ndim[0]*Ndim[1]*numberZ[reali]);
				//gpu_kernel3
				fft_Divide_mpi(plan[i].d_Data,d_blockone[i],plan[i].realsize,pad_size*pad_size,pad_size,pad_size,pad_size,Fconv.xdim,max_r2,offsetZ[reali]);
			}

			RCTOCREC(ReconTimer,ReconS_6);

	#ifdef DEBUG_RECONSTRUCT
			std::cerr << " PREWEIGHTING ITERATION: "<< iter + 1 << " OF " << max_iter_preweight << std::endl;
			// report of maximum and minimum values of current conv_weight
			std::cerr << " corr_avg= " << corr_avg / corr_nn << std::endl;
			std::cerr << " corr_min= " << corr_min << std::endl;
			std::cerr << " corr_max= " << corr_max << std::endl;
	#endif

		}

		printf("End iter \n ");
		for(int i=0;i<GPU_N; i++){
			cudaSetDevice(plan[i].devicenum);
			cudaMemcpy(tempdata+plan[i].selfoffset,d_blockone[i],plan[i].realsize*sizeof(double),cudaMemcpyDeviceToHost);
		}

		printf("before temp \n ");
		if(ranknum!=0)
		{
			MPI_Send(tempdata+pad_size*pad_size*prodataoffZ[ranknum],pad_size*pad_size*prodatanumZ[ranknum],MPI_DOUBLE,realrankarray[0],0,MPI_COMM_WORLD);
		}
		else
		{
			for(int i=1;i<ranksize;i++)
				MPI_Recv(tempdata+(pad_size*pad_size*prodataoffZ[i]),pad_size*pad_size*prodatanumZ[i],MPI_DOUBLE,realrankarray[i],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			layoutchangeback(tempdata,Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim,pad_size,Fnewweight.data);
		}
		printf("After tmp \n ");
		if(ranknum==0)
		{
			for(int i=1;i<ranksize;i++)
				MPI_Send(Fnewweight.data,Fnewweight.nzyxdim,MPI_DOUBLE,realrankarray[i],0,MPI_COMM_WORLD);
		}
		else
		{
			MPI_Recv(Fnewweight.data,Fnewweight.nzyxdim,MPI_DOUBLE,realrankarray[0],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}

		printf("before free\n ");

		for (int i = 0; i < GPU_N; i++) {
			cudaSetDevice(plan[i].devicenum);
			cufftDestroy(xyplan[i]);
			cufftDestroy(zplan[i]);
			cudaFree(d_blockone[i]);
			cudaFree(d_blocktwo[i]);
		}

		free(tempdata);
		free(d_blockone);
		free(d_blocktwo);

		for(int i=0;i<10;i++)
				printf("%f ",Fnewweight.data[i]);

		RCTICREC(ReconTimer,ReconS_7);
	#ifdef DEBUG_RECONSTRUCT
		Image<double> tttt;
		tttt()=Fnewweight;
		tttt.write("reconstruct_gridding_weight.spi");
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			DIRECT_MULTIDIM_ELEM(ttt(), n) = abs(DIRECT_MULTIDIM_ELEM(Fconv, n));
		}
		ttt.write("reconstruct_gridding_correction_term.spi");
	#endif


		// Clear memory
		Fweight.clear();

		// Note that Fnewweight now holds the approximation of the inverse of the weights on a regular grid

		// Now do the actual reconstruction with the data array
		// Apply the iteratively determined weight
		Fconv.initZeros(); // to remove any stuff from the input volume
		decenter(data, Fconv, max_r2);
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
#ifdef  RELION_SINGLE_PRECISION
			// Prevent numerical instabilities in single-precision reconstruction with very unevenly sampled orientations
			if (DIRECT_MULTIDIM_ELEM(Fnewweight, n) > 1e20)
				DIRECT_MULTIDIM_ELEM(Fnewweight, n) = 1e20;
#endif
			DIRECT_MULTIDIM_ELEM(Fconv, n) *= DIRECT_MULTIDIM_ELEM(Fnewweight, n);
		}

		// Clear memory
		Fnewweight.clear();
		RCTOCREC(ReconTimer,ReconS_7);
	} // end if skip_gridding


// Gridding theory says one now has to interpolate the fine grid onto the coarse one using a blob kernel
// and then do the inverse transform and divide by the FT of the blob (i.e. do the gridding correction)
// In practice, this gives all types of artefacts (perhaps I never found the right implementation?!)
// Therefore, window the Fourier transform and then do the inverse transform
//#define RECONSTRUCT_CONVOLUTE_BLOB
#ifdef RECONSTRUCT_CONVOLUTE_BLOB

	// Apply the same blob-convolution as above to the data array
	// Mask real-space map beyond its original size to prevent aliasing in the downsampling step below
	RCTICREC(ReconTimer,ReconS_8);
	convoluteBlobRealSpace(transformer, true);
	RCTOCREC(ReconTimer,ReconS_8);
	RCTICREC(ReconTimer,ReconS_9);
	// Now just pick every 3rd pixel in Fourier-space (i.e. down-sample)
	// and do a final inverse FT
	if (ref_dim == 2)
		vol_out.resize(ori_size, ori_size);
	else
		vol_out.resize(ori_size, ori_size, ori_size);
	RCTOCREC(ReconTimer,ReconS_9);
	RCTICREC(ReconTimer,ReconS_10);
	FourierTransformer transformer2;
	MultidimArray<Complex > Ftmp;
	transformer2.setReal(vol_out); // cannot use the first transformer because Fconv is inside there!!
	transformer2.getFourierAlias(Ftmp);
	RCTOCREC(ReconTimer,ReconS_10);
	RCTICREC(ReconTimer,ReconS_11);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Ftmp)
	{
		if (kp * kp + ip * ip + jp * jp < r_max * r_max)
		{
			DIRECT_A3D_ELEM(Ftmp, k, i, j) = FFTW_ELEM(Fconv, kp * padding_factor, ip * padding_factor, jp * padding_factor);
		}
		else
		{
			DIRECT_A3D_ELEM(Ftmp, k, i, j) = 0.;
		}
	}
	RCTOCREC(ReconTimer,ReconS_11);
	RCTICREC(ReconTimer,ReconS_12);
	// inverse FFT leaves result in vol_out
	transformer2.inverseFourierTransform();
	RCTOCREC(ReconTimer,ReconS_12);
	RCTICREC(ReconTimer,ReconS_13);
	// Shift the map back to its origin
	CenterFFT(vol_out, false);
	RCTOCREC(ReconTimer,ReconS_13);
	RCTICREC(ReconTimer,ReconS_14);
	// Un-normalize FFTW (because original FFTs were done with the size of 2D FFTs)
	if (ref_dim==3)
		vol_out /= ori_size;
	RCTOCREC(ReconTimer,ReconS_14);
	RCTICREC(ReconTimer,ReconS_15);
	// Mask out corners to prevent aliasing artefacts
	softMaskOutsideMap(vol_out);
	RCTOCREC(ReconTimer,ReconS_15);
	RCTICREC(ReconTimer,ReconS_16);
	// Gridding correction for the blob
	RFLOAT normftblob = tab_ftblob(0.);
	FOR_ALL_ELEMENTS_IN_ARRAY3D(vol_out)
	{

		RFLOAT r = sqrt((RFLOAT)(k*k+i*i+j*j));
		RFLOAT rval = r / (ori_size * padding_factor);
		A3D_ELEM(vol_out, k, i, j) /= tab_ftblob(rval) / normftblob;
		//if (k==0 && i==0)
		//	std::cerr << " j= " << j << " rval= " << rval << " tab_ftblob(rval) / normftblob= " << tab_ftblob(rval) / normftblob << std::endl;
	}
	RCTOCREC(ReconTimer,ReconS_16);

#else

	// rather than doing the blob-convolution to downsample the data array, do a windowing operation:
	// This is the same as convolution with a SINC. It seems to give better maps.
	// Then just make the blob look as much as a SINC as possible....
	// The "standard" r1.9, m2 and a15 blob looks quite like a sinc until the first zero (perhaps that's why it is standard?)
	//for (RFLOAT r = 0.1; r < 10.; r+=0.01)
	//{
	//	RFLOAT sinc = sin(PI * r / padding_factor ) / ( PI * r / padding_factor);
	//	std::cout << " r= " << r << " sinc= " << sinc << " blob= " << blob_val(r, blob) << std::endl;
	//}

	// Now do inverse FFT and window to original size in real-space
	// Pass the transformer to prevent making and clearing a new one before clearing the one declared above....
	// The latter may give memory problems as detected by electric fence....
	RCTICREC(ReconTimer,ReconS_17);
#endif

	if(ranknum==0)
	{
		system(cmd);
	}

	// Size of padded real-space volume
	size_t padoridim = ROUND(padding_factor * ori_size);
	// make sure padoridim is even
	padoridim += padoridim%2;

	fullsize = padoridim *padoridim*padoridim;

	dividetask(numberZ,offsetZ,padoridim,sumranknum);
	padtrans_size = numberZ[0]*sumranknum;
	multi_plan_init_transpose_multi(plan,GPU_N,numberZ,offsetZ,padoridim,ranknum,ranksize);

	dividetask(numbertmpZ,offsettmpZ,padtrans_size,sumranknum);
	dividetask(processnumberZ,processoffsetZ,padtrans_size,ranksize);

	if(padoridim > pad_size)
	{
		for (int i = 0; i < GPU_N; ++i) {
			cudaSetDevice(plan[i].devicenum);
			cudaFree((plan[i].d_Data));
			cudaFree((plan[i].temp_Data));
			cudaMalloc((void**) &(plan[i].d_Data),sizeof(cufftComplex) * plan[i].datasize);
			cudaMalloc((void**) &(plan[i].temp_Data),sizeof(cufftComplex) * plan[i].tempsize);
		}

		cpu_data= (cufftComplex *)realloc(cpu_data,padoridim*padtrans_size*padtrans_size * sizeof(cufftComplex));
	}


	if(ranknum==0)
	{
		system(cmd);
	}

	cufftComplex *cpu_data_real;
	cpu_data_real= (cufftComplex *)malloc(padoridim*padoridim*padoridim * sizeof(cufftComplex));
	printf("%lld %lld \n",padoridim*padtrans_size*padtrans_size,fullsize);

	if(ranknum==0)
	{
		system(cmd);
	}

	if(padoridim*padtrans_size*padtrans_size >= fullsize)
	{
		windowFourierTransform(Fconv, padoridim);
		layoutchangecomp(Fconv.data,Fconv.xdim,Fconv.ydim,Fconv.zdim,padoridim,cpu_data);
	}
	else
	{
		printf("Need debug for fullsize with padsize");
	}


//init plan and for even data


	xyN[0] = padoridim;
	xyN[1] = padoridim;
	Ndim[0]=padoridim;Ndim[1]=padoridim;Ndim[2]=padoridim;
    //==================2d fft plan

	for (int i = 0; i < GPU_N; i++) {
	    cudaSetDevice(plan[i].devicenum);
	    int reali= i+ranknum*GPU_N;
		cufftPlanMany(&xyplan[i], 2, xyN, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, numberZ[reali]);
	}

	for (int i = 0; i < GPU_N; i++) {
	    cudaSetDevice(plan[i].devicenum);
		int rank=1;
		int nrank[1];
		nrank[0]=Ndim[1];
		int inembed[2];
		inembed[0]=Ndim[0];
		inembed[1]=Ndim[1];
		cufftPlanMany(&zplan[i], rank, nrank, inembed, Ndim[0] , 1, inembed, Ndim[0], 1, CUFFT_C2C, Ndim[0]);
	}
	for(int i=0;i < GPU_N; i++)
	{
		cudaSetDevice(plan[i].devicenum);
		cudaMemcpy(plan[i].d_Data , cpu_data + plan[i].selfoffset, (plan[i].datasize) * sizeof(cufftComplex),cudaMemcpyHostToDevice);
	}

	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(plan[i].devicenum);
		cufftExecC2C(xyplan[i], plan[i].d_Data ,plan[i].d_Data , CUFFT_INVERSE);
	}
	multi_sync(plan,GPU_N);

	printf("%d %d padtrans_size: %d \n",padoridim,pad_size,padtrans_size);

	yzlocal_transpose_multicard(plan,GPU_N,padoridim,offsetZ,numberZ,offsettmpZ,ranknum,sumranknum);
	transpose_exchange_intra(plan,GPU_N,padoridim,offsetZ,numberZ,offsettmpZ,ranknum);
	memset(cpu_data,0,padoridim*padtrans_size*padtrans_size* sizeof(cufftComplex));
	data_exchange_gputocpu(plan,cpu_data,GPU_N,padoridim,offsettmpZ,ranknum);
	cpu_alltoalltrans_multinode(plan,cpu_data,padoridim,processnumberZ,processoffsetZ,ranknum,padtrans_size,ranksize,realrankarray);
	data_exchange_cputogpu(plan, cpu_data, GPU_N, padoridim,offsetZ,numberZ, offsettmpZ, numbertmpZ, ranknum,sumranknum,padtrans_size);

	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(plan[i].devicenum);
		int reali= i+ranknum*GPU_N;
		size_t offset = 0;
		for(int zslice=0;zslice<numberZ[reali];zslice++)
		{
			cufftExecC2C(zplan[i], plan[i].d_Data+offset, plan[i].d_Data+offset, CUFFT_INVERSE);
			offset+= padoridim*padoridim;
		}
	}
	if(ranknum==0)
	{
		system(cmd);
	}

	yzlocal_transpose_multicard(plan,GPU_N,padoridim,offsetZ,numberZ,offsettmpZ,ranknum,sumranknum);
	memset(cpu_data,0,padoridim*padtrans_size*padtrans_size* sizeof(cufftComplex));
	data_exchange_gputocpu(plan,cpu_data,GPU_N,padoridim,offsettmpZ,ranknum);
	//cpu_alltoalltrans_multinode(plan,cpu_data,padoridim,processnumberZ,processoffsetZ,ranknum,padtrans_size,ranksize,realrankarray);
	cpu_alltozero_multinode(cpu_data,padoridim, processnumberZ,processoffsetZ,ranknum, padtrans_size, ranksize,realrankarray);

	if(ranknum == 0)
		cpu_datarearrange_multinode(cpu_data,cpu_data_real, padoridim,  numberZ, offsetZ,numbertmpZ, offsettmpZ,  ranknum, padtrans_size, sumranknum);

	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(plan[i].devicenum);
		cufftDestroy(zplan[i]);
		cufftDestroy(xyplan[i]);

		cudaFree(plan[i].d_Data);
		cudaFree(plan[i].temp_Data);
		cudaDeviceSynchronize();
	}
	free(xyplan);
	free(zplan);
    free(cpu_data);
	if(ranknum==0)
	{
		system(cmd);
	}

	if(ranknum == 0)
	{
	    vol_out.reshape(padoridim, padoridim, padoridim);
	    vol_out.setXmippOrigin();
	//copy data
	//	memcpy(vol_out.data,cpu_data,sizeof(float)*padoridim *padoridim*padoridim);
    for(size_t i=0;i<padoridim *padoridim*padoridim;i++)
    	vol_out.data[i]=cpu_data_real[i].x;


    system(cmd);
	printwhole(vol_out.data, 10 ,ranknum);

    CenterFFT(vol_out,true);

	// Window in real-space
	if (ref_dim==2)
	{
		vol_out.window(FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size),
				       LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size));
	}
	else
	{
		vol_out.window(FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size), FIRST_XMIPP_INDEX(ori_size),
				       LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size), LAST_XMIPP_INDEX(ori_size));
	}
	vol_out.setXmippOrigin();

	// Normalisation factor of FFTW
	// The Fourier Transforms are all "normalised" for 2D transforms of size = ori_size x ori_size
	float normfft = (RFLOAT)(padding_factor * padding_factor * padding_factor * ori_size);
	vol_out /= normfft;
	// Mask out corners to prevent aliasing artefacts
	softMaskOutsideMap(vol_out);
	//windowToOridimRealSpace(transformer, vol_out, nr_threads, printTimes);
	RCTOCREC(ReconTimer,ReconS_17);
	}

    free(cpu_data_real); // all process need


#ifdef DEBUG_RECONSTRUCT
	ttt()=vol_out;
	ttt.write("reconstruct_before_gridding_correction.spi");
#endif

	if(ranknum == 0)
	{
	// Correct for the linear/nearest-neighbour interpolation that led to the data array
	RCTICREC(ReconTimer,ReconS_18);
	griddingCorrect(vol_out);
	RCTOCREC(ReconTimer,ReconS_18);
	// If the tau-values were calculated based on the FSC, then now re-calculate the power spectrum of the actual reconstruction
	if (update_tau2_with_fsc)
	{

		// New tau2 will be the power spectrum of the new map
		MultidimArray<RFLOAT> spectrum, count;

		// Calculate this map's power spectrum
		// Don't call getSpectrum() because we want to use the same transformer object to prevent memory trouble....
		RCTICREC(ReconTimer,ReconS_19);
		spectrum.initZeros(XSIZE(vol_out));
	    count.initZeros(XSIZE(vol_out));
		RCTOCREC(ReconTimer,ReconS_19);
		RCTICREC(ReconTimer,ReconS_20);
	    // recycle the same transformer for all images
        transformer.setReal(vol_out);
		RCTOCREC(ReconTimer,ReconS_20);
		RCTICREC(ReconTimer,ReconS_21);
        transformer.FourierTransform();
		RCTOCREC(ReconTimer,ReconS_21);
		RCTICREC(ReconTimer,ReconS_22);
	    FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
	    {
	    	long int idx = ROUND(sqrt(kp*kp + ip*ip + jp*jp));
	    	spectrum(idx) += norm(dAkij(Fconv, k, i, j));
	        count(idx) += 1.;
	    }
	    spectrum /= count;

		// Factor two because of two-dimensionality of the complex plane
		// (just like sigma2_noise estimates, the power spectra should be divided by 2)
		RFLOAT normfft = (ref_dim == 3 && data_dim == 2) ? (RFLOAT)(ori_size * ori_size) : 1.;
		spectrum *= normfft / 2.;

		// New SNR^MAP will be power spectrum divided by the noise in the reconstruction (i.e. sigma2)
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data_vs_prior)
		{
			DIRECT_MULTIDIM_ELEM(tau2, n) =  tau2_fudge * DIRECT_MULTIDIM_ELEM(spectrum, n);
		}
		RCTOCREC(ReconTimer,ReconS_22);
	}
	RCTICREC(ReconTimer,ReconS_23);
	// Completely empty the transformer object
	transformer.cleanup();
    // Now can use extra mem to move data into smaller array space
    vol_out.shrinkToFit();

	RCTOCREC(ReconTimer,ReconS_23);
#ifdef TIMINGREC
    if(printTimes)
    	ReconTimer.printTimes(true);
#endif


#ifdef DEBUG_RECONSTRUCT
    std::cerr<<"done with reconstruct"<<std::endl;
#endif

	tau2_io = tau2;
    sigma2_out = sigma2;
    data_vs_prior_out = data_vs_prior;
    fourier_coverage_out = fourier_coverage;
	}
	else
	{
		vol_out.clear();
		vol_out.resize(rawvoln,rawvolz,rawvoly,rawvolx);
		//vol_out.reshape(rawvoln,rawvolz,rawvoly,rawvolx);
	}

}

void BackProjector::reconstruct_test(MultidimArray<RFLOAT> &vol_out,
                                int max_iter_preweight,
                                bool do_map,
                                RFLOAT tau2_fudge,
                                MultidimArray<RFLOAT> &tau2_io, // can be input/output
                                MultidimArray<RFLOAT> &sigma2_out,
                                MultidimArray<RFLOAT> &data_vs_prior_out,
                                MultidimArray<RFLOAT> &fourier_coverage_out,
                                const MultidimArray<RFLOAT> &fsc, // only input
                                RFLOAT normalise,
                                bool update_tau2_with_fsc,
                                bool is_whole_instead_of_half,
                                int nr_threads,
                                int minres_map,
                                bool printTimes,
								bool do_fsc0999,
								int realranknum,
								int ranksize)
{

#ifdef TIMINGREC
	Timer ReconTimer;
	printTimes=1;
	int ReconS_1 = ReconTimer.setNew(" RcS1_Init ");
	int ReconS_2 = ReconTimer.setNew(" RcS2_Shape&Noise ");
	int ReconS_2_5 = ReconTimer.setNew(" RcS2.5_Regularize ");
	int ReconS_3 = ReconTimer.setNew(" RcS3_skipGridding ");
	int ReconS_4 = ReconTimer.setNew(" RcS4_doGridding_norm ");
	int ReconS_5 = ReconTimer.setNew(" RcS5_doGridding_init ");
	int ReconS_6 = ReconTimer.setNew(" RcS6_doGridding_iter ");
	int ReconS_7 = ReconTimer.setNew(" RcS7_doGridding_apply ");
	int ReconS_8 = ReconTimer.setNew(" RcS8_blobConvolute ");
	int ReconS_9 = ReconTimer.setNew(" RcS9_blobResize ");
	int ReconS_10 = ReconTimer.setNew(" RcS10_blobSetReal ");
	int ReconS_11 = ReconTimer.setNew(" RcS11_blobSetTemp ");
	int ReconS_12 = ReconTimer.setNew(" RcS12_blobTransform ");
	int ReconS_13 = ReconTimer.setNew(" RcS13_blobCenterFFT ");
	int ReconS_14 = ReconTimer.setNew(" RcS14_blobNorm1 ");
	int ReconS_15 = ReconTimer.setNew(" RcS15_blobSoftMask ");
	int ReconS_16 = ReconTimer.setNew(" RcS16_blobNorm2 ");
	int ReconS_17 = ReconTimer.setNew(" RcS17_WindowReal ");
	int ReconS_18 = ReconTimer.setNew(" RcS18_GriddingCorrect ");
	int ReconS_19 = ReconTimer.setNew(" RcS19_tauInit ");
	int ReconS_20 = ReconTimer.setNew(" RcS20_tausetReal ");
	int ReconS_21 = ReconTimer.setNew(" RcS21_tauTransform ");
	int ReconS_22 = ReconTimer.setNew(" RcS22_tautauRest ");
	int ReconS_23 = ReconTimer.setNew(" RcS23_tauShrinkToFit ");
	int ReconS_24 = ReconTimer.setNew(" RcS24_extra ");
#endif

    // never rely on references (handed to you from the outside) for computation:
    // they could be the same (i.e. reconstruct(..., dummy, dummy, dummy, dummy, ...); )

	// process change
	int ranknum;
	int realrankarray[5];
	long int  rawvolx,rawvoly,rawvolz,rawvoln;
    //mpi table
	//for(int i=0;i<ranksize;i++)
	{
		int flag=0;
		if(realranknum == 1)
			ranknum =0;
		if(realranknum == 5)
			ranknum =1;


		if(realranknum ==2 )
		{
			ranknum =0;flag=1;
		}
		if(realranknum ==6 )
		{
			ranknum =1;flag=1;
		}
		if(flag ==0)
		{
			realrankarray[0]=1;
			realrankarray[1]=5;
		}
		if(flag==1)
		{
			realrankarray[0]=2;
			realrankarray[1]=6;
		}
		rawvolx=vol_out.xdim;
		rawvoly=vol_out.ydim;
		rawvolz=vol_out.zdim;
		rawvoln=vol_out.ndim;
	}
	printf("ranknum is %d and ranksize is %d\n",ranknum,ranksize);
	//end rank change
    MultidimArray<RFLOAT> sigma2, data_vs_prior, fourier_coverage;
	MultidimArray<RFLOAT> tau2 = tau2_io;
    FourierTransformer transformer;
	MultidimArray<RFLOAT> Fweight;
	int max_r2 = ROUND(r_max * padding_factor) * ROUND(r_max * padding_factor);
    MultidimArray<Complex>& Fconv = transformer.getFourierReference();
	// Fnewweight can become too large for a float: always keep this one in double-precision
	MultidimArray<double> Fnewweight;


	RCTICREC(ReconTimer,ReconS_1);


//#define DEBUG_RECONSTRUCT
#ifdef DEBUG_RECONSTRUCT
	Image<RFLOAT> ttt;
	FileName fnttt;
	ttt()=weight;
	ttt.write("reconstruct_initial_weight.spi");
	std::cerr << " pad_size= " << pad_size << " padding_factor= " << padding_factor << " max_r2= " << max_r2 << std::endl;
#endif

    // Set Fweight, Fnewweight and Fconv to the right size
    if (ref_dim == 2)
        vol_out.setDimensions(pad_size, pad_size, 1, 1);
    else
        // Too costly to actually allocate the space
        // Trick transformer with the right dimensions
        vol_out.setDimensions(pad_size, pad_size, pad_size, 1);

    transformer.setReal(vol_out); // Fake set real. 1. Allocate space for Fconv 2. calculate plans.
    vol_out.clear(); // Reset dimensions to 0

    RCTOCREC(ReconTimer,ReconS_1);
    RCTICREC(ReconTimer,ReconS_2);

    Fweight.reshape(Fconv);
    if (!skip_gridding)
    	Fnewweight.reshape(Fconv);

	// Go from projector-centered to FFTW-uncentered
	decenter(weight, Fweight, max_r2);

	// Take oversampling into account
	RFLOAT oversampling_correction = (ref_dim == 3) ? (padding_factor * padding_factor * padding_factor) : (padding_factor * padding_factor);
	MultidimArray<RFLOAT> counter;

	// First calculate the radial average of the (inverse of the) power of the noise in the reconstruction
	// This is the left-hand side term in the nominator of the Wiener-filter-like update formula
	// and it is stored inside the weight vector
	// Then, if (do_map) add the inverse of tau2-spectrum values to the weight
	sigma2.initZeros(ori_size/2 + 1);
	counter.initZeros(ori_size/2 + 1);
	FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
	{
		int r2 = kp * kp + ip * ip + jp * jp;
		if (r2 < max_r2)
		{
			int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
			RFLOAT invw = oversampling_correction * DIRECT_A3D_ELEM(Fweight, k, i, j);
			DIRECT_A1D_ELEM(sigma2, ires) += invw;
			DIRECT_A1D_ELEM(counter, ires) += 1.;
		}
    }

	// Average (inverse of) sigma2 in reconstruction
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
	{
        if (DIRECT_A1D_ELEM(sigma2, i) > 1e-10)
            DIRECT_A1D_ELEM(sigma2, i) = DIRECT_A1D_ELEM(counter, i) / DIRECT_A1D_ELEM(sigma2, i);
        else if (DIRECT_A1D_ELEM(sigma2, i) == 0)
            DIRECT_A1D_ELEM(sigma2, i) = 0.;
		else
		{
			std::cerr << " DIRECT_A1D_ELEM(sigma2, i)= " << DIRECT_A1D_ELEM(sigma2, i) << std::endl;
			REPORT_ERROR("BackProjector::reconstruct: ERROR: unexpectedly small, yet non-zero sigma2 value, this should not happen...a");
        }
    }

	if (update_tau2_with_fsc)
    {
        tau2.reshape(ori_size/2 + 1);
        data_vs_prior.initZeros(ori_size/2 + 1);
		// Then calculate new tau2 values, based on the FSC
		if (!fsc.sameShape(sigma2) || !fsc.sameShape(tau2))
		{
			fsc.printShape(std::cerr);
			tau2.printShape(std::cerr);
			sigma2.printShape(std::cerr);
			REPORT_ERROR("ERROR BackProjector::reconstruct: sigma2, tau2 and fsc have different sizes");
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(sigma2)
        {
			// FSC cannot be negative or zero for conversion into tau2
			RFLOAT myfsc = XMIPP_MAX(0.001, DIRECT_A1D_ELEM(fsc, i));
			if (is_whole_instead_of_half)
			{
				// Factor two because of twice as many particles
				// Sqrt-term to get 60-degree phase errors....
				myfsc = sqrt(2. * myfsc / (myfsc + 1.));
			}
			myfsc = XMIPP_MIN(0.999, myfsc);
			RFLOAT myssnr = myfsc / (1. - myfsc);
			// Sjors 29nov2017 try tau2_fudge for pulling harder on Refine3D runs...
            myssnr *= tau2_fudge;
			RFLOAT fsc_based_tau = myssnr * DIRECT_A1D_ELEM(sigma2, i);
			DIRECT_A1D_ELEM(tau2, i) = fsc_based_tau;
			// data_vs_prior is merely for reporting: it is not used for anything in the reconstruction
			DIRECT_A1D_ELEM(data_vs_prior, i) = myssnr;
		}
	}
    RCTOCREC(ReconTimer,ReconS_2);
    RCTICREC(ReconTimer,ReconS_2_5);
	// Apply MAP-additional term to the Fnewweight array
	// This will regularise the actual reconstruction
    if (do_map)
	{

    	// Then, add the inverse of tau2-spectrum values to the weight
		// and also calculate spherical average of data_vs_prior ratios
		if (!update_tau2_with_fsc)
			data_vs_prior.initZeros(ori_size/2 + 1);
		fourier_coverage.initZeros(ori_size/2 + 1);
		counter.initZeros(ori_size/2 + 1);
		FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
 		{
			int r2 = kp * kp + ip * ip + jp * jp;
			if (r2 < max_r2)
			{
				int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
				RFLOAT invw = DIRECT_A3D_ELEM(Fweight, k, i, j);

				RFLOAT invtau2;
				if (DIRECT_A1D_ELEM(tau2, ires) > 0.)
				{
					// Calculate inverse of tau2
					invtau2 = 1. / (oversampling_correction * tau2_fudge * DIRECT_A1D_ELEM(tau2, ires));
				}
				else if (DIRECT_A1D_ELEM(tau2, ires) == 0.)
				{
					// If tau2 is zero, use small value instead
					invtau2 = 1./ ( 0.001 * invw);
				}
				else
				{
					std::cerr << " sigma2= " << sigma2 << std::endl;
					std::cerr << " fsc= " << fsc << std::endl;
					std::cerr << " tau2= " << tau2 << std::endl;
					REPORT_ERROR("ERROR BackProjector::reconstruct: Negative or zero values encountered for tau2 spectrum!");
				}

				// Keep track of spectral evidence-to-prior ratio and remaining noise in the reconstruction
				if (!update_tau2_with_fsc)
					DIRECT_A1D_ELEM(data_vs_prior, ires) += invw / invtau2;

				// Keep track of the coverage in Fourier space
				if (invw / invtau2 >= 1.)
					DIRECT_A1D_ELEM(fourier_coverage, ires) += 1.;

				DIRECT_A1D_ELEM(counter, ires) += 1.;

				// Only for (ires >= minres_map) add Wiener-filter like term
				if (ires >= minres_map)
				{
					// Now add the inverse-of-tau2_class term
					invw += invtau2;
					// Store the new weight again in Fweight
					DIRECT_A3D_ELEM(Fweight, k, i, j) = invw;
				}
			}
		}

		// Average data_vs_prior
		if (!update_tau2_with_fsc)
		{
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(data_vs_prior)
			{
				if (i > r_max)
					DIRECT_A1D_ELEM(data_vs_prior, i) = 0.;
				else if (DIRECT_A1D_ELEM(counter, i) < 0.001)
					DIRECT_A1D_ELEM(data_vs_prior, i) = 999.;
				else
					DIRECT_A1D_ELEM(data_vs_prior, i) /= DIRECT_A1D_ELEM(counter, i);
			}
		}

		// Calculate Fourier coverage in each shell
		FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY1D(fourier_coverage)
		{
			if (DIRECT_A1D_ELEM(counter, i) > 0.)
				DIRECT_A1D_ELEM(fourier_coverage, i) /= DIRECT_A1D_ELEM(counter, i);
		}

	} //end if do_map
    else if (do_fsc0999)
    {

     	// Sjors 9may2018: avoid numerical instabilities with unregularised reconstructions....
        FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM(Fconv)
        {
            int r2 = kp * kp + ip * ip + jp * jp;
            if (r2 < max_r2)
            {
                int ires = ROUND( sqrt((RFLOAT)r2) / padding_factor );
                if (ires >= minres_map)
                {
                    // add 1/1000th of the radially averaged sigma2 to the Fweight, to avoid having zeros there...
                	DIRECT_A3D_ELEM(Fweight, k, i, j) += 1./(999. * DIRECT_A1D_ELEM(sigma2, ires));
                }
            }
        }

    }


	//==============================================================================add multi- GPU version


	int Ndim[3];
	int GPU_N;
	//GPU_N =2 ;
	cudaGetDeviceCount(&GPU_N);
	Ndim[0]=pad_size;
	Ndim[1]=pad_size;
	Ndim[2]=pad_size;
	size_t fullsize= pad_size*pad_size*pad_size;

	int sumranknum = ranksize*GPU_N ;
	initgpu_mpi(GPU_N);

	//divide task
	//process divide data
	int *numberZ;
	int *offsetZ;
	numberZ = (int *)malloc(sizeof(int)*sumranknum);
	offsetZ = (int *)malloc(sizeof(int)*sumranknum);

	dividetask(numberZ,offsetZ,pad_size,sumranknum);

	int padtrans_size = numberZ[0]*sumranknum;


	int *processnumberZ;
	int *processoffsetZ;
	processnumberZ = (int *)malloc(sizeof(int)*ranksize);
	processoffsetZ = (int *)malloc(sizeof(int)*ranksize);
	memset(processnumberZ,0,sizeof(int)*ranksize);
	memset(processoffsetZ,0,sizeof(int)*ranksize);

	dividetask(processnumberZ,processoffsetZ,padtrans_size,ranksize);

	int *numbertmpZ;
	int *offsettmpZ;
	numbertmpZ = (int *)malloc(sizeof(int)*sumranknum);
	offsettmpZ = (int *)malloc(sizeof(int)*sumranknum);
	dividetask(numbertmpZ,offsettmpZ,padtrans_size,sumranknum);

	for(int i=0;i<sumranknum;i++)
	{
		printf("%d ",offsettmpZ[i]);
	}
	printf("\n");


	MultiGPUplan *plan;
	plan = (MultiGPUplan *)malloc(sizeof(MultiGPUplan)*GPU_N);
	multi_plan_init_transpose_multi(plan,GPU_N,numberZ,offsetZ,pad_size,ranknum,ranksize);


	printf("datasize and tmpsize : %d %d \n",plan[0].datasize,plan[0].datasize);
	for (int i = 0; i < GPU_N; ++i) {
		cudaSetDevice(plan[i].devicenum);
		cudaMalloc((void**) & (plan[i].d_Data),sizeof(cufftComplex) * plan[i].datasize);
		cudaMalloc((void**) & (plan[i].temp_Data), sizeof(cufftComplex) * plan[i].tempsize);
	}

	int xyN[2];
	xyN[0] = Ndim[0];
	xyN[1] = Ndim[1];
	cufftHandle *xyplan;
	xyplan = (cufftHandle*) malloc(sizeof(cufftHandle)*GPU_N);
	cufftHandle *zplan;
	zplan = (cufftHandle*) malloc(sizeof(cufftHandle)*GPU_N);


	cufftComplex *cpu_data;//*c_Fconv2;
//	cpu_data= (cufftComplex *)malloc(pad_size*(numberZ[0]*sumranknum)*processnumberZ[ranknum]* sizeof(cufftComplex));

	cpu_data= (cufftComplex *)malloc(pad_size*padtrans_size*padtrans_size* sizeof(cufftComplex));
	//c_Fconv2 = (cufftComplex *)malloc(fullsize * sizeof(cufftComplex));
	//cudaMallocHost((void **) &c_Fconv2, sizeof(cufftComplex) * fullsize);

	cufftComplex *cpu_data2;
	cpu_data2 = (cufftComplex *)malloc(fullsize* sizeof(cufftComplex));



    RCTOCREC(ReconTimer,ReconS_2_5);
	if (skip_gridding)
	{


	    RCTICREC(ReconTimer,ReconS_3);
		std::cerr << "Skipping gridding!" << std::endl;
		Fconv.initZeros(); // to remove any stuff from the input volume
		decenter(data, Fconv, max_r2);

		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			if (DIRECT_MULTIDIM_ELEM(Fweight, n) > 0.)
				DIRECT_MULTIDIM_ELEM(Fconv, n) /= DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		RCTOCREC(ReconTimer,ReconS_3);
#ifdef DEBUG_RECONSTRUCT
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			DIRECT_MULTIDIM_ELEM(ttt(), n) = DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		ttt.write("reconstruct_skipgridding_correction_term.spi");
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fconv)
		{
			if (DIRECT_MULTIDIM_ELEM(Fweight, n) > 0.)
				DIRECT_MULTIDIM_ELEM(ttt(), n) = 1./DIRECT_MULTIDIM_ELEM(Fweight, n);
		}
		ttt.write("reconstruct_skipgridding_correction_term_inverse.spi");
#endif

	}
	else
	{

		RCTICREC(ReconTimer,ReconS_4);
		// Divide both data and Fweight by normalisation factor to prevent FFT's with very large values....
	#ifdef DEBUG_RECONSTRUCT
		std::cerr << " normalise= " << normalise << std::endl;
	#endif
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Fweight)
		{
			DIRECT_MULTIDIM_ELEM(Fweight, n) /= normalise;
		}
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(data)
		{
			DIRECT_MULTIDIM_ELEM(data, n) /= normalise;
		}
		RCTOCREC(ReconTimer,ReconS_4);
		RCTICREC(ReconTimer,ReconS_5);
        // Initialise Fnewweight with 1's and 0's. (also see comments below)
		FOR_ALL_ELEMENTS_IN_ARRAY3D(weight)
		{
			if (k * k + i * i + j * j < max_r2)
				A3D_ELEM(weight, k, i, j) = 1.;
			else
				A3D_ELEM(weight, k, i, j) = 0.;
		}



		decenter(weight, Fnewweight, max_r2);
		RCTOCREC(ReconTimer,ReconS_5);
		// Iterative algorithm as in  Eq. [14] in Pipe & Menon (1999)
		// or Eq. (4) in Matej (2001)



		long int Fconvnum=Fconv.nzyxdim;


		long int normsize= Ndim[0]*Ndim[1]*Ndim[2];
		int halfxdim=Fconv.xdim;

		printf("%ld %ld %ld \n",Fconv.xdim,Fconv.ydim,Fconv.zdim);
		printf("Fnewweight: %ld %ld %ld \n",Fnewweight.xdim,Fnewweight.ydim,Fnewweight.zdim);


		multi_enable_access(plan,GPU_N);

		for (int i = 0; i < GPU_N; i++) {
				cudaSetDevice(plan[i].devicenum);
				cufftPlanMany(&xyplan[i], 2, xyN, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, numberZ[i]);
			}

		for (int i = 0; i < GPU_N; i++) {
		    cudaSetDevice(plan[i].devicenum);
			int rank=1;
			int nrank[1];
			nrank[0]=Ndim[1];
			int inembed[2];
			inembed[0]=Ndim[0];
			inembed[1]=Ndim[1];
			cufftPlanMany(&zplan[i], rank, nrank, inembed, Ndim[0] , 1, inembed, Ndim[0], 1, CUFFT_C2C, Ndim[0]);
		}


		for(int i=0;i<fullsize;i++)
		{
			cpu_data2[i].x=i;
			cpu_data2[i].y=0;
		}



		for(int i = 0; i < GPU_N; i++)
		{
			 cudaSetDevice(plan[i].devicenum);
			 int reali= i+ranknum*GPU_N;
			 cudaMemcpy(plan[i].d_Data, cpu_data2 + pad_size*pad_size*offsetZ[reali], pad_size*pad_size*numberZ[reali]*sizeof(cufftComplex),cudaMemcpyHostToDevice);
		}

		memset(cpu_data2,0,sizeof(cufftComplex)*fullsize);


//		yzlocal_transpose(plan, GPU_N, pad_size, offsetZ);

		cufftComplex *cpu_data_real;
		cpu_data_real= (cufftComplex *)malloc(pad_size*pad_size*pad_size * sizeof(cufftComplex));


		if(ranknum==0)
		{
			cudaMemcpy(cpu_data,plan[0].d_Data,pad_size*pad_size*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

			printf("1\n");
			for(int i=0;i<pad_size;i++)
				//for(int j=0;j<pad_size;j++)
				printf("%f ",cpu_data[i*pad_size].x);
			printf("\n");
		}



		yzlocal_transpose_multicard(plan,GPU_N,pad_size,offsetZ,numberZ,offsettmpZ,ranknum,sumranknum);
		transpose_exchange_intra(plan,GPU_N,pad_size,offsetZ,numberZ,offsettmpZ,ranknum);
		memset(cpu_data,0,pad_size*padtrans_size*padtrans_size* sizeof(cufftComplex));
		data_exchange_gputocpu(plan,cpu_data,GPU_N,pad_size,offsettmpZ,ranknum);
		cpu_alltoalltrans_multinode(plan,cpu_data,pad_size,processnumberZ,processoffsetZ,ranknum,padtrans_size,ranksize,realrankarray);
		data_exchange_cputogpu(plan, cpu_data, GPU_N, pad_size,offsetZ,numberZ, offsettmpZ, numbertmpZ, ranknum,sumranknum,padtrans_size);

		if(ranknum==0)
		{
			cudaMemcpy(cpu_data,plan[0].d_Data,pad_size*pad_size*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

			printf("2\n");
			for(int i=0;i<pad_size;i++)
				//for(int j=0;j<pad_size;j++)
				printf("%f ",cpu_data[i*pad_size].x);
			printf("\n");
		}

		printf("%d \n",pad_size);




/*		for(int i = 0; i < GPU_N; i++)
		{
			 cudaSetDevice(plan[i].devicenum);
			 int reali= i+ranknum*GPU_N;
			 cudaMemcpy(cpu_data+plan[i].selfoffset , plan[i].d_Data ,plan[i].realsize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
		}*/









//		cpu_alltozero_multinode(cpu_data,pad_size, processnumberZ,processoffsetZ,ranknum, padtrans_size, ranksize,realrankarray);
/*		for(int i = 0; i < GPU_N; i++)
		{
			 cudaSetDevice(plan[i].devicenum);
			 int reali= i+ranknum*GPU_N;
			 cudaMemcpy(cpu_data+plan[i].selfoffset , plan[i].d_Data ,plan[i].realsize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
		}*/

/*
		memset(cpu_data_real,0,sizeof(cufftComplex)*pad_size*pad_size*pad_size);

		if(ranknum==0)
		{
			printf("data from rank %d and GPU num : %d\n",ranknum,GPU_N);
			for(int gpui = 0; gpui < GPU_N; gpui++)
			{
				 cudaSetDevice(plan[gpui].devicenum);
				 int reali= gpui+ranknum*GPU_N;
				 cudaMemcpy(cpu_data, plan[gpui].temp_Data,numberZ[0] *pad_size*padtrans_size*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

					for(int i=0;i<numberZ[0];i++)
					{
						for(int j=0;j<padtrans_size;j++)
						{	for(int k=0;k<pad_size;k++)
							{
								int index=i*padtrans_size*pad_size+ j * pad_size + k;
								if(k==0)
									printf("%f ",cpu_data[index].x);
							}
						}
						printf("\n");
					}
			}
		}*/


//		if(ranknum == 0)
//			cpu_datarearrange_multinode(cpu_data,cpu_data_real, pad_size,  numberZ, offsetZ,numbertmpZ, offsettmpZ,  ranknum, padtrans_size, sumranknum);
/*		if(ranknum==0)
		{
			for(int i=0;i<padtrans_size;i++)
			{
				for(int j=0;j<padtrans_size;j++)
				{	for(int k=0;k<pad_size;k++)
					{
						int index=i*padtrans_size*pad_size+ j * pad_size + k;
						if(k==0)
							printf("%f ",cpu_data[index].x);
					}
				}
				printf("\n");

			}
		}*/

/*
		if(ranknum==0)
		{
			for(int i=0;i<pad_size;i++)
			{
				for(int j=0;j<pad_size;j++)
				{	for(int k=0;k<pad_size;k++)
					{
						int index=i*pad_size*pad_size+ j * pad_size + k;
						if(k==0)
							printf("%f ",cpu_data_real[index].x);
					}
				}
				printf("\n");

			}
		}*/


		//if(ranknum == 0)
		//	cpu_datarearrange_multinode(cpu_data,cpu_data_real, padoridim,  numberZ, offsetZ,numbertmpZ, offsettmpZ,  ranknum, padtrans_size, sumranknum);


/*

		for(int i = 0; i < GPU_N; i++)
		{
			 cudaSetDevice(plan[i].devicenum);
			 int reali= i+ranknum*GPU_N;
			 cudaMemcpy(cpu_data + pad_size*padtrans_size*offsettmpZ[reali], plan[i].temp_Data ,pad_size*padtrans_size*numbertmpZ[reali]*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
		}*/

		//	int reali= 0+ ranknum*GPU_N;
		//	int base= pad_size*pad_size*offsetZ[reali];

/*
		if(ranknum==0)
		{
			for(int i=0;i<padtrans_size;i++)
			{
				for(int j=0;j<padtrans_size;j++)
				{	for(int k=0;k<pad_size;k++)
					{
						int index=i*padtrans_size*pad_size+ j * pad_size + k;
						if(k==0)
							printf("%f ",cpu_data[index].x);
					}
				}
				printf("\n");

			}
		}
		*/
/*
			int reali= 0+ ranknum*GPU_N;
			int base= pad_size*padtrans_size*offsettmpZ[reali];
			int index=base;
			for(int i=0;i<padtrans_size;i++)
			{
				printf("%f ",cpu_data[index].x);
				index+=pad_size;
			}
			printf("\n");
			reali= 1 + ranknum*GPU_N;
			base= pad_size*padtrans_size*offsettmpZ[reali];
			index=base;
			for(int i=0;i<padtrans_size;i++)
			{
				printf("%f ",cpu_data[index].x);
				index+=pad_size;
			}
			printf("\n");*/



	}

}

