


#include "reconstructor.h"
#include "cufft.h"
#include "complex.h"


int main(int argc, char *argv[])
{
	int ori_size=360;
	int ref_dim=3;
	BackProjector backprojector(ori_size,ref_dim,"C1");
	backprojector.pad_size=723;
//set back project para :
	backprojector.data.coreAllocate(1,723,723,362);
	backprojector.weight.coreAllocate(1,723,723,362);
	backprojector.data.setXmippOrigin();
	backprojector.data.xinit=0;
	backprojector.weight.setXmippOrigin();
	backprojector.weight.xinit=0;
	backprojector.r_max=180;  //add by self
	int _blob_order = 0;
	RFLOAT _blob_radius = 1.9;
	RFLOAT _blob_alpha = 15;
	backprojector.tab_ftblob.initialise(_blob_radius * 2., _blob_alpha, _blob_order, 10000);

	printf(" %d %d %f %d\n", backprojector.ori_size, backprojector.data_dim,backprojector.padding_factor,backprojector.pad_size);
	printf("%d \n ",backprojector.r_min_nn);
	printf("%ld %ld %ld \n",backprojector.weight.xdim,backprojector.weight.ydim,backprojector.weight.zdim);

	MultidimArray<RFLOAT> dummy;
	Image<RFLOAT> Iunreg, Itmp;


	FileName fn_root = "run_ct5kdata_half1";
	int iclass=0;


	fn_root.compose(fn_root+"_class", iclass+1, "", 3);

	// Read temporary arrays back in
	Itmp.read(fn_root+"_data_real.mrc");

	Itmp().setXmippOrigin();
	Itmp().xinit=0;
	if (!Itmp().sameShape(backprojector.data))
	{
		backprojector.data.printShape(std::cerr);
		Itmp().printShape(std::cerr);
		REPORT_ERROR("Incompatible size of "+fn_root+"_data_real.mrc");
	}
	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.data, k, i, j).real = A3D_ELEM(Itmp(), k, i, j);
	}

	Itmp.read(fn_root+"_data_imag.mrc");
	Itmp().setXmippOrigin();
	Itmp().xinit=0;
	if (!Itmp().sameShape(backprojector.data))
	{
		backprojector.data.printShape(std::cerr);
		Itmp().printShape(std::cerr);
		REPORT_ERROR("Incompatible size of "+fn_root+"_data_imag.mrc");
	}
	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.data, k, i, j).imag = A3D_ELEM(Itmp(), k, i, j);
	}

	Itmp.read(fn_root+"_weight.mrc");
	Itmp().setXmippOrigin();
	Itmp().xinit=0;
	if (!Itmp().sameShape(backprojector.weight))
	{
		backprojector.weight.printShape(std::cerr);
		Itmp().printShape(std::cerr);
		REPORT_ERROR("Incompatible size of "+fn_root+"_weight.mrc");
	}
	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.weight, k, i, j) = A3D_ELEM(Itmp(), k, i, j);
	}

	// Now perform the unregularized reconstruction
	int gridding_nr_iter=1;
	bool do_fsc0999 = false;
//	backprojector.reconstruct(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
	backprojector.reconstruct_gpu(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);


/*
	// Update header information
	Iunreg.setStatisticsInHeader();
	Iunreg.setSamplingRateInHeader(1);
	// And write the resulting model to disc
	Iunreg.write(fn_root+"_unfil.mrc");*/

}
