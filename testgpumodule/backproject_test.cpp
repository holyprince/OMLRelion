


#include "../reconstructor.h"
#include "cufft.h"
#include "complex.h"
#include "mpi.h"

// MPI version
/*
int main(int argc, char *argv[])
{
	int numprocs;
	int my_rank;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	printf("process %d: %d\n",my_rank,numprocs);

    int ori_size=100;
 	FileName fn_root = "gpu3_half1";
 	int ref_dim=3;
 	int pad_size= 2* ori_size + 3;
 	BackProjector backprojector(ori_size,ref_dim,"D2");

//	int ori_size=360;
//	FileName fn_root = "run_ct5kdata_half1";
//	int ref_dim=3;
//	int pad_size= 2* ori_size + 3;
//	BackProjector backprojector(ori_size,ref_dim,"C1");

//set back project para :
	backprojector.pad_size= pad_size;
	backprojector.data.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	backprojector.weight.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	backprojector.data.setXmippOrigin();
	backprojector.data.xinit=0;
	backprojector.weight.setXmippOrigin();
	backprojector.weight.xinit=0;
	backprojector.r_max=ori_size/2;  //add by self
	int _blob_order = 0;
	RFLOAT _blob_radius = 1.9;
	RFLOAT _blob_alpha = 15;
	backprojector.tab_ftblob.initialise(_blob_radius * 2., _blob_alpha, _blob_order, 10000);

	//printf(" %d %d %f %d\n", backprojector.ori_size, backprojector.data_dim,backprojector.padding_factor,backprojector.pad_size);
	//printf("%d \n ",backprojector.r_min_nn);
	//printf("%ld %ld %ld \n",backprojector.weight.xdim,backprojector.weight.ydim,backprojector.weight.zdim);
	MultidimArray<RFLOAT> dummy;
	Image<RFLOAT> Iunreg, Itmp;

	//if(my_rank ==1 || my_rank ==3) {
	//if(my_rank ==0 || my_rank ==1 || my_rank ==2 || my_rank ==3) {
	if(my_rank ==1 || my_rank ==3) {
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
	int gridding_nr_iter=10;
	bool do_fsc0999 = false;
//	backprojector.reconstruct(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpumpi(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,2);
//	backprojector.reconstruct_gpustd(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,numprocs);

	if (my_rank == 0) {
		// Update header information
		Iunreg.setStatisticsInHeader();
		Iunreg.setSamplingRateInHeader(1);
		// And write the resulting model to disc
		Iunreg.write(fn_root + "_unfil.mrc");
	}

	}

	MPI_Finalize();
	return 0;
}
*/


int main(int argc, char *argv[])
{

    int ori_size=600;
	FileName fn_root = "gpu3_half1";
	int ref_dim=3;
	int pad_size= 2* ori_size + 3;
	BackProjector backprojector(ori_size,ref_dim,"D2");

//	int ori_size=360;
//	FileName fn_root = "run_ct5kdata_half1";
//	int ref_dim=3;
//	int pad_size= 2* ori_size + 3;
//	BackProjector backprojector(ori_size,ref_dim,"C1");

	int fullsize=pad_size*pad_size*pad_size;

//set back project para :
	backprojector.pad_size= pad_size;
	backprojector.data.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	backprojector.weight.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	backprojector.data.setXmippOrigin();
	backprojector.data.xinit=0;
	backprojector.weight.setXmippOrigin();
	backprojector.weight.xinit=0;
	backprojector.r_max=ori_size/2;  //add by self
	int _blob_order = 0;
	RFLOAT _blob_radius = 1.9;
	RFLOAT _blob_alpha = 15;
	backprojector.tab_ftblob.initialise(_blob_radius * 2., _blob_alpha, _blob_order, 10000);

	printf(" %d %d %f %d\n", backprojector.ori_size, backprojector.data_dim,backprojector.padding_factor,backprojector.pad_size);
	printf("%d \n ",backprojector.r_min_nn);
	printf("%ld %ld %ld \n",backprojector.weight.xdim,backprojector.weight.ydim,backprojector.weight.zdim);
	MultidimArray<RFLOAT> dummy;
	Image<RFLOAT> Iunreg, Itmp;



	int iclass=0;
//	fn_root.compose(fn_root+"_class", iclass+1, "", 3);
	// Read temporary arrays back in
	//Itmp.read(fn_root+"_data_real.mrc");
	Itmp.data.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	//Itmp().coreAllocate(1,pad_size,pad_size,pad_size/2+1);
//	printf("%d %d %d \n",Itmp().xdim,Itmp().ydim,Itmp().zdim);
	Itmp().setXmippOrigin();
	Itmp().xinit=0;


	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.data, k, i, j).real =  (k*k+i*i+j*j)%fullsize;
	}

	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.data, k, i, j).imag =  (k*k+i*i+j*j)%fullsize;
	}

	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.weight, k, i, j) = (k*k+i*i+j*j) %fullsize;
	}
	printf("Finished \n");

	// Now perform the unregularized reconstruction
	int gridding_nr_iter=10;
	bool do_fsc0999 = false;

//	backprojector.reconstruct_gpu_raw(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu_single(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
	backprojector.reconstruct_gpu_transpose(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);

	// Update header information
	Iunreg.setStatisticsInHeader();
	Iunreg.setSamplingRateInHeader(1);
	// And write the resulting model to disc
	Iunreg.write(fn_root+"_unfil.mrc");

	printf("Finished \n");
}
/*
int main(int argc, char *argv[])
{

    int ori_size=100;
	FileName fn_root = "gpu3_half1";
	int ref_dim=3;
	int pad_size= 2* ori_size + 3;
	BackProjector backprojector(ori_size,ref_dim,"D2");

//	int ori_size=360;
//	FileName fn_root = "run_ct5kdata_half1";
//	int ref_dim=3;
//	int pad_size= 2* ori_size + 3;
//	BackProjector backprojector(ori_size,ref_dim,"C1");


//set back project para :
	backprojector.pad_size= pad_size;
	backprojector.data.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	backprojector.weight.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	backprojector.data.setXmippOrigin();
	backprojector.data.xinit=0;
	backprojector.weight.setXmippOrigin();
	backprojector.weight.xinit=0;
	backprojector.r_max=ori_size/2;  //add by self
	int _blob_order = 0;
	RFLOAT _blob_radius = 1.9;
	RFLOAT _blob_alpha = 15;
	backprojector.tab_ftblob.initialise(_blob_radius * 2., _blob_alpha, _blob_order, 10000);

	printf(" %d %d %f %d\n", backprojector.ori_size, backprojector.data_dim,backprojector.padding_factor,backprojector.pad_size);
	printf("%d \n ",backprojector.r_min_nn);
	printf("%ld %ld %ld \n",backprojector.weight.xdim,backprojector.weight.ydim,backprojector.weight.zdim);
	MultidimArray<RFLOAT> dummy;
	Image<RFLOAT> Iunreg, Itmp;



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
	int gridding_nr_iter=10;
	bool do_fsc0999 = false;
//	backprojector.reconstruct(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu_single(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu_transpose(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);

	// Update header information
	Iunreg.setStatisticsInHeader();
	Iunreg.setSamplingRateInHeader(1);
	// And write the resulting model to disc
	Iunreg.write(fn_root+"_unfil.mrc");

}*/
// data -   100
/*
 *
	int ori_size=100;
	FileName fn_root = "gpu3_half1";
	int ref_dim=3;
	int pad_size= 2* ori_size + 3;
	BackProjector backprojector(ori_size,ref_dim,"D2");

//set back project para :
	backprojector.pad_size= pad_size;
	backprojector.data.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	backprojector.weight.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	backprojector.data.setXmippOrigin();
	backprojector.data.xinit=0;
	backprojector.weight.setXmippOrigin();
	backprojector.weight.xinit=0;
	backprojector.r_max=ori_size/2;  //add by self
	int _blob_order = 0;
	RFLOAT _blob_radius = 1.9;
	RFLOAT _blob_alpha = 15;
	backprojector.tab_ftblob.initialise(_blob_radius * 2., _blob_alpha, _blob_order, 10000);

	printf(" %d %d %f %d\n", backprojector.ori_size, backprojector.data_dim,backprojector.padding_factor,backprojector.pad_size);
	printf("%d \n ",backprojector.r_min_nn);
	printf("%ld %ld %ld \n",backprojector.weight.xdim,backprojector.weight.ydim,backprojector.weight.zdim);

*/


/*
 *
    int ori_size=360;
	FileName fn_root = "run_ct5kdata_half1";
	int ref_dim=3;
	int pad_size= 2* ori_size + 3;
	BackProjector backprojector(ori_size,ref_dim,"C1");

//set back project para :
	backprojector.pad_size= pad_size;
	backprojector.data.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	backprojector.weight.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	backprojector.data.setXmippOrigin();
	backprojector.data.xinit=0;
	backprojector.weight.setXmippOrigin();
	backprojector.weight.xinit=0;
	backprojector.r_max=ori_size/2;  //add by self
	int _blob_order = 0;
	RFLOAT _blob_radius = 1.9;
	RFLOAT _blob_alpha = 15;
	backprojector.tab_ftblob.initialise(_blob_radius * 2., _blob_alpha, _blob_order, 10000);

	printf(" %d %d %f %d\n", backprojector.ori_size, backprojector.data_dim,backprojector.padding_factor,backprojector.pad_size);
	printf("%d \n ",backprojector.r_min_nn);
	printf("%ld %ld %ld \n",backprojector.weight.xdim,backprojector.weight.ydim,backprojector.weight.zdim);
 *
 */
