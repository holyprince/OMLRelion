


#include "../reconstructor.h"
#include "cufft.h"
#include "complex.h"
#include "mpi.h"

// MPI version
/*
#define NUMP 2
int main(int argc, char *argv[])
{
	int numprocs;
	int my_rank;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	printf("process %d: %d\n",my_rank,numprocs);

//  int ori_size=100;
// 	FileName fn_root = "gpu3_half1";
// 	int ref_dim=3;
// 	int pad_size= 2* ori_size + 3;
// 	BackProjector backprojector(ori_size,ref_dim,"D2");

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

	//printf(" %d %d %f %d\n", backprojector.ori_size, backprojector.data_dim,backprojector.padding_factor,backprojector.pad_size);
	//printf("%d \n ",backprojector.r_min_nn);
	//printf("%ld %ld %ld \n",backprojector.weight.xdim,backprojector.weight.ydim,backprojector.weight.zdim);
	MultidimArray<RFLOAT> dummy;
	Image<RFLOAT> Iunreg, Itmp;

	if(my_rank ==1 || my_rank ==5) {
	//if(my_rank ==1 || my_rank ==3) {
	//if(my_rank ==1) {
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
//	backprojector.reconstruct_gpu(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpumpi(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);
//	backprojector.reconstruct_gpustd(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,numprocs);
//	backprojector.reconstruct_test(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);
	backprojector.reconstruct_gpumpicard(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);


//	if (my_rank == 0) {
//		// Update header information
//		Iunreg.setStatisticsInHeader();
//		Iunreg.setSamplingRateInHeader(1);
//		// And write the resulting model to disc
//		Iunreg.write(fn_root + "_unfil.mrc");
//	}
//	backprojector.reconstruct_test(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);
	}

	MPI_Finalize();
	return 0;
}*/



//MPI any dimension

#define NUMP 2
int main(int argc, char *argv[])
{
	int numprocs;
	int my_rank;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	char hostname[100];
	gethostname(hostname,sizeof(hostname));
	printf("process %d: %d on computer %s \n",my_rank,numprocs,hostname);

    int ori_size=750;

 	int ref_dim=3;
 	int pad_size= 2* ori_size + 3;
 	BackProjector backprojector(ori_size,ref_dim,"I3");
//	int ori_size=360;
//	FileName fn_root = "run_ct5kdata_half1";
//	int ref_dim=3;
//	int pad_size= 2* ori_size + 3;
//	BackProjector backprojector(ori_size,ref_dim,"C1");

//set back project para :

	if(my_rank ==1 || my_rank == 5) {


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
	//if(my_rank ==1) {
	int iclass=0;

	Itmp.data.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	Itmp().setXmippOrigin();
	Itmp().xinit=0;

	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.data, k, i, j).real = 10;// (k*k+i*i+j*j)%fullsize;
	}

	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.data, k, i, j).imag = 10;// (k*k+i*i+j*j)%fullsize;
	}

	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.weight, k, i, j) = 10;//(k*k+i*i+j*j) %fullsize;
	}
	printf("Finished \n");
	Itmp().clear();
	// Now perform the unregularized reconstruction
	int gridding_nr_iter=1;
	bool do_fsc0999 = false;
//	backprojector.reconstruct(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpumpi(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);
//	backprojector.reconstruct_gpustd(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,numprocs);
//	backprojector.reconstruct_test(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);
	backprojector.reconstruct_gpumpicard(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);


//	if (my_rank == 0) {
//		// Update header information
//		Iunreg.setStatisticsInHeader();
//		Iunreg.setSamplingRateInHeader(1);
//		// And write the resulting model to disc
//		Iunreg.write(fn_root + "_unfil.mrc");
//	}
//	backprojector.reconstruct_test(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);
	}

	MPI_Finalize();
	return 0;
}

/*

 // any dimension
int main(int argc, char *argv[])
{

    int ori_size=10;
	//FileName fn_root = "gpu3_half1";
	int ref_dim=3;
	int pad_size= 2* ori_size + 3;
	BackProjector backprojector(ori_size,ref_dim,"I3");

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
		A3D_ELEM(backprojector.data, k, i, j).real = 10;// (k*k+i*i+j*j)%fullsize;
	}

	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.data, k, i, j).imag = 10;// (k*k+i*i+j*j)%fullsize;
	}

	FOR_ALL_ELEMENTS_IN_ARRAY3D(Itmp())
	{
		A3D_ELEM(backprojector.weight, k, i, j) = 10;//(k*k+i*i+j*j) %fullsize;
	}
	printf("Finished \n");

	// Now perform the unregularized reconstruction
	int gridding_nr_iter=1;
	bool do_fsc0999 = false;
//	backprojector.reconstruct(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu_raw(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu_single(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
	backprojector.reconstruct_gpu_transpose(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu_transpose_test(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);

	// Update header information
//	Iunreg.setStatisticsInHeader();
//	Iunreg.setSamplingRateInHeader(1);
//	// And write the resulting model to disc
//	Iunreg.write(fn_root+"_unfil.mrc");

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
	int gridding_nr_iter=1;
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

}
*/


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
/*
#define NUMP 2
int main(int argc, char *argv[])
{
	int numprocs;
	int my_rank;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	printf("process %d: %d\n",my_rank,numprocs);

    int ori_size=700;

 	int ref_dim=3;
 	int pad_size= 2* ori_size + 3;
 	BackProjector backprojector(ori_size,ref_dim,"I3");
 	int fullsize=pad_size*pad_size*pad_size;
//	int ori_size=360;
//	FileName fn_root = "run_ct5kdata_half1";
//	int ref_dim=3;
//	int pad_size= 2* ori_size + 3;
//	BackProjector backprojector(ori_size,ref_dim,"C1");

//set back project para :

	if(my_rank ==1 || my_rank == 5) {
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
	//if(my_rank ==1) {
	int iclass=0;

	Itmp.data.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
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


	MultidimArray<Complex > test1;
	MultidimArray<Complex > test2;
	MultidimArray<Complex > test3;
	MultidimArray<Complex > test4;
	MultidimArray<Complex > test5;
	MultidimArray<Complex > test6;
	MultidimArray<Complex > test7;
	MultidimArray<Complex > test8;
	MultidimArray<Complex > test9;
	MultidimArray<Complex > test10;
	MultidimArray<Complex > test11;
	MultidimArray<Complex > test12;
	MultidimArray<Complex > test13;
	MultidimArray<Complex > test14;
	MultidimArray<Complex > test15;
	MultidimArray<Complex > test16;
	MultidimArray<Complex > test17;
	MultidimArray<Complex > test18;
	MultidimArray<Complex > test19;
	MultidimArray<Complex > test20;
	MultidimArray<Complex > test21;
	MultidimArray<Complex > test22;
	MultidimArray<Complex > test23;
	MultidimArray<Complex > test24;
	MultidimArray<Complex > test25;
	Complex temp;
	temp.real=100; temp.imag=-100;
	test1.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test1, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test2.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test2, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test3.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test3, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test4.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test4, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test5.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test5, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test6.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test6, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test7.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test7, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test8.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test8, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test9.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test9, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test10.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test10, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test11.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test11, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test12.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test12, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test13.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test13, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test14.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test14, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test15.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test15, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test16.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test16, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test17.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test17, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test18.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test18, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test19.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test19, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test20.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test20, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test21.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test21, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test22.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test22, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test23.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test23, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test24.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test24, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	test25.coreAllocate(1,pad_size,pad_size,pad_size/2+1);
	DIRECT_A3D_ELEM(test25, pad_size-1, pad_size-1, pad_size/2+1-1) = temp;
	printf("Finished \n");
	Itmp().clear();
	// Now perform the unregularized reconstruction
	int gridding_nr_iter=10;
	bool do_fsc0999 = false;
//	backprojector.reconstruct(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpu(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999);
//	backprojector.reconstruct_gpumpi(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);
//	backprojector.reconstruct_gpustd(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,numprocs);
//	backprojector.reconstruct_test(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);
//	backprojector.reconstruct_gpumpicard(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);


//	if (my_rank == 0) {
//		// Update header information
//		Iunreg.setStatisticsInHeader();
//		Iunreg.setSamplingRateInHeader(1);
//		// And write the resulting model to disc
//		Iunreg.write(fn_root + "_unfil.mrc");
//	}
//	backprojector.reconstruct_test(Iunreg(), gridding_nr_iter, false, 1., dummy, dummy, dummy, dummy, dummy, 1., false, true, 1, -1, false, do_fsc0999,my_rank,NUMP);
	}

	MPI_Finalize();
	return 0;
}*/
