#include "backproject_impl.h"



void printwholeres(cufftComplex *out, int dimx,int dimy,int dimz) {

        printf("=====================\n");
        for (int i = 0; i < 10; i++)
                printf("%f %f \n", out[i].x, out[i].y);  //a

        for (int i = 0 + (dimx * dimy / 2); i < 10 + (dimx * dimy / 2); i++)  //b
                printf("%f %f \n", out[i].x, out[i].y);

        for (int i = dimx * dimy *(dimz/2); i < dimx * dimy *(dimz/2) + 10; i++) //c
                printf("%f %f \n", out[i].x, out[i].y);

        for (int i = dimx * dimy / 2 * dimz + (dimx * dimy / 2);
                        i < dimx * dimy / 2 * dimz + (dimx * dimy / 2) + 10; i++) //dd
                printf("%f %f \n", out[i].x, out[i].y);

        for(int i= dimx*dimy*dimz -10 ; i<dimx*dimy*dimz;i++)  // end
                printf("%f %f \n",out[i].x, out[i].y);

        for(int i= dimx*dimy*(dimz-1) ; i<dimx*dimy*(dimz-1)+10;i++)  // last slice
                printf("%f %f \n",out[i].x, out[i].y);

}

void datainit(cufftComplex *data,int dimxYZ)
{
        for (int i = 0; i < dimxYZ; i++) {
                data[i].x = i % 5000 ;
                data[i].y= 0;
        }
}

void multi_plan_init(MultiGPUplan *plan, int GPU_N, size_t fullsize, int dimx,int dimy,int dimz)
{
	//MultiGPUplan plan[MAXGPU];
	for (int i = 0; i < GPU_N; i++) {
		//deviceNum[i] = i;
		plan[i].devicenum = i;
		plan[i].datasize = fullsize;
	}
	plan[0].selfoffset = 0;
	plan[1].selfoffset = dimx * dimy * (dimz / 2);
}


void multi_enable_access(MultiGPUplan *plan,int GPU_N)
{
	int can_access_peer = -100;
	cudaDeviceCanAccessPeer(&can_access_peer, plan[0].devicenum,
			plan[1].devicenum);
	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(plan[i].devicenum);
		cudaDeviceEnablePeerAccess((GPU_N - 1) - plan[i].devicenum, 0);
	}
	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(plan[i].devicenum);
		cudaDeviceSynchronize();
	}
}
void multi_memcpy_data(MultiGPUplan *plan, cufftComplex *f,int GPU_N,int dimx,int dimy )
{
	int offset=0;
	for (int i = 0; i < GPU_N; ++i) {
		cudaSetDevice(plan[i].devicenum);
		cudaMemcpyAsync(plan[i].d_Data + plan[i].selfoffset,
				f + offset,(plan[i].selfZ * dimx * dimy) * sizeof(cufftComplex),cudaMemcpyHostToDevice);
		offset += plan[0].selfZ  *dimx * dimy ;
	}
	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(plan[i].devicenum);
		cudaDeviceSynchronize();
	}
}

void multi_memcpy_data_gpu(MultiGPUplan *plan,int GPU_N,int dimx,int dimy )
{
	int offset=0;
	// expect 0 GPU because it hold data self
	for (int i = 0; i < GPU_N; ++i) {
		cudaSetDevice(plan[i].devicenum);
		if(i!=0)
		{
			cudaMemcpyAsync(plan[i].d_Data + plan[i].selfoffset,	plan[0].d_Data + offset,
					(plan[i].selfZ * dimx * dimy) * sizeof(cufftComplex),cudaMemcpyDeviceToDevice);
		}
		offset += plan[i].selfZ  *dimx * dimy ;
	}
	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(plan[i].devicenum);
		cudaDeviceSynchronize();
	}
}

void multi_memcpy_databack(MultiGPUplan *plan, cufftComplex *out,int GPU_N,int dimx,int dimy)
{
	int offset = 0;
	cudaSetDevice(0);
	cudaMemcpyAsync(out, plan[0].d_Data,(plan[0].selfZ * dimx * dimy) * sizeof(cufftComplex),cudaMemcpyDeviceToHost);
	offset += plan[0].selfZ * dimx * dimy;
	cudaMemcpyAsync(out + offset, plan[1].d_Data + offset,(plan[1].selfZ * dimx * dimy) * sizeof(cufftComplex),cudaMemcpyDeviceToHost);

	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(plan[i].devicenum);
		cudaDeviceSynchronize();
	}
}



void mulit_alltoall_one(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ)
{
	cudaSetDevice(0);
	int dimxy = dimx * dimy;
	int halfslice1 = (offsetZ[0]) * dimx; //Z reperesent Y
	int halfslice2 = (offsetZ[0]) * dimx * dimy;
	int sliceoffset021;
	int sliceoffset120;

	int deltadimxy = 0;
	int cpysize01 = (offsetZ[1]) * dimx * sizeof(cufftComplex);
	int cpysize10 = (offsetZ[0]) * dimx * sizeof(cufftComplex);
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	for (int j = 0; j < offsetZ[0]; j++) {
		sliceoffset021 = halfslice1 + deltadimxy;
		sliceoffset120 = halfslice2 + deltadimxy;

		cudaMemcpyAsync(plan[1].d_Data + sliceoffset021, plan[0].d_Data + sliceoffset021, cpysize01,cudaMemcpyDeviceToDevice, stream1);
		cudaMemcpyAsync(plan[0].d_Data + sliceoffset120, plan[1].d_Data + sliceoffset120, cpysize10,cudaMemcpyDeviceToDevice, stream2);

		deltadimxy += dimxy;
	}
	//extra for 1 z
	deltadimxy = 0;

	for (int j = (dimz - extraz); j < dimz; j++) {
		cudaMemcpyAsync(plan[0].d_Data + (dimz - extraz) * dimx * dimy + deltadimxy,
				plan[1].d_Data + (dimz - extraz) * dimx * dimy + deltadimxy,cpysize10, cudaMemcpyDeviceToDevice, stream1);
		deltadimxy += dimxy;
	}
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);


}
void mulit_alltoall_two(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ)
{
	int dimxy = dimx * dimy;
	int halfslice1 = (offsetZ[0]) * dimx; //Z reperesent Y
	int halfslice2 = (offsetZ[0]) * dimx * dimy;
	int cpysize01 = (offsetZ[1]) * dimx * sizeof(cufftComplex);
	int cpysize10 = (offsetZ[0]) * dimx * sizeof(cufftComplex);
	int deltadimxy = 0;
	int sliceoffset021,sliceoffset120;
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	for (int j = 0; j < offsetZ[0]; j++) {
		sliceoffset021 = halfslice1 + deltadimxy;
		sliceoffset120 = halfslice2 + deltadimxy;

		cudaMemcpyAsync(plan[0].d_Data + sliceoffset021,
				plan[1].d_Data + sliceoffset021, cpysize01,
				cudaMemcpyDeviceToDevice, stream1);
		cudaMemcpyAsync(plan[1].d_Data + sliceoffset120,
				plan[0].d_Data + sliceoffset120, cpysize10,
				cudaMemcpyDeviceToDevice, stream2);
		deltadimxy += dimxy;
	}
	deltadimxy = 0;
	for (int j = (dimz - extraz); j < dimz; j++) {
		cudaMemcpyAsync(plan[1].d_Data + (dimz - extraz) * dimx * dimy + deltadimxy,
				plan[0].d_Data + (dimz - extraz) * dimx * dimy + deltadimxy,
				cpysize10, cudaMemcpyDeviceToDevice, stream1);
		deltadimxy += dimxy;
	}
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

}

void mulit_alltoall_all1to0(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ)
{
	int dimxy = dimx * dimy;
	int halfslice1 = (offsetZ[0]) * dimx; //Z reperesent Y
	int cpysize01 = (offsetZ[1]) * dimx * sizeof(cufftComplex);
	int deltadimxy = 0;
	int sliceoffset021;

	for (int j = 0; j < dimz; j++) {
		sliceoffset021 = halfslice1 + deltadimxy;
		cudaMemcpyAsync(plan[0].d_Data + sliceoffset021, plan[1].d_Data + sliceoffset021, cpysize01,cudaMemcpyDeviceToDevice);

		deltadimxy += dimxy;
	}

}
void mulit_datacopy_0to1(MultiGPUplan *plan, int dimx,int dimy,int *offsetZ)
{

	int sliceoffset = (offsetZ[0]) * dimx * dimy;
	int cpysize = (offsetZ[1]) * dimx *dimy * sizeof(cufftComplex);
	cudaMemcpy(plan[1].d_Data + sliceoffset, plan[0].d_Data + sliceoffset, cpysize,cudaMemcpyDeviceToDevice);

}

void multi_sync(MultiGPUplan *plan,int GPU_N)
{
	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(plan[i].devicenum);
		cudaDeviceSynchronize();
	}
}

void multi_plan_init_mpi(MultiGPUplan *plan, size_t fullsize,size_t realznum,size_t offsetz,int cardnum,int dimx,int dimy)
{

	plan[0].devicenum = cardnum;
	plan[0].datasize = fullsize;
	plan[0].realsize = dimx* dimy * realznum;
	plan[0].selfZ = realznum;  //self num
	plan[0].selfoffset =dimx* dimy* offsetz;

}
void gpu_to_cpu(MultiGPUplan *plan,cufftComplex *cpu_data)
{
	cudaMemcpy(cpu_data + plan[0].selfoffset,plan[0].d_Data + plan[0].selfoffset,plan[0].realsize * sizeof(cufftComplex),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}
void gpu_to_cpu_inverse(MultiGPUplan *plan,cufftComplex *cpu_data)
{
	/*
	if(ranknum==0)
	int recvsliceoffset=padsize*padsize*numberZ[0];
	int recvslicesize=padsize*numberZ[0];
	for(int i=0;i<numberZ[1];i++)
	{
		MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, desranknum, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
		recvsliceoffset += padsize*padsize;
	}
	cudaDeviceSynchronize();*/
}


void cpu_alltoall(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize)
{
// 4.2 : cpu all to all
	//send numberZ[0] to 1
	//recv numberZ[1] from 1
	int rawranknum,desranknum;
	if(ranknum == 0)
		rawranknum =1;
	if(ranknum == 1)
		rawranknum =3;

	if(rawranknum ==1)
		desranknum=3;
	if(rawranknum ==3)
		desranknum=1;
	if(ranknum==0)
	{
		int sendsliceoffset=padsize*numberZ[0];
		int sendslicesize=padsize*numberZ[1];
		for(int i=0;i<numberZ[0];i++)
		{
			MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD);
			sendsliceoffset+=padsize*padsize;
		}
		int recvsliceoffset=padsize*padsize*numberZ[0];
		int recvslicesize=padsize*numberZ[0];
		for(int i=0;i<numberZ[1];i++)
		{
			MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, desranknum, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
			recvsliceoffset += padsize*padsize;
		}
	}
	if(ranknum==1)
	{

		int recvsliceoffset=padsize*numberZ[0];
		int recvslicesize=padsize*numberZ[1];
		for(int i=0;i<numberZ[0];i++)
		{
			MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, desranknum, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
			cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
			recvsliceoffset += padsize*padsize;
		}
		int sendsliceoffset = padsize *padsize*numberZ[0];
		int sendslicesize =padsize*numberZ[0];
		for(int i=0;i<numberZ[1];i++)
		{
			MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD);
			sendsliceoffset+=padsize*padsize;
		}
	}
}

void cpu_alltoall_inverse(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize)
{
// 4.2 : cpu all to all
	//send numberZ[0] to 1
	//recv numberZ[1] from 1
	int rawranknum,desranknum;
	if(ranknum == 0)
		rawranknum =1;
	if(ranknum == 1)
		rawranknum =3;

	if(rawranknum ==1)
		desranknum=3;
	if(rawranknum ==3)
		desranknum=1;
	if(ranknum==1)
	{
		int sendsliceoffset=padsize*numberZ[0];
		int sendslicesize=padsize*numberZ[1];
		for(int i=0;i<numberZ[0];i++)
		{
			cudaMemcpy(cpu_data+sendsliceoffset,plan[0].d_Data + sendsliceoffset,sendslicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
			MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD);
			sendsliceoffset+=padsize*padsize;
		}
		int recvsliceoffset=padsize*padsize*numberZ[0];
		int recvslicesize=padsize*numberZ[0];
		for(int i=0;i<numberZ[1];i++)
		{
			MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, desranknum, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
			recvsliceoffset += padsize*padsize;
		}
	}
	if(ranknum==0)
	{

		int recvsliceoffset=padsize*numberZ[0];
		int recvslicesize=padsize*numberZ[1];
		for(int i=0;i<numberZ[0];i++)
		{
			MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, desranknum, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
			cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
			recvsliceoffset += padsize*padsize;
		}
		int sendsliceoffset = padsize *padsize*numberZ[0];
		int sendslicesize =padsize*numberZ[0];
		for(int i=0;i<numberZ[1];i++)
		{
			cudaMemcpy(cpu_data+sendsliceoffset,plan[0].d_Data + sendsliceoffset,sendslicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
			MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD);
			sendsliceoffset+=padsize*padsize;
		}
	}
}



void cpu_alltoalltozero(cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize)
{
	if(ranknum==0)
	{
		int recvsliceoffset = padsize*numberZ[0];
		int recvslicesize =padsize*numberZ[1];
		for(int i=0;i<padsize;i++)
		{
			MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,3,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			recvsliceoffset+=padsize*padsize;
		}
	}
	if(ranknum==1)
	{
		int sendsliceoffset = padsize*numberZ[0];
		int sendslicesize =padsize*numberZ[1];
		for(int i=0;i<padsize;i++)
		{
			MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,1,0,MPI_COMM_WORLD);
			sendsliceoffset+=padsize*padsize;
		}
	}
}
void cpu_allcombine(cufftComplex *cpu_data,int ranknum, int *numberZ, int *offsetZ,int padsize)
{
	if(ranknum==0)
	{
		int recvsliceoffset = numberZ[0]*padsize*padsize;
		int recvslicesize = numberZ[1]*padsize*padsize;

		MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,3,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	}
	if(ranknum==1)
	{
		int sendsliceoffset = numberZ[0]*padsize*padsize;
		int sendslicesize= numberZ[1]*padsize*padsize;

		MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,1,0,MPI_COMM_WORLD);

	}
}
/*
void multi_plan_init_mpi(MultiGPUplan *plan, int GPU_N, size_t fullsize, int dimx,int dimy,int dimz,int ranknum)
{

	plan[0].devicenum = ranknum;
	plan[0].datasize = fullsize;
	if(ranknum==0)
		plan[0].selfoffset = 0;
	if(ranknum==1)
		plan[0].selfoffset = dimx * dimy * (dimz / 2);
}*/
void printres(cufftComplex *cpu_data,  int *numberZ ,int *offsetZ,int pad_size,int ranknum)
{
	int starindex=(0);
	int endindex=numberZ[ranknum]*pad_size*pad_size;
	int nonzeronum=0;
	float sumnum=0;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i].x !=0)
			nonzeronum++;
		if(i<10)
		printf("%f ",cpu_data[i].x);
		sumnum+=cpu_data[i].x;
	}
	printf("block1  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);

	nonzeronum=0;sumnum=0;
	starindex= endindex;
	endindex = pad_size*pad_size*pad_size;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i].x !=0)
			nonzeronum++;
		sumnum+=cpu_data[i].x;
	}
	printf("block2  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);
}
void printwhole(double *cpu_data,  int fullszie ,int ranknum)
{
	int starindex=(0);
	int endindex=fullszie/2;
	int nonzeronum=0;
	float sumnum=0;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i] !=0)
			nonzeronum++;
		if(i<10)
		printf("%f ",cpu_data[i]);
		sumnum+=cpu_data[i];
	}
	printf("block1  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);

	nonzeronum=0;sumnum=0;
	starindex= endindex;
	endindex = fullszie;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i] !=0)
			nonzeronum++;
		sumnum+=cpu_data[i] ;
	}
	printf("block2  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);
}
void printwhole(cufftComplex *cpu_data,  int fullszie ,int ranknum)
{
	int starindex=(0);
	int endindex=fullszie/2;
	int nonzeronum=0;
	float sumnum=0;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i].x !=0)
			nonzeronum++;
		if(i<10)
		printf("%f ",cpu_data[i].x);
		sumnum+=cpu_data[i].x;
	}
	printf("block1  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);

	nonzeronum=0;sumnum=0;
	starindex= endindex;
	endindex = fullszie;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i].x !=0)
			nonzeronum++;
		sumnum+=cpu_data[i].x ;
	}
	printf("block2  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);
}

