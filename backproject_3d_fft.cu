#include "backproject_impl.h"

int self_MPI_Send(void *buf, size_t count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {

    int result(0);
    RFLOAT start_time = MPI_Wtime();

//#define ONLY_NORMAL_SEND
//#ifdef ONLY_NORMAL_SEND
    int unitsize(0);
    MPI_Type_size(datatype, &unitsize);
    size_t blocksize(512*1024*1024);
    size_t totalsize(count*unitsize);
    if (totalsize <= blocksize ) {
        result = MPI_Send(buf, count, datatype, dest, tag, comm);

    } else {
        char * const buffer(reinterpret_cast<char*>(buf));
        size_t ntimes(totalsize/blocksize);
        size_t nremain(totalsize%blocksize);
//        if(rank == 1)
//        printf("send divide block %d \n",ntimes);
        std::ptrdiff_t i(0);
        for(; i<ntimes; ++i) {
            result = MPI_Send(buffer+i*blocksize, blocksize, MPI_CHAR, dest, tag, comm);

        }
        if(nremain>0) {
            result = MPI_Send(buffer+i*blocksize, nremain, MPI_CHAR, dest, tag, comm);

        }
    }


#ifdef VERBOSE_MPISENDRECV
        if (count > 100)
                std::cerr <<" relion_MPI_Send: message to " << dest << " of size "<< count << " arrived in " << MPI_Wtime() - start_time << " seconds" << std::endl;
#endif
        return result;

}

int self_MPI_Recv(void *buf, size_t count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) {
    int result=0;


    int unitsize(0);
    MPI_Type_size(datatype, &unitsize);
    size_t blocksize(512*1024*1024);
    size_t totalsize(count*unitsize);
    if (totalsize <= blocksize ) {
        int result_irecv = MPI_Recv(buf, count, datatype, source, tag, comm,status);

    } else {
        char * const buffer(reinterpret_cast<char*>(buf));
        size_t ntimes(totalsize/blocksize);
        size_t nremain(totalsize%blocksize);
        size_t i(0);
        for(; i<ntimes; ++i) {
            int result_irecv = MPI_Recv(buffer+i*blocksize, blocksize, MPI_CHAR, source, tag, comm,status);

        }
        if(nremain>0) {
            int result_irecv = MPI_Recv(buffer+i*blocksize, nremain, MPI_CHAR, source, tag, comm,status);
        }
    }
#ifdef VERBOSE_MPISENDRECV
        if (count > 100)
                std::cerr <<" relion_MPI_Recv: message from "<<source << " of size "<< count <<" arrived in " << MPI_Wtime() - start_time << " seconds" << std::endl;
#endif
        return result;

}

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

void multi_plan_init(MultiGPUplan *plan, int GPU_N, size_t fullsize, int *numberZ, int *offsetZ, int pad_size)
{
	for (int i = 0; i < GPU_N; i++) {
		//deviceNum[i] = i;
		plan[i].devicenum = i;
		plan[i].datasize = fullsize;
		plan[i].selfoffset = pad_size* pad_size * offsetZ[i];
		plan[i].selfZ =  numberZ[i];
		plan[i].realsize= pad_size* pad_size* numberZ[i];
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

void dividetask(int *numberZ, int *offsetZ,int pad_size,int ranksize)
{
	//divide task
	//process divide data

	int baseznum= pad_size / ranksize;
	int extranum= pad_size % ranksize;
	for(int i=0;i<ranksize;i++)
	{
		numberZ[i]=baseznum;
		if(i<extranum)
			numberZ[i]++;
		if(i == 0)
			offsetZ[0]=0;
		else
			offsetZ[i] = offsetZ[i-1]+numberZ[i-1];
	}

}


void multi_plan_init_transpose(MultiGPUplan *plan, int GPU_N, int *numberZ, int *offsetZ, int pad_size)
{
	//MultiGPUplan plan[MAXGPU];
	for (int i = 0; i < GPU_N; i++) {
		//deviceNum[i] = i;
		plan[i].devicenum = i;
		plan[i].datasize = pad_size* pad_size * numberZ[i];
		plan[i].selfoffset = pad_size* pad_size * offsetZ[i];
		plan[i].selfZ =  numberZ[i];
		plan[i].tempydim = numberZ[0]*GPU_N;
		plan[i].tempsize = pad_size * plan[i].tempydim *numberZ[0]; // use the max dimy and dimz
		plan[i].realsize= plan[i].datasize;
	}

}
void multi_plan_init_transpose_multi(MultiGPUplan *plan, int GPU_N, int *numberZ, int *offsetZ, size_t pad_size,int ranknum,int ranksize)
{
	//MultiGPUplan plan[MAXGPU];
	int reali;
	for (int i = 0; i < GPU_N; i++) {
		//deviceNum[i] = i;
		reali = i+ranknum*GPU_N;
		plan[i].devicenum = i;
		plan[i].datasize = pad_size* pad_size * numberZ[reali];
		plan[i].selfoffset = pad_size* pad_size * offsetZ[reali];
		plan[i].selfZ =  numberZ[reali];
		plan[i].tempydim = numberZ[0]*GPU_N*ranksize;
		plan[i].tempsize = pad_size * plan[i].tempydim *numberZ[0]; // use the max dimy and dimz
		plan[i].realsize= plan[i].datasize;
	}

}


void multi_enable_access(MultiGPUplan *plan,int GPU_N)
{
	int can_access_peer = -100;
	for (int i = 0; i < GPU_N; i++) {
		for(int j=0;j< GPU_N;j++)
		{
			if(i!=j)
			{
				cudaDeviceCanAccessPeer(&can_access_peer, plan[i].devicenum,plan[j].devicenum);
				printf("%d to %d : access %d\n",i,j,can_access_peer);
			}
		}
	}

	for (int i = 0; i < GPU_N; i++) {
		cudaSetDevice(plan[i].devicenum);
		for(int j=0;j<GPU_N;j++)
		{
			if(i!=j)
				cudaDeviceEnablePeerAccess(j, 0);
		}
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

void gpu_alltoall_multinode(MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ)
{
	int NX=pad_size;
	int NY=pad_size;
	for(int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(plan[i].devicenum);
		int baseoffset=offsetZ[i]*pad_size*pad_size;
		for(int desnum=0;desnum<GPU_N;desnum++)  // i -> desnum
		{
			if(i == desnum) continue;
			int offsetcpydes=offsetZ[desnum]*NX + baseoffset;     //src == dev?
			int offsetcpysrc=offsetZ[desnum]*NX + baseoffset;
			for(int j=0;j<plan[i].selfZ;j++)
			{
				cudaMemcpy(plan[desnum].d_Data+offsetcpydes,plan[i].d_Data+offsetcpysrc,NX*plan[desnum].selfZ*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
				offsetcpydes+= NX*NY;
				offsetcpysrc+= NX*NY;
			}
		}
		cudaDeviceSynchronize();
	}
}
void gpu_alltoall_multinode_inverse(MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ)
{
	int NX=pad_size;
	int NY=pad_size;
	for(int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(plan[i].devicenum);

		for(int desnum=0;desnum<GPU_N;desnum++)  // i -> desnum
		{
			int baseoffset=offsetZ[desnum]*pad_size*pad_size;
			if(i == desnum) continue;
			int offsetcpydes=offsetZ[i]*NX+ baseoffset;    //src == dev?
			int offsetcpysrc=offsetZ[i]*NX+ baseoffset;
			for(int j=0;j<plan[desnum].selfZ;j++)
			{
				cudaMemcpy(plan[desnum].d_Data+offsetcpysrc,plan[i].d_Data+offsetcpydes, NX*plan[i].selfZ*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
				offsetcpydes+= NX*NY;
				offsetcpysrc+= NX*NY;
			}
		}
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
	//printgpures(cpu_data + plan[0].selfoffset,plan[0].realsize,0);

}
void gpu_to_cpu_1dfft(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int *offsetZ,int padsize,int ranknum)
{

	int sliceoffset = padsize*offsetZ[ranknum];
	int slicesize =padsize*numberZ[ranknum];
	cufftComplex *start=cpu_data + sliceoffset;
	double sum=0;
	int countzero=0;
	for(int i=0;i<padsize;i++)
	{
		cudaMemcpy(cpu_data + sliceoffset,plan[0].d_Data +sliceoffset, slicesize * sizeof(cufftComplex),cudaMemcpyDeviceToHost);
		for(int j=0;j<slicesize;j++)
		{
			if(start[j].x!=0)
				countzero++;
			sum+=start[j].x;
		}
		start+=padsize*padsize;
		sliceoffset +=padsize*padsize;
	}
	printf("Rank is %d count %d and sum %f \n",ranknum,countzero,sum);
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
		rawranknum =0;
	if(ranknum == 1)
		rawranknum =1;


	if(rawranknum ==0)
		desranknum=1;
	if(rawranknum ==1)
		desranknum=0;




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
int getrawranknum(int ranknum)
{
	int rawranknum;
	if(ranknum == 0)
		rawranknum =1;
	if(ranknum == 1)
		rawranknum =3;
/*	if(ranknum == 2)
		rawranknum =2;
	if(ranknum == 3)
		rawranknum =3;*/
	return rawranknum;
}
int getdesranknum(int ranknum)
{
	int desranknum;
	if(ranknum == 0)
		desranknum =3;
	if(ranknum == 1)
		desranknum =1;
/*	if(ranknum == 2)
		desranknum =2;
	if(ranknum == 3)
		desranknum =3;*/
	return desranknum;
}
void cpu_alltoall_multinode(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int *offsetZ,int ranknum,int padsize,int ranksize,int *realrankarray)
{
// 4.2 : cpu all to all
	//send numberZ[0] to 1
	//recv numberZ[1] from 1
	int desranknum;

	if(ranknum==0)
	{
		for(int i=1;i<ranksize;i++)
		{
			int sendsliceoffset=padsize*offsetZ[i];
			int sendslicesize=padsize*numberZ[i];
			for(int slicenum=0;slicenum<numberZ[0];slicenum++)
			{
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padsize;
			}
			int recvsliceoffset=padsize*padsize*offsetZ[i];
			int recvslicesize=padsize*numberZ[0];
			for(int slicenum=0;slicenum<numberZ[i];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[i], 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				recvsliceoffset += padsize*padsize;
			}
		}
	}
	else
	{
		desranknum = 0;  //all recv and send from 0
		int recvsliceoffset=padsize*offsetZ[ranknum];
		int recvslicesize=padsize*numberZ[ranknum];
		for(int slicenum=0;slicenum<numberZ[desranknum];slicenum++)
		{
			MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[desranknum] , 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
			cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
			recvsliceoffset += padsize*padsize;
		}
		int sendsliceoffset = padsize *padsize*offsetZ[ranknum];
		int sendslicesize =padsize*numberZ[desranknum];
		for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
		{
			MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[desranknum],0,MPI_COMM_WORLD);
			sendsliceoffset+=padsize*padsize;
		}
	}

	if(ranknum==1)
	{
		//rank 1 send
		for(int i=2;i<ranksize;i++)
		{
			int globaloffset=padsize*padsize*offsetZ[ranknum]; //globaloffset offsetZ[1] = number[0]
			int sendsliceoffset=padsize*offsetZ[i]+globaloffset;
			int sendslicesize=padsize*numberZ[i];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padsize;
			}
			globaloffset=padsize*offsetZ[ranknum];
			int recvsliceoffset=padsize*padsize*offsetZ[i]+globaloffset;
			int recvslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[i];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[i], 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				recvsliceoffset += padsize*padsize;
			}
		}
	}
	else
	{
		//rank 1 recv
		if(ranknum==2||ranknum==3)
		{
			desranknum = 1;
			int globaloffset=padsize*padsize*offsetZ[desranknum];
			int recvsliceoffset=padsize*offsetZ[ranknum]+globaloffset;
			int recvslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[desranknum];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[desranknum] , 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
				cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				recvsliceoffset += padsize*padsize;
			}
			globaloffset=padsize*offsetZ[desranknum];
			int sendsliceoffset = padsize *padsize*offsetZ[ranknum]+globaloffset;
			int sendslicesize =padsize*numberZ[desranknum];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[desranknum],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padsize;
			}
		}
	}
	if(ranknum==2)
	{
		for(int i=3;i<ranksize;i++)
		{
			int globaloffset=padsize*padsize*offsetZ[ranknum]; //globaloffset offsetZ[1] = number[0]
			int sendsliceoffset=padsize*offsetZ[i]+globaloffset;
			int sendslicesize=padsize*numberZ[i];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padsize;
			}
			globaloffset=padsize*offsetZ[ranknum];
			int recvsliceoffset=padsize*padsize*offsetZ[i]+globaloffset;
			int recvslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[i];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[i], 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				recvsliceoffset += padsize*padsize;
			}
		}
	}
	else
	{
		if(ranknum==3)
		{
			desranknum = 2;
			int globaloffset=padsize*padsize*offsetZ[desranknum];
			int recvsliceoffset=padsize*offsetZ[ranknum]+globaloffset;
			int recvslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[desranknum];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[desranknum] , 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
				cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				recvsliceoffset += padsize*padsize;
			}
			globaloffset=padsize*offsetZ[desranknum];
			int sendsliceoffset = padsize *padsize*offsetZ[ranknum]+globaloffset;
			int sendslicesize =padsize*numberZ[desranknum];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[desranknum],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padsize;
			}
		}
	}

}

void cpu_alltoalltrans_multinode(MultiGPUplan *plan,cufftComplex *cpu_data,size_t padsize, int *numberZ,int *offsetZ,int ranknum,int padtmpsize,int ranksize,int *realrankarray)
{
// 4.2 : cpu all to all
	//send numberZ[0] to 1
	//recv numberZ[1] from 1
	int desranknum;

	if(ranknum==0)
	{
		for(int i=1;i<ranksize;i++)
		{

			size_t sendsliceoffset = padsize *offsetZ[i];
			size_t sendslicesize =padsize*numberZ[i];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padtmpsize;
			}
			size_t recvsliceoffset=padsize* padtmpsize *offsetZ[i];
			size_t recvslicesize= padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[i];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[i] , 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
				recvsliceoffset += padsize*padtmpsize;
			}


		}
	}
	else
	{
		desranknum = 0;  //all recv and send from 0
		size_t recvsliceoffset=padsize*offsetZ[ranknum];
		size_t recvslicesize=padsize*numberZ[ranknum];
		for(int slicenum=0;slicenum<numberZ[desranknum];slicenum++)
		{
			MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[desranknum], 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			recvsliceoffset += padsize*padtmpsize;
		}
		size_t sendsliceoffset=padsize* padtmpsize *offsetZ[ranknum];
		size_t sendslicesize=padsize*numberZ[desranknum];
		for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
		{
			MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[desranknum],0,MPI_COMM_WORLD);
			sendsliceoffset+=padsize*padtmpsize;
		}

	}

	if(ranknum==1)
	{
		//rank 1 send
		for(int i=2;i<ranksize;i++)
		{
			size_t globaloffset=padsize*padsize*offsetZ[ranknum]; //globaloffset offsetZ[1] = number[0]
			size_t sendsliceoffset=padsize*offsetZ[i]+globaloffset;
			int sendslicesize=padsize*numberZ[i];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padsize;
			}
			globaloffset=padsize*offsetZ[ranknum];
			int recvsliceoffset=padsize*padsize*offsetZ[i]+globaloffset;
			int recvslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[i];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[i], 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				recvsliceoffset += padsize*padsize;
			}
		}
	}
	else
	{
		//rank 1 recv
		if(ranknum==2||ranknum==3)
		{
			desranknum = 1;
			int globaloffset=padsize*padsize*offsetZ[desranknum];
			int recvsliceoffset=padsize*offsetZ[ranknum]+globaloffset;
			int recvslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[desranknum];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[desranknum] , 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
				recvsliceoffset += padsize*padsize;
			}
			globaloffset=padsize*offsetZ[desranknum];
			int sendsliceoffset = padsize *padsize*offsetZ[ranknum]+globaloffset;
			int sendslicesize =padsize*numberZ[desranknum];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[desranknum],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padsize;
			}
		}
	}
	if(ranknum==2)
	{
		for(int i=3;i<ranksize;i++)
		{
			size_t globaloffset=padsize*padsize*offsetZ[ranknum]; //globaloffset offsetZ[1] = number[0]
			size_t sendsliceoffset=padsize*offsetZ[i]+globaloffset;
			int sendslicesize=padsize*numberZ[i];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padsize;
			}
			globaloffset=padsize*offsetZ[ranknum];
			int recvsliceoffset=padsize*padsize*offsetZ[i]+globaloffset;
			int recvslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[i];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[i], 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				recvsliceoffset += padsize*padsize;
			}
		}
	}
	else
	{
		if(ranknum==3)
		{
			desranknum = 2;
			int globaloffset=padsize*padsize*offsetZ[desranknum];
			int recvsliceoffset=padsize*offsetZ[ranknum]+globaloffset;
			int recvslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[desranknum];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset, recvslicesize*2, MPI_FLOAT, realrankarray[desranknum] , 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE );
				recvsliceoffset += padsize*padsize;
			}
			globaloffset=padsize*offsetZ[desranknum];
			int sendsliceoffset = padsize *padsize*offsetZ[ranknum]+globaloffset;
			int sendslicesize =padsize*numberZ[desranknum];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[desranknum],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padsize;
			}
		}
	}
}


void cpu_alltozero_multinode(cufftComplex *pad_data,size_t padsize,
		int *processnumberZ, int *processoffsetZ, int ranknum,size_t padtmpsize,int ranksize,int *realrankarray)
{


	//all process send data to zero
	int rawranknum,desranknum;

	if(ranknum==0)
	{
		for(int i=1;i<ranksize;i++)
		{
			size_t recvsliceoffset = padtmpsize*padsize*processoffsetZ[i];
			self_MPI_Recv(pad_data+recvsliceoffset,processnumberZ[i]*padtmpsize*padsize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}
	}
	else
	{
		desranknum=realrankarray[0];
		size_t sendsliceoffset = padtmpsize*padsize*processoffsetZ[ranknum];
		//MPI_Send(pad_data+sendsliceoffset,processnumberZ[ranknum]*padtmpsize*padsize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD);
		self_MPI_Send(pad_data+sendsliceoffset,processnumberZ[ranknum]*padtmpsize*padsize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD); //void *buf, std::ptrdiff_t count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm
	}
    // rearrage pad_data to cpudata
}

void cpu_datarearrange_multinode(cufftComplex *pad_data,cufftComplex *real_data,size_t padsize, int *numberZ,int *offsetZ,
		int *numbertmpZ,int *offsettmpZ, int ranknum,size_t padtmpsize,int sumranknum)
{
	size_t srcaddr=0;
	size_t desaddr=0;
	size_t slicesize=0;
	for(int i=0;i<sumranknum;i++)
	{

		for(int j=0;j<sumranknum;j++)
		{
			srcaddr=offsettmpZ[i]*padsize + offsettmpZ[j]*padsize * padtmpsize;
			desaddr = offsetZ[j]*padsize  + offsetZ[i]*padsize*padsize;
			slicesize = numberZ[j]*padsize;
			for(int slicenum=0;slicenum<numberZ[i];slicenum++)   // yiding yu i xiangguna
			{

				memcpy(real_data+desaddr,pad_data+srcaddr,sizeof(cufftComplex)*slicesize);
				srcaddr+=padsize * padtmpsize;
				desaddr+=padsize * padsize;
			}
		}
	}

}
void validatealltoall(cufftComplex *cpu_data,int *numberZ,int *offsetZ, int ranknum,int padsize)
{

	/*
		float sum=0;
		int countzero=0;
		for(int i=0;i<plan[0].realsize;i++)
		{
			if(start[i].x!=0)
				countzero++;
			sum+=start[i].x;
		}
		*/
	int sliceoffset = padsize*offsetZ[ranknum];
	int slicesize =padsize*numberZ[ranknum];
	cufftComplex *start=cpu_data + sliceoffset;
	double sum=0;
	int countzero=0;
	for(int i=0;i<padsize;i++)
	{
		for(int j=0;j<slicesize;j++)
		{
			if(start[j].x!=0)
				countzero++;
			sum+=start[j].x;
		}
		start+=padsize*padsize;
	}
	printf("Rank is %d count %d and sum %f \n",ranknum,countzero,sum);
}
void cpu_alltoall_inverse(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize)
{
// 4.2 : cpu all to all
	//send numberZ[0] to 1
	//recv numberZ[1] from 1
	int rawranknum,desranknum;
	rawranknum=getrawranknum(ranknum);
	desranknum=getdesranknum(ranknum);

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

void cpu_alltoall_inverse_multinode(MultiGPUplan *plan,cufftComplex *cpu_data,int *numberZ,int *offsetZ,int ranknum,int padsize,int ranksize,int *realrankarray)
{
// 4.2 : cpu all to all
	//send numberZ[0] to 1
	//recv numberZ[1] from 1
	int rawranknum,desranknum;


	if(ranknum==0)
	{
		for(int i=1;i<ranksize;i++)
		{
			int sendsliceoffset=padsize*padsize*offsetZ[i];
			int sendslicesize=padsize*numberZ[0];
			for(int slicenum=0;slicenum<numberZ[i];slicenum++)
			{
				cudaMemcpy(cpu_data+sendsliceoffset,plan[0].d_Data + sendsliceoffset,sendslicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
				MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD);
				sendsliceoffset+=padsize*padsize;
			}

			int recvsliceoffset=padsize*offsetZ[i];
			int recvslicesize=padsize*numberZ[i];
			for(int slicenum=0;slicenum<numberZ[0];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				recvsliceoffset+=padsize*padsize;
			}
		}
	}
	else
	{
		//desranknum = realrankarray[0];  //all recv and send from 0
		desranknum = 0;
		int recvsliceoffset = padsize *padsize*offsetZ[ranknum];
		int recvslicesize =padsize*numberZ[desranknum];
		for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
		{
			MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,realrankarray[desranknum],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
			recvsliceoffset+=padsize*padsize;
		}

		int sendsliceoffset=padsize*offsetZ[ranknum];
		int sendslicesize=padsize*numberZ[ranknum];
		for(int slicenum=0;slicenum<numberZ[0];slicenum++)
		{
			cudaMemcpy(cpu_data+sendsliceoffset,plan[0].d_Data + sendsliceoffset,sendslicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
			MPI_Send(cpu_data+sendsliceoffset, sendslicesize*2, MPI_FLOAT, realrankarray[desranknum] , 0, MPI_COMM_WORLD );
			sendsliceoffset += padsize*padsize;
		}

	}

	if(ranknum==1)
	{
		//rank 1 send
		for(int i=2;i<ranksize;i++)
		{
			int globaloffset=padsize*offsetZ[ranknum];
			int sendsliceoffset=padsize*padsize*offsetZ[i]+globaloffset;
			int sendslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[i];slicenum++)
			{
				cudaMemcpy(cpu_data+sendsliceoffset,plan[0].d_Data + sendsliceoffset,sendslicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
				MPI_Send(cpu_data+sendsliceoffset, sendslicesize*2, MPI_FLOAT, realrankarray[i], 0, MPI_COMM_WORLD);
				sendsliceoffset += padsize*padsize;
			}
			globaloffset=padsize*padsize*offsetZ[ranknum]; //globaloffset offsetZ[1] = number[0]
			int recvsliceoffset=padsize*offsetZ[i]+globaloffset;
			int recvslicesize=padsize*numberZ[i];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				recvsliceoffset+=padsize*padsize;
			}
		}
	}
	else
	{
		//rank 1 recv
		if(ranknum==2||ranknum==3)
		{
			desranknum = 1;
			int globaloffset=padsize*offsetZ[desranknum];
			int recvsliceoffset = padsize *padsize*offsetZ[ranknum]+globaloffset;
			int recvslicesize =padsize*numberZ[desranknum];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,realrankarray[desranknum],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				recvsliceoffset+=padsize*padsize;
			}

			globaloffset=padsize*padsize*offsetZ[desranknum];
			int sendsliceoffset=padsize*offsetZ[ranknum]+globaloffset;
			int sendslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[desranknum];slicenum++)
			{
				cudaMemcpy(cpu_data+sendsliceoffset,plan[0].d_Data + sendsliceoffset,sendslicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
				MPI_Send(cpu_data+sendsliceoffset, sendslicesize*2, MPI_FLOAT, realrankarray[desranknum] , 0, MPI_COMM_WORLD );
				sendsliceoffset += padsize*padsize;
			}
		}
	}
	if(ranknum==2)
	{
		for(int i=3;i<ranksize;i++)
		{
			int globaloffset=padsize*offsetZ[ranknum];
			int sendsliceoffset=padsize*padsize*offsetZ[i]+globaloffset;
			int sendslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[i];slicenum++)
			{
				cudaMemcpy(cpu_data+sendsliceoffset,plan[0].d_Data + sendsliceoffset,sendslicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
				MPI_Send(cpu_data+sendsliceoffset, sendslicesize*2, MPI_FLOAT, realrankarray[i], 0, MPI_COMM_WORLD);
				sendsliceoffset += padsize*padsize;
			}

			globaloffset=padsize*padsize*offsetZ[ranknum]; //globaloffset offsetZ[1] = number[0]
			int recvsliceoffset=padsize*offsetZ[i]+globaloffset;
			int recvslicesize=padsize*numberZ[i];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				recvsliceoffset+=padsize*padsize;
			}

		}
	}
	else
	{
		if(ranknum==3)
		{
			desranknum = 2;
			int globaloffset=padsize*offsetZ[desranknum];
			int recvsliceoffset = padsize *padsize*offsetZ[ranknum]+globaloffset;
			int recvslicesize =padsize*numberZ[desranknum];
			for(int slicenum=0;slicenum<numberZ[ranknum];slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,realrankarray[desranknum],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				cudaMemcpy(plan[0].d_Data + recvsliceoffset,cpu_data+recvsliceoffset,recvslicesize*sizeof(cufftComplex),cudaMemcpyHostToDevice);
				recvsliceoffset+=padsize*padsize;
			}


			globaloffset=padsize*padsize*offsetZ[desranknum];
			int sendsliceoffset=padsize*offsetZ[ranknum]+globaloffset;
			int sendslicesize=padsize*numberZ[ranknum];
			for(int slicenum=0;slicenum<numberZ[desranknum];slicenum++)
			{
				cudaMemcpy(cpu_data+sendsliceoffset,plan[0].d_Data + sendsliceoffset,sendslicesize*sizeof(cufftComplex),cudaMemcpyDeviceToHost);
				MPI_Send(cpu_data+sendsliceoffset, sendslicesize*2, MPI_FLOAT, realrankarray[desranknum] , 0, MPI_COMM_WORLD);
				sendsliceoffset += padsize*padsize;
			}
		}
	}

}


void cpu_alltoalltozero(cufftComplex *cpu_data,int *numberZ,int ranknum,int padsize)
{
	int rawranknum,desranknum;
	if(ranknum == 0)
		rawranknum =0;
	if(ranknum == 1)
		rawranknum =1;

	if(rawranknum ==0)
		desranknum=1;
	if(rawranknum ==1)
		desranknum=0;


	if(ranknum==0)
	{
		int recvsliceoffset = padsize*numberZ[0];
		int recvslicesize =padsize*numberZ[1];
		for(int slicenum=0;slicenum<padsize;slicenum++)
		{
			MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			recvsliceoffset+=padsize*padsize;
		}
	}
	if(ranknum==1)
	{
		int sendsliceoffset = padsize*numberZ[0];
		int sendslicesize =padsize*numberZ[1];
		for(int i=0;i<padsize;i++)
		{
			MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD);
			sendsliceoffset+=padsize*padsize;
		}
	}
}

void cpu_alltoalltozero_multi(cufftComplex *cpu_data,int *numberZ,int *offsetZ,int ranknum,int padsize,int ranksize,int *realrankarray)
{
	int rawranknum,desranknum;

	if(ranknum==0)
	{
		for(int i=1;i<ranksize;i++)
		{
			int recvsliceoffset = padsize*offsetZ[i];
			int recvslicesize =padsize*numberZ[i];
			for(int slicenum=0;slicenum<padsize;slicenum++)
			{
				MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				recvsliceoffset+=padsize*padsize;
			}
		}

	}
	else
	{
		desranknum=realrankarray[0];
		int sendsliceoffset = padsize*offsetZ[ranknum];
		int sendslicesize =padsize*numberZ[ranknum];
		for(int slicenum=0;slicenum<padsize;slicenum++)
		{
			MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD);
			sendsliceoffset+=padsize*padsize;
		}
	}

}
void cpu_allcombine(cufftComplex *cpu_data,int ranknum, int *numberZ, int *offsetZ,int padsize)
{


	int rawranknum,desranknum;
	if(ranknum == 0)
		rawranknum =0;
	if(ranknum == 1)
		rawranknum =1;

	if(rawranknum ==0)
		desranknum=1;
	if(rawranknum ==1)
		desranknum=0;

	if(ranknum==0)
	{
		int recvsliceoffset = numberZ[0]*padsize*padsize;
		int recvslicesize = numberZ[1]*padsize*padsize;

		MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	}
	if(ranknum==1)
	{
		int sendsliceoffset = numberZ[0]*padsize*padsize;
		int sendslicesize= numberZ[1]*padsize*padsize;
		MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD);

	}
}
void cpu_allcombine_multi(cufftComplex *cpu_data,int ranknum, int *numberZ, int *offsetZ,int padsize,int ranksize,int *realrankarray)
{


	int desranknum;

	if(ranknum==0)
	{
		for(int i=1;i<ranksize;i++)
		{
			int recvsliceoffset = offsetZ[i]*padsize*padsize;
			int recvslicesize = numberZ[i]*padsize*padsize;
			MPI_Recv(cpu_data+recvsliceoffset,recvslicesize*2,MPI_FLOAT,realrankarray[i],0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		}

	}
	else
	{
		desranknum =realrankarray[0];
		int sendsliceoffset =  offsetZ[ranknum]*padsize*padsize;
		int sendslicesize= numberZ[ranknum]*padsize*padsize;
		MPI_Send(cpu_data+sendsliceoffset,sendslicesize*2,MPI_FLOAT,desranknum,0,MPI_COMM_WORLD);
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
void printgpures(double *cpu_data, int fullsize,int ranknum)
{
	int starindex=0;
	int endindex=fullsize;
	int nonrealzeronum=0;
	int nonimagzeronum=0;
	double sumrealnum=0;
	double sumimagnum=0;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i] !=0)
			nonrealzeronum++;
		if(i<10)
		printf("%f ",cpu_data[i]);
		sumrealnum+=cpu_data[i];
	}
	printf("\n double data num is %d and sum is %f\n",nonrealzeronum,sumrealnum);


}
void printgpures(float *cpu_data, int fullsize,int ranknum)
{
	int starindex=0;
	int endindex=fullsize;
	int nonrealzeronum=0;
	int nonimagzeronum=0;
	double sumrealnum=0;
	double sumimagnum=0;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i] !=0)
			nonrealzeronum++;
		if(i<10)
		printf("%f ",cpu_data[i]);
		sumrealnum+=cpu_data[i];
	}
	printf("\n float data num is %d and sum is %f\n",nonrealzeronum,sumrealnum);

}
void printgpures(cufftComplex *cpu_data, int fullsize,int ranknum)
{
	int starindex=0;
	int endindex=fullsize;
	int nonrealzeronum=0;
	int nonimagzeronum=0;
	double sumrealnum=0;
	double sumimagnum=0;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i].x !=0)
			nonrealzeronum++;
		if(cpu_data[i].y !=0)
			nonimagzeronum++;
		if(i<10)
		printf("%f ",cpu_data[i].x);
		sumrealnum+=cpu_data[i].x;
		sumimagnum+=cpu_data[i].y;
	}
	printf("\n real data num is %d and sum is %f\n",nonrealzeronum,sumrealnum);
	printf("imag data num is %d and sum is %f\n",sumimagnum,sumimagnum);

}
void printwhole(double *cpu_data,  int fullszie ,int ranknum)
{
	int starindex=(0);
	int endindex=fullszie/2;
	int nonzeronum=0;
	double sumnum=0;
	for(int i=starindex;i<fullszie;i++)
	{
		if(cpu_data[i] !=0)
			nonzeronum++;
		if(i<10)
		printf("%f ",cpu_data[i]);
		sumnum+=cpu_data[i];
	}
	printf("block1  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);
/*
	nonzeronum=0;sumnum=0;
	starindex= endindex;
	endindex = fullszie;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i] !=0)
			nonzeronum++;
		sumnum+=cpu_data[i] ;
	}
	printf("block2  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);*/
}
void printwhole(float *cpu_data,  size_t fullszie ,int ranknum)
{
	int starindex=(0);
	int endindex=fullszie/2;
	int nonzeronum=0;
	double sumnum=0;
	for(int i=starindex;i<fullszie;i++)
	{
		if(cpu_data[i] !=0)
			nonzeronum++;
		if(i<10)
		printf("%f ",cpu_data[i]);
		sumnum+=cpu_data[i];
	}
	printf("\nblock1  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);
/*
	nonzeronum=0;sumnum=0;
	starindex= endindex;
	endindex = fullszie;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i] !=0)
			nonzeronum++;
		sumnum+=cpu_data[i] ;
	}
	printf("block2  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);*/
}

void printwhole_const(double tempdata, int fullsize,int ranknum)
{

}
void printwhole(cufftComplex *cpu_data,  int fullszie ,int ranknum)
{
	int starindex=(0);
	int endindex=fullszie/2;
	int nonzeronum=0;
	float sumnum=0;
	for(int i=starindex;i<fullszie;i++)
	{
		if(cpu_data[i].x !=0)
			nonzeronum++;
		if(i<10)
		printf("%.4f ",cpu_data[i].x);
		sumnum+=cpu_data[i].x;
	}
	printf("\nblock1  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);
/*
	nonzeronum=0;sumnum=0;
	starindex= endindex;
	endindex = fullszie;
	for(int i=starindex;i<endindex;i++)
	{
		if(cpu_data[i].x !=0)
			nonzeronum++;
		sumnum+=cpu_data[i].x ;
		if(i>(endindex-10))
			printf("%f ",cpu_data[i].x);
	}
	printf("block2  : %d from rank %d and sum is %f\n",nonzeronum,ranknum,sumnum);*/
}

void transpose_exchange(MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ,int *offsettmpZ)
{
	int NX=pad_size;
	int NY=pad_size;
	for(int i = 0; i < GPU_N; i++)
	{
		cudaSetDevice(plan[i].devicenum);
		for(int desnum=0;desnum<GPU_N;desnum++)  // i -> desnum
		{
			int offsetcpydes=offsetZ[i]*NX;     //from where need coresponeing i offset
			int offsetcpysrc=offsettmpZ[desnum]*NX;
			for(int j=0;j<plan[desnum].selfZ;j++)
			{
				cudaMemcpy(plan[desnum].d_Data+offsetcpydes,plan[i].temp_Data+offsetcpysrc,NX*plan[i].selfZ*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
				offsetcpysrc+= NX* plan[i].tempydim ;
				offsetcpydes+= NX*NY;
			}
		}
		cudaDeviceSynchronize();
	}
}

void transpose_exchange_intra(MultiGPUplan *plan,int GPU_N,int pad_size,int *offsetZ ,int *numberZ,
		int *offsettmpZ,int ranknum)
{
	int NX=pad_size;
	int NY=pad_size;
	for(int i = 0; i < GPU_N; i++)
	{
		int reali= i+ranknum*GPU_N;
		cudaSetDevice(plan[i].devicenum);
		for(int desnum=0;desnum<GPU_N;desnum++)  // i -> desnum
		{
			size_t realdex=desnum+ranknum*GPU_N;
			size_t offsetcpydes=offsetZ[reali]*NX;     //from where need coresponeing i offset
			size_t offsetcpysrc=offsettmpZ[realdex]*NX;
			for(int j=0;j<numberZ[realdex];j++)
			{
				cudaMemcpy(plan[desnum].d_Data+offsetcpydes,plan[i].temp_Data+offsetcpysrc,NX*numberZ[reali]*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
				offsetcpysrc+= NX* plan[i].tempydim ;
				offsetcpydes+= NX*NY;
			}
		}
		cudaDeviceSynchronize();
	}
}

void data_exchange_gputocpu(MultiGPUplan *plan, cufftComplex *cpu_data, int GPU_N,size_t pad_size,int *offsetZ ,int ranknum)
{
	//step1: tmpdata all to device
	size_t offsetcpydes=0;
	for(int i = 0; i < GPU_N; i++)
	{
		int reali= i+ ranknum*GPU_N;
		cudaSetDevice(plan[i].devicenum);
		offsetcpydes = offsetZ[reali]*pad_size*plan[i].tempydim;
		cudaMemcpy(cpu_data+offsetcpydes,plan[i].temp_Data,plan[i].tempsize*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
}
void data_exchange_cputogpu(MultiGPUplan *plan, cufftComplex *cpu_data, int GPU_N,int pad_size,int *offsetZ ,
		int *numberZ, int *offsettmpZ, int *numbertmpZ, int ranknum,int sumranknum, int padsizetmp)
{
	//step1: tmpdata all to device
	size_t offsetcpydes=0;
	size_t NX=pad_size;
	//int NY=newpadsize;
	for(int i = 0; i < GPU_N; i++)
	{
		int reali= i+ ranknum*GPU_N;
		cudaSetDevice(plan[i].devicenum);

		for(int desnum=0;desnum<sumranknum;desnum++)
		{
			size_t srcstart= offsettmpZ[desnum] * NX* padsizetmp ;
			if(desnum >= ranknum*GPU_N  && desnum < (ranknum+1)*GPU_N) continue;
			else
			{
				size_t offsetcpydes=offsetZ[desnum]*NX;
				size_t offsetcpysrc=srcstart + offsettmpZ[reali]*NX;
				for(int j=0;j< numberZ[i];j++)
				{
					cudaMemcpy(plan[i].d_Data+ offsetcpydes, cpu_data+offsetcpysrc, NX*numberZ[desnum]*sizeof(cufftComplex) ,cudaMemcpyHostToDevice);
					offsetcpysrc += NX*padsizetmp;
					offsetcpydes += NX*pad_size;
				}
			}
		}
		cudaDeviceSynchronize();
	}
}
// from  desnum other to i

