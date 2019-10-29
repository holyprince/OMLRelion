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

void datainit(cufftComplex *data,int NXYZ)
{
        for (int i = 0; i < NXYZ; i++) {
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
	int nxy = dimx * dimy;
	int halfslice1 = (offsetZ[0]) * dimx; //Z reperesent Y
	int halfslice2 = (offsetZ[0]) * dimx * dimy;
	int sliceoffset021;
	int sliceoffset120;

	int deltanxy = 0;
	int cpysize01 = (offsetZ[1]) * dimx * sizeof(cufftComplex);
	int cpysize10 = (offsetZ[0]) * dimx * sizeof(cufftComplex);
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	for (int j = 0; j < offsetZ[0]; j++) {
		sliceoffset021 = halfslice1 + deltanxy;
		sliceoffset120 = halfslice2 + deltanxy;

		cudaMemcpyAsync(plan[1].d_Data + sliceoffset021, plan[0].d_Data + sliceoffset021, cpysize01,cudaMemcpyDeviceToDevice, stream1);
		cudaMemcpyAsync(plan[0].d_Data + sliceoffset120, plan[1].d_Data + sliceoffset120, cpysize10,cudaMemcpyDeviceToDevice, stream2);

		deltanxy += nxy;
	}
	//extra for 1 z
	deltanxy = 0;

	for (int j = (dimz - extraz); j < dimz; j++) {
		cudaMemcpyAsync(plan[0].d_Data + (dimz - extraz) * dimx * dimy + deltanxy,
				plan[1].d_Data + (dimz - extraz) * dimx * dimy + deltanxy,cpysize10, cudaMemcpyDeviceToDevice, stream1);
		deltanxy += nxy;
	}
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);


}
void mulit_alltoall_two(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ)
{
	int nxy = dimx * dimy;
	int halfslice1 = (offsetZ[0]) * dimx; //Z reperesent Y
	int halfslice2 = (offsetZ[0]) * dimx * dimy;
	int cpysize01 = (offsetZ[1]) * dimx * sizeof(cufftComplex);
	int cpysize10 = (offsetZ[0]) * dimx * sizeof(cufftComplex);
	int deltanxy = 0;
	int sliceoffset021,sliceoffset120;
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	for (int j = 0; j < offsetZ[0]; j++) {
		sliceoffset021 = halfslice1 + deltanxy;
		sliceoffset120 = halfslice2 + deltanxy;

		cudaMemcpyAsync(plan[0].d_Data + sliceoffset021,
				plan[1].d_Data + sliceoffset021, cpysize01,
				cudaMemcpyDeviceToDevice, stream1);
		cudaMemcpyAsync(plan[1].d_Data + sliceoffset120,
				plan[0].d_Data + sliceoffset120, cpysize10,
				cudaMemcpyDeviceToDevice, stream2);
		deltanxy += nxy;
	}
	deltanxy = 0;
	for (int j = (dimz - extraz); j < dimz; j++) {
		cudaMemcpyAsync(plan[1].d_Data + (dimz - extraz) * dimx * dimy + deltanxy,
				plan[0].d_Data + (dimz - extraz) * dimx * dimy + deltanxy,
				cpysize10, cudaMemcpyDeviceToDevice, stream1);
		deltanxy += nxy;
	}
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

}

void mulit_alltoall_all1to0(MultiGPUplan *plan, int dimx,int dimy,int dimz, int extraz,int *offsetZ)
{
	int nxy = dimx * dimy;
	int halfslice1 = (offsetZ[0]) * dimx; //Z reperesent Y
	int cpysize01 = (offsetZ[1]) * dimx * sizeof(cufftComplex);
	int deltanxy = 0;
	int sliceoffset021;

	for (int j = 0; j < dimz; j++) {
		sliceoffset021 = halfslice1 + deltanxy;
		cudaMemcpyAsync(plan[0].d_Data + sliceoffset021, plan[1].d_Data + sliceoffset021, cpysize01,cudaMemcpyDeviceToDevice);

		deltanxy += nxy;
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

