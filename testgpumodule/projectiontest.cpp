
#include "../reconstructor.h"
#include "cufft.h"
#include "complex.h"
#include <cmath>

#include <sys/time.h>

void uncompress_projection_data(MultidimArray<RFLOAT > &desdata, MultidimArray<RFLOAT > srcdata,int *ydata, int pad_size,int xdim)
{

	size_t rawoffset=0;
	size_t compressoffset=0;
	size_t ydim=pad_size;
	size_t zdim=pad_size;


	desdata.initZeros(zdim,ydim,xdim);


	for(size_t i=0;i<zdim;i++)
		for(size_t j=0;j<ydim;j++)
		{
			size_t curiindex=i*ydim+j;
			if(ydata[curiindex] !=0 )
			{
				for(size_t k=0;k<ydata[curiindex];k++)
				{
					//desdata.data[rawoffset+k].real=BPref[0].compdatareal.data[compressoffset+k];
					//BPref[0].data.data[rawoffset+k].imag=BPref[0].compdataimag.data[compressoffset+k];
					desdata.data[rawoffset+k]=srcdata.data[compressoffset+k];
				}
			}

			compressoffset += ydata[curiindex];
			rawoffset+=xdim;
		}
/*	BPref[0].compdatareal.clear();
	BPref[0].compdataimag.clear();
	BPref[0].compweight.clear();
	free(BPref[0].ydata);
	free(BPref[0].yoffsetdata);*/

}
/*
void compress_projection_data(MultidimArray<RFLOAT > realdata,MultidimArray<RFLOAT > imagdata, MultidimArray<Complex > data,int *ydata, int pad_size,)
{

	size_t rawoffset=0;
	size_t compressoffset=0;
	size_t ydim=data.ydim;
	size_t zdim=data.zdim;
	size_t xdim=data.xdim;

	realdata.initZeros(zdim,ydim,xdim);
	imagdata.initZeros(zdim,ydim,xdim);


	printf("test weight : %ld %ld \n",realdata.yinit,realdata.zinit);


	for(size_t i=0;i<zdim;i++)
		for(size_t j=0;j<ydim;j++)
		{
			size_t curiindex=i*ydim+j;
			if(ydata[curiindex] !=0 )
			{
				//memset(data.data+);
				for(size_t k=0;k<BPref[0].ydata[curiindex];k++)
				{
					realdata.data[compressoffset+k]=data.data[rawoffset+k].real;
					imagdata.data[compressoffset+k] = data.data[rawoffset+k].imag;
				}
			}
			compressoffset += ydata[curiindex];
			rawoffset+=xdim;
		}
	data.clear();
	//free(BPref[0].ydata);
	//free(BPref[0].yoffsetdata);
}*/
void compress_projection_data(MultidimArray<RFLOAT > &desdata, MultidimArray<RFLOAT > srcdata,int *ydata, int pad_size,int sumalldata)
{

	size_t rawoffset=0;
	size_t compressoffset=0;
	size_t ydim=srcdata.ydim;
	size_t zdim=srcdata.zdim;
	size_t xdim=srcdata.xdim;

	desdata.initZeros(sumalldata);
	//imagdata.initZeros(zdim,ydim,xdim);


	printf("test weight : %ld %ld \n",srcdata.yinit,srcdata.zinit);


	for(size_t i=0;i<zdim;i++)
		for(size_t j=0;j<ydim;j++)
		{
			size_t curiindex=i*ydim+j;
			if(ydata[curiindex] !=0 )
			{
				//memset(data.data+);
				for(size_t k=0;k<ydata[curiindex];k++)
				{
					desdata.data[compressoffset+k]=srcdata.data[rawoffset+k];

					//if(srcdata.data[rawoffset+k] !=0)
					//	printf("%d %d \n",rawoffset+k,compressoffset+k);
				} //206576 132900
			}
			compressoffset += ydata[curiindex];
			rawoffset+=xdim;
		}
	//srcdata.clear();
	//free(BPref[0].ydata);
	//free(BPref[0].yoffsetdata);
}
/*
void initcompressdata(MultidimArray<RFLOAT > realdata,MultidimArray<RFLOAT > imagdata, int *ydata, int pad_size)
{


	int max_r2= ROUND((r_max+2) * padding_factor) * ROUND((r_max+2) * padding_factor);

	ydata=(int *)malloc(sizeof(int)*pad_size*pad_size);
	memset(ydata,0,sizeof(int)*pad_size*pad_size);
	for(int iz=0;iz<pad_size;iz++)
		for(int jy=0;jy<pad_size;jy++)
		{
			int xtemp=max_r2 - (iz+ data.zinit)*(iz+ data.zinit) - (jy+data.yinit)*(jy+data.yinit);
			if(xtemp<=0)
				ydata[iz*pad_size+jy]= 0;
			else
			{
				int ydatatemp=(int) sqrt(xtemp-0.01)+1;
				if(ydatatemp>data.xdim)
					ydata[iz*pad_size+jy]= data.xdim;
				else
					ydata[iz*pad_size+jy]=ydatatemp;
			}

		}
	yoffsetdata=(size_t *)malloc(sizeof(size_t)*pad_size*pad_size);
	yoffsetdata[0]=0;
	for(int cur=1;cur<pad_size*pad_size;cur++)
		yoffsetdata[cur]=yoffsetdata[cur-1]+ydata[cur-1];
	sumalldata=yoffsetdata[pad_size*pad_size-1]+ydata[pad_size*pad_size-1];
	compdatareal.resize(sumalldata);
	compdataimag.resize(sumalldata);
}*/

/*
int main()
{


//	printf("%s \n",fn_proj);

	FileName fn_root_real = "2w5cpu_itreal";
	FileName fn_root_imag = "2w5cpu_itimag";

	Image<RFLOAT> Itmpreal,Itmpimag;

	int iter=19;
	fn_root_real.compose(fn_root_real, iter, "mrc", 3);


	printf("%s \n",fn_root_real.c_str());

	// Read temporary arrays back in
	Itmpreal.read(fn_root_real);

//	Itmp().setXmippOrigin();
//	Itmp().xinit=0;

	long int x,y,z,n;
	Itmpreal().getDimensions(x,y,z,n);
	printf("%d %d %d \n",x,y,z);
	//for(int i=0;i<10;i++)

	int pad_size=y;
	int xdim=x;
	int padding_factor = 2;
	int r_max = ((pad_size - 1)/2 -1 )/2;
	printf("%d \n",r_max);
	int max_r2= ROUND((r_max+2) * padding_factor) * ROUND((r_max+2) * padding_factor);
	int *ydata;
	Itmpreal().setXmippOrigin();
	Itmpreal().xinit=0;

	ydata=(int *)malloc(sizeof(int)*pad_size*pad_size);
	memset(ydata,0,sizeof(int)*pad_size*pad_size);
	for(int iz=0;iz<pad_size;iz++)
		for(int jy=0;jy<pad_size;jy++)
		{
			int xtemp=max_r2 - (iz+ Itmpreal().zinit)*(iz+ Itmpreal().zinit) - (jy+Itmpreal().yinit)*(jy+Itmpreal().yinit);
			if(xtemp<=0)
				ydata[iz*pad_size+jy]= 0;
			else
			{
				int ydatatemp=(int)sqrt(xtemp-0.01)+1;
				if(ydatatemp>Itmpreal().xdim)
					ydata[iz*pad_size+jy]= Itmpreal().xdim;
				else
					ydata[iz*pad_size+jy]=ydatatemp;
			}

		}

	size_t *yoffsetdata=(size_t *)malloc(sizeof(size_t)*pad_size*pad_size);
	yoffsetdata[0]=0;
	for(int cur=1;cur<pad_size*pad_size;cur++)
		yoffsetdata[cur]=yoffsetdata[cur-1]+ydata[cur-1];
	size_t sumalldata=yoffsetdata[pad_size*pad_size-1]+ydata[pad_size*pad_size-1];
    MultidimArray<RFLOAT > compdatareal;
    MultidimArray<RFLOAT > compdataimag;
	compdatareal.resize(sumalldata);
	compdataimag.resize(sumalldata);
	printf("sumalldata : %d \n",sumalldata);
	compress_projection_data(compdatareal,Itmpreal(),ydata,pad_size,sumalldata);
//	compress_projection_data(compdataimag,Itmpreal(),ydata,pad_size,sumalldata);
	printf("compdatareal.data[1000] %f \n",compdatareal.data[132900]);
	MultidimArray<RFLOAT > newcomdata;

	newcomdata.resize(pad_size,pad_size,xdim);

	newcomdata.setXmippOrigin();
	newcomdata.xinit=0; // ! need for init
	uncompress_projection_data(newcomdata, compdatareal,ydata,pad_size,xdim);

	int count=0;
	printf("%d %d %d \n",newcomdata.xdim,newcomdata.ydim,newcomdata.zdim);
	printf("%d %d %d \n",newcomdata.xinit,newcomdata.yinit,newcomdata.zinit);

	Itmpreal.setStatisticsInHeader();
	Itmpreal.setSamplingRateInHeader(1);
	// And write the resulting model to disc
	Itmpreal.write("tempraw.mrc");

	Itmpimag()=newcomdata;
	Itmpimag.setStatisticsInHeader();
	Itmpimag.setSamplingRateInHeader(1);
	// And write the resulting model to disc
	Itmpimag.write("tempcomm.mrc");


	FOR_ALL_ELEMENTS_IN_ARRAY3D(newcomdata)
	{
		if(A3D_ELEM(newcomdata, k, i, j) != A3D_ELEM(Itmpreal(), k, i, j))
		{
			A3D_ELEM(Itmpreal(), k, i, j)=1;
			count++;
		}
		else
			A3D_ELEM(Itmpreal(), k, i, j)=0;
	}



	Itmpreal.setStatisticsInHeader();
	Itmpreal.setSamplingRateInHeader(1);
	// And write the resulting model to disc
	Itmpreal.write("diff.mrc");
	printf("%d Success\n",count);



}*/

int main()
{


//	printf("%s \n",fn_proj);



//	Itmp().setXmippOrigin();
//	Itmp().xinit=0;
    struct timeval tv1,tv2;
    struct timezone tz;
    long int time_use;
    MultidimArray<RFLOAT > compdatareal;
    MultidimArray<RFLOAT > compdataimag;
	int parsize=360;

	int pad_size=parsize*2+3;
	int xdim=parsize+2;
	int padding_factor = 2;
	int r_max = ((pad_size - 1)/2 -1 )/2;

	int max_r2= ROUND((r_max+2) * padding_factor) * ROUND((r_max+2) * padding_factor);
	int *ydata;



	int xinit = 0;
	int yinit=  -(long int)((float) (pad_size) / 2.0);
	int zinit=  -(long int)((float) (pad_size) / 2.0);
	printf("%d %d\n",yinit,zinit);
	 gettimeofday (&tv1, &tz);

	ydata=(int *)malloc(sizeof(int)*pad_size*pad_size);
	memset(ydata,0,sizeof(int)*pad_size*pad_size);
	for(int iz=0;iz<pad_size;iz++)
		for(int jy=0;jy<pad_size;jy++)
		{
			int xtemp=max_r2 - (iz+ zinit)*(iz+ zinit) - (jy+yinit)*(jy+yinit);
			if(xtemp<=0)
				ydata[iz*pad_size+jy]= 0;
			else
			{
				int ydatatemp=(int)sqrt(xtemp-0.01)+1;
				if(ydatatemp > xdim)
					ydata[iz*pad_size+jy]= xdim;
				else
					ydata[iz*pad_size+jy]=ydatatemp;
			}

		}

	size_t *yoffsetdata=(size_t *)malloc(sizeof(size_t)*pad_size*pad_size);
	yoffsetdata[0]=0;
	for(int cur=1;cur<pad_size*pad_size;cur++)
		yoffsetdata[cur]=yoffsetdata[cur-1]+ydata[cur-1];
	size_t sumalldata=yoffsetdata[pad_size*pad_size-1]+ydata[pad_size*pad_size-1];



	compdatareal.resize(sumalldata);
	compdataimag.resize(sumalldata);


	printf("sumalldata : %lld \n",sumalldata);

	gettimeofday (&tv2, &tz);
	time_use=(tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec);
	printf("Time : %d \n",time_use);


	 gettimeofday (&tv1, &tz);
	free(yoffsetdata);
	free(ydata);
	gettimeofday (&tv2, &tz);
	time_use=(tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec);
	printf("Time2 : %d \n",time_use);

}

