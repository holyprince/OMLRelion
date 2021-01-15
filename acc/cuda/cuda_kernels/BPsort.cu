
#ifndef CUDA_BPS_KERNELS_CUH_
#define CUDA_BPS_KERNELS_CUH_


#include "BPsort.cuh"




void cuda_kernel_backproject3D_sortbykey(int *d_index,XFLOAT *d_real,XFLOAT *d_imag,XFLOAT *d_weight,int *d_count,int numElements,int &countnum)
{
	thrust::device_ptr<int> keys(d_index);
	thrust::device_ptr<XFLOAT> vals0(d_real);
	thrust::device_ptr<XFLOAT> vals1(d_imag);
	thrust::device_ptr<XFLOAT> vals2(d_weight);


	// allocate space for the output
	thrust::device_vector<XFLOAT> sortedvals0(numElements);
	thrust::device_vector<XFLOAT> sortedvals1(numElements);
	thrust::device_vector<XFLOAT> sortedvals2(numElements);

	// initialize indices vector to [0,1,2,..]
	thrust::counting_iterator<int> iter(0);
	thrust::device_vector<int> indices(numElements);
	thrust::copy(iter, iter + indices.size(), indices.begin());

	// first sort the keys and indices by the keys
	//thrust::sort_by_key(keys.begin(), keys.end(), indices.begin());
	thrust::sort_by_key(keys, keys+numElements, indices.begin(),thrust::greater<int>());

	// Now reorder the ID arrays using the sorted indices
	thrust::gather(indices.begin(), indices.end(), vals0, sortedvals0.begin());
	thrust::gather(indices.begin(), indices.end(), vals1, sortedvals1.begin());
	thrust::gather(indices.begin(), indices.end(), vals2, sortedvals2.begin());

	d_index = thrust::raw_pointer_cast(keys);
	d_real = thrust::raw_pointer_cast((thrust::device_ptr<XFLOAT>)sortedvals0.data());
	d_imag = thrust::raw_pointer_cast((thrust::device_ptr<XFLOAT>)sortedvals1.data());
	d_weight = thrust::raw_pointer_cast((thrust::device_ptr<XFLOAT>)sortedvals2.data());

/*
	thrust::device_vector<int> reducecount(d_count,d_count+numElements);
	countnum = thrust::reduce(reducecount.begin(), reducecount.end());*/
	//printf("reduce res : %d \n",countnum);

}
#endif /* CUDA_PB_KERNELS_CUH_ */
