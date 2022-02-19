//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!

#include "bmm.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// TILEX and TILEY are used to set the number of threads in a CUDA block 
#define TILEX 32
#define TILEY 16

// you may define other parameters here!
#define TM (TILEY < TILEX ? TILEY : TILEX)
// you may define other macros here!
// you may define other functions here!

dim3 getDimGrid(const int m, const int n) {
	dim3 dimGrid(n/TILEX,n/TILEY);
	return dimGrid;
}
dim3 getDimBlock(const int m, const int n) {
	dim3 dimBlock(TILEX,TILEY);
	return dimBlock;
}

__global__ void kernelFunc(float* ad, float* bd, float* cd, const int m, const int n) {
	
	// write your GPU kernel function here
	
	__shared__ float as[TILEY][2*TM];
	__shared__ float bs[2*TM][TILEX];

	int i = TILEY*by + ty;
	int j = TILEX*bx + tx;
	//int tmp = 0;
	float s = 0;
	int k;
	int r = 0;
	int offset;
	int offsetx;
	int offsety;
	int im = (i << m);
	for (; r < n/(2*TM); r++){
		//init as,bs:
		offset = TM * 2 * r;
		offsetx = im + offset + tx;
		offsety = offset + ty;
		if(tx < TM){
			as[ty][tx] = ad[offsetx];
			as[ty][tx + TM] = ad[offsetx + TM];
		}
		if(ty < TM){
			bs[ty][tx] = bd[((offsety) << m) + j];
			bs[ty + TM][tx] = bd[((offsety + TM) << m) + j];
		}
		//tmp = tmp + 4*TM;
		__syncthreads();

		for (k = 0; k < 2*TM; k++)
			//use as,bs:
			s += as[ty][k] * bs[k][tx];
		
		__syncthreads();
	}
	cd[(i << m)+ j] = s;
}