// ONLY MODIFY THIS FILE

#include "scan2.h"
#include "gpuerrors.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// you may define other parameters here!
#define TILEX 1024

// you may define other macros here!
// you may define other functions here!

__global__ void kernelFunc1(float* a, float* w)
{
	__shared__ float as[TILEX];

    int i = 1024 * bx + tx;

    as[tx] = a[i];

    __syncthreads();
	
	int j;
	int j2;
	if(tx == 1024 - 1) as[tx] = 0;
	
	for(j = 1; j < 10; j += 1){
		j2 = 1 << j;
        if((tx % j2 == j2 - 1) && tx != 1024 - 1){
            as[tx] = as[tx] + as[tx - (j2>>1)];
        }
        __syncthreads();
    }
	
	float as_tmp;
	int offset;
	
    for(; j >= 1; j -= 1){
		j2 = 1 << j;
        offset = tx - (j2>>1);
		if(tx % j2 == j2 - 1){
            as_tmp = as[offset];
            as[offset] = as[tx];
            as[tx] = as[tx] + as_tmp;
        }
        __syncthreads();
    }	
	
	if(tx == (1024 - 1)){
        w[bx] = a[i] + as[tx];
    }

	a[i] = as[tx];
}

__global__ void kernelFunc3(float* a, int blockSize)
{
	__shared__ float as[TILEX];
	__shared__ float bs[TILEX];

    int i = blockSize * bx + tx;

	as[tx] = (tx > 0) ? a[i-1] : 0;

    __syncthreads();
	
	int j = 2;
	int k = 0;
	int offset;
	
	for(; j <= blockSize; j <<= 1){
		k = 1 - k;
		offset = tx - (j>>1);
		if(tx >= (j>>1))
			if(k) bs[tx] = as[tx] + as[offset];
			else as[tx] = bs[tx] + bs[offset];
		else
			if(k) bs[tx] = as[tx];
			else as[tx] = bs[tx];
		__syncthreads();
    }
	if(k) a[i] = bs[tx];
	else a[i] = as[tx];
}

__global__ void kernelFunc2(float* a, float* b)
{
	int i = 1024 * bx + tx;
    a[i] = b[bx] + a[i];
}

void gpuKernel(float* a, float* c,int n) {

	if(n == (1 << 20)){
		
		float* ad;
		float* w1;
	
		cudaMalloc((void**)&ad, n*sizeof(float));
		cudaMalloc((void**)&w1, (n>>10)*sizeof(float));
		
		cudaMemcpy(ad, a, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc3<<< 1 , n>>10 >>> (w1, 1024);
		
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c, ad+1, (n-1)*sizeof(float), cudaMemcpyDeviceToHost);
		c[n-1] = c[n-2] + a[n-1];
		
		cudaFree(ad);
		cudaFree(w1);
	}
	else if(n <= (1 << 25)){
		float* ad;
		float* w1;
		float* w2;
	
		cudaMalloc((void**)&ad, n*sizeof(float));
		cudaMalloc((void**)&w1, (n>>10)*sizeof(float));
		cudaMalloc((void**)&w2, (n>>20)*sizeof(float));
		
		cudaMemcpy(ad, a, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c, ad+1, (n-1)*sizeof(float), cudaMemcpyDeviceToHost);
		c[n-1] = c[n-2] + a[n-1];
		
		cudaFree(ad);
		cudaFree(w1);
		cudaFree(w2);
	}
	else if(n == (1 << 26)){
		
		n = n / 2;
		
		float* ad;
		float* w1;
		float* w2;
	
		cudaMalloc((void**)&ad, n*sizeof(float));
		cudaMalloc((void**)&w1, (n>>10)*sizeof(float));
		cudaMalloc((void**)&w2, (n>>20)*sizeof(float));
		
		cudaMemcpy(ad, a, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c, ad+1, (n-1)*sizeof(float), cudaMemcpyDeviceToHost);
		c[n-1] = c[n-2] + a[n-1];
		a[n-1] = c[n-1];
		
		cudaMemcpy(ad, a+n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 ,n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[2*n-2] = c[2*n-3] + a[2*n-2];
		c[2*n-1] = c[2*n-2] + a[2*n-1];
		
		cudaFree(ad);
		cudaFree(w1);
		cudaFree(w2);
	}
	else if(n == (1 << 27)){
		
		n = n / 4;
		
		float* ad;
		float* w1;
		float* w2;
	
		cudaMalloc((void**)&ad, n*sizeof(float));
		cudaMalloc((void**)&w1, (n>>10)*sizeof(float));
		cudaMalloc((void**)&w2, (n>>20)*sizeof(float));
		
		cudaMemcpy(ad, a, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c, ad+1, (n-1)*sizeof(float), cudaMemcpyDeviceToHost);
		c[n-1] = c[n-2] + a[n-1];
		a[n-1] = c[n-1];
		
		cudaMemcpy(ad, a+n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[2*n-2] = c[2*n-3] + a[2*n-2];
		c[2*n-1] = c[2*n-2] + a[2*n-1];
		a[2*n-1] = c[2*n-1];
		
		cudaMemcpy(ad, a+2*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+2*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[3*n-2] = c[3*n-3] + a[3*n-2];
		c[3*n-1] = c[3*n-2] + a[3*n-1];
		a[3*n-1] = c[3*n-1];
				
		cudaMemcpy(ad, a+3*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+3*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[4*n-2] = c[4*n-3] + a[4*n-2];
		c[4*n-1] = c[4*n-2] + a[4*n-1];
		
		cudaFree(ad);
		cudaFree(w1);
		cudaFree(w2);
	}
	else if(n == (1 << 28)){
		
		n = n / 8;
		
		float* ad;
		float* w1;
		float* w2;
	
		cudaMalloc((void**)&ad, n*sizeof(float));
		cudaMalloc((void**)&w1, (n>>10)*sizeof(float));
		cudaMalloc((void**)&w2, (n>>20)*sizeof(float));

		cudaMemcpy(ad, a, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c, ad+1, (n-1)*sizeof(float), cudaMemcpyDeviceToHost);
		c[n-1] = c[n-2] + a[n-1];
		a[n-1] = c[n-1];
		
		cudaMemcpy(ad, a+n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[2*n-2] = c[2*n-3] + a[2*n-2];
		c[2*n-1] = c[2*n-2] + a[2*n-1];
		a[2*n-1] = c[2*n-1];
		
		cudaMemcpy(ad, a+2*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+2*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[3*n-2] = c[3*n-3] + a[3*n-2];
		c[3*n-1] = c[3*n-2] + a[3*n-1];
		a[3*n-1] = c[3*n-1];
				
		cudaMemcpy(ad, a+3*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+3*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[4*n-2] = c[4*n-3] + a[4*n-2];
		c[4*n-1] = c[4*n-2] + a[4*n-1];
		a[4*n-1] = c[4*n-1];
		
		cudaMemcpy(ad, a+4*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+4*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[5*n-2] = c[5*n-3] + a[5*n-2];
		c[5*n-1] = c[5*n-2] + a[5*n-1];
		a[5*n-1] = c[5*n-1];
		
		cudaMemcpy(ad, a+5*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+5*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[6*n-2] = c[6*n-3] + a[6*n-2];
		c[6*n-1] = c[6*n-2] + a[6*n-1];
		a[6*n-1] = c[6*n-1];
		
		cudaMemcpy(ad, a+6*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+6*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[7*n-2] = c[7*n-3] + a[7*n-2];
		c[7*n-1] = c[7*n-2] + a[7*n-1];
		a[7*n-1] = c[7*n-1];
		
		cudaMemcpy(ad, a+7*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+7*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[8*n-2] = c[8*n-3] + a[8*n-2];
		c[8*n-1] = c[8*n-2] + a[8*n-1];
		
		
		cudaFree(ad);
		cudaFree(w1);
		cudaFree(w2);
	}
	else if(n == (1 << 29)){
		
		n = n / 16;
		
		float* ad;
		float* w1;
		float* w2;
	
		cudaMalloc((void**)&ad, n*sizeof(float));
		cudaMalloc((void**)&w1, (n>>10)*sizeof(float));
		cudaMalloc((void**)&w2, (n>>20)*sizeof(float));
		
		cudaMemcpy(ad, a, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c, ad+1, (n-1)*sizeof(float), cudaMemcpyDeviceToHost);
		c[n-1] = c[n-2] + a[n-1];
		a[n-1] = c[n-1];
		
		cudaMemcpy(ad, a+n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[2*n-2] = c[2*n-3] + a[2*n-2];
		c[2*n-1] = c[2*n-2] + a[2*n-1];
		a[2*n-1] = c[2*n-1];
		
		cudaMemcpy(ad, a+2*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20>>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+2*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[3*n-2] = c[3*n-3] + a[3*n-2];
		c[3*n-1] = c[3*n-2] + a[3*n-1];
		a[3*n-1] = c[3*n-1];
				
		cudaMemcpy(ad, a+3*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+3*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[4*n-2] = c[4*n-3] + a[4*n-2];
		c[4*n-1] = c[4*n-2] + a[4*n-1];
		a[4*n-1] = c[4*n-1];
		
		cudaMemcpy(ad, a+4*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+4*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[5*n-2] = c[5*n-3] + a[5*n-2];
		c[5*n-1] = c[5*n-2] + a[5*n-1];
		a[5*n-1] = c[5*n-1];
		
		cudaMemcpy(ad, a+5*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+5*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[6*n-2] = c[6*n-3] + a[6*n-2];
		c[6*n-1] = c[6*n-2] + a[6*n-1];
		a[6*n-1] = c[6*n-1];
		
		cudaMemcpy(ad, a+6*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+6*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[7*n-2] = c[7*n-3] + a[7*n-2];
		c[7*n-1] = c[7*n-2] + a[7*n-1];
		a[7*n-1] = c[7*n-1];
		
		cudaMemcpy(ad, a+7*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+7*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[8*n-2] = c[8*n-3] + a[8*n-2];
		c[8*n-1] = c[8*n-2] + a[8*n-1];
		a[8*n-1] = c[8*n-1];
				
		cudaMemcpy(ad, a+8*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+8*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[9*n-2] = c[9*n-3] + a[9*n-2];
		c[9*n-1] = c[9*n-2] + a[9*n-1];
		a[9*n-1] = c[9*n-1];
				
		cudaMemcpy(ad, a+9*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+9*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[10*n-2] = c[10*n-3] + a[10*n-2];
		c[10*n-1] = c[10*n-2] + a[10*n-1];
		a[10*n-1] = c[10*n-1];
				
		cudaMemcpy(ad, a+10*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20, 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+10*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[11*n-2] = c[11*n-3] + a[11*n-2];
		c[11*n-1] = c[11*n-2] + a[11*n-1];
		a[11*n-1] = c[11*n-1];
				
		cudaMemcpy(ad, a+11*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+11*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[12*n-2] = c[12*n-3] + a[12*n-2];
		c[12*n-1] = c[12*n-2] + a[12*n-1];
		a[12*n-1] = c[12*n-1];
				
		cudaMemcpy(ad, a+12*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+12*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[13*n-2] = c[13*n-3] + a[13*n-2];
		c[13*n-1] = c[13*n-2] + a[13*n-1];
		a[13*n-1] = c[13*n-1];
				
		cudaMemcpy(ad, a+13*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+13*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[14*n-2] = c[14*n-3] + a[14*n-2];
		c[14*n-1] = c[14*n-2] + a[14*n-1];
		a[14*n-1] = c[14*n-1];
				
		cudaMemcpy(ad, a+14*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+14*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[15*n-2] = c[15*n-3] + a[15*n-2];
		c[15*n-1] = c[15*n-2] + a[15*n-1];
		a[15*n-1] = c[15*n-1];
		
		cudaMemcpy(ad, a+15*n-1, n*sizeof(float), cudaMemcpyHostToDevice);
		
		kernelFunc1<<< n>>10 , 1024 >>>(ad,w1);
		kernelFunc1<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc3<<< 1 , n>>20 >>> (w2, n>>20);
		
		kernelFunc2<<< n>>20 , 1024 >>> (w1,w2);
		kernelFunc2<<< n>>10 , 1024 >>> (ad,w1);
		
		cudaMemcpy(c+15*n, ad+2, (n-2)*sizeof(float), cudaMemcpyDeviceToHost);
		c[16*n-2] = c[16*n-3] + a[16*n-2];
		c[16*n-1] = c[16*n-2] + a[16*n-1];
				
		cudaFree(ad);
		cudaFree(w1);
		cudaFree(w2);
	}
}