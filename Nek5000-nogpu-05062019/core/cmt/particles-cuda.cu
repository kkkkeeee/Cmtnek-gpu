//ll includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <sys/time.h>
#include "nvml.h"
#include "cuda_functions.h"

// includes, project
//#include "magma.h"
#include "cuda_multi_gemm_unif.cu"
//#include "cuda_add_vec.h"

//My includes
// #include "debug_fns.h"
//#include "transformations.h"

//switch the comments to toggle debug mode
//#define D
#define D for(;0;)

double get_time( void )
{
	struct timeval t;
	gettimeofday( &t, NULL );
	return t.tv_sec + t.tv_usec*1e-6;
}

__global__ void particles_in_nid(double *rpart){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int n=10;
	if(id == n){
		//double *rpart = rpart1 + id * nr;
		//int *ipart = ipart1 + id * ni;
		int ie;
		double xloc = rpart[id];
		double yloc = rpart[id];
		double zloc = rpart[id];
		rpart[0]= rpart[0]+22;
		rpart[1]= rpart[1]+id;
	}

}

extern "C" void particles_in_nid_wrapper_(double* d_rpart, int* a_, double* adee_d_temp_) {

	float time;
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);
	double results[10];
	// double *d_rfpts,  *d_xerange;

	//  int *d_ifptsmap, *d_ifpts, *d_ipart, *d_nfpts;
	int blockSize = 1024, gridSize, n=100;
	gridSize = (int)ceil((float)n/blockSize);
	printf (" particles-cuda.cu parinnidwrapper d_rpart(0) a  adee2 %lf %d \n", d_rpart[0],a_[0]);
	particles_in_nid<<<gridSize, blockSize>>>(adee_d_temp_);

	cudaError_t code = cudaPeekAtLastError();
	if (code != cudaSuccess){
		printf("cuda error particles_in_nid : %s\n",cudaGetErrorString(code));
	}
	cudaMemcpy(results, adee_d_temp_, 10*sizeof(double), cudaMemcpyDeviceToHost);

	printf("inside gpu results value %lf %lf", results[0], results[1]);
	//        if(nfpts[0]>0){
	//            cudaMemcpy(fptsmap, d_ifptsmap, nfpts[0]*sizeof(int), cudaMemcpyDeviceToHost);
	//            cudaMemcpy(rfpts, d_rfpts, nfpts[0]*nrf[0]*sizeof(double), cudaMemcpyDeviceToHost);
	//            cudaMemcpy(ifpts, d_ifpts, nfpts[0]*nif[0]*sizeof(int), cudaMemcpyDeviceToHost);

	//        }
	// printf ("print var 1st %d\n", nfpts);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&time, startEvent, stopEvent);

	code = cudaPeekAtLastError();
	if (code != cudaSuccess){
		//printf("cuda error after particles_in_nid kernel launch: %s\n",cudaGetErrorString(code));
	}


	// printf ("print var 2nd %d\n", nfpts);
	printf("particles in nid time is %f\n",time*1e-03);

}


__global__ void update_particle_location_kernel(double *rpart1, int *xdrange1, int *in_part, int *bc_part, int n, int ndim, int nr,int ni, int jx0, int jx1, int jx2, int jx3,int nbc_sum, int lr, int li,int *newn, int llpart){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	//commented by adeesha change 100 back to n*ndim
	if(id < n*ndim){

		int i = id/ndim;
		in_part[i]=0;
		int j = id%ndim;
		int off = i*lr+j;
		double *rpart = rpart1+off;
		int *xdrange = xdrange1+2*j;

		if (rpart[jx0-1] < xdrange[0]){
			if ( (bc_part[0] == 0 && j == 0) || (bc_part[2] == 0 && j == 1) || (bc_part[4] == 0 && j == 2) ){
				rpart[jx0-1] = xdrange[1] - fabs(xdrange[0] - rpart[jx0-1]);
				rpart[jx1-1] = xdrange[1] + fabs(xdrange[0] - rpart[jx1-1]);
				rpart[jx2-1] = xdrange[1] + fabs(xdrange[0] - rpart[jx2-1]);
				rpart[jx3-1] = xdrange[1] + fabs(xdrange[0] - rpart[jx3-1]);
			}
			else if ( (bc_part[0] != 0 && j == 0) || (bc_part[2] != 0 && j == 1) || (bc_part[4] != 0 && j == 2) ){
				atomicExch(in_part+i, -1);
			}
		}
		if (rpart[jx0-1] > xdrange[1]){
			if ( (bc_part[0] == 0 && j == 0) || (bc_part[2] == 0 && j == 1) || (bc_part[4] == 0 && j == 2) ){
				rpart[jx0-1] = xdrange[0] + fabs(xdrange[1] - rpart[jx0]);
				rpart[jx1-1] = xdrange[0] - fabs(xdrange[1] - rpart[jx1]);
				rpart[jx2-1] = xdrange[0] - fabs(xdrange[1] - rpart[jx2]);
				rpart[jx3-1] = xdrange[0] - fabs(xdrange[1] - rpart[jx3]);
			}
			else if ( (bc_part[0] != 0 && j == 0) || (bc_part[2] != 0 && j == 1) || (bc_part[4] != 0 && j == 2) ){
				atomicExch(in_part+i, -1);
			}

		}
		__syncthreads();
		if(nbc_sum > 0){
			int ic=-1;
			if(j==0){
				if(in_part[i]==0){
					int temploop;
					for(temploop=0;temploop<=i;temploop++){
						if(in_part[i]==0){
							ic++;
						}
					}
					if(ic!=i){
						int copynr, copyni;
						for(copynr=0;copynr<nr;copynr++){
							rpart[ic*lr+copynr]=rpart[i*lr+copynr];
						}
						for(copyni=0;copyni<ni;copyni++){
							rpart[ic*li+copyni]=rpart[i*li+copyni];
						}
					}
				}	

			}

			__syncthreads();
			if(i==n-1){
				newn[0]=ic;
			}
		}

	}
}
extern "C" void update_particle_location_wrapper_(double *d_rpart, int *d_xdrange, int *d_bc_part, int *n, int *ndim, int *nr,int *ni, int *jx0, int *jx1, int *jx2,int *jx3, int *nbc_sum, int *lr, int *li, int *llpart){

	float time;
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	int* d_newn;
	cudaMalloc((void**)&d_newn, sizeof(int));

	int  *d_in_part;
	cudaMalloc((void**)&d_in_part,llpart[0]*sizeof(int));


	printf("values in update_particle_location_wrapper n[0], ndim[0],nr[0] %d %d %d\n",n[0],ndim[0],nr[0]);  
	int blockSize = 1024, gridSize;
	gridSize = (int)ceil((float)n[0]*ndim[0]/blockSize);
	update_particle_location_kernel<<<gridSize, blockSize>>>(d_rpart,d_xdrange,d_in_part,d_bc_part,n[0],ndim[0],nr[0],ni[0],jx0[0],jx1[0],jx2[0],jx3[0],nbc_sum[0],lr[0],li[0],d_newn,llpart[0]);


	printf("GPU : particle.cu : update_aprticle_location_wrapper\n");

	cudaMemcpy(n,d_newn,sizeof(int), cudaMemcpyDeviceToHost); //need to check whether the n is updated  later. adeesha

	cudaError_t code = cudaPeekAtLastError();
	if (code != cudaSuccess){
		printf("cuda error update_particle_location : %s\n",cudaGetErrorString(code));
	}




	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&time, startEvent, stopEvent);
	// printf ("print var 2nd %d\n", nfpts);
	//printf("particles in nid time is %f\n",time*1e-03);
	cudaFree(d_newn);
	cudaFree(d_in_part);

}


__global__ void update_particle_location_kernel(double *rpart, int *xdrange, double *v_part, double *rxbo, int n, int ndim, int jv0, int jv1, int jv2, int jv3,int lr, int li){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	//commented by adeesha change 100 back to n*ndim
	if(id < n*ndim){

		int i = id/ndim;
		int j = id%ndim;

// following is from palce_particles_user. but currently does not affects to else part
		rxbo[0] = xdrange[0];// ! X-Left
		rxbo[1] = xdrange[1];// ! X-Righ;
		rxbo[2] = xdrange[2];// ! Y-Left
		rxbo[3] = xdrange[3];// ! Y-Right
		rxbo[4] = xdrange[4];// ! Z-Left
		rxbo[5] = xdrange[5];// ! Z-Right

		rpart[i*lr+jv0+j]=v_part[j]; //original code says n instead of i
		rpart[i*lr+jv1+j]=v_part[j]; //original code says n instead of i
		rpart[i*lr+jv2+j]=v_part[j]; //original code says n instead of i
		rpart[i*lr+jv3+j]=v_part[j]; //original code says n instead of i
	

	}

}



extern "C" void place_particles_else_gpu_wrapper_(double *d_rpart,int *d_xdrange,double *d_v_part,double *d_rxbo,int *n,int *ndim,int *jv0,int *jv1,int *jv2, int* jv3,int *lr,int *li){



int blockSize = 1024, gridSize;
gridSize = (int)ceil((float)n[0]*3/blockSize);
//place_particles_else_gpu_kernel<<<gridSize, blockSize>>>(d_rpart,d_xdrange,d_in_part,d_bc_part,n[0],ndim[0],nr[0],ni[0],jx0[0],jx1[0],jx2[0],jx3[0],nbc_sum[0],lr[0],li[0],d_newn,llpart[0]);


//        printf("GPU : particle.cu : update_aprticle_location_wrapper\n");

cudaError_t code = cudaPeekAtLastError();
if (code != cudaSuccess){
	printf("cuda error update_particle_location : %s\n",cudaGetErrorString(code));
}



}

