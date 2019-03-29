#include <stdio.h>
#include <cuda_runtime_api.h>
//#include <cublas.h> //comment by Kk 02/14
#include <cuda_runtime.h> //added by Kk 02/25
#include <cublas_v2.h>
#include "nvml.h"

// includes, project
//#include "magma.h"
#include "cuda_multi_gemm_unif.cu"
//#define DEBUGPRINT 0

__global__ void nekadd2(double *a, double*b, int n){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n)
		a[id]+=b[id];

}
//comment to test git
//2nd comment to test git
__global__ void neksub2(double *a, double*b, int n){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n)
		a[id]-=b[id];

}

__global__ void nekcol2(double *a, double*b, int n){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n)
		a[id]*=b[id];

}

__global__ void double_copy_gpu_kernel(double *a1, double *a2,int n){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n){
		a1[id] =a2[id];


	}

}



extern "C" void double_copy_gpu_wrapper_(int *glbblockSize2,double *a1,int *n1,double *a2,int *n2,int *n ){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	printf("CUDA: Start double_copy_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start double_copy_gpu_wrapper values glbblockSize2=%d n1=%d,n2=%d,n=%d\n", glbblockSize2[0],n1[0],n2[0],n[0]);
#endif
	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)n[0]/blockSize);
	double_copy_gpu_kernel<<<gridSize, blockSize>>>(a1+n1[0],a2+n2[0],n[0]);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End double_copy__wrapper cuda status: %s\n",cudaGetErrorString(code2));

#endif
	/*printf(" $$$ double_copy_gpu_wrapper check start ");
	  for(int b=0;b<10;b++){
	  printf("a1[%d]= %lf and a2[%d] = %lf \n",n1[0]+b,a1[n1[0]+b],n2[0]+b,a2[n2[0]+b]);
	  }
	  printf(" $$$ double_copy_gpu_wrapper check End ");*/



}
void gpu_double_copy_gpu_wrapper(int glbblockSize2,double *a1,int n1,double *a2,int n2,int n ){

	int blockSize = glbblockSize2, gridSize;
	gridSize = (int)ceil((float)n/blockSize);
	double_copy_gpu_kernel<<<gridSize, blockSize>>>(a1+n1,a2+n2,n);



}

//added by Kk 03/22 for use in artvisc_cuda.cu : entropy_residual_gpu_wrapper_
void gpu_nekadd2(int glbblockSize2,double *a, double*b, int n){
	int blockSize = glbblockSize2, gridSize;
	gridSize = (int)ceil((float)n/blockSize);
	nekadd2<<<gridSize, blockSize>>>(a,b,n);


}
void gpu_neksub2(int glbblockSize2,double *a, double*b, int n){
	int blockSize = glbblockSize2, gridSize;
	gridSize = (int)ceil((float)n/blockSize);
	neksub2<<<gridSize, blockSize>>>(a,b,n);


}
void gpu_nekcol2(int glbblockSize2,double *a, double*b, int n){

	int blockSize = glbblockSize2, gridSize;
	gridSize = (int)ceil((float)n/blockSize);
	nekcol2<<<gridSize, blockSize>>>(a,b,n);

}



__global__ void double_sub2_gpu_kernel(double *a1, double *a2,int n){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n){
		a1[id] =a1[id]-a2[id];


	}

}



extern "C" void double_sub2_gpu_wrapper_(int *glbblockSize2,double *a1,int *n1,double *a2,int *n2,int *n ){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	printf("CUDA: Start double_sub2_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start double_sub2_gpu_wrapper values glbblockSize2=%d n1=%d,n2=%d,n=%d\n", glbblockSize2[0],n1[0],n2[0],n[0]);
#endif
	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)n[0]/blockSize);
	double_sub2_gpu_kernel<<<gridSize, blockSize>>>(a1+n1[0],a2+n2[0],n[0]);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End double_copy__wrapper cuda status: %s\n",cudaGetErrorString(code2));

#endif

}


__global__ void gpu_rzero(double *arr, int n){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n)
		arr[id] = 0.0;
}

extern "C" void rzero_gpu_wrapper_(int *glbblockSize2,double *a,int *start,int *n ){

	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)n[0]/blockSize);
	gpu_rzero<<<gridSize, blockSize>>>(a+start[0],n[0]);

}


__global__ void gpu_cadd2_kernel(double *a1, double *a2, double c, int n){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n){
		a1[id] =a2[id]+c;


	}

}



extern "C" void gpu_cadd2_wrapper_(int *glbblockSize2,double *a1,int *n1,double *a2,int *n2, double *c,int *n ){

	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)n[0]/blockSize);
	gpu_cadd2_kernel<<<gridSize, blockSize>>>(a1+n1[0],a2+n2[0],c[0],n[0]);

}



void gpu_local_grad3_t(double *u, double *ur, double *us, double *ut, int nx1, double *d, double *dt, double *w, int nel){

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start gpu_local_grad3_t cuda status: %s\n",cudaGetErrorString(code1));

#endif

	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1*nx1*nel,nx1,alpha,dt, nx1, ur, nx1, beta,u,nx1);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code3 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status after Dgemm: %s\n",cudaGetErrorString(code3));

        /*double *cpu_du;
        cpu_du= (double*)malloc((nel*nx1*nx1*nx1)*sizeof(double));
        cudaMemcpy(cpu_du,u, nel*nx1*nx1*nx1*sizeof(double) , cudaMemcpyDeviceToHost);
        for(int i=0;i<nel*nx1*nx1*nx1;i++){
              printf("debug 3u %d %d %.30lf \n ",i/(nx1*nx1*nx1),i%(nx1*nx1*nx1),cpu_du[i]);
        }*/
#endif
	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1, nx1, alpha,us, nx1,nx1*nx1, d,nx1,0, beta,w ,nx1,nx1*nx1,nel*nx1);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code4 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status after Dgemmstride: %s\n",cudaGetErrorString(code4));

        /*double *cpu_dw= (double*)malloc((nel*nx1*nx1*nx1)*sizeof(double));
        cudaMemcpy(cpu_dw,w, nel*nx1*nx1*nx1*sizeof(double) , cudaMemcpyDeviceToHost);
        for(int i=0;i<nel*nx1*nx1*nx1;i++){
              printf("debug 3w %d %d %.30lf \n ",i/(nx1*nx1*nx1),i%(nx1*nx1*nx1),cpu_dw[i]);
        }*/


#endif
	int blockSize=1024, gridSize;
	gridSize = (int)ceil((float)nel*nx1*nx1*nx1/blockSize);
	nekadd2<<<gridSize, blockSize>>>(u,w, nel*nx1*nx1*nx1);

	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1*nx1, nx1, nx1, alpha,ut, nx1*nx1,nx1*nx1*nx1, d,nx1,0, beta,w ,nx1*nx1,nx1*nx1*nx1,nel);

	nekadd2<<<gridSize, blockSize>>>(u,w, nel*nx1*nx1*nx1);
        cublasDestroy(handle);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code5 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status after Dgemmstride: %s\n",cudaGetErrorString(code5));

        /*cudaMemcpy(cpu_dw,u, nel*nx1*nx1*nx1*sizeof(double) , cudaMemcpyDeviceToHost);
        for(int i=0;i<nel*nx1*nx1*nx1;i++){
              printf("debug 4du %d %d %.30lf \n ",i/(nx1*nx1*nx1),i%(nx1*nx1*nx1),cpu_dw[i]);
        }*/

#endif


}
void gpu_local_grad2_t(double *u, double *ur, double *us, int nx1, double *d, double *dt, double *w, int nel){

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start gpu_local_grad2_t cuda status: %s, nel %d \n",cudaGetErrorString(code1), nel);

#endif

	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1*nel,nx1,alpha,dt, nx1, ur, nx1, beta,u,nx1);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code3 = cudaPeekAtLastError();
	printf("CUDA: Start gpu_local_grad2_t cuda status after Dgemm: %s\n",cudaGetErrorString(code3));

#endif

	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1, nx1, alpha,us, nx1,nx1*nx1, d,nx1,0, beta,w ,nx1,nx1*nx1,nel);

        cublasDestroy(handle);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code5 = cudaPeekAtLastError();
	printf("CUDA: Start gpu_local_grad2_t cuda status after Dgemmstride: %s\n",cudaGetErrorString(code5));
#endif
	int blockSize=1024, gridSize;
	//gridSize = (int)ceil((float)nel*nx1*nx1*nx1/blockSize);

	gridSize = (int)ceil((float)nel*nx1*nx1/blockSize);
	nekadd2<<<gridSize, blockSize>>>(u,w, nel*nx1*nx1);



}
__global__ void mxm(double *a, int n1, double *b, int n2, double *c, int n3, int nelt, int aSize, int bSize, int cSize, int extraEq){

	//calculate c(n1,n3) = a(n1,n2) X b(n2,n3) in c
	//in fortran the original calculation was 
	// c(n3,n1) = b(n3,n2) X a(n2,n1)

	// a,b,cSize are single element size
	//extraEq, in case of a matrix has equation as an index
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nelt*n1*n3){
		int e = id/(n1*n3);
		int rc = id%(n1*n3);
		int i = rc/n3;
		int j = rc%n3;
		int cid = e*cSize + rc;
		int aid = e*aSize + extraEq + i*n2;
		int bid = e*bSize + j;
		c[cid] = 0;
		for(int k = 0; k<n2; k++)
			c[cid]+=a[aid+k]*b[bid+k*n3];
	}

}
void gpu_local_grad3(double * ur, double *us, double *ut, double *u, int nx1, double *d, double *dt, int nel){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start gpu_local_grad3 nott cuda status: %s\n",cudaGetErrorString(code1));

#endif

	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1*nx1*nel,nx1,alpha,d, nx1, u, nx1, beta,ur,nx1);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code3 = cudaPeekAtLastError();
	printf("CUDA: Start gpu_local_grad3 cuda status after Dgemm: %s\n",cudaGetErrorString(code3));

#endif
	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1, nx1, alpha,u, nx1,nx1*nx1, dt,nx1,0, beta,us ,nx1,nx1*nx1,nel*nx1);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code4 = cudaPeekAtLastError();
	printf("CUDA: Start gpu_local_grad3 cuda status after Dgemmstride: %s\n",cudaGetErrorString(code4));
#endif

	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1*nx1, nx1, nx1, alpha,u, nx1*nx1,nx1*nx1*nx1, dt,nx1,0, beta,ut ,nx1*nx1,nx1*nx1*nx1,nel);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code5 = cudaPeekAtLastError();
	printf("CUDA: Start gpu_local_grad3 cuda status after Dgemmstride: %s\n",cudaGetErrorString(code5));
#endif
        cublasDestroy(handle);

}

void gpu_local_grad2(double *ur, double *us, double *u, int nx1, double *d, double *dt, int nel){

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start gpu_local_grad2 cuda status: %s\n",cudaGetErrorString(code1));
 
       /* int ndlel = nx1*nx1*nel;
        int nxyz = nx1*nx1;
        double *cpu_dxm1;
        cpu_dxm1= (double*)malloc(nx1*nx1*sizeof(double));
        cudaMemcpy(cpu_dxm1,d,nx1*nx1*sizeof(double) , cudaMemcpyDeviceToHost);
        for(int i=0;i<nx1*nx1;i++){
            printf("debug d_dxm1 in local_grad2 %d %d %.17E \n ",i/nx1+1, i%nx1+1, cpu_dxm1[i]);
        }

        double *cpu_dxtm1;
        cpu_dxtm1= (double*)malloc(nx1*nx1*sizeof(double));
        cudaMemcpy(cpu_dxtm1,dt,nx1*nx1*sizeof(double) , cudaMemcpyDeviceToHost);
        for(int i=0;i<nx1*nx1;i++){
            printf("debug d_dxtm1 in local_grad2 %d %d %.17E \n ",i/nx1+1, i%nx1+1, cpu_dxtm1[i]);
        }
        double *cpu_dud;
        cpu_dud= (double*)malloc(ndlel*sizeof(double));
        cudaMemcpy(cpu_dud,u,ndlel*sizeof(double) , cudaMemcpyDeviceToHost);
        for(int i=0;i<nxyz*nel;i++){
            printf("debug ud %d %d %.17E \n ",i/(nxyz)+1, i%(nxyz)+1,cpu_dud[i]);
        }*/
                        
#endif

	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1*nel,nx1,alpha,d, nx1, u, nx1, beta,ur,nx1);

#ifdef DEBUGPRINT
        /*double *cpu_dur;
        cpu_dur= (double*)malloc(ndlel*sizeof(double));
        cudaMemcpy(cpu_dur,ur,ndlel*sizeof(double) , cudaMemcpyDeviceToHost);
        for(int i=0;i<nxyz*nel;i++){
            printf("debug ur in local_grad2 %d %d %.17E \n ",i/(nxyz)+1, i%(nxyz)+1,cpu_dur[i]);
        }*/

	cudaDeviceSynchronize();
	cudaError_t code3 = cudaPeekAtLastError();
	printf("CUDA: Start gpu_local_grad2 cuda status after Dgemm: %s\n",cudaGetErrorString(code3));

#endif

	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1, nx1, alpha,u, nx1,nx1*nx1, dt,nx1,0, beta,us ,nx1,nx1*nx1,nel);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code5 = cudaPeekAtLastError();
	printf("CUDA: Start gpu_local_grad2 cuda status after Dgemmstride: %s\n",cudaGetErrorString(code5));
#endif
        cublasDestroy(handle);

}

__global__ void nekinvcol3_gpu_kernel(double *a, double *b, double *c, int n){    
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n){
		a[id] = b[id]/c[id];

	}
}

void gpu_specmpn(double *d_b, int nb, double *d_a, int na, double * d_ba, double* d_ab, bool if3d, double * d_w, int ldw, int nel, int intermediatestride, int eq, bool second_eq){
	//intermediatestride means the stride size need to skip
#ifdef DEBUGPRINT
	cudaError_t code1 = cudaPeekAtLastError();
	if (code1 != cudaSuccess){
		printf("Start gpu_specmpn cuda error str 1: %s\n",cudaGetErrorString(code1));
	}
        printf("debug gpu_specmpn, second_eq: %s\n", second_eq ? "true" : "false");
#endif

	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	if(second_eq){// this is idir==0 case in intp_rstd function
		if(if3d){
			int nab = na*nb;
			int nbb = nb*nb;

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,na*na , na, alpha,d_ba, nb,0, d_a,na,na*na*na*intermediatestride, beta,d_w ,nb,nb*na*na,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code4 = cudaPeekAtLastError();
			printf("CUDA: Start gpu_specmpn cuda status after Dgemmstride1 idir=0, if3d = 1: %s\n",cudaGetErrorString(code4));
#endif

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,nb,na, alpha,d_w, nb,na*nb , d_ab,na,0, beta,d_w+na*na*nb*nel ,nb,nb*nb,na*nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code5 = cudaPeekAtLastError();
			printf("CUDA: Start gpu_specmpn cuda status after Dgemmstride2 idir=0, if3d = 1: %s\n",cudaGetErrorString(code5));

#endif

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb*nb,nb,na, alpha,d_w+na*na*nb*nel, nb*nb,nb*nb*na , d_ab,na,0, beta,d_b,nb*nb,nb*nb*nb,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code6 = cudaPeekAtLastError();
			printf("CUDA: Start gpu_specmpn cuda status after Dgemmstride3 idir=0, if3d = 1: %s\n",cudaGetErrorString(code6));

#endif

		}
		else{

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,na, na, alpha,d_ba, nb,0, d_a,na,na*na*intermediatestride, beta,d_w ,nb,nb*na,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code7 = cudaPeekAtLastError();
			printf("CUDA: Start gpu_specmpn cuda status after Dgemmstride1 idir=0, if3d =0:   %s\n",cudaGetErrorString(code7));

#endif

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,nb,na, alpha,d_w, nb,nb*na , d_ab,na,0, beta,d_b,nb,nb*nb,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code8 = cudaPeekAtLastError();
			printf("CUDA: Start gpu_specmpn cuda status after Dgemmstride2 idir=0, if3d = 0: %s\n",cudaGetErrorString(code8));

#endif

		}

	}
	else{//idir = 1 in intp_rstd function
		if(if3d){
			int nab = na*nb;
			int nbb = nb*nb;

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,na*na , na, alpha,d_ba, nb,0, d_a,na,na*na*na*intermediatestride, beta,d_w ,nb,nb*na*na,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code4 = cudaPeekAtLastError();
			printf("CUDA: Start gpu_specmpn cuda status after Dgemmstride1 idir=1, if3d = 1: %s\n",cudaGetErrorString(code4));

                        /*double *cpu_dw;
                        cpu_dw= (double*)malloc((nel*nb*na*na+nel*nb*nb*na)*sizeof(double));
                        cudaMemcpy(cpu_dw,d_w, nel*nb*na*na*sizeof(double) , cudaMemcpyDeviceToHost);
                        for(int i=0;i<  nel*nb*na*na;i++){
                            printf("debug 2d_w %d %d %d %.30lf \n ",i/(nb*na*na),i%(nb*na*na),eq,cpu_dw[i]);
                        }*/
#endif

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,nb,na, alpha,d_w, nb,na*nb , d_ab,na,0, beta,d_w+na*na*nb*nel ,nb,nb*nb,na*nel);
//                      cudaMemcpy(cpu_dw,d_w, (nel*nb*na*na+nel*nb*nb*na)*sizeof(double) , cudaMemcpyDeviceToHost);
//                      for(int i=0;i<  nel*nb*nb*na;i++){
//                            printf("debug 2d_w %d %d %d %.30lf \n ",i/(nb*nb*na),i%(nb*nb*na),eq,cpu_dw[nel*na*na*nb+i]);
//                      }

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code5 = cudaPeekAtLastError();
			printf("CUDA: Start gpu_specmpn cuda status after Dgemmstride2 idir=1, if3d = 1: %s\n",cudaGetErrorString(code5));
#endif

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb*nb,nb,na, alpha,d_w+na*na*nb*nel, nb*nb,nb*nb*na , d_ab,na,0, beta,d_b,nb*nb,nb*nb*nb,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code6 = cudaPeekAtLastError();
			printf("CUDA: Start gpu_specmpn cuda status after Dgemmstride3 idir=1, if3d = 1: %s\n",cudaGetErrorString(code6));
#endif
		}
		else{//idir = 1, if3d = 2d
			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,na, na, alpha,d_ba, nb,0, d_a,na,na*na*intermediatestride, beta,d_w ,nb,nb*na,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code7 = cudaPeekAtLastError();
			printf("CUDA: Start gpu_specmpn cuda status after Dgemmstride1 idir=1, if3d = 0: %s\n",cudaGetErrorString(code7));
			/*double *cpu_db;
                        cpu_db= (double*)malloc((nel*nb*na)*sizeof(double));
                        cudaMemcpy(cpu_db,d_w, nel*nb*na*sizeof(double) , cudaMemcpyDeviceToHost);
                        for(int i=0;i<  nel*nb*na;i++){
                              printf("debug w %2d %2d %3d %25.17E \n ",i/(nb*na)+1,eq, i%(nb*na)+1, cpu_db[i]);
                        }*/
#endif

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,nb,na, alpha,d_w, nb,nb*na , d_ab,na,0, beta,d_b,nb,nb*nb,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code8 = cudaPeekAtLastError();
			printf("CUDA: Start gpu_specmpn cuda status after Dgemmstride2 idir=1, if3d = 0: %s\n",cudaGetErrorString(code8));
#endif
		}
	}
        cublasDestroy(handle);

#ifdef DEBUGPRINT
	cudaError_t code = cudaPeekAtLastError();
	if (code != cudaSuccess){
		printf("cuda error str 1: %s\n",cudaGetErrorString(code));
	}
#endif
}
