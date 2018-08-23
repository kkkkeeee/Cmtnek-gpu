#include <cuda_runtime_api.h>
#include <cublas.h>
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

#endif
	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1, nx1, alpha,us, nx1,nx1*nx1, d,nx1,0, beta,w ,nx1,nx1*nx1,nel*nx1);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code4 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status after Dgemmstride: %s\n",cudaGetErrorString(code4));

#endif
	int blockSize=1024, gridSize;
	gridSize = (int)ceil((float)nel*nx1*nx1*nx1/blockSize);
	nekadd2<<<gridSize, blockSize>>>(u,w, nel*nx1*nx1*nx1);

	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1*nx1, nx1, nx1, alpha,ut, nx1*nx1,nx1*nx1*nx1, d,nx1,0, beta,w ,nx1*nx1,nx1*nx1*nx1,nel);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code5 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status after Dgemmstride: %s\n",cudaGetErrorString(code5));
#endif
	nekadd2<<<gridSize, blockSize>>>(u,w, nel*nx1*nx1*nx1);

}
void gpu_local_grad2_t(double *u, double *ur, double *us, int nx1, double *d, double *dt, double *w, int nel){

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start gpu_local_grad2_t cuda status: %s\n",cudaGetErrorString(code1));

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
	printf("CUDA: Start map_faced cuda status after Dgemm: %s\n",cudaGetErrorString(code3));

#endif

	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1, nx1, alpha,us, nx1,nx1*nx1, d,nx1,0, beta,w ,nx1,nx1*nx1,nel);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code5 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status after Dgemmstride: %s\n",cudaGetErrorString(code5));
#endif
	int blockSize=1024, gridSize;
	gridSize = (int)ceil((float)nel*nx1*nx1*nx1/blockSize);

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
	printf("CUDA: Start map_faced cuda status after Dgemm: %s\n",cudaGetErrorString(code3));

#endif
	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1, nx1, alpha,u, nx1,nx1*nx1, dt,nx1,0, beta,us ,nx1,nx1*nx1,nel*nx1);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code4 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status after Dgemmstride: %s\n",cudaGetErrorString(code4));
#endif

	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1*nx1, nx1, nx1, alpha,u, nx1*nx1,nx1*nx1*nx1, dt,nx1,0, beta,ut ,nx1*nx1,nx1*nx1*nx1,nel);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code5 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status after Dgemmstride: %s\n",cudaGetErrorString(code5));
#endif

}

void gpu_local_grad2(double * ur, double *us, double *u, int nx1, double *d, double *dt, int nel){

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start gpu_local_grad2 cuda status: %s\n",cudaGetErrorString(code1));

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
	cudaDeviceSynchronize();
	cudaError_t code3 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status after Dgemm: %s\n",cudaGetErrorString(code3));

#endif

	cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1, nx1, alpha,u, nx1,nx1*nx1, dt,nx1,0, beta,us ,nx1,nx1*nx1,nel);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code5 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status after Dgemmstride: %s\n",cudaGetErrorString(code5));
#endif


}

__global__ void nekinvcol3_gpu_kernel(double *a, double *b, double *c, int n){    
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n){
		a[id] = b[id]/c[id];

	}
}
//complete this function.  adeesha.
void gpu_gen_int(double *jgl, double *jgt,int mp,int np,double *w){
	int iz = 1;
	int id = iz + np;

	//  call zwgll (w(iz),jgt,np); need to implement this. talk with Dr.Tania adeesha
	//  call zwgl  (w(id),jgt,mp);

	int n  = np-1;
	for(int i=0;i<mp;i++){
		//  call gpu_fd_weights_full(w(id+i-1),w(iz),n,0,jgt);
		for(int j=0;j<np;j++){
			jgl[j*mp+i] = jgt[j];                 //  Interpolation matrix
		}
	}
	// call gpu_transpose(jgt,np,jgl,mp)

}

void gpu_get_int_ptr(int *ip,int  if3d, int mx, int md, int nelt,double *jgl, double *jgt,double *wkd,int lxd,int *pjgl){

	int ld= 2*lxd;

	int ij = md + ld*(mx-1);
	ip[0] = pjgl[ij-1];
	if (ip[0]==0){
		int nstore   = pjgl[0];
		pjgl[ij-1] = nstore+1;
		nstore   = nstore + md*mx;
		pjgl[0]  = nstore;
		ip[0]       = pjgl[ij-1];
		int nwrkd = mx + md;
		printf("Warning!!! ip[0] is 0. This should not happen \n");
		//gpu_gen_int(jgl+ip[0]-1,jgt+ip[0]-1,md,mx,wkd);
	}
	ip[0]=1;  // dummy for now. change to the actual value later. adeesha.
}

void gpu_gen_dgl(double *dgl,double *dgt,int mp,int np,double *w){
	int iz = 1;
	int id = iz + np;

	//  call zwgl  (w(iz),dgt,np)  ! GL points
	//  call zwgl  (w(id),dgt,mp)  ! GL points

	int  ndgt = 2*np;
	int  ldgt = mp*np;

	int  n  = np-1;
	for(int i=0;i<mp;i++){
		//call fd_weights_full(w(id+i-1),w(iz),n,1,dgt) ! 1=1st deriv.
		for(int j=0;j<np;j++){
			dgl[j*mp+i] = dgt[np+j];            //           ! Derivative matrix
		}
	}
	//  call gpu_transpose(dgt,np,dgl,mp)


}
void gpu_get_dgl_ptr (int *ip,int  if3d, int mx, int md, int nelt,double *dg, double *dgt,double *wkd, int lxd, int *pdg){

	int ld=2*lxd;
	int ij = md + ld*(mx-1);
	ip[0] = pdg [ij];

	if (ip[0]==0){

		int nstore   = pdg [0];
		pdg[ij-1] = nstore+1;
		nstore   = nstore + md*mx;
		pdg [0] = nstore;
		ip[0]       = pdg [ij-1];
		int nwrkd = mx + md;
		gpu_gen_dgl(dg+ip[0]-1,dgt+ip[0]-1,md,mx,wkd);

	}

}

void gpu_specmpn(double *d_b, int nb, double *d_a, int na, double * d_ba, double* d_ab, bool if3d, double * d_w, int ldw, int nel, int intermediatestride, int eq, bool second_eq){
	//intermediatestride means the stride size need to skip
#ifdef DEBUGPRINT
	cudaError_t code1 = cudaPeekAtLastError();
	if (code1 != cudaSuccess){
		printf("Start gpu_specmpn cuda error str 1: %s\n",cudaGetErrorString(code1));
	}
#endif

	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	if(second_eq){// this is idir in intp_rstd function

		if(if3d){
			int nab = na*nb;
			int nbb = nb*nb;

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,na*na , na, alpha,d_ba, nb,0, d_a,na,na*na*na*intermediatestride, beta,d_w ,nb,nb*na*na,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code4 = cudaPeekAtLastError();
			printf("CUDA: Start map_faced cuda status after Dgemmstride 4: %s\n",cudaGetErrorString(code4));

#endif



			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,nb,na, alpha,d_w, nb,na*nb , d_ab,na,0, beta,d_w+na*na*nb*nel ,nb,nb*nb,na*nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code5 = cudaPeekAtLastError();
			printf("CUDA: Start map_faced cuda status after Dgemmstride 5: %s\n",cudaGetErrorString(code5));

#endif

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb*nb,nb,na, alpha,d_w+na*na*nb*nel, nb*nb,nb*nb*na , d_ab,na,0, beta,d_b,nb*nb,nb*nb*nb,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code6 = cudaPeekAtLastError();
			printf("CUDA: Start map_faced cuda status after Dgemmstride 6: %s\n",cudaGetErrorString(code6));

#endif

		}
		else{

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,na, na, alpha,d_ba, nb,0, d_a,na,na*na*intermediatestride, beta,d_w ,nb,nb*na,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code7 = cudaPeekAtLastError();
			printf("CUDA: Start map_faced cuda status after Dgemmstride:  7 %s\n",cudaGetErrorString(code7));

#endif

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,nb,na, alpha,d_w, nb,nb*na , d_ab,na,0, beta,d_b,nb,nb*nb,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code8 = cudaPeekAtLastError();
			printf("CUDA: Start map_faced cuda status after Dgemmstride: 8%s\n",cudaGetErrorString(code8));

#endif

		}

	}
	else{


		if(if3d){
			int nab = na*nb;
			int nbb = nb*nb;

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,na*na , na, alpha,d_ba, nb,0, d_a,na,na*na*na*intermediatestride, beta,d_w ,nb,nb*na*na,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code4 = cudaPeekAtLastError();
			printf("CUDA: Start map_faced cuda status after Dgemmstride 4: %s\n",cudaGetErrorString(code4));

#endif



			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,nb,na, alpha,d_w, nb,na*nb , d_ab,na,0, beta,d_w+na*na*nb*nel ,nb,nb*nb,na*nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code5 = cudaPeekAtLastError();
			printf("CUDA: Start map_faced cuda status after Dgemmstride 5: %s\n",cudaGetErrorString(code5));

#endif

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb*nb,nb,na, alpha,d_w+na*na*nb*nel, nb*nb,nb*nb*na , d_ab,na,0, beta,d_b,nb*nb,nb*nb*nb,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code6 = cudaPeekAtLastError();
			printf("CUDA: Start map_faced cuda status after Dgemmstride 6: %s\n",cudaGetErrorString(code6));

#endif

		}
		else{

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,na, na, alpha,d_ba, nb,0, d_a,na,na*na*intermediatestride, beta,d_w ,nb,nb*na,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code7 = cudaPeekAtLastError();
			printf("CUDA: Start map_faced cuda status after Dgemmstride:  7 %s\n",cudaGetErrorString(code7));

#endif

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nb,nb,na, alpha,d_w, nb,nb*na , d_ab,na,0, beta,d_b,nb,nb*nb,nel);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			cudaError_t code8 = cudaPeekAtLastError();
			printf("CUDA: Start map_faced cuda status after Dgemmstride: 8%s\n",cudaGetErrorString(code8));

#endif

		}

	}

#ifdef DEBUGPRINT
	cudaError_t code = cudaPeekAtLastError();
	if (code != cudaSuccess){
		printf("cuda error str 1: %s\n",cudaGetErrorString(code));
	}
#endif


}
void gpu_gen_dgll(double *dgl,double *dgt,int mp,int np,double *w){
	int iz = 1;
	int id = iz + np;

	//  call zwgl  (w(iz),dgt,np)  ! GL points
	//  call zwgl  (w(id),dgt,mp)  ! GL points

	int  ndgt = 2*np;
	int  ldgt = mp*np;

	int  n  = np-1;
	for(int i=0;i<mp;i++){
		//call fd_weights_full(w(id+i-1),w(iz),n,1,dgt) ! 1=1st deriv.
		for(int j=0;j<np;j++){
			dgl[j*mp+i] = dgt[np+j];            //           ! Derivative matrix
		}
	}
	//  call gpu_transpose(dgt,np,dgl,mp)


}

void gpu_get_dgll_ptr (int *ip,int  if3d, int mx, int md, int nelt,double *d, double *dt,double *wkd,int  lxd, int *pdg){

	int ld=2*lxd;
	int ij = md + ld*(mx-1);
	ip[0] = pdg [ij];

	if (ip[0]==0){

		int nstore   = pdg [0];
		pdg[ij-1] = nstore+1;
		nstore   = nstore + md*mx;
		pdg [0] = nstore;
		ip[0]       = pdg [ij-1];
		int nwrkd = mx + md;
		gpu_gen_dgll(d+ip[0]-1,dt+ip[0]-1,md,mx,wkd);

	}

}




