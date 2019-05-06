#include <cuda_runtime_api.h>
#include <cublas.h>
#include "nvml.h"
// includes, project
//#include "magma.h"
#include "cuda_multi_gemm_unif.cu"


__global__ void nekadd2(double *a, double*b, int n){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n)
		a[id]+=b[id];

}
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
cudaDeviceSynchronize();
        cudaError_t code1 = cudaPeekAtLastError();
        printf("CUDA: Start double_copy_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

        printf("CUDA: Start double_copy_gpu_wrapper values glbblockSize2=%d n1=%d,n2=%d,n=%d\n", glbblockSize2[0],n1[0],n2[0],n[0]);

	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)n[0]/blockSize);
	double_copy_gpu_kernel<<<gridSize, blockSize>>>(a1+n1[0],a2+n2[0],n[0]);
	cudaDeviceSynchronize();
	 cudaError_t code2 = cudaPeekAtLastError();

        printf("CUDA: End double_copy__wrapper cuda status: %s\n",cudaGetErrorString(code2));

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
	cudaDeviceSynchronize();
        cudaError_t code1 = cudaPeekAtLastError();
        printf("CUDA: Start double_sub2_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

        printf("CUDA: Start double_sub2_gpu_wrapper values glbblockSize2=%d n1=%d,n2=%d,n=%d\n", glbblockSize2[0],n1[0],n2[0],n[0]);
	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)n[0]/blockSize);
	double_sub2_gpu_kernel<<<gridSize, blockSize>>>(a1+n1[0],a2+n2[0],n[0]);

	cudaDeviceSynchronize();
         cudaError_t code2 = cudaPeekAtLastError();

        printf("CUDA: End double_copy__wrapper cuda status: %s\n",cudaGetErrorString(code2));


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



void gpu_local_grad3_t(double *u, double *ur, double *us, double *ut, int nxd, double *d, double *dt, double *w, int nel){

 	cudaDeviceSynchronize();
        cudaError_t code1 = cudaPeekAtLastError();

        printf("CUDA: Start gpu_local_grad3 cuda status: %s\n",cudaGetErrorString(code1));
	
	int nxd_2 = nxd * nxd;
	int nxd_3 = nxd_2 * nxd;
	// u(nxd,nxd*nxd) = dt(nxd,nxd) * ur(nxd, nxd*nxd) fortran
	// u(nxd*nxd,nxd) = ur(nxd*nxd, nxd) * dt(nxd,nxd) C
	int blockSize=1024, gridSize;
	cudaStream_t stream;
	cudaStreamCreate( &stream );
	const double alpha = 1;
	const double beta = 0;

	gridSize = (int)ceil((float)nel*nxd_3/blockSize);
	//mxm<<<gridSize, blockSize>>>(ur,nxd_2, dt, nxd, u, nxd, nel, nxd_3, 0, nxd_3, 0);
	cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nxd, nxd_2, &alpha, dt, nxd, 0, ur, nxd, nxd_3, &beta, u, nxd, nxd_3, nel, gridSize);

	cudaDeviceSynchronize();
        code1 = cudaPeekAtLastError();

        printf("CUDA: gpu_local_grad3 after cuda_multi_gemm_unif 1 cuda status: %s\n",cudaGetErrorString(code1));

	for(int k = 0; k<nxd;k++){
		//wk(nxd,nxd) = usk(nxd,nxd)*D(nxd,nxd) fortran
		//wk(nxd,nxd) = D(nxd,nxd)*usk(nxd,nxd) C
		gridSize = (int)ceil((float)nel*nxd_2/blockSize);
		//mxm<<<gridSize, blockSize>>>(d,nxd, us+k*nxd_2, nxd, w+k*nxd_2, nxd, nel, 0, nxd_3, nxd_3, 0);
		cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nxd, nxd, &alpha, us+k*nxd_2, nxd, nxd_3, d, nxd, 0, &beta, w+k*nxd_2, nxd, nxd_3, nel, gridSize);
	}

	 cudaDeviceSynchronize();
         code1 = cudaPeekAtLastError();

        printf("CUDA: gpu_local_grad3 after cuda_multi_gemm_unif 2 for loop cuda status: %s\n",cudaGetErrorString(code1));

	gridSize = (int)ceil((float)nel*nxd_3/blockSize);
	nekadd2<<<gridSize, blockSize>>>(u,w, nel*nxd_3);
	//w(nxd*nxd,nxd) = ut(nxd*nxd,nxd) * D(nxd,nxd) fortran
	//w(nxd,nxd*nxd) = D(nxd,nxd) * ut(nxd,nxd*nxd) C
	//mxm<<<gridSize, blockSize>>>(d,nxd, ut, nxd, w, nxd_2, nel, 0, nxd_3, nxd_3, 0);
	cuda_multi_gemm_unif(stream, 'N', 'N', nxd_2, nxd, nxd, &alpha, ut, nxd, nxd_3, d, nxd, 0, &beta, w, nxd_2, nxd_3, nel, gridSize);

	cudaDeviceSynchronize();
        code1 = cudaPeekAtLastError();

        printf("CUDA: gpu_local_grad3 after cuda_multi_gemm_unif 3 cuda status: %s\n",cudaGetErrorString(code1));


	nekadd2<<<gridSize, blockSize>>>(u,w, nel*nxd_3);
	cudaStreamDestroy(stream);



}
void gpu_local_grad2_t(double *u, double *ur, double *us, int nxd, double *d, double *dt, double *w, int nel){

	int nxd_2 = nxd * nxd;
	int nxd_3 = nxd_2 * nxd;
	// u(nxd,nxd*nxd) = dt(nxd,nxd) * ur(nxd, nxd*nxd) fortran
	// u(nxd*nxd,nxd) = ur(nxd*nxd, nxd) * dt(nxd,nxd) C
	int blockSize=1024, gridSize;
	cudaStream_t stream;
	cudaStreamCreate( &stream );
	const double alpha = 1;
	const double beta = 0;

	gridSize = (int)ceil((float)nel*nxd_2/blockSize);
	//mxm<<<gridSize, blockSize>>>(ur,nxd_2, dt, nxd, u, nxd, nel, nxd_3, 0, nxd_3, 0);
	cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nxd, nxd, &alpha, dt, nxd, 0, ur, nxd, nxd_2, &beta, u, nxd, nxd_2, nel, gridSize);

	gridSize = (int)ceil((float)nel*nxd_2/blockSize);
	//w(nxd*nxd,nxd) = ut(nxd*nxd,nxd) * D(nxd,nxd) fortran
	//w(nxd,nxd*nxd) = D(nxd,nxd) * ut(nxd,nxd*nxd) C
	//mxm<<<gridSize, blockSize>>>(d,nxd, ut, nxd, w, nxd_2, nel, 0, nxd_3, nxd_3, 0);
	cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nxd, nxd, &alpha, us, nxd, nxd_2, d, nxd, 0, &beta, w, nxd, nxd_2, nel, gridSize);

	nekadd2<<<gridSize, blockSize>>>(u,w, nel*nxd_3);
	cudaStreamDestroy(stream);


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

void gpu_local_grad3(double * ur, double *us, double *ut, double *u, int nx, int nxd, double *d, double *dt, int nel){

	int nx_2 = nx*nx;
	int nx_3 = nx_2*nx;
	int nxd_3 = pow(nxd,3);
	//ur(nx,nx*nx) = D(nx,nx) * u(nx,nx*nx) fortran
	//ur(nx*nx,nx) = u(nx*nx,nx) * D(nx,nx) C
	int blockSize=1024, gridSize;
	gridSize = (int)ceil((float)nel*nx_3/blockSize);
	mxm<<<gridSize, blockSize>>>(u,nx_2, d, nx, ur, nx, nel, nx_3, 0, nxd_3, 0);//ur,us, ut should be indexed by nxd
	for(int k = 0; k<nx; k++){
		//usk(nx,nx) = uk(nx,nx) * dt(nx,nx) fortran
		//usk(nx,nx) = dt(nx,nx) * uk(nx,nx) C
		gridSize = (int)ceil((float)nel*nx_2/blockSize);
		mxm<<<gridSize, blockSize>>>(dt,nx, u+k*nx_2, nx, us+k*nx_2, nx, nel, 0, nx_3, nxd_3, 0);
	}
	//ut(nx_2,nx) = u(nx_2,nx) * dt(nx,nx) fortran
	//ut(nx,nx_2) = dt(nx,nx) * u(nx,nx_2) C
	gridSize = (int)ceil((float)nel*nx_3/blockSize);
	mxm<<<gridSize, blockSize>>>(dt, nx, u, nx, ut, nx_2, nel, 0, nx_3, nxd_3, 0);

}

void gpu_local_grad2(double * ur, double *us, double *u, int nx, int nxd, double *d, double *dt, int nel){

	int nx_2 = nx*nx;
	int nx_3 = nx_2*nx;
	int nxd_2 = pow(nxd,2);
	int nxd_3 = pow(nxd,3);
	//ur(nx,nx*nx) = D(nx,nx) * u(nx,nx*nx) fortran
	//ur(nx*nx,nx) = u(nx*nx,nx) * D(nx,nx) C
	int blockSize=1024, gridSize;
	gridSize = (int)ceil((float)nel*nx_2/blockSize);
	mxm<<<gridSize, blockSize>>>(u,nx, d, nx, ur, nx, nel, nx_2, 0, nxd_2, 0);//ur,us, ut should be indexed by nxd
	//ut(nx_2,nx) = u(nx_2,nx) * dt(nx,nx) fortran
	//ut(nx,nx_2) = dt(nx,nx) * u(nx,nx_2) C
	gridSize = (int)ceil((float)nel*nx_2/blockSize);
	mxm<<<gridSize, blockSize>>>(dt, nx, u, nx, us, nx, nel, 0, nx_2, nxd_2, 0);

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
		printf("GPU :cuda_helpers.cu : gpu_get_int_ptr before ip[0]\n");
	ip[0] = pjgl[ij-1];
		printf("GPU :cuda_helpers.cu : gpu_get_int_ptr after ip[0]\n");
	if (ip[0]==0){
		int nstore   = pjgl[0];
		pjgl[ij-1] = nstore+1;
		nstore   = nstore + md*mx;
		pjgl[0]  = nstore;
		ip[0]       = pjgl[ij-1];
		int nwrkd = mx + md;
		//gpu_gen_int(jgl+ip[0]-1,jgt+ip[0]-1,md,mx,wkd);
	}
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

void gpu_specmpn(double *d_b, int nb, double *d_a, int na, double * d_ba, double* d_ab, bool if3d, double * d_w, int ldw, int nel, int neq, int eq, bool second_eq){
	//d_a is array(na,na,na)*nel, d_b(nb,nb,nb)*nel, w(ldw)*nel where ldw = na*na*nb+nb*nb*na
	//d_a is array of nel each array(na,na,na)
	int blockSize, gridSize;

	// Number of threads in each thread block
	blockSize = 1024;
	cudaStream_t stream;
	cudaStreamCreate( &stream );
	const double alpha = 1;
	const double beta = 0;

	cudaError_t code1 = cudaPeekAtLastError();
        if (code1 != cudaSuccess){
                printf("Start gpu_specmpn cuda error str 1: %s\n",cudaGetErrorString(code1));
        }

	if(if3d){
		int nab = na*nb;
		int nbb = nb*nb;
		//calc w = ba*a in fortran
		//so in c calc wt = at * bat
		//call mxm(ba,nb,a,na,w,na*na)
		//in fortran calc w(nb,na*na) = ba(nb,na) * a(na,na*na)
		//in c w(na*na,nb) = a(na*na,na) * ba(na,nb)
		//neq = 1 if array not indexed by eq and eq = 0
		int aSize = neq*pow(na,3), bSize = pow(nb,3);
		gridSize = (int)ceil((float)na*na*nb*nel/blockSize);
		//mxm<<<gridSize, blockSize>>>(d_a,na*na, d_ba, na, d_w, nb, nel, aSize, 0, ldw, eq*pow(na,3));
		printf("GPU :cuda_helpers.cu : gpu_specmpn before multi_gemm_unif 1\n");

		int resultfrommultigem = cuda_multi_gemm_unif(stream, 'N', 'N', nb, na, na*na, &alpha, d_ba, nb, 0, d_a, na, aSize, &beta, d_w, nb, ldw, nel, gridSize);
		printf("GPU :cuda_helpers.cu : gpu_specmpn after multi_gemm_unif 1 result %d\n",resultfrommultigem);
		int k = 0, l = na*na*nb;
		for(int iz=0; iz<na;iz++){
			//calc in fortran wl(nb*nb) = wk(nb*na) * ab(na*nb)
			//in c wl(nb*nb) = ab(nb*na) * wk(na*nb)
			gridSize = (int)ceil((float)nb*nb*nel/blockSize);
			//mxm<<<gridSize, blockSize>>>(d_ab,nb, d_w+k, na, d_w+l, nb, nel, 0, ldw, ldw, 0);
			printf("GPU :cuda_helpers.cu : gpu_specmpn before multi_gemm_unif 2 big for loop\n");
			cuda_multi_gemm_unif(stream, 'N', 'N', nb, na, nb, &alpha, d_w+k, nb, ldw, d_ab, na, 0, &beta, d_w+l, nb, ldw, nel, gridSize);
			printf("GPU :cuda_helpers.cu : gpu_specmpn after multi_gemm_unif 2 big for loop\n");

			k = k + nab;
			l = l + nbb;
		}
		l = na*na*nb;
		//calc in fortran b(nb*nb,nb) = wl(nb*nb,na)* ab(na,nb)
		//in C b(nb,nb*nb) = ab(nb,na) * wl(na,nb*nb)
		gridSize = (int)ceil((float)nb*nb*nb*nel/blockSize);
		//mxm<<<gridSize, blockSize>>>(d_ab,nb, d_w+l, na, d_b, nb*nb, nel, 0, ldw, bSize, 0);
			printf("GPU :cuda_helpers.cu : gpu_specmpn before multi_gemm_unif 3 \n");
		cuda_multi_gemm_unif(stream, 'N', 'N', nb*nb, na, nb, &alpha, d_w+l, nb*nb, ldw, d_ab, na, 0, &beta, d_b, nb*nb, bSize, nel, gridSize);
			printf("GPU :cuda_helpers.cu : gpu_specmpn after multi_gemm_unif 3 \n");


	}
	else{
		//calc w(nb*na) = ba(nb,na) * a(na,na) in fortran,
		//in C w(na*nb) = a(na,na) * ba(na,nb)
		gridSize = (int)ceil((float)na*nb*nel/blockSize);
		mxm<<<gridSize, blockSize>>>(d_a,na, d_ba, na, d_w, nb, nel, neq*na*na, 0, ldw, eq*na*na);
		//in fortran, b(nb,nb) = w(nb,na)*ab(na,nb)
		//in C b(nb,nb) = ab(nb,na) * w(na,nb)
		gridSize = (int)ceil((float)nb*nb*nel/blockSize);
		mxm<<<gridSize, blockSize>>>(d_ab,nb, d_w, na, d_b, nb, nel, 0, ldw, nb*nb, 0);


	}
	cudaError_t code = cudaPeekAtLastError();
	if (code != cudaSuccess){
		printf("cuda error str 1: %s\n",cudaGetErrorString(code));
	}
	cudaStreamDestroy(stream);

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



void gpu_gradl_rst(double *ur, double *us, double *ut, double *u, double *d, double *dt, int md, int nel, bool if3d,int *ip,double *wkd,int *pdg,int nx, int nxd){
	int m0=md-1;
	//gpu_get_dgll_ptr(ip,if3d,md,md,nel,d,dt,wkd,nxd,pdg);
	if(if3d){
		gpu_local_grad3(ur, us, ut, u, nx, nxd, d, dt, nel);  //something is wrong with these functions. Need to do some deep digging to find out the correct values for nx and nx2. Talk with Mohamed to find out the correct way. adeesha.
	}
	else{
		gpu_local_grad2(ur, us, u, nx, nxd, d, dt, nel);
	}


}


