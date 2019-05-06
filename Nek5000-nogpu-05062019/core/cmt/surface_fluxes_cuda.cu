#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "nvml.h"

// includes, project
//#include "magma.h"
#include "cuda_multi_gemm_unif.cu"

__global__ void fluxes_full_field_gpu_kernel_fillq(double *vtrans, double *vx, double *vy, double *vz, double *pr, double *t, double *csound, double *phig, double *vdiff, double *fatface,int irho, int iux, int iuy, int iuz, int ipr, int ithm, int isnd, int iph, int icvf, int icpf, int imuf, int ikndf, int ilamf, int iwm, int iwp, int icv, int icp, int imu, int iknd, int ilam,int *iface_flux, int nnel, int nxz2ldim, int lxyz,int lxz, int ivarcoef,int leltlxyz ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){

		int e = id/nxz2ldim;
		int j = id % nxz2ldim;
		//fillq vtrans
		int i = iface_flux[id]-1;  // because forgot to -1 in the follows
		fatface[(iwp-1)+id] = vtrans[e*lxyz+i]; 
		// following works because ndg_face is same as nnel. Talk with Dr. Tania. adeesha
		fatface[(iwm-1)+(irho-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 

		//fillq vx
		fatface[(iwp-1)+id] = vx[e*lxyz+i]; 
		fatface[(iwm-1)+(iux-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq vy
		fatface[(iwp-1)+id] = vy[e*lxyz+i]; 
		fatface[(iwm-1)+(iuy-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq vz
		fatface[(iwp-1)+id] = vz[e*lxyz+i]; 
		fatface[(iwm-1)+(iuz-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq pr
		fatface[(iwp-1)+id] = pr[e*lxyz+i]; 
		fatface[(iwm-1)+(ipr-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq t
		fatface[(iwp-1)+id] = t[e*lxyz+i]; 
		fatface[(iwm-1)+(ithm-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq csound
		fatface[(iwp-1)+id] = csound[e*lxyz+i]; 
		fatface[(iwm-1)+(isnd-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq phig
		fatface[(iwp-1)+id] = phig[e*lxyz+i]; 
		fatface[(iwm-1)+(iph-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq vtrans icv
		fatface[(iwp-1)+id] = vtrans[(icv-1)*leltlxyz+e*lxyz+i]; 
		fatface[(iwm-1)+(icvf-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq vtrans icp
		fatface[(iwp-1)+id] = vtrans[(icp-1)*leltlxyz+e*lxyz+i]; 
		fatface[(iwm-1)+(icpf-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq vdiff imu
		fatface[(iwp-1)+id] = vdiff[(imu-1)*leltlxyz+e*lxyz+i]; 
		fatface[(iwm-1)+(imuf-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq vdiff iknd
		fatface[(iwp-1)+id] = vdiff[(iknd-1)*leltlxyz+e*lxyz+i]; 
		fatface[(iwm-1)+(ikndf-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 
		//fillq vdiff ilam
		fatface[(iwp-1)+id] = vdiff[(ilam-1)*leltlxyz+e*lxyz+i]; 
		fatface[(iwm-1)+(ilam-1)*ivarcoef+id] = fatface[(iwp-1)+id]; 




	}

}

__global__ void fluxes_full_field_gpu_kernel_faceu(double *fatface, double *u,int i_cvars, int nneltoteq, int nnel, int toteq, int lxyz, int iwm, int iph,int *iface_flux,int nxz2ldim){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nneltoteq){
		int ivar = id/nnel;
		int e_n =  id%(nnel);;
		int e = e_n/nxz2ldim;
		int i = iface_flux[e_n];
		//full2face_cmt
		fatface[(i_cvars-1)+id] =u[e*toteq*lxyz+ivar*lxyz+i-1];

		fatface[(i_cvars-1)+id]= fatface[(i_cvars-1)+id]/fatface[iwm-1+nnel*(iph-2)+id];// invcol2
		// check with Dr.Tania. above functions may not work properly.

	}

}


extern "C" void fluxes_full_field_gpu_wrapper_(int *glbblockSize2,double *d_fatface,double *d_vtrans,double *d_u, double *d_vx, double *d_vy, double *d_vz, double *d_pr, double *d_t, double *d_csound, double *d_phig, double *d_vdiff, int *irho, int *iux, int *iuy, int *iuz, int *ipr, int *ithm, int *isnd, int *iph, int *icvf, int *icpf, int *imuf, int *ikndf, int *ilamf, int *iwm, int *iwp, int *icv, int *icp, int *imu, int *iknd, int *ilam,int *d_iface_flux, int *nelt, int *lx1, int *ly1, int *lz1, int *ldim, int *lelt, int *i_cvars,int *toteq, int *nqq){

cudaDeviceSynchronize();
cudaError_t code1 = cudaPeekAtLastError();
//        if (code1 != cudaSuccess){
                printf("CUDA: Start fluxes_full_field_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
		printf("CUDA: Start compute_entropy_gpu_wrapper values  irho = %d, iux= %d,iuy= %d,iuz= %d,ipr= %d,ithm= %d,isnd= %d,iph= %d, icvf= %d,icpf= %d,imuf= %d,ikndf= %d,ilamf= %d,iwm= %d,iwp= %d,icv= %d, icp= %d,imu= %d,iknd= %d,ilam= %d,nelt= %d,lx1= %d,ly1= %d,lz1= %d,ldim= %d,lelt= %d,i_cvars= %d,toteq= %d,nqq=%d \n",irho[0],iux[0],iuy[0],iuz[0],ipr[0],ithm[0],isnd[0],iph[0],icvf[0],icpf[0],imuf[0],ikndf[0],ilamf[0],iwm[0],iwp[0],icv[0],icp[0],imu[0],iknd[0],ilam[0],nelt[0],lx1[0],ly1[0],lz1[0],ldim[0],lelt[0],i_cvars[0],toteq[0],nqq[0]);
  //      }

	int nxz2ldim = lx1[0]*lz1[0]*2*ldim[0];
	int lxyz = lx1[0]*ly1[0]*lz1[0];
	int leltlxyz = lelt[0]*lxyz;
	int lxz = lx1[0]*lz1[0];
	int nnel = nelt[0]*nxz2ldim;
	int ivarcoef = nxz2ldim*lelt[0];
	int nnelnqq = nnel*nqq[0];
	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)nnel/blockSize);
	fluxes_full_field_gpu_kernel_fillq<<<gridSize, blockSize>>>(d_vtrans, d_vx, d_vy, d_vz,d_pr, d_t, d_csound,d_phig,d_vdiff,d_fatface,irho[0],iux[0], iuy[0], iuz[0], ipr[0],ithm[0],isnd[0],iph[0],icvf[0],icpf[0],imuf[0],ikndf[0],ilamf[0],iwm[0],iwp[0],icv[0],icp[0],imu[0],iknd[0],ilam[0],d_iface_flux,nnel, nxz2ldim,lxyz,lxz,ivarcoef, leltlxyz);

	cudaDeviceSynchronize();
	code1 = cudaPeekAtLastError();
                printf("CUDA: fluxes_full_field_gpu_wrapper after fillq cuda status: %s\n",cudaGetErrorString(code1));

	gridSize = (int)ceil((float)nnel*toteq[0]/blockSize);
	fluxes_full_field_gpu_kernel_faceu<<<gridSize, blockSize>>>(d_fatface,d_u,i_cvars[0],nnel*toteq[0],nnel, toteq[0], lxyz,iwm[0],iph[0],d_iface_flux,nxz2ldim);

cudaDeviceSynchronize();
cudaError_t code2 = cudaPeekAtLastError();
//        if (code1 != cudaSuccess){
                printf("CUDA: End fluxes_full_field_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));


}


void map_faced(double *jgl, double *jgt, double *ju, double *u, double *w, int nx1, int nxd, int fdim, int nelt, int nfaces, int idir){

	cudaStream_t stream;
	cudaStreamCreate( &stream );
	const double alpha = 1;
	const double beta = 0;
	int nx1_2 = pow(nx1,2);
	int nxd_2 = pow(nxd,2);
	int batchSize = nelt*nfaces;


	if(idir==0){
		int blockSize = 1024, gridSize;
		//calc w(nxd,nx1) = jgl(nxd*nx1) * u(nx1,nx1) in fortran
		//calc w(nx1,nxd) = u(nx1,nx1) * jgl(nx1,nxd) in C
		gridSize = (int)ceil((float)nelt*nfaces*nx1*nxd/blockSize);
		cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nx1, nx1, &alpha, jgl, nxd, 0, u, nx1, nx1_2, &beta, w, nxd, nx1*nxd, batchSize, gridSize);
		//mxm_faces<<<gridSize, blockSize>>>(u, nx1, jgl, nx1, w, nxd, nelt, nfaces, nx1*nx1, 0, nx1*nxd);

		//calc ju(nxd,nxd) = w(nxd,nx1) * jgt(nx1,nxd) in fortran
		//calc ju(nxd,nxd) = jgt(nxd,nx1) * w(nx1,nxd)
		gridSize = (int)ceil((float)nelt*nfaces*nxd*nxd/blockSize);
		cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nx1, nxd, &alpha, w, nxd, nx1*nxd, jgt, nx1, 0, &beta, ju, nxd, nxd_2, batchSize, gridSize);
		//mxm_faces<<<gridSize, blockSize>>>(jgt, nxd, w, nx1, ju, nxd, nelt, nfaces, 0, nx1*nxd, nxd*nxd);
	}
	else{
		int blockSize = 1024, gridSize;
		//calc w(nx1,nxd) = jgt(nx1,nxd) * u(nxd,nxd) in fortran
		//calc w(nxd,nx1) = u(nxd,nxd) * jgt(nxd,nx1) in C
		gridSize = (int)ceil((float)nelt*nfaces*nx1*nxd/blockSize);
		cuda_multi_gemm_unif(stream, 'N', 'N', nx1, nxd, nxd, &alpha, jgt, nx1, 0, u, nxd, nxd_2, &beta, w, nx1, nx1*nxd, batchSize, gridSize);
		//mxm_faces<<<gridSize, blockSize>>>(u, nxd, jgt, nxd, w, nx1, nelt, nfaces, nxd*nxd, 0, nx1*nxd);

		//calc ju(nx1,nx1) = w(nx1,nxd) * jgl(nxd,nx1) in fortran
		//calc ju(nx1,nx1) = jgl(nx1,nxd) * w(nxd,nx1) in C
		gridSize = (int)ceil((float)nelt*nfaces*nx1*nx1/blockSize);
		cuda_multi_gemm_unif(stream, 'N', 'N', nx1, nxd, nx1, &alpha, w, nx1, nx1*nxd, jgl, nxd, 0, &beta, ju, nx1, nx1_2, batchSize, gridSize);
		//mxm_faces<<<gridSize, blockSize>>>(jgl, nx1, w, nxd, ju, nx1, nelt, nfaces, 0, nx1*nxd, nx1*nx1);

	}
	cudaStreamDestroy(stream);
}
extern "C" void InviscidFlux_gpu_wrapper_( ){
	/*

	//nx extended to be nx(nel,nfaces,#points_in_face)
	//irho should be irho1[0]-1, others also
	//printf("in invFlux**\n");
	int fdim = ndim-1;
	int nfaces = 2*ndim;
	int nx1_2 = nx1*nx1;
	int nxd_2 = nxd*nxd;
	double *w;
	cudaMalloc(&w,nel*nfaces*pow(nxd,2)*sizeof(double));


	//add neksub2 which is last step of face_state_commo
	int blockSize1 = 1024, gridSize1;
	gridSize1 = (int)ceil((float)nstate*nel*nfaces*nx1_2/blockSize1);

	neksub2<<<gridSize1, blockSize1>>>(qplus,qminus,nstate*nel*nfaces*nx1_2);

	cudaError_t code = cudaPeekAtLastError();
	if (code != cudaSuccess){
	printf("cuda error Inv, comp-1: %s\n",cudaGetErrorString(code));
	}


	int totpts = nel * nfaces *  nx1_2;
	map_faced(jgl, jgt, nx, unx, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, ny, uny, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, nz, unz, w, nx1, nxd, fdim, nel, nfaces, 0);
	code = cudaPeekAtLastError();
	if (code != cudaSuccess){
	printf("cuda error Inv, comp-2: %s\n",cudaGetErrorString(code));
	}


	//printf("irho=%d,iux=%d,iuy=%d,iuz=%d,ipr=%d,ithm=%d,isnd=%d,icpf=%d\n",irho,iux,iuy,iuz,ipr,ithm,isnd,icpf);
	map_faced(jgl, jgt, rl, qminus+irho*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, ul, qminus+iux*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, vl, qminus+iuy*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, wl, qminus+iuz*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, pl, qminus+ipr*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, tl, qminus+ithm*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, al, qminus+isnd*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, cpl, qminus+icpf*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	code = cudaPeekAtLastError();
	if (code != cudaSuccess){
	printf("cuda error Inv, comp-3: %s\n",cudaGetErrorString(code));
	}


	map_faced(jgl, jgt, rr, qplus+irho*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, ur, qplus+iux*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, vr, qplus+iuy*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, wr, qplus+iuz*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, pr, qplus+ipr*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, tr, qplus+ithm*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, ar, qplus+isnd*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	map_faced(jgl, jgt, cpr, qplus+icpf*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	code = cudaPeekAtLastError();
	if (code != cudaSuccess){
	printf("cuda error Inv, comp-4: %s\n",cudaGetErrorString(code));
	}


	int blockSize = 1024, gridSize;
	gridSize = (int)ceil((float)totpts/blockSize);
	invcol3_flux<<<gridSize, blockSize>>>(jaco_c,area,wghtc,nx1_2,totpts);
	map_faced(jgl, jgt, jaco_f, jaco_c, w, nx1, nxd, fdim, nel, nfaces, 0);

	int totpts_d = nel * nfaces * nxd_2;
	gridSize = (int)ceil((float)totpts_d/blockSize);
	nekcol2_flux<<<gridSize, blockSize>>>(jaco_f,wghtf,nxd_2,totpts_d);

	//Ausm
	//gridSize = (int)ceil((float)nel*nfaces*nxd_2/blockSize);
	invcol2<<<gridSize, blockSize>>>(cpl,rl,totpts_d);
	invcol2<<<gridSize, blockSize>>>(cpr,rr,totpts_d);

	//gridSize = (int)ceil((float)nel*nfaces*nxd_2/blockSize);
	Ausm_flux<<<gridSize, blockSize>>>(neq, totpts_d, nx, ny, nz, jaco_f, fs, rl, ul, vl, wl, pl, al, tl, rr, ur, vr, wr, pr, ar, tr, flx, cpl, cpr);
	code = cudaPeekAtLastError();
	if (code != cudaSuccess){
		printf("cuda error Inv, comp-5: %s\n",cudaGetErrorString(code));
	}


	map_faced(jgl, jgt, pl, qminus+iph*totpts, w, nx1, nxd, fdim, nel, nfaces, 0);
	for(int j=0; j<neq;j++){
		nekcol2<<<gridSize, blockSize>>>(flx+j*totpts_d,pl,totpts_d);
		map_faced(jgl, jgt, flux+j*totpts, flx+j*totpts_d, w, nx1, nxd, fdim, nel, nfaces, 1);
	}
	code = cudaPeekAtLastError();
	if (code != cudaSuccess){
		printf("cuda error Inv, comp-6: %s\n",cudaGetErrorString(code));
	}


	cudaFree(w);
	code = cudaPeekAtLastError();
	if (code != cudaSuccess){
		printf("cuda error Inv, comp-7: %s\n",cudaGetErrorString(code));
	}
	*/
}

__global__ void surface_integral_full_gpu_kernel(double *iface_flux, double *res1, double *flux, int ntot,int lxyz,int lxyzlelt,int nxz2ldim,int eq ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){

		int e = id/nxz2ldim;

		//add_face2full_cmt(nel,nx,ny,nz,iface,vols,faces)
		int newi=iface_flux[id];
		res1[eq*lxyzlelt+e*lxyz+newi-1]=  res1[eq*lxyzlelt+e*lxyz+newi-1]+flux[id];
	}

}


extern "C" void surface_integral_full_gpu_wrapper_(int *glbblockSize2,double *d_res1,double *d_flux,int *eq,int *ieq,int *nelt, int *lelt, int *toteq,int *lx1,int *ly1,int *lz1,int *ldim,double *d_iface_flux ){

cudaDeviceSynchronize();
cudaError_t code1 = cudaPeekAtLastError();
printf("CUDA: Start surface_integral_full_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

printf("CUDA: Start surface_integral_full_gpu_wrapper values eq = %d,ieq= %d,nelt= %d,lelt=%d,toteq= %d,lx1=%d,ly1= %d,lz1= %d,ldim= %d\n",eq[0],ieq[0],nelt[0],lelt[0],toteq[0],lx1[0],ly1[0],lz1[0],ldim[0]);

	int nxz2ldim = lx1[0]*lz1[0]*2*ldim[0];
	int lxyz = lx1[0]*ly1[0]*lz1[0];
	int lxyzlelt = lelt[0]*lxyz;
	int ntot = nelt[0]*nxz2ldim;
	int ivarcoef = nxz2ldim*lelt[0];

	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);

	surface_integral_full_gpu_kernel<<<gridSize, blockSize>>>(d_iface_flux,d_res1,d_flux+ieq[0]-1,ntot,lxyz,lxyzlelt,nxz2ldim,eq[0]);

cudaError_t code2 = cudaPeekAtLastError();

printf("CUDA: End surface_integral_full_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));

}

__global__ void igu_cmt_gpu_wrapper1_gpu_kernel(double *flux, double *graduf,int ntot,int lxz2ldimlelt,int toteq,int iwp ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){

		for(int eq=0;eq<toteq;eq++){
			flux[iwp-1+eq*lxz2ldimlelt +id]=graduf[eq*lxz2ldimlelt+id];
			graduf[eq*lxz2ldimlelt+id]=graduf[eq*lxz2ldimlelt+id]*0.5;
		}
	}

}

__global__ void igu_cmt_gpu_wrapper2_gpu_kernel(double *flux,double *graduf,char *cbc,double *unx,double *uny,double *unz,int ntot,int lxz2ldim,int lxz2ldimlelt,int lxz2ldimlelttoteq,int toteq,int iwp,int iwm,int ldim,int irho,int icvf,int ilamf,int imuf,int iux,int iuy,int iuz,int iu5,int if3d,int lxz,int a2ldim ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){
		int e = id/lxz2ldim;
		int iface= ((id/lxz)%a2ldim);
		for(int eq=0;eq<toteq;eq++){
			flux[iwp-1+eq*lxz2ldimlelt +id]=flux[iwp-1+eq*lxz2ldimlelt +id]-graduf[eq*lxz2ldimlelt+id];
		}

		//igu_dirichlet(flxscr,gdudxk)
		char cb2= cbc[e*18+iface];
		if(cb2  !='E' && cb2 !='P'){
			for (int eq=0;eq<toteq;eq++){
				graduf[eq*lxz2ldimlelt+id] =graduf[eq*lxz2ldimlelt+id]*2.0;
				flux[iwp-1+eq*lxz2ldimlelt +id]=graduf[eq*lxz2ldimlelt+id]; 
			}

		}
		//bcflux(flxscr,gdudxk,wminus)
		if (cbc[e*18+iface]!='E'&& cbc[e*18+iface]!='P'){
			char cb1= cbc[e*18+iface];
			if(cb1=='I'){
				flux[iwp-1+id]=0;
				for (int eq=1;eq<ldim+1;eq++){
					flux[iwp-1+eq*lxz2ldimlelt +id]=graduf[eq*lxz2ldimlelt+id];
				}
				flux[iwp-1+(toteq-1)*lxz2ldimlelt +id]=0;
				//a5adiabatic_wall(flux(1,1,1,toteq),f,e,agradu,qminus)
				//call a51dUadia(flxscr,f,e,dU,wstate)
				double dU1x=graduf[0*lxz2ldimlelttoteq+0*lxz2ldimlelt+id];
				double dU2x=graduf[0*lxz2ldimlelttoteq+1*lxz2ldimlelt+id];
				double dU3x=graduf[0*lxz2ldimlelttoteq+2*lxz2ldimlelt+id];
				double dU4x=graduf[0*lxz2ldimlelttoteq+3*lxz2ldimlelt+id];
				double dU5x=graduf[0*lxz2ldimlelttoteq+4*lxz2ldimlelt+id];
				double dU1y=graduf[1*lxz2ldimlelttoteq+0*lxz2ldimlelt+id];
				double dU2y=graduf[1*lxz2ldimlelttoteq+1*lxz2ldimlelt+id];
				double dU3y=graduf[1*lxz2ldimlelttoteq+2*lxz2ldimlelt+id];
				double dU4y=graduf[1*lxz2ldimlelttoteq+3*lxz2ldimlelt+id];
				double dU5y=graduf[1*lxz2ldimlelttoteq+4*lxz2ldimlelt+id];
				double dU1z=graduf[2*lxz2ldimlelttoteq+0*lxz2ldimlelt+id];
				double dU2z=graduf[2*lxz2ldimlelttoteq+1*lxz2ldimlelt+id];
				double dU3z=graduf[2*lxz2ldimlelttoteq+2*lxz2ldimlelt+id];
				double dU4z=graduf[2*lxz2ldimlelttoteq+3*lxz2ldimlelt+id];
				double dU5z=graduf[2*lxz2ldimlelttoteq+4*lxz2ldimlelt+id];
				double rho   =flux[iwm-1+(irho-1)*lxz2ldimlelt +id];
				double cv    =flux[iwm-1+(icvf-1)*lxz2ldimlelt +id]/rho;
				double lambda=flux[iwm-1+(ilamf-1)*lxz2ldimlelt +id];
				double mu    =flux[iwm-1+(imuf-1)*lxz2ldimlelt +id];
				double K     =0.0;// ADIABATIC HARDCODING
					double u1    =flux[iwm-1+(iux-1)*lxz2ldimlelt +id];
				double u2    =flux[iwm-1+(iuy-1)*lxz2ldimlelt +id];
				double u3    =flux[iwm-1+(iuz-1)*lxz2ldimlelt +id];
				double E     =flux[iwm-1+(iu5-1)*lxz2ldimlelt +id]/rho;
				double lambdamu=lambda+mu;
				double kmcvmu=K-cv*mu;
				double t_flux=(K*dU5x+cv*lambda*u1*dU4z-kmcvmu*u3*dU4x+cv*lambda*u1*dU3y-kmcvmu*u2*dU3x+cv*mu*u3*dU2z+cv*mu*u2*dU2y+(cv*lambda-K+2*cv*mu)*u1*dU2x-cv*lambdamu*u1*u3*dU1z-cv*lambdamu*u1*u2*dU1y+(K*u3*u3-cv*mu*u3*u3+K*u2*u2-cv*mu*u2*u2-cv*lambda*u1*u1+K*u1*u1-2*cv*mu*u1*u1-E*K)*dU1x)/(cv*rho);

				flux[iwp-1+(toteq-1)*lxz2ldimlelt +id]=flux[iwp-1+(toteq-1)*lxz2ldimlelt +id]+t_flux*unx[id];

				//a52dUadia_gpu(flux,f,ie,dU,wstate)
				t_flux=(K*dU5y+cv*lambda*u2*dU4z-kmcvmu*u3*dU4y+cv*mu*u3*dU3z+(cv*lambda-K+2*cv*mu)*u2*dU3y+cv*mu*u1*dU3x-kmcvmu*u1*dU2y+cv*lambda*u2*dU2x-cv*lambdamu*u2*u3*dU1z+(K*u3*u3-cv*mu*u3*u3-cv*lambda*u2*u2+K*u2*u2-2*cv*mu*u2*u2+K*u1*u1-cv*mu*u1*u1-E*K)*dU1y-cv*lambdamu*u1*u2*dU1x)/(cv*rho);
				flux[iwp-1+(toteq-1)*lxz2ldimlelt +id]=flux[iwp-1+(toteq-1)*lxz2ldimlelt +id]+t_flux*uny[id];
				
				if(if3d){
					t_flux=(K*(dU5z-E*dU1z)+cv*u3*(lambda*dU4z+2*mu*dU4z+lambda*dU3y+lambda*dU2x)-K*u3*dU4z+cv*mu*u2*(dU4y+dU3z)+cv*mu*u1*(dU4x+dU2z)-K*u2*dU3z-K*u1*dU2z-cv*(lambda+2*mu)*u3*u3*dU1z+K*u3*u3*dU1z+K*u2*u2*dU1z-cv*mu*u2*u2*dU1z+K*u1*u1*dU1z-cv*mu*u1*u1*dU1z-cv*(lambda+mu)*u2*u3*dU1y-cv*(lambda+mu)*u1*u3*dU1x)/(cv*rho);
			
					flux[iwp-1+(toteq-1)*lxz2ldimlelt +id]=flux[iwp-1+(toteq-1)*lxz2ldimlelt +id]+t_flux*unz[id];

				}



			}
			else{
				for (int eq=0;eq<toteq;eq++){
					flux[iwp-1+eq*lxz2ldimlelt+id]=0;
				}

			}			
		}
		//chsign(flxscr,ntot)
		for (int eq=0;eq<toteq;eq++){
			flux[iwp-1+eq*lxz2ldimlelt +id]=-1*flux[iwp-1+eq*lxz2ldimlelt +id];
		}
	}

}



extern "C" void  igu_cmt_gpu_wrapper1_(int *glbblockSize2,double *d_flux,double *d_graduf,int *toteq,int *iwp,int *lx1,int *ly1,int *lz1,int *ldim,int *nelt,int *lelt){

cudaError_t code1 = cudaPeekAtLastError();

printf("CUDA: Start igu_cmt_gpu_wrapper1 cuda status: %s\n",cudaGetErrorString(code1));

printf("CUDA: Start igu_cmt_gpu_wrapper1 values toteq=%d ,iwp=%d ,lx1=%d ,ly1=%d ,lz1=%d ,ldim=%d ,nelt=%d ,lelt=%d ,\n",toteq[0],iwp[0],lx1[0],ly1[0],lz1[0],ldim[0],nelt[0],lelt[0]);

	int lxz2ldim = lx1[0]*lz1[0]*2*ldim[0];
	int ntot = nelt[0]*lxz2ldim;
	int lxz2ldimlelt= lxz2ldim*lelt[0];
	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);

	igu_cmt_gpu_wrapper1_gpu_kernel<<<gridSize, blockSize>>>(d_flux,d_graduf,ntot,lxz2ldimlelt,toteq[0],iwp[0]);
cudaError_t code2 = cudaPeekAtLastError();
printf("CUDA: End igu_cmt_gpu_wrapper1 cuda status: %s\n",cudaGetErrorString(code2));


}

extern "C" void  igu_cmt_gpu_wrapper2_(int *glbblockSize2,double *d_flux,double *d_graduf,char *d_cbc,double *d_unx,double *d_uny,double *d_unz,int *toteq,int *iwp,int *iwm,int *ldim,int *irho,int *icvf,int *ilamf,int *imuf,int *iux,int *iuy,int *iuz,int *iu5,int *if3d,int *lx1,int *ly1,int *lz1,int *nelt,int *lelt){

cudaError_t code1 = cudaPeekAtLastError();

printf("CUDA: Start igu_cmt_gpu_wrapper2 cuda status: %s\n",cudaGetErrorString(code1));

printf("CUDA: Start igu_cmt_gpu_wrapper2 values toteq =%d ,iwp =%d,iwm =%d,ldim =%d,irho =%d,icvf =%d,ilamf =%d,imuf =%d,iux =%d,iuy =%d,iuz =%d,iu5 =%d,if3d =%d,lx1 =%d,ly1 =%d,lz1 =%d,nelt =%d,lelt =%d \n", toteq[0],iwp[0],iwm[0],ldim[0],irho[0],icvf[0],ilamf[0],imuf[0],iux[0],iuy[0],iuz[0],iu5[0],if3d[0],lx1[0],ly1[0],lz1[0],nelt[0],lelt[0] );

	int lxz=lx1[0]*lz1[0];
	int a2ldim= 2*ldim[0];
	int lxz2ldim = lxz*2*ldim[0];
	int ntot = nelt[0]*lxz2ldim;
	int lxz2ldimlelt =lxz2ldim*lelt[0];
	int  lxz2ldimlelttoteq= lxz2ldimlelt*toteq[0];

	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);

	igu_cmt_gpu_wrapper2_gpu_kernel<<<gridSize, blockSize>>>(d_flux,d_graduf,d_cbc,d_unx,d_uny,d_unz,ntot,lxz2ldim,lxz2ldimlelt, lxz2ldimlelttoteq,toteq[0],iwp[0],iwm[0],ldim[0],irho[0],icvf[0],ilamf[0],imuf[0],iux[0],iuy[0],iuz[0],iu5[0],if3d[0],lxz,a2ldim);

cudaError_t code2 = cudaPeekAtLastError();
printf("CUDA: End igu_cmt_gpu_wrapper2 cuda status: %s\n",cudaGetErrorString(code2));

}












