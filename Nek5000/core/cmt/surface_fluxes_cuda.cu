#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
//#include <cublas.h>
#include "nvml.h"

// includes, project
//#include "magma.h"
#include "cuda_multi_gemm_unif.cu"
#include "cuda_helpers.h"

#define DEBUGPRINT 0
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


		//		if(id<10){
		//			printf("$$$ pfc i=%d, vtrans[%d]=%lf,vx[%d]=%lf,vy[%d]=%lf,vz[%d]=%lf,pr[%d]=%lf,t[%d]=%lf,csound[%d]=%lf,phig[%d]=%lf,vtrans[%d]=%lf,vdiff[%d]=%lf,vtrans[%d]=%lf \n",i,e*lxyz+i,vtrans[e*lxyz+i],e*lxyz+i,vx[e*lxyz+i],e*lxyz+i,vy[e*lxyz+i],e*lxyz+i,vz[e*lxyz+i],e*lxyz+i,pr[e*lxyz+i],e*lxyz+i,t[e*lxyz+i],e*lxyz+i,csound[e*lxyz+i],e*lxyz+i,phig[e*lxyz+i],e*lxyz+i,vtrans[e*lxyz+i],e*lxyz+i,vdiff[e*lxyz+i],(icp-1)*leltlxyz+e*lxyz+i,vtrans[(icp-1)*leltlxyz+e*lxyz+i]);

		//		}

	}

}

__global__ void fluxes_full_field_gpu_kernel_faceu(double *flux, double *u,int i_cvars, int nneltoteq, int nnel, int toteq, int lxyz, int iwm, int iph,int *iface_flux,int nxz2ldim,int ivarcoef){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nneltoteq){
		int ivar = id/nnel;
		int e_n =  id%(nnel);;
		int e = e_n/nxz2ldim;
		int i = iface_flux[e_n];
		//full2face_cmt
		flux[(i_cvars-1)+id] =u[e*toteq*lxyz+ivar*lxyz+i-1];
		flux[(i_cvars-1)+id]= flux[(i_cvars-1)+id]/flux[iwm-1+ivarcoef*(iph-2)+id];// invcol2
		// check with Dr.Tania. above functions may not work properly.

	}

}


extern "C" void fluxes_full_field_gpu_wrapper_(int *glbblockSize2,double *d_flux,double *d_vtrans,double *d_u, double *d_vx, double *d_vy, double *d_vz, double *d_pr, double *d_t, double *d_csound, double *d_phig, double *d_vdiff, int *irho, int *iux, int *iuy, int *iuz, int *ipr, int *ithm, int *isnd, int *iph, int *icvf, int *icpf, int *imuf, int *ikndf, int *ilamf, int *iwm, int *iwp, int *icv, int *icp, int *imu, int *iknd, int *ilam,int *d_iface_flux, int *nelt, int *lx1, int *ly1, int *lz1, int *ldim, int *lelt, int *i_cvars,int *toteq, int *nqq){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	//        if (code1 != cudaSuccess){
	printf("CUDA: Start fluxes_full_field_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
	printf("CUDA: Start fluxes_full_field_gpu_wrapper values  irho = %d, iux= %d,iuy= %d,iuz= %d,ipr= %d,ithm= %d,isnd= %d,iph= %d, icvf= %d,icpf= %d,imuf= %d,ikndf= %d,ilamf= %d,iwm= %d,iwp= %d,icv= %d, icp= %d,imu= %d,iknd= %d,ilam= %d,nelt= %d,lx1= %d,ly1= %d,lz1= %d,ldim= %d,lelt= %d,i_cvars= %d,toteq= %d,nqq=%d \n",irho[0],iux[0],iuy[0],iuz[0],ipr[0],ithm[0],isnd[0],iph[0],icvf[0],icpf[0],imuf[0],ikndf[0],ilamf[0],iwm[0],iwp[0],icv[0],icp[0],imu[0],iknd[0],ilam[0],nelt[0],lx1[0],ly1[0],lz1[0],ldim[0],lelt[0],i_cvars[0],toteq[0],nqq[0]);
	//      }
#endif
	int nxz2ldim = lx1[0]*lz1[0]*2*ldim[0];
	int lxyz = lx1[0]*ly1[0]*lz1[0];
	int leltlxyz = lelt[0]*lxyz;
	int lxz = lx1[0]*lz1[0];
	int nnel = nelt[0]*nxz2ldim;
	int ivarcoef = nxz2ldim*nelt[0];
	int nnelnqq = nnel*nqq[0];
	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)nnel/blockSize);
	fluxes_full_field_gpu_kernel_fillq<<<gridSize, blockSize>>>(d_vtrans, d_vx, d_vy, d_vz,d_pr, d_t, d_csound,d_phig,d_vdiff,d_flux,irho[0],iux[0], iuy[0], iuz[0], ipr[0],ithm[0],isnd[0],iph[0],icvf[0],icpf[0],imuf[0],ikndf[0],ilamf[0],iwm[0],iwp[0],icv[0],icp[0],imu[0],iknd[0],ilam[0],d_iface_flux,nnel, nxz2ldim,lxyz,lxz,ivarcoef, leltlxyz);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	code1 = cudaPeekAtLastError();
	printf("CUDA: fluxes_full_field_gpu_wrapper after fillq cuda status: %s\n",cudaGetErrorString(code1));
#endif
	gridSize = (int)ceil((float)nnel*toteq[0]/blockSize);
	fluxes_full_field_gpu_kernel_faceu<<<gridSize, blockSize>>>(d_flux,d_u,i_cvars[0],nnel*toteq[0],nnel, toteq[0], lxyz,iwm[0],iph[0],d_iface_flux,nxz2ldim,ivarcoef);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	//        if (code1 != cudaSuccess){
	printf("CUDA: End fluxes_full_field_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));

#endif

}

__global__ void inviscidFlux_gpu_kernel1(double *jaco_c,double *area,double *wghtc,int ntot){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){
		jaco_c[id]=  area[id]/wghtc[id];
	}

}

__global__ void inviscidFlux_gpu_kernel2(double *jaco_f,double *wghtf,int ntotd){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntotd){
		jaco_f[id]=jaco_f[id]*wghtf[id];

	}

}


__global__ void inviscidFlux_gpu_kernel3(double *unx,double *uny,double *unz,double *nx,double *ny,double *nz,double *rl,double *ul,double *vl,double *wl,double *pl,double *tl,double *al,double *cpl,double *rr,double *ur,double *vr,double *wr,double *pr,double *tr,double *ar,double *cpr,double *phl,double *jaco_f,double *fatface,double *area,int iwm,int iwp,int irho,int iux,int iuy,int iuz,int ipr,int ithm,int isnd,int icpf,int iph,int lxz2ldimlelt,int ntot){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){
		nx[id]=unx[id];
		ny[id]=uny[id];
		nz[id]=unz[id];
		rl[id]=fatface[iwm-1+(irho-1)*lxz2ldimlelt+id]; // send the calculate array index for the optimizations later. adeesha.
		ul[id]=fatface[iwm-1+(iux-1)*lxz2ldimlelt+id];
		vl[id]=fatface[iwm-1+(iuy-1)*lxz2ldimlelt+id];
		wl[id]=fatface[iwm-1+(iuz-1)*lxz2ldimlelt+id];
		pl[id]=fatface[iwm-1+(ipr-1)*lxz2ldimlelt+id];
		tl[id]=fatface[iwm-1+(ithm-1)*lxz2ldimlelt+id];
		al[id]=fatface[iwm-1+(isnd-1)*lxz2ldimlelt+id];
		cpl[id]=fatface[iwm-1+(icpf-1)*lxz2ldimlelt+id];

		rr[id]=fatface[iwp-1+(irho-1)*lxz2ldimlelt+id];
		ur[id]=fatface[iwp-1+(iux-1)*lxz2ldimlelt+id];
		vr[id]=fatface[iwp-1+(iuy-1)*lxz2ldimlelt+id];
		wr[id]=fatface[iwp-1+(iuz-1)*lxz2ldimlelt+id];
		pr[id]=fatface[iwp-1+(ipr-1)*lxz2ldimlelt+id];
		tr[id]=fatface[iwp-1+(ithm-1)*lxz2ldimlelt+id];
		ar[id]=fatface[iwp-1+(isnd-1)*lxz2ldimlelt+id];
		cpr[id]=fatface[iwp-1+(icpf-1)*lxz2ldimlelt+id];

		phl[id]=fatface[iwp-1+(iph-1)*lxz2ldimlelt+id];
		jaco_f[id]=area[id];

	}

}

__global__ void inviscidFlux_gpu_kernel4(double *flx,double *phl,int ntotd){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntotd){
		flx[id]=flx[id]*phl[id];

	}

}


__global__ void Ausm_flux(int neq, int ntotd, double *nx, double *ny, double *nz, double *nm, double *fs, double *rl, double *ul, double *vl, double *wl, double *pl, double *al, double *tl, double *rr, double *ur, double *vr, double *wr, double *pr, double *ar, double *tr, double *flx, double *cpl, double *cpr,double *phl){

	int i = blockIdx.x*blockDim.x+threadIdx.x;
	//ntotd = nel * nfaces * nxzd
	if(i<ntotd){
		cpl[i]=cpl[i]/rl[i];//invcol2
		cpr[i]=cpr[i]/rl[i];//invcol2

		fs[i] = 0;// it is 0 in cmtbone but can be changed
		double af,mf,mfa,mfm,mfp,ml,mla,mlp,mr,mra,mrm,pf,ql,qr,wtl,wtr,Hl,Hr;
		Hl = cpl[i]*tl[i] + 0.5*(ul[i]*ul[i]+vl[i]*vl[i]+wl[i]*wl[i]);
		Hr = cpr[i]*tr[i] + 0.5*(ur[i]*ur[i]+vr[i]*vr[i]+wr[i]*wr[i]);

		ql = ul[i]*nx[i] + vl[i]*ny[i] + wl[i]*nz[i] - fs[i];
		qr = ur[i]*nx[i] + vr[i]*ny[i] + wr[i]*nz[i] - fs[i];

		af = 0.5*(al[i] + ar[i]);
		ml = ql/af;
		mla = abs(ml);

		mr = qr/af;
		mra = abs(mr);

		if(mla <= 1.0){
			mlp = 0.25*pow((ml+1.0),2) + 0.125*pow((ml*ml-1.0),2);
			wtl = 0.25*pow(ml+1.0,2)*(2.0-ml) + 0.1875*ml*pow(ml*ml-1.0,2);
		}
		else{
			mlp = 0.5*(ml+mla);
			wtl = 0.5*(1.0+ml/mla);
		}
		if(mra <= 1.0){
			mrm = -0.25*pow((mr-1.0),2) - 0.125*pow((mr*mr-1.0),2);
			wtr = 0.25*pow(mr-1.0,2)*(2.0+mr) - 0.1875*mr*pow(mr*mr-1.0,2);
		}
		else{
			mrm = 0.5*(mr-mra);
			wtr = 0.5*(1.0-mr/mra);
		}

		mf = mlp + mrm;
		mfa = abs(mf);
		mfp = 0.5*(mf+mfa);
		mfm = 0.5*(mf-mfa);

		pf = wtl*pl[i] + wtr*pr[i];

		//compute fluxes
		flx[i] = ((af*(mfp*rl[i] + mfm*rr[i])) * nm[i])*phl[i];
		flx[1*ntotd+i] = ((af*(mfp*rl[i]*ul[i] + mfm*rr[i]*ur[i])+pf*nx[i]) * nm[i])*phl[i];
		flx[2*ntotd+i] = ((af*(mfp*rl[i]*vl[i] + mfm*rr[i]*vr[i])+pf*ny[i]) * nm[i])*phl[i];
		flx[3*ntotd+i] = ((af*(mfp*rl[i]*wl[i] + mfm*rr[i]*wr[i])+pf*nz[i]) * nm[i])*phl[i];
		flx[4*ntotd+i] = ((af*(mfp*rl[i]*Hl + mfm*rr[i]*Hr)+pf*fs[i]) * nm[i])*phl[i];


	}
}


void map_faced(double *d_jgl, double *d_jgt, double *ju, double *u, double *d_w, int nx1, int nxd, int fdim, int nelt, int nfaces, int idir,int ip){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	printf("CUDA: Start map_faced cuda status begining: %s\n",cudaGetErrorString(code1));

#endif
	// may be creating the handle is time consiming and can pass it from the parent functions. Check this later. Adeesha
	cublasHandle_t handle;
	cublasCreate(&handle);
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;
	if(idir==0){
		if(fdim==2){
			// Do the actual multiplication
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nxd, nx1*nelt*nfaces ,nx1,alpha, d_jgl+ip-1, nxd, u, nx1, beta, d_w,nxd);

#ifdef DEBUGPRINT
        cudaDeviceSynchronize();
        cudaError_t code3 = cudaPeekAtLastError();
        printf("CUDA: Start map_faced cuda status after Dgemm: %s\n",cudaGetErrorString(code3));

#endif


			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nxd, nxd, nx1, alpha,d_w, nxd,nxd*nx1, d_jgt,nx1,0, beta,ju ,nxd,nxd*nx1,nelt*nfaces);
#ifdef DEBUGPRINT
        cudaDeviceSynchronize();
        cudaError_t code4 = cudaPeekAtLastError();
        printf("CUDA: Start map_faced cuda status after Dgemmstride: %s\n",cudaGetErrorString(code4));

#endif


		}
		else{
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nxd,1, nx1*nelt*nfaces,alpha, d_jgl+ip-1,nxd, u,nx1, beta, ju, nxd);
		}

	}
	else{
		if(fdim==2){
			// Do the actual multiplication
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nxd*nelt*nfaces ,nxd,alpha, d_jgt+ip-1, nx1, u, nxd, beta, d_w, nx1);

			cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1, nx1, nxd, alpha,d_w, nx1,nxd*nx1, d_jgl,nxd,0, beta,ju ,nx1,nxd*nx1,nelt*nfaces);
		}
		else{
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nx1,1, nxd*nelt*nfaces,alpha, d_jgt+ip-1, nx1, u, nxd, beta, ju,nx1);
		}

		// Destroy the handle
		cublasDestroy(handle);
#ifdef DEBUGPRINT
        cudaDeviceSynchronize();
        cudaError_t code2 = cudaPeekAtLastError();
        printf("CUDA: Start map_faced cuda status at the end: %s\n",cudaGetErrorString(code2));

#endif

	}	
}

// d_fatface became d_flux

extern "C" void inviscidflux_gpu_wrapper_(int *glbblockSize2,double *jgl,double *jgt,double *d_unx,double *d_uny,double *d_unz,double *d_area,double *d_fatface,double *d_wghtc,double *d_wghtf,int *irho,int *iux,int *iuy,int *iuz,int *ipr,int *ithm,int *isnd,int *icpf,int *iph,int *iwm,int *iwp,int *iflx,int *ldim,int *lxd,int *lzd,int *nelt,int *lelt,int *toteq,int *lx1,int *ly1,int *lz1,int *ip ){

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	printf("CUDA: Start inviscidFlux_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
	printf("CUDA: Start inviscidFlux_gpu_wrapper values  irho = %d, iux= %d,iuy= %d,iuz= %d,ipr= %d,ithm= %d,isnd= %d,iph= %d,icpf= %d,iwm= %d,iwp= %d,nelt= %d,lx1= %d,ly1= %d,lz1= %d,ldim= %d,lelt= %d,toteq= %d,lxd= %d, lzd = %d , ip = %d\n",irho[0],iux[0],iuy[0],iuz[0],ipr[0],ithm[0],isnd[0],iph[0],icpf[0],iwm[0],iwp[0],nelt[0],lx1[0],ly1[0],lz1[0],ldim[0],lelt[0],toteq[0],lxd[0],lzd[0],ip[0]);
#endif
	int fdim = ldim[0]-1;
	int nfaces = 2*ldim[0];
	int lxz=lx1[0]*lz1[0];
	int ntot= nelt[0]*nfaces*lxz;
	int ntotd = nelt[0] * nfaces * lxd[0]*lzd[0];
	int lxd_3= lxd[0]*lxd[0]*lxd[0];
	int nel=nelt[0];
	int lxz2ldim=lxz*nfaces;
	int lxz2ldimlelt=lxz2ldim*nelt[0];

	double *d_jgl;
	double *d_jgt;

	cudaMalloc(&d_jgl, lxd_3*sizeof(double));
	cudaMalloc(&d_jgt, lxd_3*sizeof(double));

	cudaMemcpy(d_jgl, jgl,lxd_3*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_jgt, jgt,lxd_3*sizeof(double), cudaMemcpyHostToDevice);

	double *d_w;
	double *d_nx;
	double *d_ny;
	double *d_nz;
	double *d_rl;
	double *d_ul;
	double *d_wl;
	double *d_vl;
	double *d_pl;
	double *d_tl;
	double *d_al;
	double *d_cpl;
	double *d_rr;
	double *d_ur;
	double *d_wr;
	double *d_vr;
	double *d_pr;
	double *d_tr;
	double *d_ar;
	double *d_cpr;
	double *d_jaco_c;
	double *d_jaco_f;
	double *d_phl;
	double *d_fs;
	double *d_flx;



	cudaMalloc(&d_w, ntotd*sizeof(double));
	cudaMalloc(&d_nx, ntotd*sizeof(double));
	cudaMalloc(&d_ny, ntotd*sizeof(double));
	cudaMalloc(&d_nz, ntotd*sizeof(double));

	cudaMalloc(&d_rl, ntotd*sizeof(double));
	cudaMalloc(&d_ul, ntotd*sizeof(double));
	cudaMalloc(&d_wl, ntotd*sizeof(double));
	cudaMalloc(&d_vl, ntotd*sizeof(double));
	cudaMalloc(&d_pl, ntotd*sizeof(double));
	cudaMalloc(&d_tl, ntotd*sizeof(double));
	cudaMalloc(&d_al, ntotd*sizeof(double));
	cudaMalloc(&d_cpl, ntotd*sizeof(double));

	cudaMalloc(&d_rr, ntotd*sizeof(double));
	cudaMalloc(&d_ur, ntotd*sizeof(double));
	cudaMalloc(&d_vr, ntotd*sizeof(double));
	cudaMalloc(&d_wr, ntotd*sizeof(double));
	cudaMalloc(&d_pr, ntotd*sizeof(double));
	cudaMalloc(&d_tr, ntotd*sizeof(double));
	cudaMalloc(&d_ar, ntotd*sizeof(double));
	cudaMalloc(&d_cpr, ntotd*sizeof(double));
	cudaMalloc(&d_jaco_c, ntot*sizeof(double));
	cudaMalloc(&d_jaco_f, ntotd*sizeof(double));
	cudaMalloc(&d_phl, ntotd*sizeof(double));
	cudaMalloc(&d_fs, ntotd*sizeof(double));
	cudaMalloc(&d_flx, ntotd*toteq[0]*sizeof(double));

	int totpts =lxz2ldimlelt;

	int blockSize = glbblockSize2[0], gridSize;
	if(lxd[0]>lx1[0]){
		map_faced(d_jgl, d_jgt, d_nx, d_unx, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_ny, d_uny, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_nz, d_unz, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_rl, d_fatface+iwm[0]-1+(irho[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_ul, d_fatface+iwm[0]-1+(iux[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_vl, d_fatface+iwm[0]-1+(iuy[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_wl, d_fatface+iwm[0]-1+(iuz[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_pl, d_fatface+iwm[0]-1+(ipr[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_tl, d_fatface+iwm[0]-1+(ithm[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_al, d_fatface+iwm[0]-1+(isnd[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_cpl, d_fatface+iwm[0]-1+(icpf[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);

		map_faced(d_jgl, d_jgt, d_rr, d_fatface+iwp[0]-1+(irho[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_ur, d_fatface+iwp[0]-1+(iux[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_vr, d_fatface+iwp[0]-1+(iuy[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_wr, d_fatface+iwp[0]-1+(iuz[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_pr, d_fatface+iwp[0]-1+(ipr[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_tr, d_fatface+iwp[0]-1+(ithm[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_ar, d_fatface+iwp[0]-1+(isnd[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_cpr, d_fatface+iwp[0]-1+(icpf[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);
		map_faced(d_jgl, d_jgt, d_phl, d_fatface+iwp[0]-1+(iph[0]-1)*totpts, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);

#ifdef DEBUGPRINT
		cudaDeviceSynchronize();
		code1 = cudaPeekAtLastError();
		printf("CUDA:inviscidFlux_gpu_wrapper after map_faced cuda status: %s\n",cudaGetErrorString(code1));

#endif

		gridSize = (int)ceil((float)ntot/blockSize);
		inviscidFlux_gpu_kernel1<<<gridSize, blockSize>>>(d_jaco_c,d_area,d_wghtc,ntot);

		map_faced(d_jgl, d_jgt, d_jaco_f, d_jaco_c, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 0,ip[0]);

		gridSize = (int)ceil((float)ntotd/blockSize);
		inviscidFlux_gpu_kernel2<<<gridSize, blockSize>>>(d_jaco_f,d_wghtf,ntotd);

#ifdef DEBUGPRINT
		cudaDeviceSynchronize();
		code1 = cudaPeekAtLastError();
		printf("CUDA: inviscidFlux_gpu_wrapper after kernen2 cuda status: %s\n",cudaGetErrorString(code1));

#endif

	}
	else{
		gridSize = (int)ceil((float)ntot/blockSize);
		inviscidFlux_gpu_kernel3<<<gridSize, blockSize>>>(d_unx,d_uny,d_unz,d_nx,d_ny,d_nz,d_rl,d_ul,d_vl,d_wl,d_pl,d_tl,d_al,d_cpl,d_rr,d_ur,d_vr,d_wr,d_pr,d_tr,d_ar,d_cpr,d_phl,d_jaco_f,d_fatface,d_area,iwm[0],iwp[0],irho[0],iux[0],iuy[0],iuz[0],ipr[0],ithm[0],isnd[0],icpf[0],iph[0],lxz2ldimlelt,ntot );


	}

	gridSize = (int)ceil((float)ntotd/blockSize);
	Ausm_flux<<<gridSize, blockSize>>>(toteq[0],ntotd, d_nx, d_ny, d_nz, d_jaco_f, d_fs, d_rl, d_ul, d_vl, d_wl, d_pl, d_al, d_tl, d_rr, d_ur, d_vr, d_wr, d_pr, d_ar, d_tr, d_flx, d_cpl, d_cpr,d_phl);


	if(lxd[0]>lx1[0]){
		for(int j=0; j<toteq[0];j++){
			gridSize = (int)ceil((float)ntotd/blockSize);
			map_faced(d_jgl, d_jgt,d_fatface+iflx[0]-1+j*totpts, d_flx+j*ntotd, d_w, lx1[0], lxd[0], fdim, nel, nfaces, 1,ip[0]);
		}
#ifdef DEBUGPRINT
		cudaDeviceSynchronize();
		code1 = cudaPeekAtLastError();
		printf("CUDA: inviscidFlux_gpu_wrapper after forloop cuda status: %s\n",cudaGetErrorString(code1));
#endif

	}
	else{
		gpu_double_copy_gpu_wrapper(glbblockSize2[0],d_fatface+iflx[0]-1,0,d_flx,0,ntotd*toteq[0]);	

	}

	cudaFree(d_jgl);	
	cudaFree(d_jgt);	
	cudaFree(d_nx);	
	cudaFree(d_ny);	
	cudaFree(d_nz);	
	cudaFree(d_rl);	
	cudaFree(d_ul);	
	cudaFree(d_wl);	
	cudaFree(d_vl);	
	cudaFree(d_pl);	
	cudaFree(d_tl);	
	cudaFree(d_al);	
	cudaFree(d_cpl);	
	cudaFree(d_rr);
	cudaFree(d_ur);
	cudaFree(d_wr);
	cudaFree(d_vr);
	cudaFree(d_pr);
	cudaFree(d_tr);
	cudaFree(d_ar);
	cudaFree(d_cpr);	
	cudaFree(d_jaco_c);	
	cudaFree(d_jaco_f);	
	cudaFree(d_phl);	
	cudaFree(d_fs);	
	cudaFree(d_flx);	

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	code1 = cudaPeekAtLastError();
	printf("CUDA: End inviscidFlux_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));


#endif


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

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	printf("CUDA: Start surface_integral_full_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start surface_integral_full_gpu_wrapper values eq = %d,ieq= %d,nelt= %d,lelt=%d,toteq= %d,lx1=%d,ly1= %d,lz1= %d,ldim= %d\n",eq[0],ieq[0],nelt[0],lelt[0],toteq[0],lx1[0],ly1[0],lz1[0],ldim[0]);

#endif
	int nxz2ldim = lx1[0]*lz1[0]*2*ldim[0];
	int lxyz = lx1[0]*ly1[0]*lz1[0];
	int lxyzlelt = lelt[0]*lxyz;
	int ntot = nelt[0]*nxz2ldim;

	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);

	surface_integral_full_gpu_kernel<<<gridSize, blockSize>>>(d_iface_flux,d_res1,d_flux+ieq[0]-1,ntot,lxyz,lxyzlelt,nxz2ldim,(eq[0]-1));

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End surface_integral_full_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));

#endif
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

#ifdef DEBUGPRINT
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start igu_cmt_gpu_wrapper1 cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start igu_cmt_gpu_wrapper1 values toteq=%d ,iwp=%d ,lx1=%d ,ly1=%d ,lz1=%d ,ldim=%d ,nelt=%d ,lelt=%d ,\n",toteq[0],iwp[0],lx1[0],ly1[0],lz1[0],ldim[0],nelt[0],lelt[0]);

#endif
	int lxz2ldim = lx1[0]*lz1[0]*2*ldim[0];
	int ntot = nelt[0]*lxz2ldim;
	int lxz2ldimlelt= lxz2ldim*nelt[0];
	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);

	igu_cmt_gpu_wrapper1_gpu_kernel<<<gridSize, blockSize>>>(d_flux,d_graduf,ntot,lxz2ldimlelt,toteq[0],iwp[0]);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	printf("CUDA: End igu_cmt_gpu_wrapper1 cuda status: %s\n",cudaGetErrorString(code2));

#endif

}

extern "C" void  igu_cmt_gpu_wrapper2_(int *glbblockSize2,double *d_flux,double *d_graduf,char *d_cbc,double *d_unx,double *d_uny,double *d_unz,int *toteq,int *iwp,int *iwm,int *ldim,int *irho,int *icvf,int *ilamf,int *imuf,int *iux,int *iuy,int *iuz,int *iu5,int *if3d,int *lx1,int *ly1,int *lz1,int *nelt,int *lelt){

#ifdef DEBUGPRINT
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start igu_cmt_gpu_wrapper2 cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start igu_cmt_gpu_wrapper2 values toteq =%d ,iwp =%d,iwm =%d,ldim =%d,irho =%d,icvf =%d,ilamf =%d,imuf =%d,iux =%d,iuy =%d,iuz =%d,iu5 =%d,if3d =%d,lx1 =%d,ly1 =%d,lz1 =%d,nelt =%d,lelt =%d \n", toteq[0],iwp[0],iwm[0],ldim[0],irho[0],icvf[0],ilamf[0],imuf[0],iux[0],iuy[0],iuz[0],iu5[0],if3d[0],lx1[0],ly1[0],lz1[0],nelt[0],lelt[0] );

#endif
	int lxz=lx1[0]*lz1[0];
	int a2ldim= 2*ldim[0];
	int lxz2ldim = lxz*2*ldim[0];
	int ntot = nelt[0]*lxz2ldim;
	int lxz2ldimlelt =lxz2ldim*nelt[0];
	int  lxz2ldimlelttoteq= lxz2ldimlelt*toteq[0];

	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);

	igu_cmt_gpu_wrapper2_gpu_kernel<<<gridSize, blockSize>>>(d_flux,d_graduf,d_cbc,d_unx,d_uny,d_unz,ntot,lxz2ldim,lxz2ldimlelt, lxz2ldimlelttoteq,toteq[0],iwp[0],iwm[0],ldim[0],irho[0],icvf[0],ilamf[0],imuf[0],iux[0],iuy[0],iuz[0],iu5[0],if3d[0],lxz,a2ldim);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	printf("CUDA: End igu_cmt_gpu_wrapper2 cuda status: %s\n",cudaGetErrorString(code2));

#endif
}












