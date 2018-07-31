#include <stdio.h>
#define DEBUGPRINT 0

__global__ void compute_entropy_gpu_kernel(double *tlag, double *pr, double *vtrans,int ntot, int irho, double ntol , double rgam, double gmaref,int ltot  ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){

		double rho= fmax(vtrans[ltot*(irho-1)+id],ntol);
		tlag[id]=rgam*rho*log(pr[id]/(pow(rho,gmaref) ));

	}

}


extern "C" void compute_entropy_gpu_wrapper_(int *glbblockSize1,double *d_tlag, double *d_pr, double *d_vtrans,int *ntot, int *irho, double *ntol , double *rgam, double *gmaref, int *ltot){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	// if (code1 != cudaSuccess){
	printf("CUDA: Start compute_entropy_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
	printf("CUDA: Start compute_entropy_gpu_wrapper values ntot = %d, irho = %d, ntol = %lf, rgam = %lf, gmaref = %lf \n",ntot[0],irho[0],ntol[0],rgam[0],gmaref[0] );

#endif
	//}

	int blockSize = glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)ntot[0]/blockSize);
	compute_entropy_gpu_kernel<<<gridSize, blockSize>>>(d_tlag,d_pr,d_vtrans,ntot[0],irho[0],ntol[0],rgam[0],gmaref[0],ltot[0]);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	//if (code2 != cudaSuccess){
	printf("CUDA: End compute_engropy_wrapper cuda status: %s\n",cudaGetErrorString(code1));
#endif
	//}


}


__global__ void entropy_residual_flux_gpu_kernel(double *tlag, double *res2,int ntot, double rdt, int stage, int lorder, int ltot, double *totalh, int lxyzd, double *vx, double *vy, double *vz, int if3d ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){
		if(stage==1){
			res2[id]=tlag[id]-tlag[ltot*lorder+id]   ;
		}
		else{
			res2[id]=tlag[id]-tlag[ltot+id]   ;
		}
		res2[id] = res2[id]*rdt;

		// evaluate_entropy_flux(e)
		totalh[id]= vx[id]*tlag[id];
		totalh[lxyzd+id] = vy[id]*tlag[id];
		if(if3d){totalh[lxyzd*2+id] = vz[id]*tlag[id];}

		//flux_div_mini(e)
	}


}

__global__ void flux_div_mini_gpu_kernel(double *tlag, double *res2,int ntot, double rdt, int stage, int lorder, int ltot, double *totalh, int lxyzd, double *ur1,double *us1, double *ut1, double *ur2, double *us2, double *ut2, double *ur3, double *us3, double *ut3,double *ud, int ldd, double *jacmi, double *rxm1, double *sxm1, double *txm1, double *rym1, double *sym1, double *tym1,double *rzm1, double *szm1, double *tzm1, int if3d ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int i= id % ldd;
	if(id<ntot){
		//something is wrong because ur us ut has only [i]. I think it should be [id] because I added *lelt later. Check again. adeesha
		if(if3d){
			ud[id] =  jacmi[id] *( rxm1[id]*ur1[i]+ sxm1[id]*us1[i]+txm1[id]*ut1[i]);
			ud[id] =  ud[id]+ jacmi[id] *( rym1[id]*ur2[i]+ sym1[id]*us2[i]+txm1[id]*ut2[i]);
			ud[id] =  ud[id] + jacmi[id] *( rzm1[id]*ur3[i]+ szm1[id]*us3[i]+tzm1[id]*ut3[i]);

		}
		else{
			ud[id] =  jacmi[id] *( rxm1[id]*ur1[i]+ sxm1[id]*us1[i]);
			ud[id] =  ud[id]+ jacmi[id] *( rym1[id]*ur2[i]+ sym1[id]*us2[i]);
		}
		//add 2 
		res2[id] = res2[id] + ud[id];


	}

}
//mxm multiplication
__global__ void mxm1(double *a, int n1, double *b, int n2, double *c, int n3, int nelt, int aSize, int bSize, int cSize, int extraEq){

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




extern "C" void entropy_residual_gpu_wrapper_(int *glbblockSize1,double *d_tlag, double *d_res2,int *ntot, double *rdt, int *stage, int *lorder,int *ltot, int *lxd, int *lyd, int *lzd, double *d_vx, double *d_vy, double *d_vz, int *lx1, int *ly1, int *lz1, double *d_jacmi, double *d_rxm1, double *d_sxm1, double *d_txm1, double *d_rym1, double *d_sym1, double *d_tym1,double *d_rzm1, double *d_szm1, double *d_tzm1,int *if3d,int *nelt, double *d_dxm1, double *d_dxtm1 ){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	//  if (code1 != cudaSuccess){
	printf("CUDA: Start entropy_residual_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
	printf("CUDA: Start entropy_residual_gpu_wrapper values rdt = %lf, stage = %d, lorder= %d,ltot = %d,lxd = %d, lyd = %d, lzd = %d, lx1 = %d,ly1 = %d,lz1 = %d,if3d = %d,nelt = %d \n",rdt[0], stage[0],lorder[0],ltot[0],lxd[0], lyd[0],lzd[0],lx1[0],ly1[0],lz1[0],if3d[0],nelt[0]);
#endif
	//}


	double *d_totalh_temp;  // Anyway d_totalh seems not needed. check with Dr.Tania. adeesha
	double *d_ur1;
	double *d_us1;
	double *d_ut1;
	double *d_ur2;
	double *d_us2;
	double *d_ut2;
	double *d_ur3;
	double *d_us3;
	double *d_ut3;
	double *d_ud;
	int lxyzd = lxd[0]*lyd[0]*lzd[0];
	int ldd = lx1[0]*ly1[0]*lz1[0];
	cudaMalloc((void**)&d_totalh_temp,3*lxyzd *nelt[0]*  sizeof(double));
	cudaMalloc((void**)&d_ur1,ldd *  nelt[0]*sizeof(double)); //nelt[0] added later.  need to double check.
	cudaMalloc((void**)&d_us1,ldd *  nelt[0]*sizeof(double));
	cudaMalloc((void**)&d_ut1,ldd *  nelt[0]*sizeof(double));
	cudaMalloc((void**)&d_ur2,ldd *  nelt[0]*sizeof(double));
	cudaMalloc((void**)&d_us2,ldd *  nelt[0]*sizeof(double));
	cudaMalloc((void**)&d_ut2,ldd *  nelt[0]*sizeof(double));
	cudaMalloc((void**)&d_ur3,ldd *  nelt[0]*sizeof(double));
	cudaMalloc((void**)&d_us3,ldd *  nelt[0]*sizeof(double));
	cudaMalloc((void**)&d_ut3,ldd *  nelt[0]*sizeof(double));
	cudaMalloc((void**)&d_ud,nelt[0]*ldd *  sizeof(double));
	cudaMemset(d_totalh_temp, 0.0, 3*lxyzd*nelt[0]*sizeof(double));
	cudaMemset(d_ur1, 0.0, ldd*nelt[0]*sizeof(double));
	cudaMemset(d_us1, 0.0, ldd*nelt[0]*sizeof(double));
	cudaMemset(d_ut1, 0.0, ldd*nelt[0]*sizeof(double));
	cudaMemset(d_ur2, 0.0, ldd*nelt[0]*sizeof(double));
	cudaMemset(d_us2, 0.0, ldd*nelt[0]*sizeof(double));
	cudaMemset(d_ut2, 0.0, ldd*nelt[0]*sizeof(double));
	cudaMemset(d_ur3, 0.0, ldd*nelt[0]*sizeof(double));
	cudaMemset(d_us3, 0.0, ldd*nelt[0]*sizeof(double));
	cudaMemset(d_ut3, 0.0, ldd*nelt[0]*sizeof(double));
	cudaMemset(d_ud, 0.0, ldd*nelt[0]*sizeof(double));
	int blockSize = glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)ntot[0]/blockSize);


	entropy_residual_flux_gpu_kernel<<<gridSize, blockSize>>>(d_tlag,d_res2,ntot[0],rdt[0],stage[0],lorder[0], ltot[0], d_totalh_temp, lxyzd, d_vx, d_vy, d_vz, if3d[0]);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	code1 = cudaPeekAtLastError();
	printf("CUDA: entropy_residual_gpu_wrapper after kernel 1cuda status: %s\n",cudaGetErrorString(code1));


#endif
	int mdm1 = lx1[0]-1;
	// Following is the local_grad3 function

	int nx_2 = lx1[0]*lx1[0];
	int nx_3 = nx_2*lx1[0];
	int nxd_2 = nx_2;
	int nxd_3 = nx_3;
	//ur(nx,nx*nx) = D(nx,nx) * u(nx,nx*nx) fortran
	//ur(nx*nx,nx) = u(nx*nx,nx) * D(nx,nx) C
	if(if3d[0]){

		blockSize=glbblockSize1[0], gridSize;
		// for ur1 us1 and ut1
		gridSize = (int)ceil((float)nelt[0]*nx_3/blockSize);
		mxm1<<<gridSize, blockSize>>>(d_totalh_temp,nx_2, d_dxm1, lx1[0], d_ur1, lx1[0], nelt[0], nx_3, 0, nxd_3, 0);//ur,us, ut should be indexed by nxd
#ifdef DEBUGPRINT
		cudaDeviceSynchronize();
		code1 = cudaPeekAtLastError();
		printf("CUDA: entropy_residual_gpu_wrapper after 1st mxm 1cuda status: %s\n",cudaGetErrorString(code1));

#endif
		for(int k = 0; k<lx1[0]; k++){
			//usk(nx,nx) = uk(nx,nx) * dt(nx,nx) fortran
			//usk(nx,nx) = dt(nx,nx) * uk(nx,nx) C
			gridSize = (int)ceil((float)nelt[0]*nx_2/blockSize);
			mxm1<<<gridSize, blockSize>>>(d_dxtm1,lx1[0], d_totalh_temp+k*nx_2, lx1[0],d_us1+k*nx_2, lx1[0], nelt[0], 0, nx_3, nxd_3, 0);
		}
#ifdef DEBUGPRINT
		cudaDeviceSynchronize();
		code1 = cudaPeekAtLastError();
		printf("CUDA: entropy_residual_gpu_wrapper after for loop mxm 1cuda status: %s\n",cudaGetErrorString(code1));

#endif
		//ut(nx_2,nx) = u(nx_2,nx) * dt(nx,nx) fortran
		//ut(nx,nx_2) = dt(nx,nx) * u(nx,nx_2) C
		gridSize = (int)ceil((float)nelt[0]*nx_3/blockSize);
		mxm1<<<gridSize, blockSize>>>(d_dxtm1,lx1[0], d_totalh_temp, lx1[0], d_ut1, nx_2, nelt[0], 0, nx_3, nxd_3, 0);
#ifdef DEBUGPRINT
		cudaDeviceSynchronize();
		code1 = cudaPeekAtLastError();
		printf("CUDA: entropy_residual_gpu_wrapper after 3rd mxm 1cuda status: %s\n",cudaGetErrorString(code1));

#endif

		// for ur2 us2 and ut2
		gridSize = (int)ceil((float)nelt[0]*nx_3/blockSize);
		mxm1<<<gridSize, blockSize>>>(d_totalh_temp+lxyzd*nelt[0],nx_2, d_dxm1, lx1[0], d_ur2, lx1[0], nelt[0], nx_3, 0, nxd_3, 0);//ur,us, ut should be indexed by nxd
		for(int k = 0; k<lx1[0]; k++){
			//usk(nx,nx) = uk(nx,nx) * dt(nx,nx) fortran
			//usk(nx,nx) = dt(nx,nx) * uk(nx,nx) C
			gridSize = (int)ceil((float)nelt[0]*nx_2/blockSize);
			mxm1<<<gridSize, blockSize>>>(d_dxtm1,lx1[0], d_totalh_temp+lxyzd*nelt[0]+k*nx_2, lx1[0],d_us2+k*nx_2, lx1[0], nelt[0], 0, nx_3, nxd_3, 0);
		}
		//ut(nx_2,nx) = u(nx_2,nx) * dt(nx,nx) fortran
		//ut(nx,nx_2) = dt(nx,nx) * u(nx,nx_2) C
		gridSize = (int)ceil((float)nelt[0]*nx_3/blockSize);
		mxm1<<<gridSize, blockSize>>>(d_dxtm1,lx1[0], d_totalh_temp+lxyzd*nelt[0], lx1[0], d_ut2, nx_2, nelt[0], 0, nx_3, nxd_3, 0);


		// for ur3 us3 and ut3
		gridSize = (int)ceil((float)nelt[0]*nx_3/blockSize);
		mxm1<<<gridSize, blockSize>>>(d_totalh_temp+2*lxyzd*nelt[0],nx_2, d_dxm1, lx1[0], d_ur3, lx1[0], nelt[0], nx_3, 0, nxd_3, 0);//ur,us, ut should be indexed by nxd
		for(int k = 0; k<lx1[0]; k++){
			//usk(nx,nx) = uk(nx,nx) * dt(nx,nx) fortran
			//usk(nx,nx) = dt(nx,nx) * uk(nx,nx) C
			gridSize = (int)ceil((float)nelt[0]*nx_2/blockSize);
			mxm1<<<gridSize, blockSize>>>(d_dxtm1,lx1[0], d_totalh_temp+2*lxyzd*nelt[0]+k*nx_2, lx1[0],d_us3+k*nx_2, lx1[0], nelt[0], 0, nx_3, nxd_3, 0);
		}
		//ut(nx_2,nx) = u(nx_2,nx) * dt(nx,nx) fortran
		//ut(nx,nx_2) = dt(nx,nx) * u(nx,nx_2) C
		gridSize = (int)ceil((float)nelt[0]*nx_3/blockSize);
		mxm1<<<gridSize, blockSize>>>(d_dxtm1,lx1[0], d_totalh_temp+2*lxyzd*nelt[0], lx1[0], d_ut3, nx_2, nelt[0], 0, nx_3, nxd_3, 0);


	}else{
		// for ur1 us1 and ut1
		gridSize = (int)ceil((float)nelt[0]*nx_2/blockSize);
		mxm1<<<gridSize, blockSize>>>(d_totalh_temp,lx1[0], d_dxm1, lx1[0], d_ur1, lx1[0], nelt[0], nx_2, 0, nxd_2, 0);//ur,us, ut should be indexed by nxd

		gridSize = (int)ceil((float)nelt[0]*nx_2/blockSize);
		mxm1<<<gridSize, blockSize>>>(d_dxtm1,lx1[0], d_totalh_temp, lx1[0],d_us1, lx1[0], nelt[0], 0, nx_2, nxd_2, 0);

		// for ur2 us2 and ut1
		gridSize = (int)ceil((float)nelt[0]*nx_2/blockSize);
		mxm1<<<gridSize, blockSize>>>(d_totalh_temp+lxyzd*nelt[0],lx1[0], d_dxm1, lx1[0], d_ur2, lx1[0], nelt[0], nx_2, 0, nxd_2, 0);//ur,us, ut should be indexed by nxd

		gridSize = (int)ceil((float)nelt[0]*nx_2/blockSize);
		mxm1<<<gridSize, blockSize>>>(d_dxtm1,lx1[0], d_totalh_temp+lxyzd*nelt[0], lx1[0],d_us2, lx1[0], nelt[0], 0, nx_2, nxd_2, 0);



	}
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	code1 = cudaPeekAtLastError();
	printf("CUDA: entropy_residual_gpu_wrapper before flux_div_mini_gpu_kernel cuda status: %s\n",cudaGetErrorString(code1));

#endif

	flux_div_mini_gpu_kernel<<<gridSize, blockSize>>>(d_tlag,d_res2,ntot[0],rdt[0],stage[0],lorder[0], ltot[0], d_totalh_temp, lxyzd, d_ur1,d_us1, d_ut1, d_ur2,d_us2, d_ut2, d_ur3, d_us3, d_ut3,d_ud, ldd, d_jacmi, d_rxm1, d_sxm1, d_txm1, d_rym1, d_sym1, d_tym1, d_rzm1, d_szm1, d_tzm1,if3d[0]);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	code1 = cudaPeekAtLastError();
	printf("CUDA: entropy_residual_gpu_wrapper after flux_div_mini_gpu_kernel cuda status: %s\n",cudaGetErrorString(code1));

#endif
	cudaFree(d_totalh_temp);

	cudaFree(d_ur1);
	cudaFree(d_ur2);
	cudaFree(d_ur3);
	cudaFree(d_us1);

	cudaFree(d_us2);
	cudaFree(d_us3);
	cudaFree(d_ut1);
	cudaFree(d_ut2);

	cudaFree(d_ut3);

	cudaFree(d_ud);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	// if (code2 != cudaSuccess){
	printf("CUDA: End entropy residual_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
	//}


}

__global__ void wavevisc_gpu_kernel(double *t,double *csound, double *vx, double *vy, double *vz, int ntot, double *wavespeed,int lxyz, int lx1, int ly1, int lz1, double *vtranstmp, double c_max,int ltot, double *meshh ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){
		wavespeed[id]= csound [id] +sqrtf(vx[id]*vx[id]+vy[id]*vy[id]+vz[id]*vz[id]  )	;
		// find max of wavespeed using reduction
		__syncthreads();
		unsigned int i = lxyz/2;
		int e= id/(lx1*ly1*lz1);
		int startofcurrentelement = id-e;
		while(i != 0){
			if(id-startofcurrentelement < i){
				wavespeed[id] = fmaxf(wavespeed[id], wavespeed[id + i]);
			}

			__syncthreads();
			i /= 2;
		}

		double maxeig = wavespeed[id-e];
		// find max of vtrans using reduction. But never used? check with Dr.Tania
		//i = lxyz/2;
		//int e= id/(lx1*ly1*lz1);
		//int startofcurrentelement = id-e;
		//while(i != 0){
		//	if(id-startofcurrentelement < i){
		//		vtranstmp[id] = fmaxf(vtranstmp[id], vtranstmp[id + i]);
		//	}
		//	__syncthreads();
		//	i /= 2;
		//}

		//int rhomax = vtranstmp[id-e];
		t[2*ltot+id] = c_max*maxeig*meshh[e];
//		if(id<10){
//			printf("$$$ print from cuda maxeig = %lf t[2*ltot+id]= %lf meshh[e]=%lf \n",maxeig,t[2*ltot+id],meshh[e]);
//		}
	}
}


extern "C" void wavevisc_gpu_wrapper_(int *glbblockSize1,double *d_t, double *d_csound,double *d_vx, double *d_vy, double *d_vz, int *nelt, int *lelt, int *lx1, int *ly1, int *lz1, double *d_vtrans, double *c_max, double *d_meshh,  int *irho  ){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	// if (code1 != cudaSuccess){
	printf("CUDA: Start wavevisc_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
	printf("CUDA: Start compute_entropy_gpu_wrapper values nelt= %d,lelt= %d,lx1= %d,ly1= %d, lz1= %d,c_max= %lf,irho= %d \n",nelt[0],lelt[0],lx1[0],ly1[0],lz1[0],c_max[0],irho[0]);       
#endif
	// }


	int ntot = nelt[0]*lx1[0]*ly1[0]*lz1[0];
	int lxyz = lx1[0]*ly1[0]*lz1[0];
	int ltot = lelt[0]*lxyz;
	double *d_wavespeed; 
	cudaMalloc((void**)&d_wavespeed,nelt[0]*lxyz* sizeof(double));
	// cudaMemset(d_totalh_temp, 0.0, 3*lxyzd*nelt*sizeof(double));
	double *d_vtranstemp; 
	cudaMalloc((void**)&d_vtranstemp,nelt[0]*lxyz* sizeof(double));
	cudaMemcpy(d_vtranstemp, &d_vtrans[(irho[0]-1)*lelt[0]*lxyz], nelt[0]*lxyz* sizeof(double), cudaMemcpyDeviceToDevice);
	int blockSize = glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);
	wavevisc_gpu_kernel<<<gridSize, blockSize>>>(d_t,d_csound, d_vx, d_vy, d_vz,ntot,d_wavespeed, lxyz,lx1[0],ly1[0],lz1[0],d_vtranstemp,c_max[0], ltot, d_meshh);

	cudaFree(d_wavespeed);
	cudaFree(d_vtranstemp);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	// if (code2 != cudaSuccess){
	printf("CUDA: End Wavevisc_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
	// }

}

__global__ void max_to_trilin_gpu_kernel(double *t,int ntot,int lxyz, int lx1, int ly1, int lz1,int ltot, int lxy, double *xm1, double *ym1, double *zm1, int if3d ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){

		int e= id/(lxyz);
		double p000 = t[2*ltot+e*lxyz];
		double p100 = t[2*ltot+e*lxyz+(lx1-1)];
		double p010 = t[2*ltot+e*lxyz+(ly1-1)*lx1];
		double p110 = t[2*ltot+e*lxyz+(ly1-1)*lx1+(lx1-1)];
		double p001 = t[2*ltot+e*lxyz+(lz1-1)*lxy];
		double p101 = t[2*ltot+e*lxyz+(lz1-1)*lxy+(lx1-1)];
		double p011 = t[2*ltot+e*lxyz+(lz1-1)*lxy+(ly1-1)*lx1];
		double p111 = t[2*ltot+e*lxyz+(lz1-1)*lxy+(ly1-1)*lx1+(lx1-1)];

		double  c1=p100-p000;
		double c2=p010-p000;
		double  c3=p001-p000;
		double c4=p110-p010-p100+p000;
		double c5=p011-p001-p010+p000;
		double c6=p101-p001-p100+p000;
		double  c7=p111-p011-p101-p110+p100+p001+p010-p000;
		double rdx=1.0/(xm1[e*lxyz+(lx1-1)]-xm1[e*lxyz]); // cubes only!!!
		double rdy=1.0/(ym1[e*lxyz+(ly1-1)*lx1]-ym1[e*lxyz]);
		double rdz=0.0;

		if(if3d){ rdz=1.0/(zm1[e*lxyz+(lz1-1)*lxy]-zm1[e*lxyz]); }

		int firstlx = id%lxyz;

		double deltax=rdx*(xm1[id]-xm1[e*lxyz]) ;//! cubes only!!!
		double deltay=rdy*(ym1[id]-ym1[e*lxyz]);
		double deltaz=0.0;
		if (if3d){ deltaz=rdz*(zm1[id]-zm1[e*lxyz]);}
		t[2*ltot+id] =p000+c1*deltax+c2*deltay+c3*deltaz+ c4*deltax*deltay+c5*deltay*deltaz+ c6*deltaz*deltax+c7*deltay*deltaz*deltax;
	}

}


extern "C" void max_to_trilin_gpu_wrapper_(int *glbblockSize1,double *d_t, int *nelt, int *lelt, int *lx1, int *ly1, int *lz1, double *d_xm1, double *d_ym1, double *d_zm1, int *if3d ){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	// if (code1 != cudaSuccess){
	printf("CUDA: Start max_to_trilin_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
	printf("CUDA: Start compute_entropy_gpu_wrapper values nelt=%d,lelt=%d,lx1=%d,ly1=%d,lz1=%d,if3d=%d \n",nelt[0],lelt[0],lx1[0],ly1[0],lz1[0],if3d[0]);       
#endif
	//}

	int ntot = nelt[0]*lx1[0]*ly1[0]*lz1[0];
	int lxyz = lx1[0]*ly1[0]*lz1[0];
	int ltot = lelt[0]*lxyz;

	int blockSize = glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);


	max_to_trilin_gpu_kernel<<<gridSize, blockSize>>>(d_t,ntot,lxyz,lx1[0],ly1[0],lz1[0], ltot,lx1[0]*ly1[0], d_xm1, d_ym1,  d_zm1, if3d[0]);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	//if (code2 != cudaSuccess){
	printf("CUDA: End max_to_trilin_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
	//}


}

__global__ void resvisc_gpu_kernel1(double *res2,int ntot,int lxyz, int lx1, int ly1, int lz1,int ltot, int lxy,double *meshh ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){

		int e= id/(lx1*ly1*lz1);
		res2[id] = res2[id]*meshh[e]*meshh[e];
	}

}


extern "C" void resvisc_gpu_wrapper1_(int *glbblockSize1,double *d_res2, int *nelt, int *lelt, int *lx1, int *ly1, int *lz1, double *d_meshh){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	//if (code1 != cudaSuccess){
	printf("CUDA: Start resvisc_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
	printf("CUDA: Start compute_entropy_gpu_wrapper values nelt= %d,lelt= %d,lx1= %d,ly1= %d,lz1 = %d,\n", nelt[0],lelt[0],lx1[0],ly1[0],lz1[0]);	        
#endif
	//}

	int ntot = nelt[0]*lx1[0]*ly1[0]*lz1[0];
	int lxyz = lx1[0]*ly1[0]*lz1[0];
	int ltot = lelt[0]*lxyz;

	int blockSize = glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);


	resvisc_gpu_kernel1<<<gridSize, blockSize>>>(d_res2,ntot,lxyz,lx1[0],ly1[0],lz1[0], ltot,lx1[0]*ly1[0], d_meshh);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	//if (code2 != cudaSuccess){
	printf("CUDA: End resvisc_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
	//}

}

__global__ void resvisc_gpu_kernel2(double *res2,int ntot,int lxyz, int lx1, int ly1, int lz1,int ltot, int lxy,double c_sub_e, double maxdiff ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){

		int e= id/(lx1*ly1*lz1);
		res2[id] = fabs(res2[id]);
		res2[id] = res2[id]*c_sub_e;  // cmult
		if(maxdiff !=0){
			double consta = 1/maxdiff;
			res2[id] = res2[id]*consta;
		}
	}

}


extern "C" void resvisc_gpu_wrapper2_(int *glbblockSize1,double *d_res2, int *nelt, int *lelt, int *lx1, int *ly1, int *lz1, double *c_sub_e, double *maxdiff){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	//        if (code1 != cudaSuccess){
	printf("CUDA: Start resvisc_gpu_wrapper2 cuda status: %s\n",cudaGetErrorString(code1));
	printf("CUDA: Start compute_entropy_gpu_wrapper values  nelt=%d,lelt=%d,lx1=%d,ly1=%d,lz1=%d,c_sub_e=%lf,maxdiff= %.20lf,  \n",nelt[0],lelt[0],lx1[0],ly1[0],lz1[0],c_sub_e[0],maxdiff[0]);  
#endif
	//      }

	int ntot = nelt[0]*lx1[0]*ly1[0]*lz1[0];
	int lxyz = lx1[0]*ly1[0]*lz1[0];
	int ltot = lelt[0]*lxyz;

	int blockSize =glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);


	resvisc_gpu_kernel2<<<gridSize, blockSize>>>(d_res2,ntot,lxyz,lx1[0],ly1[0],lz1[0], ltot,lx1[0]*ly1[0], c_sub_e[0], maxdiff[0]);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	// if (code2 != cudaSuccess){
	printf("CUDA: End resvisc_gpu_wrapper2 cuda status: %s\n",cudaGetErrorString(code2));
#endif
	// }

}

__global__ void evnsmooth_gpu_kernel(double *res2, double *t, int ntot,int lxyz, int lx1, int ly1, int lz1,int ltot, int lxy, int kstart, int kend, int jstart, int jend, int istart, int iend,int ldim , double rldim, double *rtmp, int if3d ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){

		int e= id/(lx1*ly1*lz1);
		if(t[2*ltot+id] <= res2[id]){
			res2[id] = t[2*ltot+id];// wavevisc and resvisc are really res2 and t. but the dimensions are different. As I understand this will start from 0 and works well.  Need to check with Dr.Tania . adeesha
		}

		//global syncthread is needed here. check with Dr.Tania. adeesha. 
		rtmp[id] = res2[id];
		int ix= id % lx1;
		int iy= (id/lx1)%ly1;
		int iz = (id / (lx1*ly1))%lz1;

		if((kstart<=iz && iz<=kend)&& (jstart<= iy && iy<= jend) && (istart<=ix && ix<=iend)){
			int izm,izp;
			if(if3d){
				int km1=iz-1;
				int kp1=iz+1;
				int izm=km1;
				if (km1 < 0){ izm=kp1;} // Guermond symmetry
				izp=kp1;
				if (kp1 > (lz1-1)){ izp=km1;} // Guermond symmetry
			}
			else{
				izm=iz;
				izp=iz;
			}
			int jm1=iy-1;
			int jp1=iy+1;
			int iym=jm1;
			if (jm1 < 0){ iym=jp1;}// Guermond symmetry
			int iyp=jp1;
			if (jp1 > (ly1-1)){ iyp=jm1;} // Guermond symmetry

			int im1=ix-1;
			int ip1=ix+1;
			int ixm=im1;
			if (im1 < 0){ ixm=ip1;} // Guermond symmetry
			int ixp=ip1;
			if (ip1 > (lx1-1)) {ixp=im1 ;} // Guermond symmetry
			double x0 = res2[e*lxyz+iz*lxy+iy*lx1+ix];
			double  x1 = res2[e*lxyz+iz*lxy+iy*lx1+ixm];
			double x2 = res2[e*lxyz+iz*lxy+iy*lx1+ixp];
			double x3 = res2[e*lxyz+iz*lxy+iym*lx1+ix];
			double x4 = res2[e*lxyz+iz*lxy+iyp*lx1+ix];
			double x5,x6;
			if (if3d){
				x5 = res2[e*lxyz+izm*lxy+iy*lx1+ixp];
				x6 = res2[e*lxyz+izp*lxy+iy*lx1+ixp];
			}
			else	{
				x5=0.0;
				x6=0.0;
			}
			rtmp[id]=0.25*(2.0*ldim*x0+x1+x2+x3+x4+x5+x6)*rldim;// check whether this is same as rtmp [id]. adeesha



		}
		res2[id]=rtmp[id];

	}
}

extern "C" void evnsmooth_gpu_wrapper_(int *glbblockSize1,double *d_res2, double *d_t, int *nelt, int *lelt, int *lx1, int *ly1, int *lz1,int *kstart, int *kend,  int *jstart, int *jend, int *istart, int *iend, int *ldim , double *rldim,  int *if3d ){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	// if (code1 != cudaSuccess){
	printf("CUDA: Start evnsmooth_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
	printf("CUDA: Start compute_entropy_gpu_wrapper values nelt =%d ,lelt=%d,lx1=%d,ly1=%d,lz1=%d,kstart=%d,kend=%d,jstart=%d,jend=%d,istart=%d,iend=%d,ldim=%d ,rldim=%lf,if3d=%d,\n", nelt[0],lelt[0],lx1[0],ly1[0],lz1[0],kstart[0],kend[0],jstart[0],jend[0],istart[0],iend[0],ldim[0] ,rldim[0],if3d[0]);
#endif
	// }

	int ntot = nelt[0]*lx1[0]*ly1[0]*lz1[0];
	int lxyz = lx1[0]*ly1[0]*lz1[0];
	int ltot = lelt[0]*lxyz;

	double *d_rtmp;
	cudaMalloc((void**)&d_rtmp,nelt[0]*lxyz* sizeof(double));


	int blockSize = glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);


	evnsmooth_gpu_kernel<<<gridSize, blockSize>>>(d_res2,d_t,ntot,lxyz,lx1[0],ly1[0],lz1[0], ltot,lx1[0]*ly1[0],(kstart[0]-1),(kend[0]-1),(jstart[0]-1),(jend[0]-1),(istart[0]-1),(iend[0]-1),ldim[0] ,rldim[0], d_rtmp,  if3d[0] );

	cudaFree(d_rtmp);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	// if (code2 != cudaSuccess){
	printf("CUDA: End evnsmooth_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
	// }

}



