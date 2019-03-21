#include <stdio.h>
//#define DEBUGPRINT 0

__global__ void compute_transport_props_gpu_kernel(double *vx, double *vy, double *vz, double *u, int nelt, int nxyz,int nnel,int irpu, int irpv, int irpw, int iret, int irg, int toteq,int if3d,double *vtrans, int irho, double *phig, int lx1, int ly1, int lz1, int *lglel, double *xm1, double *ym1, double *zm1, double *t,int ldimt, int npscal, double *pr, double p0th, double *sii, double *siii, double *vdiff, int ifield,char *cb, int icv, int icp, double *csound, int imu,int ilam, double cpgref, double cvgref, double gmaref, double rgasref, int *gllel, double *res2, int iknd, int inus, int lxy,int nlel ){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){

		int e = id/nxyz;
		int ieg= lglel[e];
		int k =  (id / (lx1*ly1))%lz1;
		int j =  (id/lx1)%ly1;
		int newi = id % lx1;

		// nekasgn
		double x = xm1[e*nxyz+k*lxy+j*lx1+newi];
		double y = ym1[e*nxyz+k*lxy+j*lx1+newi];
		double z = zm1[e*nxyz+k*lxy+j*lx1+newi];
		double r = x*x+y*y;
		double theta=0.0;
		if (r>0.0){ r = sqrtf(r);}
		if ( x != 0.0 || y!= 0.0){theta = atan2(y,x);   }
		double ux= vx[e*nxyz+k*lxy+j*lx1+newi];
		double uy= vy[e*nxyz+k*lxy+j*lx1+newi];
		double uz= vz[e*nxyz+k*lxy+j*lx1+newi];
		double temp = t [ e*nxyz+k*lxy+j*lx1+newi ];
		int ips;
		double ps[10]; // ps is size of ldimt which is 3. Not sure npscal is also 3. Need to check with Dr.Tania
		for (ips=0;ips<npscal;ips++){
			ps[ips]=t[(ips+1)*nlel+e*nxyz+k*lxy+j*lx1+newi ]; // 5 th dimension of t is idlmt which is 3. Not sure how the  nekasgn access ips+1. Need to check with Dr.Tania
		}
		double pa = pr [e*nxyz+k*lxy+j*lx1+newi];
		double p0= p0th;
		double si2 =  sii[e*nxyz+k*lxy+j*lx1+newi];
		double si3 =  siii[e*nxyz+k*lxy+j*lx1+newi];
		double udiff =  vdiff[(ifield-1)*nlel+e*nxyz+k*lxy+j*lx1+newi];
		double utrans =  vtrans[(ifield-1)*nlel+e*nxyz+k*lxy+j*lx1+newi];
		char cbu1 = cb[0];
		char cbu2 = cb[1];
		char cbu3 = cb[2];

		//cmtasgn
		int eqnum;
		double varsic[10];
		int e_offset = toteq*nxyz;
		for (eqnum=0;eqnum<toteq;eqnum++){
			varsic[eqnum] = u[e*e_offset+eqnum*nxyz+k*lxy+j*lx1+newi];

		}
		double phi = phig[e*nxyz+k*lxy+j*lx1+newi];
		double rho = vtrans[(irho-1)*nlel +e*nxyz+k*lxy+j*lx1+newi];
		double pres = pr[e*nxyz+k*lxy+j*lx1+newi];
		double cv=0.0,cp=0.0;
		if(rho!=0){
			cv=vtrans[(icv-1)*nlel +e*nxyz+k*lxy+j*lx1+newi]/rho;
			cp=vtrans[(icp-1)*nlel +e*nxyz+k*lxy+j*lx1+newi]/rho;
		}
		double asnd = csound [e*nxyz+k*lxy+j*lx1+newi];
		double mu = vdiff[(imu-1)*nlel+e*nxyz+k*lxy+j*lx1+newi];
		//udiff = vdiff[(imu-1)*nlel+e*nxyz+k*lxy+j*lx1+newi];// this overrides the udiff in nekasgn (line 63 in this function). Need to check withDr.Tania
		udiff = vdiff[(iknd-1)*nlel+e*nxyz+k*lxy+j*lx1+newi];//correct by Kk 02/01/2019
		double lambda = vdiff[(ilam-1)*nlel+e*nxyz+k*lxy+j*lx1+newi];

		// uservp
		int uservpe = gllel[ieg-1];
		mu=rho*res2[(uservpe-1)*nxyz+k*lxy+j*lx1+newi] ;//! finite c_E;
		double nu_s=0.75*mu/rho;
//		if(e==7){
//			printf("mu in compute transport is %lf rho = %lf uservpe =  %d ieg=%d \n", mu,rho,uservpe,ieg);
///		}
		mu=0.5*mu ;
		lambda=0.0;
		udiff=0.0;  // uservp makes these to zero. vdiff get zeros. Check with Dr. Tania. adeesha
		utrans=0.0; 	

		vdiff[(imu-1)*nlel +e*nxyz+k*lxy+j*lx1+newi]=mu; // vidff [  + id] is same as this. check later . adeesha
		vdiff[(ilam-1)*nlel +e*nxyz+k*lxy+j*lx1+newi]=lambda;
		vdiff[(iknd-1)*nlel +e*nxyz+k*lxy+j*lx1+newi]=udiff;
		vdiff[(inus-1)*nlel +e*nxyz+k*lxy+j*lx1+newi]=nu_s;



	}

}


extern "C" void compute_transport_props_gpu_wrapper_(int *glbblockSize1,double *d_vx, double *d_vy, double *d_vz, double *d_u,int *nelt,int *irpu, int *irpv, int *irpw, int* iret,  int *irg, int *toteq, int *if3d, double *d_vtrans, int *irho, double *d_phig, int *lx1, int *ly1, int *lz1, int *d_lglel, double *d_xm1, double *d_ym1, double *d_zm1, double *d_t,int *ldimt, int *npscal, double *d_pr, double *p0th, double *d_sii, double *d_siii, double *d_vdiff, int *ifield,char *d_cb, int *icv, int *icp, double *d_csound, int *imu,int *ilam, double *cpgref, double *cvgref, double *gmaref, double *rgasref,  int *d_gllel, double *d_res2, int *iknd, int *inus,int *lelt){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start compute_transport_props_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start compute_transport_props_gpu_wrapper values nelt =%d,irpu=%d,irpv=%d,irpw=%d,iret=%d,irg=%d,toteq=%d,if3d=%d,irho=%d,lx1=%d,ly1=%d,lz1=%d,ldimt=%d,npscal=%d,p0th=%lf,ifield=%d,icv=%d,icp=%d,imu=%d,ilam=%d,cpgref=%lf,cvgref=%lf,gmaref=%lf,rgasref=%lf,iknd=%d,inus=%d,lelt=%d ,\n", nelt[0],irpu[0],irpv[0],irpw[0],iret[0],irg[0],toteq[0],if3d[0],irho[0],lx1[0],ly1[0],lz1[0],ldimt[0],npscal[0],p0th[0],ifield[0],icv[0],icp[0],imu[0],ilam[0],cpgref[0],cvgref[0],gmaref[0],rgasref[0],iknd[0],inus[0],lelt[0]);
#endif

	int blockSize = glbblockSize1[0], gridSize;
	int lxy = lx1[0]*ly1[0];
	int nxyz= lxy*lz1[0];
	int nnel=nxyz*nelt[0];
	int nlel= nxyz*lelt[0];
	gridSize = (int)ceil((float)nnel/blockSize);
	compute_transport_props_gpu_kernel<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz, d_u, nelt[0], nxyz,nnel,  irpu[0], irpv[0], irpw[0], iret[0],irg[0],toteq[0],if3d[0],d_vtrans, irho[0],d_phig ,lx1[0], ly1[0],lz1[0], d_lglel, d_xm1, d_ym1,d_zm1, d_t,ldimt[0], npscal[0], d_pr,p0th[0], d_sii,d_siii,d_vdiff, ifield[0],d_cb, icv[0], icp[0],d_csound,imu[0],ilam[0],  cpgref[0], cvgref[0], gmaref[0], rgasref[0], d_gllel, d_res2, iknd[0], inus[0], lxy,nlel);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	printf("CUDA: End compute_transport_props_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));

#endif


}


__global__ void imqqtu_gpu_kernel(double *flux, int nf, int lf, int iuj, int ium, int iup,int totthreads){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<totthreads){
		flux[(iuj-1)+id] =flux[(ium-1)+id] + flux[(iup-1)+id] ; //add3
		flux[(iuj-1)+id] =flux[(iuj-1)+id] * -0.5; //cmult
		flux[(iuj-1)+id] =  flux[(iuj-1)+id] + flux[(ium-1)+id] ;// add 2
		//above 3 can be simplified iuj= (ium-iup) /2 check with Dr.Tania adeesha
                /*if(id ==0) {
                    printf("debug imqqtu %25.16E %25.16E %25.16E %25.16E %25.16E\n", flux[(ium-1)+id], flux[(iup-1)+id], (flux[(ium-1)+id] + flux[(iup-1)+id])*(-0.5),  (flux[(ium-1)+id] + flux[(iup-1)+id])*(-0.5)+flux[(ium-1)+id], flux[(iuj-1)+id]);

                }*/


	}

}

extern "C" void imqqtu_gpu_wrapper_(int *glbblockSize2,double *d_flux,int *iuj,int *ium,int *iup,int *lx1, int *ly1, int *lz1, int *ldim, int *nelt, int *lelt, int *toteq){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();

	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start imqqtu_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start imqqtu_gpu_wrapper values iuj =%d ,ium=%d,iup=%d,lx1=%d,ly1=%d,lz1=%d,ldim=%d,nelt=%d,lelt=%d,toteq=%d\n",iuj[0],ium[0],iup[0],lx1[0],ly1[0],lz1[0],ldim[0],nelt[0],lelt[0],toteq[0]);

#endif
	int nf= lx1[0]*lz1[0]*2*ldim[0]*nelt[0];
	int lf= lx1[0]*lz1[0]*2*ldim[0]*lelt[0];
	int totthreads = nf*toteq[0];


	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)totthreads/blockSize);
	imqqtu_gpu_kernel<<<gridSize, blockSize>>>(d_flux,nf,lf,iuj[0],ium[0],iup[0],totthreads);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End imqqtu_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));


#endif

}

__global__ void imqqtu_dirichlet_gpu_kernel(double *flux, int ntot, int ifield, int ltot, int ilam, int irho, int icv, int icp, int imu, double molmass, int iwp, int iwm, int iuj, int iux,int iuy,int iuz, int iph, int ithm, int iu1, int iu2, int iu3, int iu4, int iu5, int icvf,int toteq,int lx1,int ly1,int lz1,int lxy, int lxz,int nxyz, int lxz2ldim, int lxz2ldimlelt, int a2ldim, char *cbc, int *lglel, double *xm1,double *ym1, double *zm1, double *vx, double *vy, double *vz, double *t, double *pr, double *sii, double *siii, double *vdiff, double *vtrans, char *cb, double *u, double *phig, double *pres, double *csound,int npscal,double p0th,int e_offset,int nlel,int nqq){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){
		int i1 = id % lx1;
		int i2 = (id/lx1)%lz1;
		int iface= ((id/lxz)%a2ldim);
		int e = id/lxz2ldim;


		char cb1 = cbc[e*18+iface*3];
		char cb2 = cbc[e*18+iface*3+1];
		char cb3 = cbc[e*18+iface*3+2]; //iface -> iface*3 by Kk 03/18
		if(cb1!='E' &&  cb1!='P'){
			int iy,iz,ix,l;
			if(cb1 =='W'|| cb1=='I'){
				int ieg=lglel[e];

				if(iface==0){
					iy=0;
					iz=i2;
					ix=i1;
					l=lx1*iz+ix;
				}

				else if(iface==3){
					ix=0;
					iz=i2;
					iy=i1;
					l=ly1*iz+iy;
				}
				else if(iface==4){
					iz=0;
					iy=i2;
					ix=i1;
					l=lx1*iy+ix;
				}
				else if(iface==1){
					ix=lx1-1;
					iz=i2;
					iy=i1;
					l=ly1*iz+iy;
				}
				else if(iface==2){
					iy=ly1-1;
					iz=i2;
					ix=i1;
					l=lx1*iz+ix;
				}
				else if(iface==5){
					iz=lz1-1;
					iy=i2;
					ix=i1;
					l=lx1*iy+ix;
				}



				//nekasgn
				//				double x = xm1[e*nxyz+iz*lxy+iy*lx1+ix];
				//				double y = ym1[e*nxyz+iz*lxy+iy*lx1+ix];
				//				double z = zm1[e*nxyz+iz*lxy+iy*lx1+ix];
				//				double r = x*x+y*y;
				//				double theta=0.0;
				//				if (r>0.0){ r = sqrtf(r);}
				//				if ( x != 0.0 || y!= 0.0){theta = atan2(y,x);   }
				double ux= vx[e*nxyz+iz*lxy+iy*lx1+ix];
				double uy= vy[e*nxyz+iz*lxy+iy*lx1+ix];
				double uz= vz[e*nxyz+iz*lxy+iy*lx1+ix];
				double temp = t [ e*nxyz+iz*lxy+iy*lx1+ix];
				//				int ips;
				//				double ps[10]; // ps is size of ldimt which is 3. Not sure npscal is also 3. Need to check with Dr.Tania
				//				for (ips=0;ips<npscal;ips++){
				//					ps[ips]=t[(ips+1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix]; // 5 th dimension of t is idlmt which is 3. Not sure how the  nekasgn access ips+1. Need to check with Dr.Tania
				//				}
				//				double pa = pr [e*nxyz+iz*lxy+iy*lx1+ix];
				//				double p0= p0th;
				//				double si2 =  sii[e*nxyz+iz*lxy+iy*lx1+ix];
				//				double si3 =  siii[e*nxyz+iz*lxy+iy*lx1+ix];
				//				double udiff =  vdiff[(ifield-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];
				//				double utrans =  vtrans[(ifield-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];
				//				char cbu1 = cb[0];
				//				char cbu2 = cb[1];
				//				char cbu3 = cb[2];

				//cmtasgn
				//				int eqnum;
				//				double varsic[10];
				//				for (eqnum=0;eqnum<toteq;eqnum++){
				//					varsic[eqnum] = u[e*e_offset+eqnum*nxyz+iz*lxy+iy*lx1+ix];
				//
				//				}
				double phi = phig[e*nxyz+iz*lxy+iy*lx1+ix];
				//				double rho = vtrans[(irho-1)*nlel +e*nxyz+iz*lxy+iy*lx1+ix];
				//				double pres = pr[e*nxyz+iz*lxy+iy*lx1+ix];
				//				double cv=0.0,cp=0.0;
				//				if(rho!=0){
				//					cv=vtrans[(icv-1)*nlel +e*nxyz+iz*lxy+iy*lx1+ix]/rho;
				//					cp=vtrans[(icp-1)*nlel +e*nxyz+iz*lxy+iy*lx1+ix]/rho;
				//				}
				//				double asnd = csound [e*nxyz+iz*lxy+iy*lx1+ix];
				//				double mu = vdiff[(imu-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];
				//				udiff = vdiff[(imu-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];// this overrides the udiff in nekasgn (line 63 in this function). Need to check withDr.Tania
				//				double lambda = vdiff[(ilam-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];
				// userbc
				//				double molarmass = molmass;

				if(fabs(vdiff[(ilam-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix]) > 0.0000000001){
					flux[(iwp-1)+(iux-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l] = ux;
					flux[(iwp-1)+(iuy-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l] = uy;
					flux[(iwp-1)+(iuz-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l] = uz;
					flux[(iwp-1)+(iph-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l] = phi;
					flux[(iwp-1)+(ithm-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l] = temp;
					flux[(iwp-1)+(iu1-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l] = flux[(iwm-1)+(iu1-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];// can remove later and just muliitply with followings. adeesha.
					flux[(iwp-1)+(iu2-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l] = flux[(iwp-1)+(iu1-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]*ux;
					flux[(iwp-1)+(iu3-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l] = flux[(iwp-1)+(iu1-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]*uy;
					flux[(iwp-1)+(iu4-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l] = flux[(iwp-1)+(iu1-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]*uz;

					if(cb1=='W'){

						flux[(iwp-1)+(iu5-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l] = phi*flux[(iwm-1)+(icvf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]*temp+0.5/flux[(iwp-1)+(iu1-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]*( (flux[(iwp-1)+(iu2-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]*flux[(iwp-1)+(iu2-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l])+(flux[(iwp-1)+(iu3-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]*flux[(iwp-1)+(iu3-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l])+(flux[(iwp-1)+(iu4-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]*flux[(iwp-1)+(iu4-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]));
					}
					else if(cb1=='I'){
						flux[(iwp-1)+(iu5-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]=flux[(iwm-1)+(iu5-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
					}

				}
				else{
					for(int m=0;m<nqq;m++){
						flux[(iwp-1)+m*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= flux[(iwm-1)+m*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
					}

				}



			}
			//			if(id<10){
			//				printf("iuj = %d, iwm = %d, iwp = %d, lxz2ldimlelt = %d, lxz2ldim = %d, lxz = %d, iface= %d id %d  \n",iuj,iwm,iwp,lxz2ldimlelt,lxz2ldim,lxz,iface,id );
			//			}
				for(int ivar=0;ivar<toteq;ivar++){
					flux[(iuj-1)+ivar*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+lx1*i2+i1] = flux[(iwm-1)+ivar*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+lx1*i2+i1]-flux[(iwp-1)+ivar*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+lx1*i2+i1];
				}
//						if(id<500){
//							printf("iuj = %d, iwm = %d, iwp = %d, lxz2ldimlelt = %d, lxz2ldim = %d, lxz = %d, iface= %d id %d l = %d ix = %d, iy =%d, iz = %d, i1 = %d, i2=%d,lx1=%d, ly1=%d,lz1=%d, cb1= %d \n",iuj,iwm,iwp,lxz2ldimlelt,lxz2ldim,lxz,iface,id ,l,ix,iy,iz,i1,i2,lx1,ly1,lz1,cb1);
//						}

		}

	}
}


extern "C" void imqqtu_dirichlet_gpu_wrapper_(int *glbblockSize2,double *d_flux, int *ifield,  int *ilam, int *irho, int *icv, int *icp, int *imu, double *molmass, int *iwp, int *iwm, int *iuj, int *iux,int *iuy,int *iuz, int *iph, int *ithm, int *iu1, int *iu2, int *iu3, int *iu4, int *iu5, int *icvf,int *toteq,int *lx1,int *ly1,int *lz1, char *d_cbc, int *d_lglel, double *d_xm1,double *d_ym1, double *d_zm1, double *d_vx, double *d_vy, double *d_vz, double *d_t, double *d_pr, double *d_sii, double *d_siii, double *d_vdiff, double *d_vtrans, char *d_cb, double *d_u, double *d_phig, double *d_pres, double *d_csound, int *ldim, int *lelt, int *nelt,int *npscal, double *p0th,int *nqq  ){

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start imqqtu_dirichlet_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start imqqtu_dirichlet_gpu_wrapper values ifield=%d,ilam=%d,irho=%d,icv=%d,icp=%d,imu=%d,molmass=%lf,iwp=%d,iwm=%d,iuj=%d,iux=%d,iuy=%d,iuz=%d,iph=%d,ithm=%d,iu1=%d,iu2=%d,iu3=%d,iu4=%d,iu5=%d,icvf=%d,toteq=%d,lx1=%d,ly1=%d,lz1=%d,ldim=%d,lelt=%d,nelt=%d,npscal=%d,p0th=%lf,nqq=%d,\n",ifield[0],ilam[0],irho[0],icv[0],icp[0],imu[0],molmass[0],iwp[0],iwm[0],iuj[0],iux[0],iuy[0],iuz[0],iph[0],ithm[0],iu1[0],iu2[0],iu3[0],iu4[0],iu5[0],icvf[0],toteq[0],lx1[0],ly1[0],lz1[0],ldim[0],lelt[0],nelt[0],npscal[0],p0th[0],nqq[0]);

#endif
	int lxz2ldim = lx1[0]*lz1[0]*2*ldim[0];
	int nxyz = lx1[0]*ly1[0]*lz1[0];
	int lxy = lx1[0]*ly1[0];
	int lxz = lx1[0]*lz1[0];
	int ntot = nelt[0]*lxz2ldim;
	int lxz2ldimlelt = lxz2ldim*nelt[0];
	int ltot= lxz2ldim*lelt[0];
	int e_offset=toteq[0]*nxyz;
	int  nlel= nxyz*lelt[0];

	int a2ldim= 2*ldim[0];
	int blockSize = glbblockSize2[0], gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);


	imqqtu_dirichlet_gpu_kernel<<<gridSize, blockSize>>>(d_flux, ntot, ifield[0], ltot,ilam[0],irho[0],icv[0],icp[0],imu[0],molmass[0],iwp[0],iwm[0],iuj[0],iux[0],iuy[0],iuz[0],iph[0],ithm[0],iu1[0],iu2[0],iu3[0],iu4[0],iu5[0],icvf[0],toteq[0],lx1[0],ly1[0],lz1[0],lxy,lxz,nxyz,lxz2ldim,lxz2ldimlelt,a2ldim,d_cbc, d_lglel,d_xm1,d_ym1, d_zm1, d_vx,d_vy,d_vz,d_t,d_pr,d_sii,d_siii,d_vdiff,d_vtrans, d_cb,d_u, d_phig,d_pres,d_csound,npscal[0],p0th[0],e_offset,nlel,nqq[0] );




#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End imqqtu_dirichlet_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));


#endif
}

