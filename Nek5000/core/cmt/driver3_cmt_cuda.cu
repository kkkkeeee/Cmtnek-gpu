#include <stdio.h>
#include <cuda_runtime_api.h>
#include "cuda_helpers.h" //for set_convect_cons_gpu_wrapper_ calling specmpn

//#define DEBUGPRINT 0

__global__ void compute_primitive_vars_kernel (double *vx, double *vy, double *vz, double *u, int nelt, int nxyz,int ntot,int irpu, int irpv, int irpw, int iret, int irg, int toteq,int if3d,double *scr, double* energy, double *vtrans, int irho, double *phig, int lx1, int ly1, int lz1, int *lglel, double *xm1, double *ym1, double *zm1, double *t,int ldimt, int npscal, double *pr, double p0th, double *sii, double *siii, double *vdiff, int ifield,char *cb, int icv, int icp, double *csound, int imu, int iknd, int ilam, double cpgref, double cvgref, double gmaref, double rgasref,  int ltot,int lxy){//added iknd as parameter by kk 04/26    
	int id = blockIdx.x*blockDim.x+threadIdx.x;
        //if(id<nelt*nxyz){
	if(id<ntot){//ntot = nelt *nxyz, added by kk 04/23

		int e = id/nxyz;
		int i = id%nxyz;
		int e_offset = toteq*nxyz;
		/*double c = u[e*e_offset+(irg-1)*nxyz+i];
		vx[id] = u[e*e_offset+(irpu-1)*nxyz+i]/c;//invcol3
		vy[id] = u[e*e_offset+(irpv-1)*nxyz+i]/c;
		vz[id] = u[e*e_offset+(irpw-1)*nxyz+i]/c;*/

                //added by Kk 04/09/19 for performance
                double irg_c = u[e*e_offset+(irg-1)*nxyz+i];
                double irpu_c = u[e*e_offset+(irpu-1)*nxyz+i];
                double irpv_c = u[e*e_offset+(irpv-1)*nxyz+i];
                double irpw_c = u[e*e_offset+(irpw-1)*nxyz+i];
                vx[id] = irpu_c/irg_c;
                vy[id] = irpv_c/irg_c;
                vz[id] = irpw_c/irg_c;
                //end add

		if(if3d){
			//Compute a Cartesian vector dot product. 3-d version  vdot3
                        /* for performance replace with the following
			scr[id] = u[e*e_offset+(irpu-1)*nxyz+i]*u[e*e_offset+(irpu-1)*nxyz+i]+u[e*e_offset+(irpv-1)*nxyz+i]*u[e*e_offset+(irpv-1)*nxyz+i]+u[e*e_offset+(irpw-1)*nxyz+i]*u[e*e_offset+(irpw-1)*nxyz+i];*/
                       scr[id] = irpu_c*irpu_c + irpv_c*irpv_c + irpw_c*irpw_c;
		}
		else{
			// compute vector dot product 2d version vdot2
                        /* for performance replace with the following
			scr[id] = u[e*e_offset+(irpu-1)*nxyz+i]*u[e*e_offset+(irpu-1)*nxyz+i]+u[e*e_offset+(irpv-1)*nxyz+i]*u[e*e_offset+(irpv-1)*nxyz+i];*/
                        scr[id] = irpu_c*irpu_c + irpv_c*irpv_c;
		}


		/*scr[id] = scr[id]/irg_c; //invcol2
		//scr[id] = scr[id]/c; //invcol2
		scr[id] = scr[id] * 0.5; //cmult*/
                scr[id] = scr[id]/irg_c *0.5; //invcol2, above merge to 1, by kk 04/26

		energy[id] =  u[e*e_offset+(iret-1)*nxyz+i] -scr[id];// sub3	
		energy[id] = energy[id]/irg_c;// invcol2; replace c with irg_c by Kk 04/09/2019
		vtrans[(irho-1)*ltot+id ] = irg_c / phig[id];  //invcol3; replace c with irg_c by Kk 04/09/2019

		// subroutine tdstate

		int eg= lglel[e]; // this never uses.  Check with Dr.Tania
		int k =  (id / (lx1*ly1))%lz1;
		int j =  (id/lx1)%ly1;
		int newi = id % lx1;

		double x = xm1[id]; //xm1[e*nxyz+k*lxy+j*lx1+newi];
		double y = ym1[id]; //ym1[e*nxyz+k*lxy+j*lx1+newi];
		double z = zm1[id]; //zm1[e*nxyz+k*lxy+j*lx1+newi];
		double r = x*x+y*y;
		double theta=0.0;
		if (r>0.0){ r = sqrtf(r);}
		if ( x != 0.0 || y!= 0.0){theta = atan2(y,x);	}
		double ux= vx[id]; //vx[e*nxyz+k*lxy+j*lx1+newi];
		double uy= vy[id]; //vy[e*nxyz+k*lxy+j*lx1+newi];
		double uz= vz[id]; //vz[e*nxyz+k*lxy+j*lx1+newi];
		double temp = t[id]; // t [ e*nxyz+k*lxy+j*lx1+newi ];
		int ips;
		double ps[10]; // ps is size of ldimt which is 3. Not sure npscal is also 3. Need to check with Dr.Tania
		for (ips=0;ips<npscal;ips++){
			ps[ips]=t[(ips+1)*ltot+e*nxyz+k*lxy+j*lx1+newi ]; // 5 th dimension of t is idlmt which is 3. Not sure how the  nekasgn access ips+1. Need to check with Dr.Tania
		}
		double pa = pr[id]; //pr [e*nxyz+k*lxy+j*lx1+newi];
		double p0= p0th;
		double si2 =  sii[id]; //sii[e*nxyz+k*lxy+j*lx1+newi];
		double si3 =  siii[id]; //siii[e*nxyz+k*lxy+j*lx1+newi];
		double udiff = vdiff[(ifield-1)*ltot+id]; // vdiff[(ifield-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];
		double utrans = vtrans[(ifield-1)*ltot+id];  // vtrans[(ifield-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];
		char cbu1 = cb[0];
		char cbu2 = cb[1];
		char cbu3 = cb[2];

		// subroutine cmtasgn
		int eqnum;
		double varsic[10];
		for (eqnum=0;eqnum<toteq;eqnum++){
			varsic[eqnum] = u[e*e_offset+eqnum*nxyz+k*lxy+j*lx1+newi];
		}
		double phi = phig[id]; //phig[e*nxyz+k*lxy+j*lx1+newi];
		double rho = vtrans[(irho-1)*ltot + id]; //vtrans[(irho-1)*ltot +e*nxyz+k*lxy+j*lx1+newi];
		double pres = pr[id]; // pr[e*nxyz+k*lxy+j*lx1+newi];
		double cv=0.0,cp=0.0;
		if(rho!=0){
			cv=vtrans[(icv-1)*ltot+ id]/rho; // vtrans[(icv-1)*ltot +e*nxyz+k*lxy+j*lx1+newi]/rho;
			cp=vtrans[(icp-1)*ltot + id]/rho; // vtrans[(icp-1)*ltot +e*nxyz+k*lxy+j*lx1+newi]/rho;
		}
		double asnd = csound[id]; //csound [e*nxyz+k*lxy+j*lx1+newi];
		double mu = vdiff[(imu-1)*ltot+id]; // vdiff[(imu-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];
		//udiff = vdiff[(imu-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];// this overrides the udiff in nekasgn (line 63 in this function). Need to check withDr.Tania
		udiff = vdiff[(iknd-1)*ltot+id]; //vdiff[(imu-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];// imu should be iknd, corrected by Kk 04/26
		double lambda = vdiff[(ilam-1)*ltot+id]; //vdiff[(ilam-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];

		double e_internal = energy[id]; //energy[e*nxyz+k*lxy+j*lx1+newi];
		//subroutine cmt_userEOS
		cp=cpgref;
		cv=cvgref;
		temp=e_internal/cv; // overrides
		// function MixtPerf_C_GRT
		asnd=sqrt(gmaref*rgasref*temp);  //overrides
		// function MixtPerf_P_DRT

		pres=rho*rgasref*temp;//overrides

		/* comment to replace with id
                vtrans[(icp-1)*ltot +e*nxyz+k*lxy+j*lx1+newi]=cp*rho;
		vtrans[(icv-1)*ltot +e*nxyz+k*lxy+j*lx1+newi]=cv*rho;
		t [ e*nxyz+k*lxy+j*lx1+newi ]= temp;
		csound [e*nxyz+k*lxy+j*lx1+newi]=asnd;*/
                vtrans[(icp-1)*ltot +id]=cp*rho;
                vtrans[(icv-1)*ltot +id]=cv*rho;
                t [id]= temp;
                csound [id]=asnd;
	}
}

extern "C" void compute_primitive_vars_gpu_wrapper_(int *glbblockSize1,double *d_vx, double *d_vy, double *d_vz, double *d_u, int *nxyz, int *ntot, int *nelt,int *irpu, int *irpv, int *irpw, int* iret,  int *irg, int *toteq, int *if3d, double *d_vtrans, int *irho, double *d_phig, int *lx1, int *ly1, int *lz1, int *d_lglel, double *d_xm1, double *d_ym1, double *d_zm1, double *d_t,int *ldimt, int *npscal, double *d_pr, double *p0th, double *d_sii, double *d_siii, double *d_vdiff, int *ifield,char *d_cb, int *icv, int *icp, double *d_csound, int *imu, int *iknd, int *ilam, double *cpgref, double *cvgref, double *gmaref, double *rgasref, int *ltot){
#ifdef DEBUGPRINT
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start compute_primitive_vars_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start compute_primitive_vars_gpu_wrapper values nxyz = %d,ntot = %d,nelt = %d,irpu = %d,irpv = %d,irpw = %d,iret = %d,irg=%d ,toteq = %d,if3d = %d,irho = %d,lx1 = %d,ly1 = %d,lz1 = %d,ldimt = %d,npscal = %d,p0th = %lf,ifield = %d,icv = %d,icp = %d,imu = %d,ilam = %d,cpgref = %lf,cvgref = %.20lf,gmaref = %lf,rgasref = %lf,ltot = %d,  \n", nxyz[0],ntot[0],nelt[0],irpu[0],irpv[0],irpw[0],iret[0],irg[0],toteq[0],if3d[0],irho[0],lx1[0],ly1[0],lz1[0],ldimt[0],npscal[0],p0th[0],ifield[0],icv[0],icp[0],imu[0],ilam[0],cpgref[0],cvgref[0],gmaref[0],rgasref[0],ltot[0]);
#endif

	double *d_scr;  // I think this is a tempory variable. need to check with Dr.Tania. adeesha
	double *d_energy;
	cudaMalloc((void**)&d_scr,ntot[0] *  sizeof(double));
	cudaMalloc((void**)&d_energy,ntot[0] *  sizeof(double));

	int lxy=lx1[0]*ly1[0];
	int blockSize =glbblockSize1[0], gridSize;
	//gridSize = (int)ceil((float)nelt[0]*nxyz[0]/blockSize);
        gridSize = (int)ceil((float)ntot[0]/blockSize); //ntot = nxyz * nelt, changed by kk 04/26
	compute_primitive_vars_kernel<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz, d_u, nelt[0], nxyz[0],ntot[0],  irpu[0], irpv[0], irpw[0], iret[0],irg[0],toteq[0],if3d[0],d_scr,d_energy,d_vtrans, irho[0],d_phig ,lx1[0], ly1[0],lz1[0], d_lglel, d_xm1, d_ym1,d_zm1, d_t,ldimt[0], npscal[0], d_pr,p0th[0], d_sii,d_siii,d_vdiff, ifield[0],d_cb, icv[0], icp[0],d_csound,imu[0], iknd[0], ilam[0],  cpgref[0], cvgref[0], gmaref[0], rgasref[0],ltot[0],lxy);


	cudaFree(d_scr);
	cudaFree(d_energy);
#ifdef DEBUGPRINT
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End compute_primitive_vars_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
}

__global__ void update_u_gpu_kernel(double *u, double *bm1, double *tcoef, double *res3, double *res1, int nelt, int lelt, int lx1, int ly1, int lz1, int toteq, int stage, int lxyz, int lxyznelttoteq, int lxyznelt, int lxyzlelt){//added parameter lxyzlelt by Kk 03/18
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = id % lx1;
	int iy = (id/lx1)%ly1;
	int iz = (id / (lx1*ly1))%lz1;
	int e = (id/lxyz) % nelt;
	int eq = id/lxyznelt;

	if(id<lxyznelttoteq){



	  u[e*lxyz*toteq + eq*lxyz + ix + iy*lx1 + iz*lx1*ly1] = bm1[ix + iy*lx1 + iz*lx1*ly1 + e*lxyz] * tcoef[(stage-1)*3] * res3[e*lxyz*toteq + eq*lxyz + ix + iy*lx1 + iz*lx1*ly1] + bm1[ix + iy*lx1 + iz*lx1*ly1 + e*lxyz] *  tcoef[(stage-1)*3+1] * u[e*lxyz*toteq + eq*lxyz + ix + iy*lx1 + iz*lx1*ly1] - tcoef[(stage-1)*3+2] * res1[eq*lxyzlelt + e*lxyz + ix+iy*lx1+iz*lx1*ly1];

	  u[e*lxyz*toteq + eq*lxyz + ix + iy*lx1 + iz*lx1*ly1] = u[e*lxyz*toteq + eq*lxyz + ix + iy*lx1 + iz*lx1*ly1] / bm1[ix + iy*lx1 + iz*lx1*ly1 + e*lxyz];

          /*if(eq ==1 && e == 0 && id%lxyz==10){
              printf("debug u here %25.16E %25.16E %25.16E %25.16E %25.16E %25.16E %25.16E %25.16E %25.16E %25.16E %d %d %d %d %d %d \n", bm1[ix + iy*lx1 + iz*lx1*ly1 + e*lxyz], tcoef[(stage-1)*3], res3[e*lxyz*toteq + eq*lxyz + ix + iy*lx1 + iz*lx1*ly1], tcoef[(stage-1)*3+1],  u[e*lxyz*toteq + eq*lxyz + ix + iy*lx1 + iz*lx1*ly1], tcoef[(stage-1)*3+2], res1[eq*lxyzlelt + e*lxyz + ix+iy*lx1+iz*lx1*ly1], bm1[ix + iy*lx1 + iz*lx1*ly1 + e*lxyz] * tcoef[(stage-1)*3] * res3[e*lxyz*toteq + eq*lxyz + ix + iy*lx1 + iz*lx1*ly1], bm1[ix + iy*lx1 + iz*lx1*ly1 + e*lxyz] *  tcoef[(stage-1)*3+1] * u[e*lxyz*toteq + eq*lxyz + ix + iy*lx1 + iz*lx1*ly1], tcoef[(stage-1)*3+2] * res1[eq*lxyzlelt + e*lxyz + ix+iy*lx1+iz*lx1*ly1], id, ix, iy, iz, e, eq);

          }*/
//	  printf("debug update_u u  : %.30lf %d %d %d %d %d %d %d %d %d\n",u[e*lxyz*toteq + eq*lxyz + ix + iy*lx1 + iz*lx1*ly1],id,e,eq,ix,iy,iz,blockDim.x,blockIdx.x,threadIdx.x );


	  }


}
extern "C" void update_u_gpu_wrapper_(int *glbblockSize1, double *d_u, double *d_bm1, double *d_tcoef, double *d_res3, double *d_res1, int *nelt, int *lelt, int *lx1, int *ly1, int *lz1, int *toteq, int *stage){

#ifdef DEBUGPRINT
	printf("stagem %d \n",stage[0]);
	printf("values  %d %d %d %d %d %d %d \n",lx1[0],ly1[0],nelt[0],lelt[0],toteq[0],lz1[0],stage[0]);
#endif

	int lxyz = lx1[0]*ly1[0]*lz1[0];
	int lxyznelt = lx1[0]*ly1[0]*lz1[0]*nelt[0];
	int lxyzlelt = lx1[0]*ly1[0]*lz1[0]*lelt[0]; //added by Kk 03/18
	int lxyznelttoteq = lx1[0]*ly1[0]*lz1[0]*nelt[0]*toteq[0];
	int blockSize =glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)nelt[0]*lxyz*toteq[0]/blockSize);
	//printf("gridsize ddd %d %d \n",gridSize,blockSize);

	update_u_gpu_kernel<<<gridSize, blockSize>>>(d_u, d_bm1, d_tcoef, d_res3, d_res1, nelt[0], lelt[0], lx1[0], ly1[0], lz1[0], toteq[0], stage[0], lxyz, lxyznelttoteq, lxyznelt, lxyzlelt);


#ifdef DEBUGPRINT
 cudaDeviceSynchronize();
       cudaError_t  code1 = cudaPeekAtLastError();
        printf("CUDA: update_u_gpu_wrapper: cuda status: %s\n",cudaGetErrorString(code1));
	cudaError_t code2 = cudaPeekAtLastError();
	printf("CUDA: End compute_primitive_vars_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif

}

extern "C" void set_convect_cons_gpu_wrapper_(double *d_vxd, double *d_vyd, double *d_vzd, double *d_vx, double *d_vy, double *d_vz, int *lx1, int *lxd, double *d_jgl, double *d_jgt, int *if3d, int *ldim, int *nelt){

        int ld = 2*lxd[0];
        int ldw=2*pow(ld,ldim[0]);
        double *d_w;
        cudaMalloc((void**)&d_w, nelt[0]*ldw*sizeof(double));
        //create handle
        cublasHandle_t handle;
        cublasCreate(&handle);

#ifdef DEBUGPRINT
        cudaDeviceSynchronize();
        cudaError_t code1 = cudaPeekAtLastError();
        printf("CUDA: Start set_convect_cons_gpu_wrapper_ cuda status: %s\n",cudaGetErrorString(code1));

        printf("CUDA: Start set_convect_cons_gpu_wrapper_ values lx1=%d , lxd=%d,if3d=%d,nelt=%d,ldim=%d \n", lx1[0],lxd[0],if3d[0], nelt[0],ldim[0]);
#endif

        gpu_specmpn(handle, d_vxd, lxd[0], d_vx, lx1[0], d_jgl, d_jgt, if3d[0], d_w, ldw, nelt[0], 1,1,true);

#ifdef DEBUGPRINT
        cudaDeviceSynchronize();
        cudaError_t code3 = cudaPeekAtLastError();
        printf("CUDA: set_convect_cons_gpu_wrapper_ first gpu_specmpn cuda status: %s\n",cudaGetErrorString(code3));
#endif

        gpu_specmpn(handle, d_vyd, lxd[0], d_vy, lx1[0], d_jgl, d_jgt, if3d[0], d_w, ldw, nelt[0], 1,1,true);

#ifdef DEBUGPRINT
        cudaDeviceSynchronize();
        cudaError_t code4 = cudaPeekAtLastError();
        printf("CUDA: set_convect_cons_gpu_wrapper_ second gpu_specmpn cuda status: %s\n",cudaGetErrorString(code4));
#endif

        if(if3d[0]){
           gpu_specmpn(handle, d_vzd, lxd[0], d_vz, lx1[0], d_jgl, d_jgt, if3d[0], d_w, ldw, nelt[0], 1,1,true);

#ifdef DEBUGPRINT
        cudaDeviceSynchronize();
        cudaError_t code5 = cudaPeekAtLastError();
        printf("CUDA: set_convect_cons_gpu_wrapper_ third gpu_specmpn cuda status: %s\n",cudaGetErrorString(code5));
#endif

        }
        cudaFree(d_w);
        //destroy handle
        cublasDestroy(handle);

#ifdef DEBUGPRINT
        cudaDeviceSynchronize();
        cudaError_t code2 = cudaPeekAtLastError();
        printf("CUDA: End set_convect_cons_gpu_wrapper_ cuda status: %s\n",cudaGetErrorString(code2));
#endif
        


}
