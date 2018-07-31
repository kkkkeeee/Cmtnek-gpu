#include <stdio.h>
#define DEBUGPRINT 0

__global__ void compute_primitive_vars_kernel (double *vx, double *vy, double *vz, double *u, int nelt, int nxyz,int ntot,int irpu, int irpv, int irpw, int iret, int irg, int toteq,int if3d,double *scr, double* energy, double *vtrans, int irho, double *phig, int lx1, int ly1, int lz1, int *lglel, double *xm1, double *ym1, double *zm1, double *t,int ldimt, int npscal, double *pr, double p0th, double *sii, double *siii, double *vdiff, int ifield,char *cb, int icv, int icp, double *csound, int imu,int ilam, double cpgref, double cvgref, double gmaref, double rgasref,  int ltot,int lxy){    
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nelt*nxyz){


		int e = id/nxyz;
		int i = id%nxyz;
		int e_offset = toteq*nxyz;
		double c = u[e*e_offset+(irg-1)*nxyz+i];
		vx[id] = u[e*e_offset+(irpu-1)*nxyz+i]/c;//invcol3
		vy[id] = u[e*e_offset+(irpv-1)*nxyz+i]/c;
		vz[id] = u[e*e_offset+(irpw-1)*nxyz+i]/c;

		if(if3d){
			//Compute a Cartesian vector dot product. 3-d version  vdot3
			scr[id] = u[e*e_offset+(irpu-1)*nxyz+i]*u[e*e_offset+(irpu-1)*nxyz+i]+u[e*e_offset+(irpv-1)*nxyz+i]*u[e*e_offset+(irpv-1)*nxyz+i]+u[e*e_offset+(irpw-1)*nxyz+i]*u[e*e_offset+(irpw-1)*nxyz+i];

		}
		else{
			// compute vector dot product 2d version vdot2
			scr[id] = u[e*e_offset+(irpu-1)*nxyz+i]*u[e*e_offset+(irpu-1)*nxyz+i]+u[e*e_offset+(irpv-1)*nxyz+i]*u[e*e_offset+(irpv-1)*nxyz+i];


		}
  
                scr[id] = scr[id]/c; //invcol2
                scr[id] = scr[id] * 0.5; //cmult
	
		energy[id] =  u[e*e_offset+(iret-1)*nxyz+i] -scr[id];// sub3	
		energy[id] = energy[id]/c;// invcol2
		
		vtrans[(irho-1)*ltot+id ] = c / phig[id];  //invcol3

                // subroutine tdstate

		int eg= lglel[e]; // this never uses.  Check with Dr.Tania
		int k =  (id / (lx1*ly1))%lz1;
		int j =  (id/lx1)%ly1;
		int newi = id % lx1;

		double x = xm1[e*nxyz+k*lxy+j*lx1+newi];
		double y = ym1[e*nxyz+k*lxy+j*lx1+newi];
		double z = zm1[e*nxyz+k*lxy+j*lx1+newi];
		double r = x*x+y*y;
		double theta=0.0;
		if (r>0.0){ r = sqrtf(r);}
		if ( x != 0.0 || y!= 0.0){theta = atan2(y,x);	}
		double ux= vx[e*nxyz+k*lxy+j*lx1+newi];
		double uy= vy[e*nxyz+k*lxy+j*lx1+newi];
		double uz= vz[e*nxyz+k*lxy+j*lx1+newi];
		double temp = t [ e*nxyz+k*lxy+j*lx1+newi ];
		int ips;
		double ps[10]; // ps is size of ldimt which is 3. Not sure npscal is also 3. Need to check with Dr.Tania
		for (ips=0;ips<npscal;ips++){
			ps[ips]=t[(ips+1)*ltot+e*nxyz+k*lxy+j*lx1+newi ]; // 5 th dimension of t is idlmt which is 3. Not sure how the  nekasgn access ips+1. Need to check with Dr.Tania
		}
		double pa = pr [e*nxyz+k*lxy+j*lx1+newi];
		double p0= p0th;
		double si2 =  sii[e*nxyz+k*lxy+j*lx1+newi];
		double si3 =  siii[e*nxyz+k*lxy+j*lx1+newi];
		double udiff =  vdiff[(ifield-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];
		double utrans =  vtrans[(ifield-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];
		char cbu1 = cb[0];
		char cbu2 = cb[1];
		char cbu3 = cb[2];
		
		// subroutine cmtasgn
		int eqnum;
		double varsic[10];
		for (eqnum=0;eqnum<toteq;eqnum++){
			varsic[eqnum] = u[e*e_offset+eqnum*nxyz+k*lxy+j*lx1+newi];

		}
		double phi = phig[e*nxyz+k*lxy+j*lx1+newi];
		double rho = vtrans[(irho-1)*ltot +e*nxyz+k*lxy+j*lx1+newi];
		double pres = pr[e*nxyz+k*lxy+j*lx1+newi];
		double cv=0.0,cp=0.0;
		if(rho!=0){
			cv=vtrans[(icv-1)*ltot +e*nxyz+k*lxy+j*lx1+newi]/rho;
			cp=vtrans[(icp-1)*ltot +e*nxyz+k*lxy+j*lx1+newi]/rho;
		}
		double asnd = csound [e*nxyz+k*lxy+j*lx1+newi];
		double mu = vdiff[(imu-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];
		udiff = vdiff[(imu-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];// this overrides the udiff in nekasgn (line 63 in this function). Need to check withDr.Tania
		double lambda = vdiff[(ilam-1)*ltot+e*nxyz+k*lxy+j*lx1+newi];
	
		double e_internal = energy[e*nxyz+k*lxy+j*lx1+newi];
		//subroutine cmt_userEOS
		cp=cpgref;
      		cv=cvgref;
      		temp=e_internal/cv; // overrides
		// function MixtPerf_C_GRT
      		asnd=sqrtf(gmaref*rgasref*temp);  //overrides
		// function MixtPerf_P_DRT
      		pres=rho*rgasref*temp;//overrides
		
		vtrans[(icp-1)*ltot +e*nxyz+k*lxy+j*lx1+newi]=cp*rho;
		vtrans[(icv-1)*ltot +e*nxyz+k*lxy+j*lx1+newi]=cv*rho;
		t [ e*nxyz+k*lxy+j*lx1+newi ]= temp;
		csound [e*nxyz+k*lxy+j*lx1+newi]=asnd;
		
	}
}

extern "C" void compute_primitive_vars_gpu_wrapper_(int *glbblockSize1,double *d_vx, double *d_vy, double *d_vz, double *d_u, int *nxyz, int *ntot, int *nelt,int *irpu, int *irpv, int *irpw, int* iret,  int *irg, int *toteq, int *if3d, double *d_vtrans, int *irho, double *d_phig, int *lx1, int *ly1, int *lz1, int *d_lglel, double *d_xm1, double *d_ym1, double *d_zm1, double *d_t,int *ldimt, int *npscal, double *d_pr, double *p0th, double *d_sii, double *d_siii, double *d_vdiff, int *ifield,char *d_cb, int *icv, int *icp, double *d_csound, int *imu,int *ilam, double *cpgref, double *cvgref, double *gmaref, double *rgasref, int *ltot){
#ifdef DEBUGPRINT
 cudaError_t code1 = cudaPeekAtLastError();

printf("CUDA: Start compute_primitive_vars_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

printf("CUDA: Start compute_primitive_vars_gpu_wrapper values nxyz = %d,ntot = %d,nelt = %d,irpu = %d,irpv = %d,irpw = %d,iret = %d,irg=%d ,toteq = %d,if3d = %d,irho = %d,lx1 = %d,ly1 = %d,lz1 = %d,ldimt = %d,npscal = %d,p0th = %lf,ifield = %d,icv = %d,icp = %d,imu = %d,ilam = %d,cpgref = %lf,cvgref = %lf,gmaref = %lf,rgasref = %lf,ltot = %d,  \n", nxyz[0],ntot[0],nelt[0],irpu[0],irpv[0],irpw[0],iret[0],irg[0],toteq[0],if3d[0],irho[0],lx1[0],ly1[0],lz1[0],ldimt[0],npscal[0],p0th[0],ifield[0],icv[0],icp[0],imu[0],ilam[0],cpgref[0],cvgref[0],gmaref[0],rgasref[0],ltot[0]);
#endif

	double *d_scr;  // I think this is a tempory variable. need to check with Dr.Tania. adeesha
	double *d_energy;
	cudaMalloc((void**)&d_scr,ntot[0] *  sizeof(double));
	cudaMalloc((void**)&d_energy,ntot[0] *  sizeof(double));

	int lxy=lx1[0]*ly1[0];
	int blockSize =glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)nelt[0]*nxyz[0]/blockSize);
	compute_primitive_vars_kernel<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz, d_u, nelt[0], nxyz[0],ntot[0],  irpu[0], irpv[0], irpw[0], iret[0],irg[0],toteq[0],if3d[0],d_scr,d_energy,d_vtrans, irho[0],d_phig ,lx1[0], ly1[0],lz1[0], d_lglel, d_xm1, d_ym1,d_zm1, d_t,ldimt[0], npscal[0], d_pr,p0th[0], d_sii,d_siii,d_vdiff, ifield[0],d_cb, icv[0], icp[0],d_csound,imu[0],ilam[0],  cpgref[0], cvgref[0], gmaref[0], rgasref[0],ltot[0],lxy);


        cudaFree(d_scr);
	cudaFree(d_energy);
#ifdef DEBUGPRINT
cudaError_t code2 = cudaPeekAtLastError();

printf("CUDA: End compute_primitive_vars_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif



}
