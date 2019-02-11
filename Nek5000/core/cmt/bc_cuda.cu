#include <stdio.h>
#define DEBUGPRINT 0
__global__ void InviscidBC_gpu_kernel1(int *lglel, double *fatface,char *cbc, double *xm1,double *ym1,double *zm1,double *vx,double *vy,double *vz,double *t,double *pr,double *sii,double *siii,double *vdiff,double *vtrans,char *cb,double *u,double *phig,double *csound,double *unx,double *uny,double *unz,double molarmass,int iwm,int iwp,int irho,int iux,int iuy,int iuz,int iph,int ipr,int isnd,int ithm,int icpf,int icvf,int iu1,int iu2,int iu3,int iu4,int iu5,int lx1,int lz1,int lxz,int ldim,int lxz2ldim,int nxyz,int lxy,int lxz2ldimlelt,int ntot,int toteq,int e_offset,int p0th,int ifield,int ltot,int icv,int icp,int imu,int ilam,double molmass,int nlel,int npscal,int if3d,int ly1,int outflsub,double pinfty, int lelt){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){
		int i1 = id % lx1;
		int i2 = (id/lx1)%lz1;
		int iface= ((id/lxz)%(2*ldim));
		int e = id/lxz2ldim;

		char cb1 = cbc[ifield*lelt*18+e*18+iface*3]; //corrected by Kk 02/07/2019 by adding ifield*lelt*18 and iface*3
		char cb2 = cbc[ifield*lelt*18+e*18+iface*3+1];
		char cb3 = cbc[ifield*lelt*18+e*18+iface*3+2];
		if(cb1 =='v'|| cb1=='V'){
			int ieg=lglel[e];
			int iy,iz,ix,l;

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
			double x = xm1[e*nxyz+iz*lxy+iy*lx1+ix];
			double y = ym1[e*nxyz+iz*lxy+iy*lx1+ix];
			double z = zm1[e*nxyz+iz*lxy+iy*lx1+ix];
			double r = x*x+y*y;
			double theta=0.0;
			if (r>0.0){ r = sqrtf(r);}
			if ( x != 0.0 || y!= 0.0){theta = atan2(y,x);   }
			double ux= vx[e*nxyz+iz*lxy+iy*lx1+ix];
			double uy= vy[e*nxyz+iz*lxy+iy*lx1+ix];
			double uz= vz[e*nxyz+iz*lxy+iy*lx1+ix];
			double temp = t [ e*nxyz+iz*lxy+iy*lx1+ix];
			int ips;
			double pa = pr [e*nxyz+iz*lxy+iy*lx1+ix];
			double p0= p0th;
			double si2 =  sii[e*nxyz+iz*lxy+iy*lx1+ix];
			double si3 =  siii[e*nxyz+iz*lxy+iy*lx1+ix];
			double udiff =  vdiff[(ifield-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];
			double utrans =  vtrans[(ifield-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];
			char cbu1 = cb[0];
			char cbu2 = cb[1];
			char cbu3 = cb[2];

			//cmtasgn
			double phi = phig[e*nxyz+iz*lxy+iy*lx1+ix];
			double rho = vtrans[(irho-1)*nlel +e*nxyz+iz*lxy+iy*lx1+ix];
			double pres = pr[e*nxyz+iz*lxy+iy*lx1+ix];
			double cv=0.0,cp=0.0;
			if(rho!=0){
				cv=vtrans[(icv-1)*nlel +e*nxyz+iz*lxy+iy*lx1+ix]/rho;
				cp=vtrans[(icp-1)*nlel +e*nxyz+iz*lxy+iy*lx1+ix]/rho;
			}
			double asnd = csound [e*nxyz+iz*lxy+iy*lx1+ix];
			double mu = vdiff[(imu-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];
			udiff = vdiff[(imu-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];// this overrides the udiff in nekasgn (line 63 in this function). Need to check withDr.Tania
			double lambda = vdiff[(ilam-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];
			// userbc
			double molarmass = molmass;
			fatface[(iwp-1)+(irho-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rho;
			fatface[(iwp-1)+(iux-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = ux;
			fatface[(iwp-1)+(iuy-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = uy;
			fatface[(iwp-1)+(iuz-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = uz;
			fatface[(iwp-1)+(iph-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = phi;

			double rhob = rho*phi;
			double rhoub = rhob*ux;
			double rhovb = rhob*uy;
			double rhowb = rhob*uz;

			fatface[(iwp-1)+(iu1-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhob;
			fatface[(iwp-1)+(iu2-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhoub;
			fatface[(iwp-1)+(iu3-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhovb;
			fatface[(iwp-1)+(iu4-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhowb;
			double mach=0;
			double snz=0;
			if(if3d){
				mach = sqrtf(ux*ux+uy*uy+uz*uz)/asnd;
				snz = unz[e*6*lx1*lz1+(iface)*lxz+l];
			}
			else{
				mach = sqrt(ux*ux+uy*uy+uz*uz)/asnd;
				snz=0;
			}
			double snx = unx[e*6*lx1*lz1+(iface)*lxz+l];
			double sny = uny[e*6*lx1*lz1+(iface)*lxz+l];


			if (mach<1.0){

				pres =  fatface[(iwm-1)+(ipr-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ];
				temp = pres/rho/(cp-cv);// ! definitely too perfect!
				fatface[(iwp-1)+(ipr-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ]   = pres;
				fatface[(iwp-1)+(isnd-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = sqrt(cp/cv*pres/rho);//check the operator precedence is same as fortran . check with Dr.Tania. adeesha
				fatface[(iwp-1)+(ithm-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = temp   ;//   ! definitely too perfect!
				fatface[(iwp-1)+(icpf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rho*cp ;//! NEED EOS WITH TEMP Dirichlet, userbc
				fatface[(iwp-1)+(icvf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rho*cv ;//! NEED EOS WITH TEMP Dirichlet, userbc
			}
			else{ //supersonic inflow

				fatface[(iwp-1)+(ipr-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ]   = pres;
				fatface[(iwp-1)+(isnd-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] =asnd;
				fatface[(iwp-1)+(ithm-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = temp   ;//   ! definitely too perfect!
				fatface[(iwp-1)+(icpf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rho*cp ;//! NEED EOS WITH TEMP Dirichlet, userbc
				fatface[(iwp-1)+(icvf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rho*cv ;//! NEED EOS WITH TEMP Dirichlet, userbc

			}

			fatface[(iwp-1)+(iu5-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = phi*rho*cv*temp+0.5/rhob*(rhoub*rhoub+rhovb*rhovb+rhowb*rhowb);



		}


	}
}


__global__ void InviscidBC_gpu_kernel2(int *lglel, double *fatface,char *cbc, double *xm1,double *ym1,double *zm1,double *vx,double *vy,double *vz,double *t,double *pr,double *sii,double *siii,double *vdiff,double *vtrans,char *cb,double *u,double *phig,double *csound,double *unx,double *uny,double *unz,double molarmass,int iwm,int iwp,int irho,int iux,int iuy,int iuz,int iph,int ipr,int isnd,int ithm,int icpf,int icvf,int iu1,int iu2,int iu3,int iu4,int iu5,int lx1,int lz1,int lxz,int ldim,int lxz2ldim,int nxyz,int lxy,int lxz2ldimlelt,int ntot,int toteq,int e_offset,int p0th,int ifield,int ltot,int icv,int icp,int imu,int ilam,double molmass,int nlel,int npscal,int if3d,int ly1,int outflsub,double pinfty,double *fatfaceiwp, int lelt){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){
		int i1 = id % lx1;
		int i2 = (id/lx1)%lz1;
		int iface= ((id/lxz)%(2*ldim));
		int e = id/lxz2ldim;

		char cb1 = cbc[ifield*lelt*18+e*18+iface*3];//corrected by Kk 02/07/2019 by adding ifield*lelt*18 and iface*3
		char cb2 = cbc[ifield*lelt*18+e*18+iface*3+1];
		char cb3 = cbc[ifield*lelt*18+e*18+iface*3+2];

		/*if(id==10){
			for(int i=0;i<576;i++){
				printf("cbc values cbc[%d]=%c \n",i,cbc[i+ifield*lelt*18]); 
			}
                        printf("debugggggg iface id %d %d\n", iface, id);
		}*/
		//		printf("cb1 =%c \n",cb1);
		if(cb1 =='O'){
			int ieg=lglel[e];
			int iy,iz,ix,l;

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
			double x = xm1[e*nxyz+iz*lxy+iy*lx1+ix];
			double y = ym1[e*nxyz+iz*lxy+iy*lx1+ix];
			double z = zm1[e*nxyz+iz*lxy+iy*lx1+ix];
			//		double r = x*x+y*y;
			//		double theta=0.0;
			//		if (r>0.0){ r = sqrtf(r);}
			//		if ( x != 0.0 || y!= 0.0){theta = atan2(y,x);   }
			//		double ux= vx[e*nxyz+iz*lxy+iy*lx1+ix];
			//		double uy= vy[e*nxyz+iz*lxy+iy*lx1+ix];
			//		double uz= vz[e*nxyz+iz*lxy+iy*lx1+ix];
			//		double temp = t [ e*nxyz+iz*lxy+iy*lx1+ix];
			//		double p0= p0th;

			//cmtasgn
			double phi = phig[e*nxyz+iz*lxy+iy*lx1+ix];
			double rho = vtrans[(irho-1)*nlel +e*nxyz+iz*lxy+iy*lx1+ix];
			double pres = pr[e*nxyz+iz*lxy+iy*lx1+ix];
			double cv=0.0,cp=0.0;
			// userbc
			double molarmass = molmass;


			double sxn = unx[e*6*lxz+(iface)*lxz+l];
			double syn = uny[e*6*lxz+(iface)*lxz+l];
			double szn = unz[e*6*lxz+(iface)*lxz+l];

			double rhou = fatface[(iwm-1)+(iu2-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ]/phi;
			double rhov = fatface[(iwm-1)+(iu3-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ]/phi;
			double rhow = fatface[(iwm-1)+(iu4-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ]/phi;
			double rhoe = fatface[(iwm-1)+(iu5-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ]/phi;
			double pl = fatface[(iwm-1)+(ipr-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ]/phi;
			fatfaceiwp[(icpf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ]= fatface[(iwm-1)+(icpf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			fatfaceiwp[(icvf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ]= fatface[(iwm-1)+(icvf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			cp=fatface[(iwm-1)+(icpf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]/rho;
			cv = fatface[(iwm-1)+(icvf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]/rho;
			int idbc=0;
			if(outflsub){
				pres=pinfty;
				idbc=1;
			}
			else{
				pres=fatface[(iwm-1)+(ipr-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
				idbc=0;
			}

			//BcondOutflowPerf function

			double rgas = 8314.3/molarmass;
			double gamma = cp/(cp - rgas);
			double gam1 = gamma-1;

			double u = rhou/rho;
			double v = rhov/rho;
			double w = rhow/rho;
			double csound= sqrtf(gamma*pl/rho);
			double mach = sqrtf(u*u+v*v+w*w)/csound;
			double rhob,rhoub,rhovb,rhowb,rhoeb;
			//subsonic flow
			if(mach<1 && idbc==1 || idbc==0){ // check the precendence of and and or in fortran check with Dr.Tania adeesha
				double rrhoc = 1.0 / (rho*csound);
				double deltp = pl - pres;
				rhob = rho - deltp/(csound*csound);
				double ub = u+sxn*deltp*rrhoc;	
				double vb = v+syn*deltp*rrhoc;	
				double wb = w+szn*deltp*rrhoc;	
				double vnd = ub*sxn + vb*syn + wb*szn;

				if(vnd<0.0){
					ub = copysignf(1.0,u)*fmax(fabs(ub),fabs(u));	
					vb = copysignf(1.0,v)*fmax(fabs(vb),fabs(v));	
					wb = copysignf(1.0,w)*fmax(fabs(wb),fabs(w));	
				}	
				rhoub = rhob*ub;
				rhovb = rhob*vb;	
				rhowb = rhob*wb;	
				rhoeb = rhob*( pres/(rhob*(gamma - 1.0)) + 0.5*(ub*ub +vb*vb + wb*wb));	

			}
			else{
				rhob= rho;
				rhoub = rhou;
				rhovb = rhov;
				rhowb =rhow;
				rhoeb = rhoe;

			}

//			printf("inviscid  iu1 %d iu2 %d iu3 %d iu4 %d iu5 %d iph %d ipr %d e %d lxz2ldimlelt %d iface %d lxz2ldim %d rhob %lf phi %lf rhoub %lf rhovb %lf rhowb %lf rhoeb %lf pres %lf l= %d \n",iu1,iu2,iu3,iu4,iu5,iph,ipr,e,lxz2ldimlelt,iface,lxz2ldim,rhob,phi,rhoub,rhovb,rhowb,rhoeb,pres,l);


			fatfaceiwp[(irho-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhob;
			fatfaceiwp[(iux-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhoub/rhob;
			fatfaceiwp[(iuy-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhovb/rhob;
			fatfaceiwp[(iuz-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhowb/rhob;
			fatfaceiwp[(ithm-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = (rhoeb-0.5*(rhoub*rhoub+rhovb*rhovb+rhowb*rhowb)/rhob)/cv;

			fatfaceiwp[(iu1-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhob*phi;
			fatfaceiwp[(iu2-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhoub*phi;
			fatfaceiwp[(iu3-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhovb*phi;
			fatfaceiwp[(iu4-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhowb*phi;
			fatfaceiwp[(iu5-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = rhoeb*phi;
			fatfaceiwp[(iph-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = phi;
			fatfaceiwp[(ipr-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = pres;
			fatfaceiwp[(isnd-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l ] = sqrtf(cp/cv*pres/rho);

		} 

	}





}



__global__ void InviscidBC_gpu_kernel3(int *lglel, double *fatface,char *cbc, double *xm1,double *ym1,double *zm1,double *vx,double *vy,double *vz,double *t,double *pr,double *sii,double *siii,double *vdiff,double *vtrans,char *cb,double *u,double *phig,double *csound,double *unx,double *uny,double *unz,double molarmass,int iwm,int iwp,int irho,int iux,int iuy,int iuz,int iph,int ipr,int isnd,int ithm,int icpf,int icvf,int iu1,int iu2,int iu3,int iu4,int iu5,int lx1,int lz1,int lxz,int ldim,int lxz2ldim,int nxyz,int lxy,int lxz2ldimlelt,int ntot,int toteq,int e_offset,int p0th,int ifield,int ltot,int icv,int icp,int imu,int ilam,double molmass,int nlel,int npscal,int if3d,int ly1,int outflsub,double pinfty, int lelt){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){
		int i1 = id % lx1;
		int l = id % lxz; //added by Kk 02/07/19
		int i2 = (id/lx1)%lz1;
		int iface= ((id/lxz)%(2*ldim));
		int e = id/lxz2ldim;

		char cb1 = cbc[ifield*lelt*18+e*18+iface*3];//corrected by Kk 02/07/2019 by adding ifield*lelt*18 and iface*3
		char cb2 = cbc[ifield*lelt*18+e*18+iface*3+1];
		char cb3 = cbc[ifield*lelt*18+e*18+iface*3+2];


		if(cb1 =='W'|| cb1=='I' || (cb1=='S' && cb2=='Y' && cb3=='M')){
			int ieg=lglel[e];

			/* following 4 line is commented out by Kk 02/07/2019 since subroutine facind is not used in CPU part
                        int iy=0;
			int iz=i2;
			int ix=i1;

			int l=lx1*iz+ix;*/ // this is the parallelized version of l = l+1 in every thread. Check with Dr.Tania . adeesha

			// ************************ e*lxz2ldim+(f-1)*lx1*lz1+l is same as id. change this later. ******


                           //printf("debug inviscidBc: lxz2ldimlelt %d, lxz2ldim %d, iface %d, l %d, fatface %.30lf , id %d, e %d, index1 %d, index2 %d\n", lxz2ldimlelt, lxz2ldim, iface, l, fatface[(iwm-1)+(irho-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l], id, e, (iwm-1)+(irho-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l, (irho-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l);
			double nx = unx[e*lxz2ldim+(iface)*lxz+l];
			double ny = uny[e*lxz2ldim+(iface)*lxz+l];
			double nz = unz[e*lxz2ldim+(iface)*lxz+l];
			double rl = fatface[(iwm-1)+(irho-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			double rr=rl;
			double ul = fatface[(iwm-1)+(iux-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			double vl = fatface[(iwm-1)+(iuy-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			double wl = fatface[(iwm-1)+(iuz-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			double fs=0.0;
			double udotn = ul*nx+vl*ny+wl*nz;
			double ur = ul-2.0*udotn*nx;
			double vr = vl-2.0*udotn*ny;
			double wr = wl-2.0*udotn*nz;

			fatface[(iwp-1)+(irho-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= rr;
			fatface[(iwp-1)+(iux-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= ur;
			fatface[(iwp-1)+(iuy-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= vr;
			fatface[(iwp-1)+(iuz-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= wr;
			fatface[(iwp-1)+(ipr-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= fatface[(iwm-1)+(ipr-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			fatface[(iwp-1)+(ithm-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= fatface[(iwm-1)+(ithm-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			fatface[(iwp-1)+(isnd-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= fatface[(iwm-1)+(isnd-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			fatface[(iwp-1)+(iph-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= fatface[(iwm-1)+(iph-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			fatface[(iwp-1)+(icvf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= fatface[(iwm-1)+(icvf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			fatface[(iwp-1)+(icpf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= fatface[(iwm-1)+(icpf-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			fatface[(iwp-1)+(iu1-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= fatface[(iwm-1)+(iu1-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
			fatface[(iwp-1)+(iu2-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= rr*ur;
			fatface[(iwp-1)+(iu3-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= rr*vr;
			fatface[(iwp-1)+(iu4-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= rr*wr;
			fatface[(iwp-1)+(iu5-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l]= fatface[(iwm-1)+(iu5-1)*lxz2ldimlelt+e*lxz2ldim+(iface)*lxz+l];
		}
	}
}



extern "C" void inviscidbc_gpu_wrapper_(int *glbblockSize2,int *d_lglel,double *d_fatface,char *d_cbc,double *d_xm1,double *d_ym1,double *d_zm1,double *d_vx,double *d_vy,double *d_vz,double *d_t,double *d_pr,double *d_sii,double *d_siii,double *d_vdiff,double *d_vtrans,char *d_cb,double *d_u,double *d_phig,double *d_csound,double *d_unx,double *d_uny,double *d_unz,double *molarmass,int *iwm,int *iwp,int *irho,int *iux,int *iuy,int *iuz,int *iph,int *ipr,int *isnd,int *ithm,int *icpf,int *icvf,int *iu1,int *iu2,int *iu3,int *iu4,int *iu5,int *lx1,int *lz1,int *toteq,int *ldim,int *nelt, int *lelt,double *p0th,int *ifield,int *icv, int *icp,int *imu,int *ilam,double *molmass,int *npscal,int *if3d,int *outflsub,double *pinfty,int *ly1){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start inviscidbc_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));


	printf("CUDA: Start inviscidbc_gpu_wrapper values molarmass =%lf ,iwm=%d,iwp=%d,irho=%d,iux=%d,iuy=%d,iuz=%d,iph=%d,ipr=%d,isnd=%d,ithm=%d,icpf=%d,icvf=%d,iu1=%d,iu2=%d,iu3=%d,iu4=%d,iu5=%d,lx1=%d,lz1=%d,itoteq=%d,ldim=%d,nelt=%d,lelt=%d,p0th=%lf,ifield=%d,icv=%d,icp=%d,imu=%d,ilam=%d,molmass=%lf,npscal=%d,if3d=%d,outflsub=%d,pinfty=%lf,ly1=%d \n",  molarmass[0],iwm[0],iwp[0],irho[0],iux[0],iuy[0],iuz[0],iph[0],ipr[0],isnd[0],ithm[0],icpf[0],icvf[0],iu1[0],iu2[0],iu3[0],iu4[0],iu5[0],lx1[0],lz1[0],toteq[0],ldim[0],nelt[0],lelt[0],p0th[0],ifield[0],icv[0],icp[0],imu[0],ilam[0],molmass[0],npscal[0],if3d[0],outflsub[0],pinfty[0],ly1[0]);

#endif
	int lxz = lx1[0]*lz1[0];
	int nxyz =lxz*ly1[0];
	int lxz2ldim=lxz*2*ldim[0];	
	int lxy = lx1[0]*ly1[0];
	int lxz2ldimlelt=lxz2ldim*nelt[0];
	int e_offset=nxyz*toteq[0];
	int nlel=nxyz*lelt[0];

	int ntot = nelt[0]*lxz2ldim;
	int  ltot = lelt[0]*lxz2ldim;
	int blockSize = 256, gridSize;
	gridSize = (int)ceil((float)ntot/blockSize);

        //inflow
	InviscidBC_gpu_kernel1<<<gridSize, blockSize>>>(d_lglel,d_fatface,d_cbc,d_xm1,d_ym1,d_zm1,d_vx,d_vy,d_vz,d_t,d_pr,d_sii,d_siii,d_vdiff,d_vtrans,d_cb,d_u,d_phig,d_csound,d_unx,d_uny,d_unz,molarmass[0],iwm[0],iwp[0],irho[0],iux[0],iuy[0],iuz[0],iph[0],ipr[0],isnd[0],ithm[0],icpf[0],icvf[0],iu1[0],iu2[0],iu3[0],iu4[0],iu5[0],lx1[0],lz1[0],lxz,ldim[0],lxz2ldim,nxyz,lxy,lxz2ldimlelt,ntot,toteq[0],e_offset,p0th[0],ifield[0],ltot, icv[0],icp[0],imu[0],ilam[0],molmass[0],nlel,npscal[0],if3d[0], ly1[0],outflsub[0], pinfty[0], lelt[0]);
#ifdef DEBUGPRINT	
	cudaDeviceSynchronize();
	code1 = cudaPeekAtLastError();
	printf("CUDA:inviscidbc_gpu_wrapper after 1 cuda status: %s\n",cudaGetErrorString(code1));
#endif

        //outflow
	InviscidBC_gpu_kernel2<<<gridSize, blockSize>>>(d_lglel,d_fatface,d_cbc,d_xm1,d_ym1,d_zm1,d_vx,d_vy,d_vz,d_t,d_pr,d_sii,d_siii,d_vdiff,d_vtrans,d_cb,d_u,d_phig,d_csound,d_unx,d_uny,d_unz,molarmass[0],iwm[0],iwp[0],irho[0],iux[0],iuy[0],iuz[0],iph[0],ipr[0],isnd[0],ithm[0],icpf[0],icvf[0],iu1[0],iu2[0],iu3[0],iu4[0],iu5[0],lx1[0],lz1[0],lxz,ldim[0],lxz2ldim,nxyz,lxy,lxz2ldimlelt,ntot,toteq[0],e_offset,p0th[0],ifield[0],ltot, icv[0],icp[0],imu[0],ilam[0],molmass[0],nlel,npscal[0],if3d[0], ly1[0],outflsub[0], pinfty[0],d_fatface+iwp[0]-1, lelt[0]);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	code1 = cudaPeekAtLastError();
	printf("CUDA:inviscidbc_gpu_wrapper after 2 cuda status: %s\n",cudaGetErrorString(code1));
#endif

        //wallbc_inviscid
	InviscidBC_gpu_kernel3<<<gridSize, blockSize>>>(d_lglel,d_fatface,d_cbc,d_xm1,d_ym1,d_zm1,d_vx,d_vy,d_vz,d_t,d_pr,d_sii,d_siii,d_vdiff,d_vtrans,d_cb,d_u,d_phig,d_csound,d_unx,d_uny,d_unz,molarmass[0],iwm[0],iwp[0],irho[0],iux[0],iuy[0],iuz[0],iph[0],ipr[0],isnd[0],ithm[0],icpf[0],icvf[0],iu1[0],iu2[0],iu3[0],iu4[0],iu5[0],lx1[0],lz1[0],lxz,ldim[0],lxz2ldim,nxyz,lxy,lxz2ldimlelt,ntot,toteq[0],e_offset,p0th[0],ifield[0],ltot, icv[0],icp[0],imu[0],ilam[0],molmass[0],nlel,npscal[0],if3d[0], ly1[0],outflsub[0], pinfty[0], lelt[0]);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End nviscidbc_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif

}


