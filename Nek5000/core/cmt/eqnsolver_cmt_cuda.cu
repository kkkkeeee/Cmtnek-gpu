#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <sys/time.h>
#include "nvml.h"
#include "cuda_helpers.h"
//#include "cuda_helpers.h"
#define DEBUGPRINT 0

__global__ void igtu_cmt_gpu_kernel1(double *flux, int nfq, int toteq,int toteqlxz2ldimlelt,int lxz2ldimlelt,int iuj,double *graduf,double *area,double *unx, double *uny, double *unz,int if3d,int ldim){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nfq){

		// int ix = id % lx1;
		// int iy = (id/lx1)%ly1;
		// int iz = (id / (lx1*ly1))%lz1;
		// int e = id/nxyz;


		for(int eq=0;eq<toteq;eq++){

			graduf[2*toteqlxz2ldimlelt+eq*lxz2ldimlelt+id]= flux[(iuj-1)+eq*lxz2ldimlelt+id]*area[id];
			graduf[0*toteqlxz2ldimlelt+eq*lxz2ldimlelt+id]= graduf[2*toteqlxz2ldimlelt+eq*lxz2ldimlelt+id]*unx[id];
			graduf[1*toteqlxz2ldimlelt+eq*lxz2ldimlelt+id]= graduf[2*toteqlxz2ldimlelt+eq*lxz2ldimlelt+id]*uny[id];


			if(if3d){
				graduf[2*toteqlxz2ldimlelt+eq*lxz2ldimlelt+id]=graduf[2*toteqlxz2ldimlelt+eq*lxz2ldimlelt+id]*unz[id];

			}
		}
	}
}

__global__ void igtu_cmt_gpu_kernel2(double *gradu,double *graduf, int eq, int nfq,int *iface_flux,int toteqlxyzlelt,int toteqlxz2ldimlelt,int lxyzlelt,int nf,int ldim,int toteq,int lxyz,int lxz2ldimlelt){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nfq){

		int e = id/nf;

		for(int j=0;j<ldim;j++){
			for(int eq2=0;eq2<toteq;eq2++){
				//add_face2full_cmt(nel,nx,ny,nz,iface,vols,faces)
				//something is wrong with the original functions. gradu and graduf does not consider e. lot of overrides in the for loops vols(i,1,1,ie). check with Dr.Tania. adeesha  . gradu has nelt but not in original.

				int newi = iface_flux[id];
				gradu[j*toteqlxyzlelt+eq2*lxyzlelt+e*lxyz+newi ] =  gradu[j*toteqlxyzlelt+eq2*lxyzlelt+e*lxyz+newi ]  + graduf[j*toteqlxz2ldimlelt+eq2*lxz2ldimlelt+id]; 
			}



		}


	}

}

// igtu_cmt_gpu_kernel3<<<gridSize3, blockSize1>>>(d_diffh,d_gradu, d_vtrans,d_vdiff,d_vx,d_vy,d_vz,d_u,d_viscscr,d_superhugeh, d_ur, d_us,d_ut, d_jacmi,d_rxm1,d_rym1, d_rzm1,d_sxm1, d_sym1,d_szm1,d_txm1, d_tym1,d_tzm1, lx1[0],ly1[0],lz1[0], lxy,nxyz,toteq[0], nnel,lxyzlelt,toteqlxyz, toteqlxyzlelt, irho[0], ilam[0],imu[0],icv[0], iknd[0],inus[0],eq,if3d[0],ldim[0] );



__global__ void igtu_cmt_gpu_kernel3(double *diffh,double *gradu, double *vtrans, double *vdiff, double * vx, double *vy, double *vz, double *u, double *viscscr, double *superhugeh, double *ur, double *us, double *ut, double *jacmi, double *rxm1,double *rym1, double *rzm1, double *sxm1, double *sym1, double *szm1,double *txm1, double *tym1, double *tzm1, int lx1,int ly1,int lz1, int lxy, int nxyz,  int toteq, int nnel, int lxyzlelt, int toteqlxyz, int toteqlxyzlelt, int irho, int ilam, int imu, int icv, int iknd, int inus,int eq,int  if3d,int ldim){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){

		int ix = id % lx1;
		int iy = (id/lx1)%ly1;
		int iz = (id / (lxy))%lz1;
		int e = id/nxyz;

		//agradu(diffh,gradu,e,eq) // flux =diffh change later. adeesha
		// call fluxj_ns (flux,du,e,eq)
		if(eq < toteq-1){
			if(eq==1){
				//A21kldUldxk(flux(1,1),gradu,e)
				double dU1x = gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
				double  dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
				double  dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
				double  rho   =vtrans[(irho-1)*lxyzlelt+id];
				double  lambda=vdiff[(ilam-1)*lxyzlelt+id];
				double  mu    =vdiff[(imu-1)*lxyzlelt+id];
				double  u1    =vx[id];
				double  u2    =vy[id];
				double  u3    =vz[id];
				double  lambdamu=lambda+2.0*mu;
				diffh[0*lxyzlelt+id]=(lambda*(dU4z+dU3y-u3*dU1z-u2*dU1y)+lambdamu*(dU2x-u1*dU1x))/rho;

				//A22kldUldxk(flux(1,2),gradu,e)
				dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double   dU3x=gradu[0*toteqlxyzlelt+2*lxyzlelt+id ];
				dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double   dU2y=gradu[1*toteqlxyzlelt+1*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];
				u1    =vx[id];
				u2    =vy[id];
				diffh[1*lxyzlelt+id]=mu*(dU3x+dU2y-u1*dU1y-u2*dU1x)/rho;

				//A23kldUldxk(flux(1,3),gradu,e)

				dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU4x=gradu[0*toteqlxyzlelt+3*lxyzlelt+id ];
				dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU2z=gradu[2*toteqlxyzlelt+1*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];
				u1    =vx[id];
				u3    =vz[id];
				diffh[2*lxyzlelt+id]=mu*(dU4x+dU2z-u1*dU1z-u3*dU1x)/rho;

			}
			else if(eq==2){
				//A31kldUldxk(flux(1,1),gradu,e)
				double dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU3x=gradu[0*toteqlxyzlelt+2*lxyzlelt+id ];
				double dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU2y=gradu[1*toteqlxyzlelt+1*lxyzlelt+id ];
				double rho   =vtrans[(irho-1)*lxyzlelt+id];
				double mu    =vdiff[(imu-1)*lxyzlelt+id];
				double u1    =vx[id];
				double u2    =vy[id];
				diffh[0*lxyzlelt+id] = mu*(dU3x+dU2y-u1*dU1y-u2*dU1x)/rho;

				//A32kldUldxk(flux(1,2),gradu,e)
				dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
				dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
				double dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double   dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];;
				double lambda=vdiff[(ilam-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];
				u1    =vx[id];
				u2    =vy[id];
				double u3    =vz[id];
				double lambdamu=lambda+2.0*mu;
				diffh[1*lxyzlelt+id]=(lambda*(dU4z+dU2x-u3*dU1z-u1*dU1x)+lambdamu*(dU3y-u2*dU1y))/rho;

				//A33kldUldxk(flux(1,3),gradu,e)
				dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU4y=gradu[1*toteqlxyzlelt+3*lxyzlelt+id ];
				dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double   dU3z=gradu[2*toteqlxyzlelt+2*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];
				u2    =vy[id];
				u3    =vz[id];
				diffh[2*lxyzlelt+id]=mu*(dU4y+dU3z-u2*dU1z-u3*dU1y)/rho;	



			}
			else if(eq==3){
				//A41kldUldxk(flux(1,1),gradu,e)
				double dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU4x=gradu[0*toteqlxyzlelt+3*lxyzlelt+id ];
				double dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU2z=gradu[2*toteqlxyzlelt+1*lxyzlelt+id ];
				double rho   =vtrans[(irho-1)*lxyzlelt+id];
				double mu    =vdiff[(imu-1)*lxyzlelt+id];

				double u1    =vx[id];
				double u3    =vz[id];
				diffh[0*lxyzlelt+id]=mu*(dU4x+dU2z-u1*dU1z-u3*dU1x)/rho;

				//A42kldUldxk(flux(1,2),gradu,e)
				double dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU4y=gradu[1*toteqlxyzlelt+3*lxyzlelt+id ];
				dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU3z=gradu[2*toteqlxyzlelt+2*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];

				double u2    =vy[id];
				u3    =vz[id];
				diffh[1*lxyzlelt+id]=mu*(dU4y+dU3z-u2*dU1z-u3*dU1y)/rho;

				//A43kldUldxk(flux(1,3),gradu,e)
				dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
				dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
				dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];
				double lambda=vdiff[(ilam-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];
				u1    =vx[id];
				u2    =vy[id];
				u3    =vz[id];
				double lambdamu=lambda+2.0*mu;
				diffh[2*lxyzlelt+id]=(lambda*(dU3y+dU2x-u2*dU1y-u1*dU1x)+lambdamu*(dU4z-u3*dU1z))/rho;	
			}

		}
		else{
			if(if3d){
				//a53kldUldxk(flux(1,3),gradu,e)
				double dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
				double  dU3x=gradu[0*toteqlxyzlelt+2*lxyzlelt+id ];
				double  dU4x=gradu[0*toteqlxyzlelt+3*lxyzlelt+id ];
				double  dU5x=gradu[0*toteqlxyzlelt+4*lxyzlelt+id ];
				double  dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU2y=gradu[1*toteqlxyzlelt+1*lxyzlelt+id ];
				double  dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
				double  dU4y=gradu[1*toteqlxyzlelt+3*lxyzlelt+id ];
				double  dU5y=gradu[1*toteqlxyzlelt+4*lxyzlelt+id ];
				double dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU2z=gradu[2*toteqlxyzlelt+1*lxyzlelt+id ];
				double  dU3z=gradu[2*toteqlxyzlelt+2*lxyzlelt+id ];
				double  dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
				double  dU5z=gradu[2*toteqlxyzlelt+4*lxyzlelt+id ];
				double  rho   =vtrans[(irho-1)*lxyzlelt+id];
				double  cv    =vtrans[(icv-1)*lxyzlelt+id]/rho;
				double  lambda=vdiff[(ilam-1)*lxyzlelt+id];
				double  mu    =vdiff[(imu-1)*lxyzlelt+id];
				double K     =vdiff[(iknd-1)*lxyzlelt+id];;
				double  u1    =vx[id];
				double  u2    =vy[id];
				double  u3    =vz[id];
				double  E     =u[e*toteqlxyz+(toteq-1)*nxyz+iz*lxy+iy*lx1+ix]/rho;
				double lambdamu=lambda+mu;
				double kmcvmu=K-cv*mu;
				diffh[2*lxyzlelt+id]=(K*(dU5z-E*dU1z)+cv*u3*(lambda*dU4z+2*mu*dU4z+lambda*dU3y+lambda*dU2x)-K*u3*dU4z+cv*mu*u2*(dU4y+dU3z)+cv*mu*u1*(dU4x+dU2z)-K*u2*dU3z-K*u1*dU2z-cv*(lambda+2*mu)*u3*u3*dU1z+K*u3*u3*dU1z+ K*u2*u2*dU1z-cv*mu*u2*u2*dU1z+K*u1*u1*dU1z-cv*mu*u1*u1*dU1z-cv*(lambda+mu)*u2*u3*dU1y-cv*(lambda+mu)*u1*u3*dU1x)/(cv*rho);

			}
			else{
				for(int kfortoteq=0;kfortoteq<toteq;kfortoteq++){
					gradu[2*toteqlxyzlelt+kfortoteq*lxyzlelt+id]=0;
				}
				vz[id]=0;

			}


		}
		//a51kldUldxk(flux(1,1),gradu,e)
		double dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
		double dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
		double  dU3x=gradu[0*toteqlxyzlelt+2*lxyzlelt+id ];
		double  dU4x=gradu[0*toteqlxyzlelt+3*lxyzlelt+id ];
		double  dU5x=gradu[0*toteqlxyzlelt+4*lxyzlelt+id ];
		double  dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
		double  dU2y=gradu[1*toteqlxyzlelt+1*lxyzlelt+id ];
		double  dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
		double  dU4y=gradu[1*toteqlxyzlelt+3*lxyzlelt+id ];
		double  dU5y=gradu[1*toteq*toteqlxyzlelt+4*lxyzlelt+id ];
		double dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
		double  dU2z=gradu[2*toteqlxyzlelt+1*lxyzlelt+id ];
		double  dU3z=gradu[2*toteqlxyzlelt+2*lxyzlelt+id ];
		double  dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
		double  dU5z=gradu[2*toteqlxyzlelt+4*lxyzlelt+id ];
		double  rho   =vtrans[(irho-1)*lxyzlelt+id];
		double  cv    =vtrans[(icv-1)*lxyzlelt+id]/rho;
		double  lambda=vdiff[(ilam-1)*lxyzlelt+id];
		double  mu    =vdiff[(imu-1)*lxyzlelt+id];
		double K     =vdiff[(iknd-1)*lxyzlelt+id];;
		double  u1    =vx[id];
		double  u2    =vy[id];
		double  u3    =vz[id];
		double  E     =u[e*toteqlxyz+(toteq-1)*nxyz+iz*lxy+iy*lx1+ix]/rho;
		double lambdamu=lambda+mu;
		double kmcvmu=K-cv*mu;
		diffh[0*lxyzlelt+id]=(K*dU5x+cv*lambda*u1*dU4z-kmcvmu*u3*dU4x+cv*lambda*u1*dU3y-kmcvmu*u2*dU3x+cv*mu*u3*dU2z+cv*mu*u2*dU2y+(cv*lambda-K+2*cv*mu)*u1*dU2x-cv*lambdamu*u1*u3*dU1z-cv*lambdamu*u1*u2*dU1y+(K*u3*u3-cv*mu*u3*u3+K*u2*u2-cv*mu*u2*u2-cv*lambda*u1*u1+K*u1*u1-2*cv*mu*u1*u1-E*K)*dU1x)/(cv*rho);

		//a52kldUldxk(flux(1,2),gradu,e)
		dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
		dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
		dU3x=gradu[0*toteqlxyzlelt+2*lxyzlelt+id ];
		dU4x=gradu[0*toteqlxyzlelt+3*lxyzlelt+id ];
		dU5x=gradu[0*toteqlxyzlelt+4*lxyzlelt+id ];
		dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
		dU2y=gradu[1*toteqlxyzlelt+1*lxyzlelt+id ];
		dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
		dU4y=gradu[1*toteqlxyzlelt+3*lxyzlelt+id ];
		dU5y=gradu[1*toteqlxyzlelt+4*lxyzlelt+id ];
		dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
		dU2z=gradu[2*toteqlxyzlelt+1*lxyzlelt+id ];
		dU3z=gradu[2*toteqlxyzlelt+2*lxyzlelt+id ];
		dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
		dU5z=gradu[2*toteqlxyzlelt+4*lxyzlelt+id ];
		rho   =vtrans[(irho-1)*lxyzlelt+id];
		cv    =vtrans[(icv-1)*lxyzlelt+id]/rho;
		lambda=vdiff[(ilam-1)*lxyzlelt+id];
		mu    =vdiff[(imu-1)*lxyzlelt+id];
		K     =vdiff[(iknd-1)*lxyzlelt+id];;
		u1    =vx[id];
		u2    =vy[id];
		u3    =vz[id];
		E     =u[e*toteqlxyz+(toteq-1)*nxyz+iz*lxy+iy*lx1+ix]/rho;
		lambdamu=lambda+mu;
		kmcvmu=K-cv*mu;
		diffh[1*lxyzlelt+id]=(K*dU5y+cv*lambda*u2*dU4z-kmcvmu*u3*dU4y+cv*mu*u3*dU3z+(cv*lambda-K+2*cv*mu)*u2*dU3y+cv*mu*u1*dU3x-kmcvmu*u1*dU2y+cv*lambda*u2*dU2x-cv*lambdamu*u2*u3*dU1z+(K*u3*u3-cv*mu*u3*u3-cv*lambda*u2*u2+K*u2*u2-2*cv*mu*u2*u2+K*u1*u1-cv*mu*u1*u1-E*K)*dU1y-cv*lambdamu*u1*u2*dU1x)/(cv*rho);


		//call fluxj_evm(flux,du,e,eq)

		if(eq==0){
			for(int jj=0;jj<ldim;jj++){
				diffh[jj*lxyzlelt+id]=  diffh[jj*lxyzlelt+id]+vdiff[(inus-1)*lxyzlelt+id]*gradu[jj*toteqlxyzlelt+id];
			}

		}
		else{
			if(eq<toteq-1){
				viscscr[id]=gradu[0*toteqlxyzlelt+(eq-1)*lxyzlelt+id ]; // problem with du indices. du(1,1,eq-1) third is for ldim check wih Dr.Tania adeesha.
				viscscr[id]=viscscr[id]*vdiff[(inus-1)*lxyzlelt+id];
				diffh[0*lxyzlelt+id]=  diffh[0*lxyzlelt+id]+viscscr[id]*vx[id]; 
				diffh[1*lxyzlelt+id]=  diffh[1*lxyzlelt+id]+viscscr[id]*vy[id];
				if(if3d){
					diffh[2*lxyzlelt+id]=  diffh[2*lxyzlelt+id]+viscscr[id]*vz[id];
				} 

			}
			else{
				if(if3d){
					viscscr[id] = vx[id]*vx[id]+vy[id]*vy[id]+vz[id]*vz[id];
				}
				else{
					viscscr[id] = vx[id]*vx[id]+vy[id]*vy[id];
				}
				viscscr[id] =  viscscr[id] *vdiff[(inus-1)*lxyzlelt+id];
				for(int jj=0;jj<ldim;jj++){
					diffh[jj*lxyzlelt+id]=  diffh[jj*lxyzlelt+id]+viscscr[id] *gradu[jj*toteqlxyzlelt+id];
				}
				for(int jj=0;jj<ldim;jj++){
					for(int eq2=1;eq2<ldim+1;eq2++){
						viscscr[id]=gradu[jj*toteqlxyzlelt+eq2*lxyzlelt+id ]* u[e*toteqlxyz+eq2*nxyz+iz*lxy+iy*lx1+ix]+vdiff[(inus-1)*lxyzlelt+id] ;
						viscscr[id]=viscscr[id]/vtrans[(irho-1)*lxyzlelt+id];	
						diffh[jj*lxyzlelt+id]=diffh[jj*lxyzlelt+id]-viscscr[id];
						diffh[jj*lxyzlelt+id]=  diffh[jj*lxyzlelt+id]+vdiff[(inus-1)*lxyzlelt+id] *gradu[jj*toteqlxyzlelt+(toteq-1)*lxyzlelt+id ];
					}

				} 



			}
		}
		//end of agradu
		for(int j=0;j<ldim;j++){
			superhugeh[j*lxyzlelt+id] = diffh[j*lxyzlelt+id];
		}

		if(if3d){
			ur[id] =  jacmi[id] *( rxm1[id]*superhugeh[0*lxyzlelt+id]+ rym1[id]*superhugeh[1*lxyzlelt+id]+rzm1[id]*superhugeh[2*lxyzlelt+id]);
			us[id] =  jacmi[id] *( sxm1[id]*superhugeh[0*lxyzlelt+id]+ sym1[id]*superhugeh[1*lxyzlelt+id]+szm1[id]*superhugeh[2*lxyzlelt+id]);
			ut[id] =  jacmi[id] *( txm1[id]*superhugeh[0*lxyzlelt+id]+ tym1[id]*superhugeh[1*lxyzlelt+id]+tzm1[id]*superhugeh[2*lxyzlelt+id]);

		}
		else{
			ur[id] =  jacmi[id] *( rxm1[id]*superhugeh[0*lxyzlelt+id]+ rym1[id]*superhugeh[1*lxyzlelt+id]);
			us[id] =  jacmi[id] *( sxm1[id]*superhugeh[0*lxyzlelt+id]+sym1[id]*superhugeh[1*lxyzlelt+id]);
		}





	}

}

__global__ void igtu_cmt_gpu_kernel4(double *res1, double *gradm1_t_overwrites, int nnel, int lx1, int ly1, int lz1,int lxy, int nxyz, double consta, int eq, int lxyzlelt){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){

		int ix = id % lx1;
		int iy = (id/lx1)%ly1;
		int iz = (id / (lxy))%lz1;
		int e = id/nxyz;
		gradm1_t_overwrites[id]= gradm1_t_overwrites[id]*consta;
		res1[eq*lxyzlelt+id]  = res1[eq*lxyzlelt+id] +gradm1_t_overwrites[id];
	}
}
/*void gpu_local_grad3_t(double *u, double *ur, double *us, double *ut, int nxd, double *d, double *dt, double *w, int nel){

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

for(int k = 0; k<nxd;k++){
//wk(nxd,nxd) = usk(nxd,nxd)*D(nxd,nxd) fortran
//wk(nxd,nxd) = D(nxd,nxd)*usk(nxd,nxd) C
gridSize = (int)ceil((float)nel*nxd_2/blockSize);
//mxm<<<gridSize, blockSize>>>(d,nxd, us+k*nxd_2, nxd, w+k*nxd_2, nxd, nel, 0, nxd_3, nxd_3, 0);
cuda_multi_gemm_unif(stream, 'N', 'N', nxd, nxd, nxd, &alpha, us+k*nxd_2, nxd, nxd_3, d, nxd, 0, &beta, w+k*nxd_2, nxd, nxd_3, nel, gridSize);


}
gridSize = (int)ceil((float)nel*nxd_3/blockSize);
nekadd2<<<gridSize, blockSize>>>(u,w, nel*nxd_3);
//w(nxd*nxd,nxd) = ut(nxd*nxd,nxd) * D(nxd,nxd) fortran
//w(nxd,nxd*nxd) = D(nxd,nxd) * ut(nxd,nxd*nxd) C
//mxm<<<gridSize, blockSize>>>(d,nxd, ut, nxd, w, nxd_2, nel, 0, nxd_3, nxd_3, 0);
cuda_multi_gemm_unif(stream, 'N', 'N', nxd_2, nxd, nxd, &alpha, ut, nxd, nxd_3, d, nxd, 0, &beta, w, nxd_2, nxd_3, nel, gridSize);

nekadd2<<<gridSize, blockSize>>>(u,w, nel*nxd_3);
cudaStreamDestroy(stream);



}

void gpu_local_grad2_t(double *u, double *ur, double *us, double *ut, int nxd, double *d, double *dt, double *w, int nel){

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

gridSize = (int)ceil((float)nel*nxd_3/blockSize);
//w(nxd*nxd,nxd) = ut(nxd*nxd,nxd) * D(nxd,nxd) fortran
//w(nxd,nxd*nxd) = D(nxd,nxd) * ut(nxd,nxd*nxd) C
//mxm<<<gridSize, blockSize>>>(d,nxd, ut, nxd, w, nxd_2, nel, 0, nxd_3, nxd_3, 0);
cuda_multi_gemm_unif(stream, 'N', 'N', nxd_2, nxd, nxd, &alpha, ut, nxd, nxd_3, d, nxd, 0, &beta, w, nxd_2, nxd_3, nel, gridSize);

nekadd2<<<gridSize, blockSize>>>(u,w, nel*nxd_3);
cudaStreamDestroy(stream);


}

 */


extern "C" void igtu_cmt_gpu_wrapper_(int *glbblockSize1,int *glbblockSize2,double *d_flux,double *d_gradu,double *d_graduf, int *d_iface_flux,double *d_diffh,double *d_vtrans,double *d_vdiff,double *d_vx,double *d_vy,double *d_vz,double *d_u,double *d_viscscr, double *d_jacmi,double *d_rxm1,double *d_rym1, double *d_rzm1,double *d_sxm1, double *d_sym1,double *d_szm1,double *d_txm1, double *d_tym1,double *d_tzm1,double *d_dxm1,double *d_dxtm1,double *d_res1,int *toteq,int *iuj,int *lx1,int *ly1,int *lz1,int *irho, int *ilam,int *imu,int *icv, int *iknd,int *inus,int *nelt,int *lelt, int *ldim,int *ifsip,double *d_area,double *d_unx,double *d_uny,double *d_unz,int *if3d){


	int nxz = lx1[0]*lz1[0];
	int nfaces = 2*ldim[0];
	int nf = nxz*nfaces;// ! 1 element's face points
	int nfq = nf*nelt[0];// ! all points in a pile of faces
	int nnel = lx1[0]*ly1[0]*lz1[0]*nelt[0];
	int nlel = lx1[0]*ly1[0]*lz1[0]*lelt[0];
	int lxy=lx1[0]*ly1[0];
	int lxyzlelt=nlel;

	int nxyz  =lx1[0]*ly1[0]*lz1[0];
	int toteqlxyz=toteq[0]*nxyz;
	int nvol  =nxyz*nelt[0];
	int ngradu=nxyz*toteq[0]*3;

	int lxz2ldimlelt=nf*lelt[0];
	int toteqlxz2ldimlelt= toteq[0]*lxz2ldimlelt;
	int toteqlxyzlelt= toteq[0]*nlel;
	double consta;
	if (ifsip[0]){
		consta=-1.0;// ! SIP
	}
	else{
		consta=1.0;// ! Baumann-Oden
	}



	int blockSize1 = glbblockSize1[0], blockSize2= glbblockSize2[0],gridSize1,gridSize2,gridSize3;
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();

	cudaError_t code1 = cudaPeekAtLastError();
	//        if (code1 != cudaSuccess){
	printf("CUDA: Start igtu_cmt_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
	printf("CUDA: Start igtu_cmt_gpu_wrapper values toteq=%d,iuj=%d,lx1=%d,ly1=%d,lz1=%d,irho=%d,ilam=%d,imu=%d,icv=%d,iknd=%d,inus=%d,nelt=%d,lelt=%d,ldim=%d,ifsip=%d,if3d=%d \n",toteq[0],iuj[0],lx1[0],ly1[0],lz1[0],irho[0],ilam[0],imu[0],icv[0],iknd[0],inus[0],nelt[0],lelt[0],ldim[0],ifsip[0],if3d[0]);
#endif
	//      }

	gridSize1 = (int)ceil((float)nfq/blockSize2);
	igtu_cmt_gpu_kernel1<<<gridSize1, blockSize2>>>(d_flux, nfq, toteq[0],toteqlxz2ldimlelt,lxz2ldimlelt,iuj[0],d_graduf,d_area,d_unx,d_uny,d_unz,if3d[0],ldim[0]);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	code1 = cudaPeekAtLastError();
	printf("CUDA: igtu_cmt_gpu_wrapper after kernel1 cuda status: %s\n",cudaGetErrorString(code1));
#endif

	double *d_superhugeh;
	double *d_gradm1_t_overwrites;
	double *d_ur;
	double *d_us;
	double *d_ut;
	double *d_tmp;

	cudaMalloc((void**)&d_gradm1_t_overwrites,nlel*sizeof(double));
	cudaMalloc((void**)&d_superhugeh,nlel*3*sizeof(double));
	cudaMalloc((void**)&d_us,nlel*sizeof(double));
	cudaMalloc((void**)&d_ut,nlel*sizeof(double));
	cudaMalloc((void**)&d_ur,nlel*sizeof(double));
	cudaMalloc((void**)&d_tmp,nlel*sizeof(double));
	for(int eq=0;eq<toteq[0];eq++){

		cudaMemset(d_superhugeh,0.0, nlel*3*sizeof(double));
		if (eq == 3 && !if3d){}
		else{
			cudaMemset(d_gradu,0.0, toteq[0]*3*nlel*sizeof(double));
			cudaMemset(d_diffh,0.0, nlel*3*sizeof(double));
			cudaMemset(d_ur, 0.0, nlel*sizeof(double));
			cudaMemset(d_us, 0.0, nlel*sizeof(double));
			cudaMemset(d_ut, 0.0, nlel*sizeof(double));
			cudaMemset(d_tmp, 0.0, nlel*sizeof(double));




			gridSize2 = (int)ceil((float)nfq/blockSize2);
			igtu_cmt_gpu_kernel2<<<gridSize2, blockSize2>>>(d_gradu,d_graduf,eq, nfq,d_iface_flux,toteqlxyzlelt,toteqlxz2ldimlelt,lxyzlelt,nf,ldim[0],toteq[0],nxyz,lxz2ldimlelt);


#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: igtu_cmt_gpu_wrapper after kernel2 cuda status: %s\n",cudaGetErrorString(code1));

#endif

			gridSize3 = (int)ceil((float)nnel/blockSize1);
			igtu_cmt_gpu_kernel3<<<gridSize3, blockSize1>>>(d_diffh,d_gradu, d_vtrans,d_vdiff,d_vx,d_vy,d_vz,d_u,d_viscscr,d_superhugeh, d_ur, d_us,d_ut, d_jacmi,d_rxm1,d_rym1, d_rzm1,d_sxm1, d_sym1,d_szm1,d_txm1, d_tym1,d_tzm1, lx1[0],ly1[0],lz1[0], lxy,nxyz,toteq[0], nnel,lxyzlelt,toteqlxyz, toteqlxyzlelt, irho[0], ilam[0],imu[0],icv[0], iknd[0],inus[0],eq,if3d[0],ldim[0] );

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: igtu_cmt_gpu_wrapper after kernel3 cuda status: %s\n",cudaGetErrorString(code1));
#endif
			//computation of ur us ut are in the kernel3
			//gradm1_t
			if(if3d){
				gpu_local_grad3_t(d_gradm1_t_overwrites, d_ur, d_us,d_ut,lx1[0],d_dxm1,d_dxtm1, d_tmp, nelt[0]);		
			}
			else{
				gpu_local_grad2_t(d_gradm1_t_overwrites, d_ur, d_us,lx1[0],d_dxm1,d_dxtm1, d_tmp, nelt[0]);		

			}
#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: igtu_cmt_gpu_wrapper after local_grad cuda status: %s\n",cudaGetErrorString(code1));
#endif

			gridSize3 = (int)ceil((float)nnel/blockSize1);
			igtu_cmt_gpu_kernel4<<<gridSize3, blockSize1>>>(d_res1, d_gradm1_t_overwrites,  nnel, lx1[0], ly1[0],lz1[0],lxy,nxyz, consta, eq, lxyzlelt);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: igtu_cmt_gpu_wrapper after kernel4 cuda status: %s\n",cudaGetErrorString(code1));
#endif

		}	
	}
	cudaFree(d_ur);
	cudaFree(d_us);
	cudaFree(d_ut);
	cudaFree(d_tmp);
	cudaFree(d_superhugeh);
	cudaFree(d_gradm1_t_overwrites);

#ifdef DEBUGPRINT
	cudaError_t code2 = cudaPeekAtLastError();
	//if (code2 != cudaSuccess){
	printf("End igtu_cmt_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
	// }



}
__global__ void cmtusrf_gpu_kernel(double *usrf,double *xm1,double *ym1,double *zm1,double *vx,double *vy,double *vz,double *t,double *pr,double *sii,double *siii,double *vdiff,double *vtrans,char *cb,double *ptw,int *lglel,int *gllel,double *rhs_fluidp,double *u,double *phig,int nnel,int lx1,int ly1,int lz1,int lxy,int nxyz,int lxyzlelt,int toteqlxyz,int  istep,int  npscal,int two_way,int time_delay,int icmtp,int nlel,int p0th,int ifield){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){

		int ix = id % lx1;
		int iy = (id/lx1)%ly1;
		int iz = (id / (lxy))%lz1;
		int e = id/nxyz;

		if(istep==0&&e==0){
			usrf[5*(iz*(lxy)+iy*lx1+ix)+0]=0;
			usrf[5*(iz*(lxy)+iy*lx1+ix)+1]=0;
			usrf[5*(iz*(lxy)+iy*lx1+ix)+2]=0;
			usrf[5*(iz*(lxy)+iy*lx1+ix)+3]=0;
			usrf[5*(iz*(lxy)+iy*lx1+ix)+4]=0;
		}

		int eg=lglel[e];
		//nek assign
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
		double ps[10]; // ps is size of ldimt which is 3. Not sure npscal is also 3. Need to check with Dr.Tania
		for (ips=0;ips<npscal;ips++){
			ps[ips]=t[(ips+1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix]; // 5 th dimension of t is idlmt which is 3. Not sure how the  nekasgn access ips+1. Need to check with Dr.Tania
		}
		double pa = pr [e*nxyz+iz*lxy+iy*lx1+ix];
		double p0= p0th;
		double si2 =  sii[e*nxyz+iz*lxy+iy*lx1+ix];
		double si3 =  siii[e*nxyz+iz*lxy+iy*lx1+ix];
		double udiff =  vdiff[(ifield-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];
		double utrans =  vtrans[(ifield-1)*nlel+e*nxyz+iz*lxy+iy*lx1+ix];
		char cbu1 = cb[0];
		char cbu2 = cb[1];
		char cbu3 = cb[2];
		//userf(i,j,k,eg)
		int egg=gllel[eg];// use glleg twice. check with Dr.Tania to find the real effect. If it reverses can use just id for the following arrays indices. adeesha.
		double ffx,ffy,ffz,qvol;  //actually these things should copy back to the cpu varibales. adeesha.
		if (two_way >=2) {
			if (istep > time_delay) {
				ffx =  ptw[eg*nxyz+iz*lxy+iy*lx1+ix]/vtrans[eg*nxyz+iz*lxy+iy*lx1+ix] /(1.0-ptw[3*lxyzlelt+eg*nxyz+iz*lx1*ly1+iy*lx1+ix]);
				ffy =  ptw[1*lxyzlelt+eg*lx1*nxyz+iz*lxy+iy*lx1+ix]/vtrans[eg*nxyz+iz*lxy+iy*lx1+ix] /(1.0-ptw[3*lxyzlelt+eg*nxyz+iz*lxy+iy*lx1+ix]);
				ffz =  ptw[2*lxyzlelt+eg*nxyz+iz*lx1*ly1+iy*lx1+ix]/vtrans[eg*nxyz+iz*lxy+iy*lx1+ix] /(1.0-ptw[3*lxyzlelt+eg*nxyz+iz*lxy+iy*lx1+ix]);
				if (icmtp == 1){
					qvol= ptw[4*lxyzlelt+eg*nxyz+iz*lx1*ly1+iy*lx1+ix] + rhs_fluidp[4*lxyzlelt+eg*nxyz+iz*lxy+iy*lx1+ix];
				}
				else{
					qvol=0.0;
				}
			}
			else{
				ffx = 0.0;
				ffy = 0.0;
				ffz = 0.0;
			}

		}
		else{
			ffx = 0.0;
			ffy = 0.0;
			ffz = 0.0;

		}			

		usrf[1*nxyz+iz*(lxy)+iy*lx1+ix] = ffx*u[e*toteqlxyz+iz*(lxy)+iy*lx1+ix]*phig[id];
		usrf[2*nxyz+iz*(lxy)+iy*lx1+ix] = ffy*u[e*toteqlxyz+iz*(lxy)+iy*lx1+ix]*phig[id];
		usrf[3*nxyz+iz*(lxy)+iy*lx1+ix] = ffz*u[e*toteqlxyz+iz*(lxy)+iy*lx1+ix]*phig[id];
		usrf[4*nxyz+iz*(lxy)+iy*lx1+ix] = qvol;




	}
}

extern "C" void cmtusrf_gpu_wrapper_(int *glbblockSize1,double *d_usrf,double *d_xm1,double *d_ym1,double *d_zm1,double *d_vx,double *d_vy,double *d_vz,double *d_t,double *d_pr,double *d_sii,double *d_siii,double *d_vdiff,double *d_vtrans,char *d_cb,double *d_ptw,int *d_lglel,int *d_gllel,double *d_rhs_fluidp,double *d_u,double *d_phig,int *lx1,int *ly1,int *lz1,int *toteq,int  *istep,int  *npscal,int *two_way,int *time_delay,int *icmtp,int *nelt, int *lelt,double *p0th,int *ifield){

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start cmtusrf_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start cmtusrf_gpu_wrapper values lx1=%d,ly1=%d,lz1=%d,toteq=%d,istep=%d,npscal=%d,two_way=%d,time_delay=%d,icmtp=%d,nelt=%d,lelt=%d,p0th=%lf,ifield=%d\n",lx1[0],ly1[0],lz1[0],toteq[0],istep[0],npscal[0],two_way[0],time_delay[0],icmtp[0],nelt[0],lelt[0],p0th[0],ifield[0]);

#endif
	int lxy = lx1[0]*ly1[0];
	int nxyz = lxy*lz1[0];
	int lxyzlelt = nxyz*lelt[0];
	int nnel = nxyz*nelt[0];
	int nlel=nxyz*lelt[0];
	int toteqlxyz=toteq[0]*nxyz;

	int blockSize = glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)nnel/blockSize);
	cmtusrf_gpu_kernel<<<gridSize, blockSize>>>(d_usrf,d_xm1,d_ym1,d_zm1,d_vx,d_vy,d_vz,d_t,d_pr,d_sii,d_siii,d_vdiff,d_vtrans,d_cb,d_ptw,d_lglel,d_gllel,d_rhs_fluidp,d_u,d_phig,nnel,lx1[0],ly1[0],lz1[0],lxy,nxyz,lxyzlelt,toteqlxyz,istep[0],npscal[0],two_way[0],time_delay[0],icmtp[0],nlel,p0th[0],ifield[0]);

#ifdef DEBUGPRINT
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End cmtusrf_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
}

__global__ void compute_gradients_gpu_kernel1(double *ud, double *u, double *phig, int nnel,int lx1,int ly1,int lz1, int lxy,int nxyz,int toteqlxyz,int eq){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){
		int ix = id % lx1;
		int iy = (id/lx1)%ly1;
		int iz = (id / (lxy))%lz1;
		int e = id/nxyz;

		ud[id] = u[e*toteqlxyz+eq*nxyz+iz*lxy+iy*lx1+ix]/phig[id];

	}
}

__global__ void compute_gradients_gpu_kernel2(double *ur,double *us,double *ut,double *gradu, double *jacmi,double *rxm1,double *rym1, double *rzm1, double *sxm1, double *sym1, double *szm1,double *txm1, double *tym1, double *tzm1, int lx1, int ly1, int lz1, int lxy, int nxyz, int toteqlxyz,int eq,int nnel){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){
		int ix = id % lx1;
		int iy = (id/lx1)%ly1;
		int iz = (id / (lxy))%lz1;
		int e = id/nxyz;

		gradu[0*toteqlxyz+eq*nxyz+id] =  jacmi[id] *( rxm1[id]*ur[id]+ sxm1[id]*us[id]+txm1[id]*ut[id]);
		gradu[1*toteqlxyz+eq*nxyz+id] =  jacmi[id] *( rym1[id]*ur[id]+ sym1[id]*us[id]+tym1[id]*ut[id]);
		gradu[2*toteqlxyz+eq*nxyz+id] =  jacmi[id] *( rzm1[id]*ur[id]+ szm1[id]*us[id]+tzm1[id]*ut[id]);

	}
}

__global__ void compute_gradients_gpu_kernel3(double *ur,double *us,double *gradu, double *jacmi,double *rxm1, double *rym1, double *sxm1, double *sym1, int nnel,int lx1, int ly1, int lxy,int nxyz, int toteqlxyz, int eq,int lz1){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){
		int ix = id % lx1;
		int iy = (id/lx1)%ly1;
		int iz = (id / (lxy))%lz1;
		int e = id/nxyz;

		gradu[0*toteqlxyz+eq*nxyz+id] =  jacmi[id] *( rxm1[id]*ur[id]+ sxm1[id]*us[id]);
		gradu[1*toteqlxyz+eq*nxyz+id] =  jacmi[id] *( rym1[id]*ur[id]+ sym1[id]*us[id]);

	}
}


// original  compute_gradients is in intpdiff.f file
extern "C" void compute_gradients_gpu_wrapper_(int *glbblockSize1,double *d_u,double *d_phig, double *d_dxm1,double *d_dxtm1, double *d_gradu,double *d_jacmi,double *d_rxm1,double *d_rym1,double *d_rzm1,double *d_sxm1,double *d_sym1,double *d_szm1,double *d_txm1,double *d_tym1,double *d_tzm1,int *lx1, int *ly1, int *lz1, int *nelt, int *lelt,int *toteq,int *lxd,int *lyd,int *lzd,int *if3d){

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start compute_gradients_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start compute_gradients_gpu_wrapper values lx1,ly1,lz1,nelt,lelt,toteq,lxd,lyd,lzd,if3d\n",lx1[0],ly1[0],lz1[0],nelt[0],lelt[0],toteq[0],lxd[0],lyd[0],lzd[0],if3d[0]);

#endif
	int nnel = lx1[0]*ly1[0]*lz1[0]*nelt[0];

	int lxy  = lx1[0]*ly1[0];
	int  lyz  = ly1[0]*lz1[0];
	int nxyz = lxy*lz1[0];
	int m0 = lx1[0]-1;
	int ndlel=lxd[0]*lyd[0]*lzd[0]*lelt[0];
	int toteqlxyz= nxyz*toteq[0];

	double *d_ur;
	double *d_us;
	double *d_ut;
	double *d_ud;

	cudaMalloc((void**)&d_us,ndlel*sizeof(double));
	cudaMalloc((void**)&d_ut,ndlel*sizeof(double));
	cudaMalloc((void**)&d_ur,ndlel*sizeof(double));
	cudaMalloc((void**)&d_ud,ndlel*sizeof(double));
	cudaMemset(d_ur, 0.0, ndlel*sizeof(double));
	cudaMemset(d_us, 0.0, ndlel*sizeof(double));
	cudaMemset(d_ut, 0.0, ndlel*sizeof(double));
	cudaMemset(d_ud, 0.0, ndlel*sizeof(double));


	int blockSize = glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)nnel/blockSize);

	for(int eq=0; eq<toteq[0];eq++){


		compute_gradients_gpu_kernel1<<<gridSize, blockSize>>>(d_ud,d_u,d_phig,nnel,lx1[0],ly1[0],lz1[0],lxy,nxyz,toteqlxyz,eq);

		if(if3d[0]){
			gpu_local_grad3(d_ur,d_us,d_ut,d_ud,m0,1,d_dxm1,d_dxtm1,nelt[0]);// why define  d_ur .. to ldd if only using lx1. check with Dr.Tania. adeesha.
			compute_gradients_gpu_kernel2<<<gridSize, blockSize>>>(d_ur,d_us,d_ut,d_gradu,d_jacmi,d_rxm1,d_rym1,d_rzm1,d_sxm1,d_sym1,d_szm1,d_txm1,d_tym1,d_tzm1,lx1[0],ly1[0],lz1[0],lxy,nxyz,toteqlxyz,eq,nnel);

		}
		else{
			gpu_local_grad2(d_ur,d_us,d_ud,m0,1,d_dxm1,d_dxtm1,nelt[0]);
			compute_gradients_gpu_kernel3<<<gridSize, blockSize>>>(d_ur,d_us,d_gradu,d_jacmi,d_rxm1,d_rym1,d_sxm1,d_sym1,nnel,lx1[0],ly1[0],lxy,nxyz,toteqlxyz,eq,lz1[0]);


		}	
	}
	cudaFree(d_ur);
	cudaFree(d_us);
	cudaFree(d_ut);
	cudaFree(d_ud);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End compute_gradients_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
}

__global__ void convective_cmt_gpu_kernel1(double *convh,double *vxd,double *vyd,double *vzd,double *ju1,double *ju2,int eq, int ndlel,int if3d){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ndlel){
		convh[ndlel+id] = convh[id];
		convh[2*ndlel+id] = convh[id];
		convh[id]=convh[id]*vxd[id];
		convh[ndlel+id]=convh[ndlel+id]*vyd[id];
		if(if3d){
			convh[2*ndlel+id]=convh[2*ndlel+id]*vzd[id];
		}
		convh[2*(eq-1)+id]=convh[2*(eq-1)+id]+ju1[id]*ju2[id]; // works only when toteq <=5. Otherwise eq-1 will be larger than ldim

	}
}

__global__ void convective_cmt_gpu_kernel2(double *convh,double *vxd,double *vyd,double *vzd,double *ju1,double *ju2,int eq, int ndlel){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ndlel){

		convh[id]=convh[id]+ju1[id]*ju2[id]; // works only when toteq <=5. Otherwise eq-1 will be larger than ldim
		convh[ndlel+id] = convh[id];
		convh[2*ndlel+id] = convh[id];
		convh[id]=convh[id]*vxd[id];
		convh[ndlel+id]=convh[ndlel+id]*vyd[id];
		convh[2*ndlel+id]=convh[2*ndlel+id]*vzd[id];

	}
}

__global__ void convective_cmt_flux_div_integral_dealiased_gpu_kernel1(double *totalh,double *rx,double *ur,double *us,double *ut,int lxd, int lyd, int lzd, int lxyd,  int lxyzd, int lxyzdldimldim,int ndlel,int if3d){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ndlel){
		int ix = id % lxd;
		int iy = (id/lxd)%lyd;
		int iz = (id / (lxyd))%lzd;
		int e = id/lxyzd;
		ur[id]=ur[id]+totalh[0*lxyzd+id]*rx[e*lxyzdldimldim+0*lxd*lyd*lzd+iz*lxd*lyd+iy*lxd+ix]; // this rx seems to be collection of rxm1 to tzm1. all 9 of them. so this should be  changed to seperate arrays. but original version is  implemented in this way.  need to check with Dr.Tania. adeesha.
		ur[id]=ur[id]+totalh[1*lxyzd+id]*rx[e*lxyzdldimldim+1*lxyzd+iz*lxyd+iy*lxd+ix];
		ur[id]=ur[id]+totalh[2*lxyzd+id]*rx[e*lxyzdldimldim+2*lxyzd+iz*lxyd+iy*lxd+ix];

		us[id]=us[id]+totalh[0*lxyzd+id]*rx[e*lxyzdldimldim+3*lxyzd+iz*lxyd+iy*lxd+ix];
		us[id]=us[id]+totalh[1*lxyzd+id]*rx[e*lxyzdldimldim+4*lxyzd+iz*lxyd+iy*lxd+ix];
		us[id]=us[id]+totalh[2*lxyzd+id]*rx[e*lxyzdldimldim+5*lxyzd+iz*lxyd+iy*lxd+ix];

		if(if3d){
			ut[id]=ut[id]+totalh[0*lxyzd+id]*rx[e*lxyzdldimldim+6*lxyzd+iz*lxyd+iy*lxd+ix];
			ut[id]=ut[id]+totalh[1*lxyzd+id]*rx[e*lxyzdldimldim+7*lxyzd+iz*lxyd+iy*lxd+ix];
			ut[id]=ut[id]+totalh[2*lxyzd+id]*rx[e*lxyzdldimldim+8*lxyzd+iz*lxyd+iy*lxd+ix];
		}
	}
}

__global__ void convective_cmt_evaluate_aliased_conv_h_gpu_kernel1(double *ju1,double *ju2,double *phig,double *pr, double *convh, double *u,double *totalh,int lxd,int lyd,int lzd,int lxyd,int lxyzd,int lxyzdlelt,int lx1,int ly1,int lz1,int lxy,int lxyz,int toteqlxyz,int ndlel,int eq,int ldim,double *vxd,  double *vyd,double *vzd,int if3d){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ndlel){
		int ix = id % lxd;
		int iy = (id/lxd)%lyd;
		int iz = (id / (lxyd))%lzd;
		int e = id/lxyzd;
		ju1[id]=phig[id];
		ju2[id]=pr[id];
		if(eq<4){
			convh[id]=u[e*toteqlxyz+eq*lxyz+iz*lxy+iy*lx1+ix];				     for(int j=1;j<ldim;j++){
				convh[j*lxyzdlelt+id]=convh[id];
			}
			convh[id]=convh[id]*vxd[id];
			convh[lxyzdlelt+id]=convh[lxyzdlelt+id]*vyd[id];
			if(if3d){
				convh[2*lxyzdlelt+id]=convh[2*lxyzdlelt+id]*vzd[id];
			}
			if(eq>0){
				convh[(eq-1)*lxyzdlelt+id] = convh[(eq-1)*lxyzdlelt+id]+ju1[id]*ju2[id];	
			}
		}
		else if(eq==4){
			convh[id]=u[e*toteqlxyz+eq*lxyz+iz*lxy+iy*lx1+ix];
			convh[id] = convh[id]+ju1[id]*ju2[id];
			for(int j=1;j<ldim;j++){
				convh[j*lxyzdlelt+id]=convh[id];
			}
			convh[id]=convh[id]*vxd[id];
			convh[lxyzdlelt+id]=convh[lxyzdlelt+id]*vyd[id];
			convh[2*lxyzdlelt+id]=convh[2*lxyzdlelt+id]*vzd[id];

		}
		else{
			//send error message back to the fortran program.
		}

		totalh[id]=convh[id];
		totalh[lxyzdlelt+id]=convh[lxyzdlelt+id];
		totalh[2*lxyzdlelt+id]=convh[2*lxyzdlelt+id];



	}
}


extern "C" void convective_cmt_gpu_wrapper_(int *glbblockSize1,int *glbblockSize2,double *d_wkd,double *d_convh,double *d_vxd,double *d_vyd,double *d_vzd,double *d_totalh,double *d_rx,double *d_dg,double *d_dgt,double *d_res1,int *lx1, int *ly1, int *lz1, int *nelt, int *lelt,int *toteq, int *lxd, int *lyd, int *lzd, int *ldim,int *if3d,double *d_u,double *d_phig,double *d_pr ){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start convective_cmt_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start convective_cmt_gpu_wrapper values lx1=%d ,ly1=%d,lz1=%d,nelt=%d,lelt=%d,toteq=%d,lxd=%d,lyd=%d, lzd=%d,ldim=%d,if3d=%d \n", lx1[0],ly1[0],lz1[0],nelt[0],lelt[0],toteq[0],lxd[0],lyd[0], lzd[0],ldim[0],if3d[0]);
#endif
	int nnel = lx1[0]*ly1[0]*lz1[0]*nelt[0];

	int lxy  = lx1[0]*ly1[0];
	int  lxyz  = lxy*lz1[0];
	int toteqlxyz= lxyz*toteq[0];
	int m0 = lx1[0]-1;
	int ndlel=lxd[0]*lyd[0]*lzd[0]*lelt[0];
	int ld = 2*lxd[0];
	int ldd=lxd[0]*lyd[0]*lzd[0];
	int ldw=2*pow(ld,ldim[0]);

	int lxyd= lxd[0]*lyd[0];
	int lxyzd= lxyd*lzd[0];
	int lxyzdlelt=lxyzd*lelt[0];
	int lxyzdldimldim= lxyzd*ldim[0]*ldim[0];

	double *d_ju1;
	double *d_ju2;
	double *d_jgl;
	double *d_jgt;
	double *d_w;


	double *d_ur;
	double *d_us;
	double *d_ut;
	double *d_ud;
	double *d_tu;

	int *d_pjgl;
	int *d_pdg;

	cudaMalloc((void**)&d_ju1,nelt[0]*ldd*sizeof(double));
	cudaMalloc((void**)&d_ju2,nelt[0]*ldd*sizeof(double));
	cudaMalloc((void**)&d_jgl,ldd*sizeof(double));// No nelt[0] here. Need to talk with Mohamed to find the reason or Talk with Dr.Tania. adeesha
	cudaMalloc((void**)&d_jgt,ldd*sizeof(double)); // same as above
	cudaMalloc(&d_w, nelt[0]*ldw*sizeof(double));


	cudaMalloc((void**)&d_ur,nelt[0]*ldd*sizeof(double));
	cudaMalloc((void**)&d_us,nelt[0]*ldd*sizeof(double));
	cudaMalloc((void**)&d_ut,nelt[0]*ldd*sizeof(double));
	cudaMalloc((void**)&d_ud,nelt[0]*ldd*sizeof(double));
	cudaMalloc((void**)&d_tu,nelt[0]*ldd*sizeof(double));

	cudaMalloc((void**)&d_pjgl,nelt[0]*2*lxd[0]*sizeof(int));
	cudaMalloc((void**)&d_pdg,nelt[0]*2*lxd[0]*sizeof(int));

	cudaMemset(d_ju1, 0.0, nelt[0]*ldd*sizeof(double));
	cudaMemset(d_ju2, 0.0, nelt[0]*ldd*sizeof(double));
	cudaMemset(d_jgl, 0.0, ldd*sizeof(double));// May need to copy from cpu to gpu. check with Dr.Tania. adeesha. 
	cudaMemset(d_jgt, 0.0, ldd*sizeof(double));


	int blockSize = glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)nnel/blockSize);

	int i=0;

	//printf("GPU :eqnsolver.cu : convective_cmt_gpu_wrapper after malloc\n");

	for(int eq=0; eq<toteq[0];eq++){

		if(lxd[0]>lx1[0]){
			//evaluate_dealiased_conv_h(e,eq)
			//printf("GPU :eqnsolver.cu : convective_cmt_gpu_wrapper start if lxd?lx1\n");
			if(eq==0){
				for(int j=0;j<ldim[0];j++){
					//intp_rstd(convh(1,j),u(1,1,1,eq+j,e),lx1,lxd,if3d,0)
					// call get_int_ptr (i,mx,md) // a function that goes very deep. Try to  do something about this. Talk with Dr.Tania. adeesha. this function is important because it fills jgt and jgl.
					//printf("GPU :eqnsolver.cu : convective_cmt_gpu_wrapper start if lxd>lx1 before gpu_gett_int_ptr\n");
					//gpu_get_int_ptr(&i,if3d[0], lx1[0], lxd[0], nelt[0],d_jgl, d_jgt,d_wkd,lxd[0],d_pjgl);
					//printf("GPU :eqnsolver.cu : convective_cmt_gpu_wrapper start if lxd>lx1 after gpu_gett_int_ptr\n");
					gpu_specmpn(d_convh+j*ldd, lxd[0], d_u+(j+1)*lxyz ,lx1[0], d_jgl, d_jgt, if3d[0], d_w, ldw, nelt[0], toteq[0], j+1, true ); // this may have toteq in aSize. which is confusing. Talk with Mohamed and Dr.Tania about this. adeesha.
					//also check d_u(j+1) or d_u(j+1+eq).  adeesha.
#ifdef DEBUGPRINT
					cudaDeviceSynchronize();
					code1 = cudaPeekAtLastError();
					printf("CUDA: eq=0 and j=%d convective_cmt_gpu_wrapper after gpu_specmpn cuda status: %s\n",j,cudaGetErrorString(code1));
#endif
				}
				//printf("GPU :eqnsolver.cu : convective_cmt_gpu_wrapper end if  eq==0 toteq= %d\n",eq);
			}
			else{
				//printf("GPU :eqnsolver.cu : convective_cmt_gpu_wrapper start of else  eq==0 toteq= %d\n",eq);
				//gpu_get_int_ptr(&i,if3d[0], lx1[0], lxd[0], nelt[0],d_jgl, d_jgt,d_wkd,lxd[0],d_pjgl);

				gpu_specmpn(d_ju1, lxd[0], d_phig, lx1[0], d_jgl, d_jgt, if3d, d_w, ldw, nelt[0], 1, 0,true);
				//gpu_get_int_ptr(&i,if3d[0], lx1[0], lxd[0], nelt[0],d_jgl, d_jgt,d_wkd,lxd[0],d_pjgl);
#ifdef DEBUGPRINT
				cudaDeviceSynchronize();
				code1 = cudaPeekAtLastError();
				printf("CUDA: eq!=0 convective_cmt_gpu_wrapper after gpu_specmpn with d_ju1 cuda status: %s\n",cudaGetErrorString(code1));

#endif

				gpu_specmpn(d_ju2, lxd[0], d_pr, lx1[0], d_jgl, d_jgt, if3d, d_w, ldw, nelt[0], 1, 0,true);
#ifdef DEBUGPRINT
				cudaDeviceSynchronize();
				code1 = cudaPeekAtLastError();
				printf("CUDA: eq!=0 convective_cmt_gpu_wrapper after gpu_specmpn with d_ju2 cuda status: %s\n",cudaGetErrorString(code1));

#endif

				if(eq<4){
					//gpu_get_int_ptr(&i,if3d[0], lx1[0], lxd[0], nelt[0],d_jgl, d_jgt,d_wkd,lxd[0],d_pjgl);

					gpu_specmpn(d_convh, lxd[0], d_u+eq*lxyz,lx1[0], d_jgl, d_jgt, if3d[0], d_w, ldw, nelt[0], toteq[0], eq,true);
					int blockSize, gridSize;

					// Number of threads in each thread block
					blockSize = glbblockSize1[0];
					gridSize = (int)ceil((float)ndlel/blockSize);
					convective_cmt_gpu_kernel1<<<gridSize, blockSize>>>(d_convh,d_vxd,d_vyd,d_vzd,d_ju1,d_ju2,eq,ndlel,if3d[0]);
				}	
				else if(eq==4){
					//gpu_get_int_ptr(&i,if3d[0], lx1[0], lxd[0], nelt[0],d_jgl, d_jgt,d_wkd,lxd[0],d_pjgl);

					gpu_specmpn(d_convh, lxd[0], d_u+eq*lxyz, lx1[0], d_jgl, d_jgt, if3d[0], d_w, ldw, nelt[0],toteq[0],eq,true);
#ifdef DEBUGPRINT
					cudaDeviceSynchronize();
					code1 = cudaPeekAtLastError();
					printf("CUDA: convective_cmt_gpu_wrapper eq==4 else after gpu_specmpn cuda status: %s\n",cudaGetErrorString(code1));

#endif
					convective_cmt_gpu_kernel2<<<gridSize, blockSize>>>(d_convh,d_vxd,d_vyd,d_vzd,d_ju1,d_ju2,eq,ndlel);
#ifdef DEBUGPRINT
					cudaDeviceSynchronize();
					code1 = cudaPeekAtLastError();
					printf("CUDA: convective_cmt_gpu_wrapper eq==4 else after convective_cmt_gpu_kernel2 cuda status: %s\n",cudaGetErrorString(code1));

#endif


				}


			}

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: convective_cmt_gpu_wrapper after first if cuda status: %s\n",cudaGetErrorString(code1));

#endif
			//			printf("GPU :eqnsolver.cu : convective_cmt_gpu_wrapper after if eq before else toteq= %d\n",eq);
			gpu_double_copy_gpu_wrapper(glbblockSize2[0],d_totalh,0,d_convh,0,3*ldd);//this is calling an extern functions. Check how this works. adeesha
			//flux_div_integral_dealiased(e,eq)

			cudaMemset(d_ur, 0.0, nelt[0]*ldd*sizeof(double));
			cudaMemset(d_us, 0.0, nelt[0]*ldd*sizeof(double));
			cudaMemset(d_ut, 0.0, nelt[0]*ldd*sizeof(double));
			cudaMemset(d_ud, 0.0, nelt[0]*ldd*sizeof(double));
			cudaMemset(d_tu, 0.0, nelt[0]*ldd*sizeof(double));
			//call get_dgl_ptr(ip,lxd,lxd) ! fills dg, dgt  need to implement this function somehow. adeesha.
			int ip=0;
			//gpu_get_dgl_ptr ( &ip,if3d[0], lx1[0],lxd[0], nelt[0],d_jgl, d_jgt,d_wkd,lxd[0],d_pdg);
			blockSize = glbblockSize1[0];
			gridSize = (int)ceil((float)ndlel/blockSize);
			convective_cmt_flux_div_integral_dealiased_gpu_kernel1<<<gridSize, blockSize>>>(d_totalh,d_rx,d_ur,d_us,d_ut,lxd[0],lyd[0],lzd[0],lxyd,lxyzd,lxyzdldimldim,ndlel,if3d[0]);

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: convective_cmt_gpu_wrapper after convective_cmt_flux_div_integral_dealiased_gpu_kernel1 cuda status: %s\n",cudaGetErrorString(code1));

#endif
			if(if3d){
				//uncooment after fix gpu_local_grad3_t
				gpu_local_grad3_t(d_ud, d_ur, d_us, d_ut, lxd[0], d_dg, d_dgt, d_w, nelt[0]);
			}
			else{
				gpu_local_grad2_t(d_ud, d_ur, d_us, lxd[0], d_dg, d_dgt, d_w, nelt[0]);

			}
#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: convective_cmt_gpu_wrapper after gpu_local_grad_kernel1 cuda status: %s\n",cudaGetErrorString(code1));

#endif
			//gpu_get_int_ptr(&i,if3d[0], lx1[0], lxd[0], nelt[0],d_jgl, d_jgt,d_wkd,lxd[0],d_pjgl);
			gpu_specmpn(d_tu,lx1[0],d_ud,lxd[0],d_jgt,d_jgl,if3d[0],d_w,ldw,nelt[0],1,0,false);
			gpu_neksub2(glbblockSize2[0],d_res1+eq*nnel,d_tu,nnel);
#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: convective_cmt_gpu_wrapper end of lxd>lx1 if_gpu_kernel1 cuda status: %s\n",cudaGetErrorString(code1));

#endif
		}


		else{
			//evaluate_aliased_conv_h(e,eq)
			blockSize = glbblockSize1[0];
			gridSize = (int)ceil((float)ndlel/blockSize);
			convective_cmt_evaluate_aliased_conv_h_gpu_kernel1<<<gridSize, blockSize>>>(d_ju1,d_ju2,d_phig,d_pr,d_convh,d_u,d_totalh,lxd[0],lyd[0],lzd[0],lxyd,lxyzd,lxyzdlelt,lx1[0],ly1[0], lz1[0],lxy,lxyz,toteqlxyz,ndlel,eq,ldim[0],d_vxd,d_vyd,d_vzd,if3d[0]);

			cudaMemset(d_ur, 0.0, nelt[0]*ldd*sizeof(double));
			cudaMemset(d_us, 0.0, nelt[0]*ldd*sizeof(double));
			cudaMemset(d_ut, 0.0, nelt[0]*ldd*sizeof(double));
			cudaMemset(d_ud, 0.0, nelt[0]*ldd*sizeof(double));
			cudaMemset(d_tu, 0.0, nelt[0]*ldd*sizeof(double));
			// implement the missing part.  adeesha.
			if(if3d){
				//uncomment after fix gpu_local_grad3_t
				gpu_local_grad3_t(d_ud, d_ur, d_us, d_ut, lxd[0], d_dg, d_dgt, d_w, nelt[0]);
			}
			else{
				gpu_local_grad2_t(d_ud, d_ur, d_us, lxd[0], d_dg, d_dgt, d_w, nelt[0]);

			}
			gpu_neksub2(glbblockSize2[0],d_res1+eq*nnel,d_tu,nnel);

		}
#ifdef DEBUGPRINT
		cudaDeviceSynchronize();
		code1 = cudaPeekAtLastError();
		printf("CUDA: End eq for loop eq = %d convective_cmt_wrapper cuda status: %s\n",eq,cudaGetErrorString(code1));
#endif
	}
	cudaFree(d_ur);
	cudaFree(d_us);
	cudaFree(d_ut);
	cudaFree(d_ud);
	cudaFree(d_tu);

	cudaFree(d_ju1);
	cudaFree(d_ju2);
	cudaFree(d_jgl);
	cudaFree(d_jgt);
	cudaFree(d_w);

	cudaFree(d_pjgl);
	cudaFree(d_pdg);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();
	printf("CUDA: End convective_cmt_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
}

__global__ void viscous_cmt_gpu_kernel1(double *diffh,double *gradu, double *vtrans, double *vdiff, double * vx, double *vy, double *vz, double *u, double *viscscr, int lx1,int ly1,int lz1, int lxy, int nxyz,  int toteq, int nnel, int lxyzlelt, int toteqlxyz, int toteqlxyzlelt, int irho, int ilam, int imu, int icv, int iknd, int inus,int eq,int  if3d,int ldim){

	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){

		int ix = id % lx1;
		int iy = (id/lx1)%ly1;
		int iz = (id / (lxy))%lz1;
		int e = id/nxyz;

		//agradu(diffh,gradu,e,eq) // flux =diffh change later. adeesha
		// call fluxj_ns (flux,du,e,eq)
		if(eq < toteq-1){
			if(eq==1){
				//A21kldUldxk(flux(1,1),gradu,e)
				double dU1x = gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
				double  dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
				double  dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
				double  rho   =vtrans[(irho-1)*lxyzlelt+id];
				double  lambda=vdiff[(ilam-1)*lxyzlelt+id];
				double  mu    =vdiff[(imu-1)*lxyzlelt+id];
				double  u1    =vx[id];
				double  u2    =vy[id];
				double  u3    =vz[id];
				double  lambdamu=lambda+2.0*mu;
				diffh[0*lxyzlelt+id]=(lambda*(dU4z+dU3y-u3*dU1z-u2*dU1y)+lambdamu*(dU2x-u1*dU1x))/rho;

				//A22kldUldxk(flux(1,2),gradu,e)
				dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double   dU3x=gradu[0*toteqlxyzlelt+2*lxyzlelt+id ];
				dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double   dU2y=gradu[1*toteqlxyzlelt+1*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];
				u1    =vx[id];
				u2    =vy[id];
				diffh[1*lxyzlelt+id]=mu*(dU3x+dU2y-u1*dU1y-u2*dU1x)/rho;

				//A23kldUldxk(flux(1,3),gradu,e)

				dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU4x=gradu[0*toteqlxyzlelt+3*lxyzlelt+id ];
				dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU2z=gradu[2*toteqlxyzlelt+1*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];
				u1    =vx[id];
				u3    =vz[id];
				diffh[2*lxyzlelt+id]=mu*(dU4x+dU2z-u1*dU1z-u3*dU1x)/rho;

			}
			else if(eq==2){
				//A31kldUldxk(flux(1,1),gradu,e)
				double dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU3x=gradu[0*toteqlxyzlelt+2*lxyzlelt+id ];
				double dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU2y=gradu[1*toteqlxyzlelt+1*lxyzlelt+id ];
				double rho   =vtrans[(irho-1)*lxyzlelt+id];
				double mu    =vdiff[(imu-1)*lxyzlelt+id];
				double u1    =vx[id];
				double u2    =vy[id];
				diffh[0*lxyzlelt+id] = mu*(dU3x+dU2y-u1*dU1y-u2*dU1x)/rho;

				//A32kldUldxk(flux(1,2),gradu,e)
				dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
				dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
				double dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double   dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];;
				double lambda=vdiff[(ilam-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];
				u1    =vx[id];
				u2    =vy[id];
				double u3    =vz[id];
				double lambdamu=lambda+2.0*mu;
				diffh[1*lxyzlelt+id]=(lambda*(dU4z+dU2x-u3*dU1z-u1*dU1x)+lambdamu*(dU3y-u2*dU1y))/rho;

				//A33kldUldxk(flux(1,3),gradu,e)
				dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU4y=gradu[1*toteqlxyzlelt+3*lxyzlelt+id ];
				dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double   dU3z=gradu[2*toteqlxyzlelt+2*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];
				u2    =vy[id];
				u3    =vz[id];
				diffh[2*lxyzlelt+id]=mu*(dU4y+dU3z-u2*dU1z-u3*dU1y)/rho;	



			}
			else if(eq==3){
				//A41kldUldxk(flux(1,1),gradu,e)
				double dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU4x=gradu[0*toteqlxyzlelt+3*lxyzlelt+id ];
				double dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU2z=gradu[2*toteqlxyzlelt+1*lxyzlelt+id ];
				double rho   =vtrans[(irho-1)*lxyzlelt+id];
				double mu    =vdiff[(imu-1)*lxyzlelt+id];

				double u1    =vx[id];
				double u3    =vz[id];
				diffh[0*lxyzlelt+id]=mu*(dU4x+dU2z-u1*dU1z-u3*dU1x)/rho;

				//A42kldUldxk(flux(1,2),gradu,e)
				double dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU4y=gradu[1*toteqlxyzlelt+3*lxyzlelt+id ];
				dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU3z=gradu[2*toteqlxyzlelt+2*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];

				double u2    =vy[id];
				u3    =vz[id];
				diffh[1*lxyzlelt+id]=mu*(dU4y+dU3z-u2*dU1z-u3*dU1y)/rho;

				//A43kldUldxk(flux(1,3),gradu,e)
				dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
				dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
				dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
				rho   =vtrans[(irho-1)*lxyzlelt+id];
				double lambda=vdiff[(ilam-1)*lxyzlelt+id];
				mu    =vdiff[(imu-1)*lxyzlelt+id];
				u1    =vx[id];
				u2    =vy[id];
				u3    =vz[id];
				double lambdamu=lambda+2.0*mu;
				diffh[2*lxyzlelt+id]=(lambda*(dU3y+dU2x-u2*dU1y-u1*dU1x)+lambdamu*(dU4z-u3*dU1z))/rho;	
			}

		}
		else{
			if(if3d){
				//a53kldUldxk(flux(1,3),gradu,e)
				double dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
				double dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
				double  dU3x=gradu[0*toteqlxyzlelt+2*lxyzlelt+id ];
				double  dU4x=gradu[0*toteqlxyzlelt+3*lxyzlelt+id ];
				double  dU5x=gradu[0*toteqlxyzlelt+4*lxyzlelt+id ];
				double  dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU2y=gradu[1*toteqlxyzlelt+1*lxyzlelt+id ];
				double  dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
				double  dU4y=gradu[1*toteqlxyzlelt+3*lxyzlelt+id ];
				double  dU5y=gradu[1*toteqlxyzlelt+4*lxyzlelt+id ];
				double dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
				double  dU2z=gradu[2*toteqlxyzlelt+1*lxyzlelt+id ];
				double  dU3z=gradu[2*toteqlxyzlelt+2*lxyzlelt+id ];
				double  dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
				double  dU5z=gradu[2*toteqlxyzlelt+4*lxyzlelt+id ];
				double  rho   =vtrans[(irho-1)*lxyzlelt+id];
				double  cv    =vtrans[(icv-1)*lxyzlelt+id]/rho;
				double  lambda=vdiff[(ilam-1)*lxyzlelt+id];
				double  mu    =vdiff[(imu-1)*lxyzlelt+id];
				double K     =vdiff[(iknd-1)*lxyzlelt+id];;
				double  u1    =vx[id];
				double  u2    =vy[id];
				double  u3    =vz[id];
				double  E     =u[e*toteqlxyz+(toteq-1)*nxyz+iz*lxy+iy*lx1+ix]/rho;
				double lambdamu=lambda+mu;
				double kmcvmu=K-cv*mu;
				diffh[2*lxyzlelt+id]=(K*(dU5z-E*dU1z)+cv*u3*(lambda*dU4z+2*mu*dU4z+lambda*dU3y+lambda*dU2x)-K*u3*dU4z+cv*mu*u2*(dU4y+dU3z)+cv*mu*u1*(dU4x+dU2z)-K*u2*dU3z-K*u1*dU2z-cv*(lambda+2*mu)*u3*u3*dU1z+K*u3*u3*dU1z+ K*u2*u2*dU1z-cv*mu*u2*u2*dU1z+K*u1*u1*dU1z-cv*mu*u1*u1*dU1z-cv*(lambda+mu)*u2*u3*dU1y-cv*(lambda+mu)*u1*u3*dU1x)/(cv*rho);

			}
			else{
				for(int kfortoteq=0;kfortoteq<toteq;kfortoteq++){
					gradu[2*toteqlxyzlelt+kfortoteq*lxyzlelt+id]=0;
				}
				vz[id]=0;

			}


		}
		//a51kldUldxk(flux(1,1),gradu,e)
		double dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
		double dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
		double  dU3x=gradu[0*toteqlxyzlelt+2*lxyzlelt+id ];
		double  dU4x=gradu[0*toteqlxyzlelt+3*lxyzlelt+id ];
		double  dU5x=gradu[0*toteqlxyzlelt+4*lxyzlelt+id ];
		double  dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
		double  dU2y=gradu[1*toteqlxyzlelt+1*lxyzlelt+id ];
		double  dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
		double  dU4y=gradu[1*toteqlxyzlelt+3*lxyzlelt+id ];
		double  dU5y=gradu[1*toteq*toteqlxyzlelt+4*lxyzlelt+id ];
		double dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
		double  dU2z=gradu[2*toteqlxyzlelt+1*lxyzlelt+id ];
		double  dU3z=gradu[2*toteqlxyzlelt+2*lxyzlelt+id ];
		double  dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
		double  dU5z=gradu[2*toteqlxyzlelt+4*lxyzlelt+id ];
		double  rho   =vtrans[(irho-1)*lxyzlelt+id];
		double  cv    =vtrans[(icv-1)*lxyzlelt+id]/rho;
		double  lambda=vdiff[(ilam-1)*lxyzlelt+id];
		double  mu    =vdiff[(imu-1)*lxyzlelt+id];
		double K     =vdiff[(iknd-1)*lxyzlelt+id];;
		double  u1    =vx[id];
		double  u2    =vy[id];
		double  u3    =vz[id];
		double  E     =u[e*toteqlxyz+(toteq-1)*nxyz+iz*lxy+iy*lx1+ix]/rho;
		double lambdamu=lambda+mu;
		double kmcvmu=K-cv*mu;
		diffh[0*lxyzlelt+id]=(K*dU5x+cv*lambda*u1*dU4z-kmcvmu*u3*dU4x+cv*lambda*u1*dU3y-kmcvmu*u2*dU3x+cv*mu*u3*dU2z+cv*mu*u2*dU2y+(cv*lambda-K+2*cv*mu)*u1*dU2x-cv*lambdamu*u1*u3*dU1z-cv*lambdamu*u1*u2*dU1y+(K*u3*u3-cv*mu*u3*u3+K*u2*u2-cv*mu*u2*u2-cv*lambda*u1*u1+K*u1*u1-2*cv*mu*u1*u1-E*K)*dU1x)/(cv*rho);

		//a52kldUldxk(flux(1,2),gradu,e)
		dU1x=gradu[0*toteqlxyzlelt+0*lxyzlelt+id ];
		dU2x=gradu[0*toteqlxyzlelt+1*lxyzlelt+id ];
		dU3x=gradu[0*toteqlxyzlelt+2*lxyzlelt+id ];
		dU4x=gradu[0*toteqlxyzlelt+3*lxyzlelt+id ];
		dU5x=gradu[0*toteqlxyzlelt+4*lxyzlelt+id ];
		dU1y=gradu[1*toteqlxyzlelt+0*lxyzlelt+id ];
		dU2y=gradu[1*toteqlxyzlelt+1*lxyzlelt+id ];
		dU3y=gradu[1*toteqlxyzlelt+2*lxyzlelt+id ];
		dU4y=gradu[1*toteqlxyzlelt+3*lxyzlelt+id ];
		dU5y=gradu[1*toteqlxyzlelt+4*lxyzlelt+id ];
		dU1z=gradu[2*toteqlxyzlelt+0*lxyzlelt+id ];
		dU2z=gradu[2*toteqlxyzlelt+1*lxyzlelt+id ];
		dU3z=gradu[2*toteqlxyzlelt+2*lxyzlelt+id ];
		dU4z=gradu[2*toteqlxyzlelt+3*lxyzlelt+id ];
		dU5z=gradu[2*toteqlxyzlelt+4*lxyzlelt+id ];
		rho   =vtrans[(irho-1)*lxyzlelt+id];
		cv    =vtrans[(icv-1)*lxyzlelt+id]/rho;
		lambda=vdiff[(ilam-1)*lxyzlelt+id];
		mu    =vdiff[(imu-1)*lxyzlelt+id];
		K     =vdiff[(iknd-1)*lxyzlelt+id];;
		u1    =vx[id];
		u2    =vy[id];
		u3    =vz[id];
		E     =u[e*toteqlxyz+(toteq-1)*nxyz+iz*lxy+iy*lx1+ix]/rho;
		lambdamu=lambda+mu;
		kmcvmu=K-cv*mu;
		diffh[1*lxyzlelt+id]=(K*dU5y+cv*lambda*u2*dU4z-kmcvmu*u3*dU4y+cv*mu*u3*dU3z+(cv*lambda-K+2*cv*mu)*u2*dU3y+cv*mu*u1*dU3x-kmcvmu*u1*dU2y+cv*lambda*u2*dU2x-cv*lambdamu*u2*u3*dU1z+(K*u3*u3-cv*mu*u3*u3-cv*lambda*u2*u2+K*u2*u2-2*cv*mu*u2*u2+K*u1*u1-cv*mu*u1*u1-E*K)*dU1y-cv*lambdamu*u1*u2*dU1x)/(cv*rho);


		//call fluxj_evm(flux,du,e,eq)

		if(eq==0){
			for(int jj=0;jj<ldim;jj++){
				diffh[jj*lxyzlelt+id]=  diffh[jj*lxyzlelt+id]+vdiff[(inus-1)*lxyzlelt+id]*gradu[jj*toteqlxyzlelt+id];
			}

		}
		else{
			if(eq<toteq-1){
				viscscr[id]=gradu[0*toteqlxyzlelt+(eq-1)*lxyzlelt+id ]; // problem with du indices. du(1,1,eq-1) third is for ldim check wih Dr.Tania adeesha.
				viscscr[id]=viscscr[id]*vdiff[(inus-1)*lxyzlelt+id];
				diffh[0*lxyzlelt+id]=  diffh[0*lxyzlelt+id]+viscscr[id]*vx[id]; 
				diffh[1*lxyzlelt+id]=  diffh[1*lxyzlelt+id]+viscscr[id]*vy[id];
				if(if3d){
					diffh[2*lxyzlelt+id]=  diffh[2*lxyzlelt+id]+viscscr[id]*vz[id];
				} 

			}
			else{
				if(if3d){
					viscscr[id] = vx[id]*vx[id]+vy[id]*vy[id]+vz[id]*vz[id];
				}
				else{
					viscscr[id] = vx[id]*vx[id]+vy[id]*vy[id];
				}
				viscscr[id] =  viscscr[id] *vdiff[(inus-1)*lxyzlelt+id];
				for(int jj=0;jj<ldim;jj++){
					diffh[jj*lxyzlelt+id]=  diffh[jj*lxyzlelt+id]+viscscr[id] *gradu[jj*toteqlxyzlelt+id];
				}
				for(int jj=0;jj<ldim;jj++){
					for(int eq2=1;eq2<ldim+1;eq2++){
						viscscr[id]=gradu[jj*toteqlxyzlelt+eq2*lxyzlelt+id ]* u[e*toteqlxyz+eq2*nxyz+iz*lxy+iy*lx1+ix]+vdiff[(inus-1)*lxyzlelt+id] ;
						viscscr[id]=viscscr[id]/vtrans[(irho-1)*lxyzlelt+id];	
						diffh[jj*lxyzlelt+id]=diffh[jj*lxyzlelt+id]-viscscr[id];
						diffh[jj*lxyzlelt+id]=  diffh[jj*lxyzlelt+id]+vdiff[(inus-1)*lxyzlelt+id] *gradu[jj*toteqlxyzlelt+(toteq-1)*lxyzlelt+id ];
					}

				} 



			}
		}
	}

}

__global__ void viscous_cmt_gpu_kernel2(double *graduf,double *normal,double *unx,double *uny, double *unz,double *iface_flux,double *hface,double *diffh,int ntot,int lxz2ldim,int lxz2ldimlelt,int lxyz,int eq,int lxyzlelt,double *area){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<ntot){
		int e = id/lxz2ldim;
		//diffh2graduf(e,eq,graduf)
		graduf[eq*lxz2ldimlelt+id]=0;
		for(int j=0;j<3;j++){
			if(j==1){normal[id]=unx[id];}
			if(j==2){normal[id]=uny[id];}
			if(j==3){normal[id]=unz[id];}
			int newi = iface_flux[id];
			hface[id] =diffh[j*lxyzlelt+e*lxyz+newi];
			graduf[eq*lxz2ldimlelt+id]=graduf[eq*lxz2ldimlelt+id]+hface[id]*normal[id];
		}
		graduf[eq*lxz2ldimlelt+id]= graduf[eq*lxz2ldimlelt+id]*area[id];
	}
}

__global__ void viscous_cmt_gpu_kernel3( double *ur, double *us, double *ut, double *jacmi, double *rxm1,double *rym1, double *rzm1, double *sxm1, double *sym1, double *szm1,double *txm1, double *tym1, double *tzm1,double *diffh,int ldim, int nnel,int lxyzlelt,double *bm1,int if3d){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){
		//half_iku_cmt(res1(1,1,1,e,eq),diffh,e)
		//half_iku_cmt(res,diffh,e) 
		for(int j=0;j<ldim;j++){
			diffh[j*lxyzlelt+id]= diffh[j*lxyzlelt+id]*bm1[id];
		}
		//gradm11_t(res,diffh,const,e)	
		if(if3d){
			ur[id] =  jacmi[id] *( rxm1[id]*diffh[0*lxyzlelt+id]+ rym1[id]*diffh[1*lxyzlelt+id]+rzm1[id]*diffh[2*lxyzlelt+id]);
			us[id] =  jacmi[id] *( sxm1[id]*diffh[0*lxyzlelt+id]+ sym1[id]*diffh[1*lxyzlelt+id]+szm1[id]*diffh[2*lxyzlelt+id]);
			ut[id] =  jacmi[id] *( txm1[id]*diffh[0*lxyzlelt+id]+ tym1[id]*diffh[1*lxyzlelt+id]+tzm1[id]*diffh[2*lxyzlelt+id]);

		}
		else{
			ur[id] =  jacmi[id] *( rxm1[id]*diffh[0*lxyzlelt+id]+ rym1[id]*diffh[1*lxyzlelt+id]);
			us[id] =  jacmi[id] *( sxm1[id]*diffh[0*lxyzlelt+id]+sym1[id]*diffh[1*lxyzlelt+id]);
		}			
	}
}

__global__ void viscous_cmt_gpu_kernel4(double *res1,double *ud,int nnel, int  eq,int lxyzlelt){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){
		ud[id]=ud[id]*1.0;
		res1[eq*lxyzlelt+id]=res1[eq*lxyzlelt+id]+ud[id];
	}

}


extern "C" void viscous_cmt_gpu_wrapper_(int *glbblockSize1,int *glbblockSize2,double *d_diffh,double *d_gradu,double *d_vtrans,double *d_vdiff,double *d_vx,double *d_vy,double *d_vz,double *d_u,double *d_viscscr,double *d_jacmi,double *d_rxm1,double *d_rym1,double *d_rzm1,double *d_sxm1,double *d_sym1,double *d_szm1,double *d_txm1,double *d_tym1,double *d_tzm1,double *d_graduf,double *d_unx,double *d_uny,double *d_unz,double *d_iface_flux,double *d_dxm1,double *d_dxtm1,double *d_res1,double *d_area,double *d_bm1,int *lx1,int *ly1,int *lz1,int *toteq,int *irho,int *ilam,int *imu,int *icv,int *iknd,int *inus,int *if3d,int *ldim,int *nelt,int *lelt){

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();

	printf("CUDA: Start viscous_cmt_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start viscous_cmt_gpu_wrapper values lx1=%d,ly1=%d,lz1=%d,toteq=%d,irho=%d,ilam=%d,imu=%d,icv=%d,iknd=%d,inus=%d,if3d=%d,ldim=%d,nelt=%d,lelt=%d \n", lx1[0],ly1[0],lz1[0],toteq[0],irho[0],ilam[0],imu[0],icv[0],iknd[0],inus[0],if3d[0],ldim[0],nelt[0],lelt[0]);

#endif
	int lxy  = lx1[0]*ly1[0];
	int  lxz  = ly1[0]*lz1[0];
	int nxyz = lxy*lz1[0];
	int m0 = lx1[0]-1;
	int nnel = nxyz*nelt[0];
	int lxyzlelt = nxyz*lelt[0];
	int toteqlxyz= toteq[0]*nxyz;
	int toteqlxyzlelt=toteqlxyz*lelt[0];
	int lxz2ldim=lxz*2*ldim[0];
	int ntot=lxz2ldim*lelt[0];
	int lxz2ldimlelt=ntot;


	double *d_ur;
	double *d_us;
	double *d_ut;
	double *d_ud;
	double *d_tmp;

	double *d_hface;
	double *d_normal;

	cudaMalloc((void**)&d_ur,nnel*sizeof(double));
	cudaMalloc((void**)&d_us,nnel*sizeof(double));
	cudaMalloc((void**)&d_ut,nnel*sizeof(double));
	cudaMalloc((void**)&d_ud,nnel*sizeof(double));
	cudaMalloc((void**)&d_tmp,nnel*sizeof(double));
	cudaMalloc((void**)&d_hface,lxz2ldimlelt*sizeof(double));
	cudaMalloc((void**)&d_normal,lxz2ldimlelt*sizeof(double));



	for(int eq=0; eq<toteq[0];eq++){

		cudaMemset(d_ud, 0.0, nnel*sizeof(double));

		int blockSize1 = glbblockSize1[0], blockSize2=glbblockSize2[0], gridSize1,gridSize2;
		gridSize1 = (int)ceil((float)nnel/blockSize1);
		viscous_cmt_gpu_kernel1<<<gridSize1, blockSize1>>>(d_diffh,d_gradu, d_vtrans,d_vdiff,d_vx,d_vy,d_vz,d_u,d_viscscr,lx1[0],ly1[0],lz1[0], lxy,nxyz,toteq[0], nnel,lxyzlelt,toteqlxyz, toteqlxyzlelt, irho[0], ilam[0],imu[0],icv[0], iknd[0],inus[0],eq,if3d[0],ldim[0] );


		gridSize2= (int)ceil((float)ntot/blockSize2);

		viscous_cmt_gpu_kernel2<<<gridSize2, blockSize2>>>(d_graduf,d_normal,d_unx,d_uny,d_unz,d_iface_flux,d_hface,d_diffh,ntot,lxz2ldim,lxz2ldimlelt,nxyz,eq,lxyzlelt,d_area);

		viscous_cmt_gpu_kernel3<<<gridSize1, blockSize1>>>(d_ur,d_us,d_ut, d_jacmi,d_rxm1,d_rym1,d_rzm1,d_sxm1,d_sym1,d_szm1,d_txm1,d_tym1,d_tzm1,d_diffh,ldim[0],nnel,lxyzlelt,d_bm1,if3d[0]);

		if(if3d){
			gpu_local_grad3_t(d_ud, d_ur, d_us,d_ut,lx1[0],d_dxm1,d_dxtm1, d_tmp, nelt[0]);		
		}
		else{
			gpu_local_grad2_t(d_ud, d_ur, d_us,lx1[0],d_dxm1,d_dxtm1, d_tmp, nelt[0]);		

		}
		viscous_cmt_gpu_kernel4<<<gridSize1, blockSize1>>>(d_res1,d_ud,nnel, eq,lxyzlelt);


	}
	cudaFree(d_ur);
	cudaFree(d_us);
	cudaFree(d_ut);
	cudaFree(d_ud);
	cudaFree(d_tmp);
	cudaFree(d_hface);
	cudaFree(d_normal);
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End viscous_cmt_wrapper cuda status: %s\n",cudaGetErrorString(code2));
#endif
}


__global__ void compute_forcing_gpu_kernel1(double *ur,double *us,double *ut,double *rm1,double *sm1, double *tm1,double *rdumz, double *jacmi,int nnel){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){

		rdumz[id]= 1.0/jacmi[id]*(ur[id]*rm1[id]+us[id]*sm1[id]+ut[id]*tm1[id]);

	}
}

__global__ void compute_forcing_gpu_kernel2(double *ur,double *us,double *rm1,double *sm1,double *rdumz, double *jacm1,int nnel){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){

		rdumz[id]= 1.0/jacm1[id]*(ur[id]*rm1[id]+us[id]*sm1[id]);

	}
}
__global__ void compute_forcing_gpu_kernel3(double *res1,double *usrf,double *rdumz,double *bm1, int nnel, int lx1,int ly1,int lz1, int lxy, int nxyz, int lxyzlelt,int eq){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){
		int ix = id % lx1;
		int iy = (id/lx1)%ly1;
		int iz = (id / (lxy))%lz1;
		int e = id/nxyz;
		int ixyz=id%nxyz;

		res1[eq*lxyzlelt+id]=res1[eq*lxyzlelt+id]-rdumz[id]*bm1[id];
		res1[eq*lxyzlelt+id]=res1[eq*lxyzlelt+id]-usrf[eq*nxyz+ixyz]*bm1[id];

	}
}
__global__ void compute_forcing_gpu_kernel4(double *res1,double *usrf,double *rdumz,double *bm1, int nnel, int lx1,int ly1,int lz1, int lxy, int nxyz, int lxyzlelt, int eq){
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if(id<nnel){
		int ix = id % lx1;
		int iy = (id/lx1)%ly1;
		int iz = (id / (lx1*ly1))%lz1;
		int e = id/nxyz;
		int ixyz=id%nxyz;

		res1[eq*lxyzlelt+id]=res1[eq*lxyzlelt+id]-usrf[eq*nxyz+ixyz]*bm1[id];

	}
}

extern "C" void compute_forcing_gpu_wrapper_(int *glbblockSize1,double *d_phig,double *d_rxm1,double *d_sxm1,double *d_txm1,double *d_rym1,double *d_sym1,double *d_tym1,double *d_rzm1,double *d_szm1,double *d_tzm1,double *d_jacmi,double *d_pr,double *d_res1,double *d_usrf,double *d_bm1,int *lx1,int *ly1,int *lz1,int *lelt,int *nelt,int *if3d,int *lxd,int *lyd,int *lzd,int *toteq,int *ldim,double *d_wkd){
#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code1 = cudaPeekAtLastError();
	printf("CUDA: Start compute_forcing_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

	printf("CUDA: Start compute_forcing_gpu_wrapper values, lx1=%d,ly1=%d,lz1=%d,lelt=%d,nelt=%d,if3d=%d,lxd=%d,lyd=%d,lzd=%d,toteq=%d,ldim=%d \n", lx1[0],ly1[0],lz1[0],lelt[0],nelt[0],if3d[0],lxd[0],lyd[0],lzd[0],toteq[0],ldim[0]);

#endif
	int nnel = lx1[0]*ly1[0]*lz1[0]*nelt[0];

	int lxy  = lx1[0]*ly1[0];
	int nxyz= lx1[0]*ly1[0]*lz1[0];
	int m0 = lx1[0]-1;
	int ldd= lxd[0]*lyd[0]*lzd[0];
	int lxyzlelt= nxyz*lelt[0];

	double *d_ur;
	double *d_us;
	double *d_ut;
	double *d_rdumz;
	double *d_d;
	double *d_dt;

	int *d_pdg;

	cudaMalloc((void**)&d_pdg,nelt[0]*2*lxd[0]*sizeof(int));


	cudaMalloc((void**)&d_ur,nnel*sizeof(double));
	cudaMalloc((void**)&d_us,nnel*sizeof(double));
	cudaMalloc((void**)&d_ut,nnel*sizeof(double));
	cudaMalloc((void**)&d_rdumz,nnel*sizeof(double));
	cudaMalloc((void**)&d_d,ldd*sizeof(double));  // no lelt here. check  .  adeesha
	cudaMalloc((void**)&d_dt,ldd*sizeof(double)); // same as above


	int blockSize = glbblockSize1[0], gridSize;
	gridSize = (int)ceil((float)nnel/blockSize);


	for(int eq=0; eq<toteq[0];eq++){
		int ip=0;
		if(eq!=0&&eq!=4){
#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: before gpu_gradl_rst compute_forcing_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));

#endif
			gpu_gradl_rst(d_ur,d_us,d_ut,d_phig, d_d, d_dt,lxd[0], nelt[0], if3d[0],&ip,d_wkd,d_pdg,lx1[0],lxd[0]);
#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: after gpu_gradl_rst compute_forcing_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
#endif
			if(if3d){
				gridSize = (int)ceil((float)nnel/blockSize);
				if(eq==1){
					compute_forcing_gpu_kernel1<<<gridSize, blockSize>>>(d_ur,d_us,d_ut,d_rxm1,d_sxm1,d_txm1,d_rdumz,d_jacmi,nnel);
#ifdef DEBUGPRINT
					cudaDeviceSynchronize();
					code1 = cudaPeekAtLastError();
					printf("CUDA: eq=1 after forcing_gpu_kernel1 compute_forcing_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
#endif

				}
				else if(eq==2){
					compute_forcing_gpu_kernel1<<<gridSize, blockSize>>>(d_ur,d_us,d_ut,d_rym1,d_sym1,d_tym1,d_rdumz,d_jacmi,nnel);
#ifdef DEBUGPRINT
					cudaDeviceSynchronize();
					code1 = cudaPeekAtLastError();
					printf("CUDA: eq=2 after forcing_gpu_kernel1 compute_forcing_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
#endif

				}
				else if(eq==3){
					compute_forcing_gpu_kernel1<<<gridSize, blockSize>>>(d_ur,d_us,d_ut,d_rzm1,d_szm1,d_tzm1,d_rdumz,d_jacmi,nnel);
#ifdef DEBUGPRINT
					cudaDeviceSynchronize();
					code1 = cudaPeekAtLastError();
					printf("CUDA: eq=3 after forcing_gpu_kernel1 compute_forcing_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
#endif

				}
			}
			else{
				gridSize = (int)ceil((float)nnel/blockSize);
				if(eq==1){
					compute_forcing_gpu_kernel2<<<gridSize, blockSize>>>(d_ur,d_us,d_rxm1,d_sxm1,d_rdumz,d_jacmi,nnel);

				}
				else if(eq==2){
					compute_forcing_gpu_kernel2<<<gridSize, blockSize>>>(d_ur,d_us,d_rym1,d_sym1,d_rdumz,d_jacmi,nnel);

				}

			}

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: after first if compute_forcing_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
#endif
			gpu_nekcol2(glbblockSize1[0],d_rdumz,d_pr,nnel);

			if(eq!=3 || ldim[0]!=2){
				gridSize = (int)ceil((float)nnel/blockSize);
				compute_forcing_gpu_kernel3<<<gridSize, blockSize>>>(d_res1, d_usrf,d_rdumz,d_bm1,nnel,lx1[0],ly1[0],lz1[0],lxy,nxyz,lxyzlelt,eq);

			}

#ifdef DEBUGPRINT
			cudaDeviceSynchronize();
			code1 = cudaPeekAtLastError();
			printf("CUDA: eq<4 end if compute_forcing_gpu_wrapper cuda status: %s\n",cudaGetErrorString(code1));
#endif
		}
		else if (eq==4){
			gridSize = (int)ceil((float)nnel/blockSize);
			compute_forcing_gpu_kernel4<<<gridSize, blockSize>>>(d_res1, d_usrf,d_rdumz,d_bm1,nnel,lx1[0],ly1[0],lz1[0],lxy,nxyz,lxyzlelt,eq);

		}
#ifdef DEBUGPRINT

		cudaDeviceSynchronize();
		code1 = cudaPeekAtLastError();
		printf("CUDA: eq= %d gpu_gradl_rst ccmpute_forcing_gpu_wrapper cuda status: %s\n",eq,cudaGetErrorString(code1));
#endif

	}
	cudaFree(d_ur);
	cudaFree(d_us);
	cudaFree(d_ut);
	cudaFree(d_d);
	cudaFree(d_dt);

	cudaFree(d_pdg);

#ifdef DEBUGPRINT
	cudaDeviceSynchronize();
	cudaError_t code2 = cudaPeekAtLastError();

	printf("CUDA: End compute_forcing_wrapper cuda status: %s\n",cudaGetErrorString(code2));

#endif
}               

