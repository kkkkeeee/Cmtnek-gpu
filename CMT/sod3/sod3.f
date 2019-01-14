c README -  First shock with entropy viscosity in EUler gas dynamics! 
c           the almighty Sod problem in section 3 of his 1978 JCP 27

      subroutine uservp (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'   ! this is not
      include 'CMTDATA' ! the best idea
      include 'NEKUSE'
      integer e,eg

      e = gllel(eg)

c-----------------------------------------------------------------------
c c_E-> \infty. wave speed visco only
c-----------------------------------------------------------------------
c     mu=rho*t(ix,iy,iz,e,3)

c-----------------------------------------------------------------------
c finite c_E. res2 viscosity clipped in subroutin resvisc (or evmsmooth;
c             I still haven't decided it.
c-----------------------------------------------------------------------
      mu=rho*res2(ix,iy,iz,e,1) ! finite c_E;
      nu_s=0.75*mu/rho

      mu=0.5*mu ! A factor of
           ! 2 lurks in agradu's evaluation of strain rate, even in EVM
      lambda=0.0
      udiff=0.0
      utrans=0.

      return
      end
c-----------------------------------------------------------------------
      subroutine userf  (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      integer e,eg

      ffx = 0.0
      ffy = 0.0
      ffz = 0.0
      return
      end
c-----------------------------------------------------------------------
      subroutine userq  (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      integer e,eg

      qvol   = 0.0

      return
      end
c-----------------------------------------------------------------------
      subroutine userchk
      include 'SIZE'
      include 'TOTAL'
      include 'TORO'
      include 'CMTDATA'

      integer  e,f
      integer  luout(6)
      character*24 zefn

      luout(1) = 59
      luout(2) = 60
      luout(3) = 61
      luout(4) = 62
      luout(5) = 63
      luout(6) = 64

      nxyz= nx1*ny1*nz1
      n = nxyz*nelt
      ifxyo=.true.
      if (istep.gt.1) ifxyo=.false.

      eps     = 0.00001
      time=time_cmt
      dt=dt_cmt

c     umin = glmin(t,n)
c     umax = glmax(t,n)
      umin = glmin(vx,n)
      umax = glmax(vx,n)
c     if (mod(kstep,100).eq.0) then
      if (nio.eq.0) then
c        write(6,2)istep,time,umin,' <T<',umax
         write(6,2)istep,time,umin,' <u<',umax
      endif
c     endif
2     format(i6,1p2e17.8,a4,1p1e17.8)

      if (time_cmt .eq. 0.0.and..not.ifrestart) then
         ! useric kludge to get the discontinuity right at the face
         do e=1,nelt
            iflag=1
            do i=1,nxyz
               if (xm1(i,1,1,e).lt.0.0) iflag=0
            enddo
            if (iflag .eq.1) then
               ux=dr*ur
               energy=pright/(gmaref-1.0)+0.5*dr*ur**2
               do i=1,nxyz
                  u(i,1,1,1,e)=dr
                  u(i,1,1,2,e)=ux
                  u(i,1,1,3,e)=0.0
                  u(i,1,1,4,e)=0.0
                  u(i,1,1,5,e)=energy
               enddo
            endif
         enddo
      endif

      if(ifoutfld.or.istep.eq.0) then
         call compute_primitive_vars

c     if (nio.eq.0)then ! badly parallelized
c$$$      write(zefn,'(a16,i5.5,a1,i1)') 'profiles/rhoprof',istep,'p',nid
c$$$      open(unit=luout(1),file=zefn,form="formatted")
c$$$      write(zefn,'(a14,i5.5,a1,i1)') 'profiles/uprof',istep,'p',nid
c$$$      open(unit=luout(2),file=zefn,form="formatted")
c$$$      write(zefn,'(a14,i5.5,a1,i1)') 'profiles/vprof',istep,'p',nid
c$$$      open(unit=luout(3),file=zefn,form="formatted")
c$$$      write(zefn,'(a14,i5.5,a1,i1)') 'profiles/Tprof',istep,'p',nid
c$$$      open(unit=luout(4),file=zefn,form="formatted")
c$$$      write(zefn,'(a17,i5.5,a1,i1)') 'profiles/rhoeprof',istep,'p',nid
c$$$      open(unit=luout(5),file=zefn,form="formatted")
c$$$c     write(zefn,'(a16,i5.5,a1,i1)') 'profiles/resprof',istep,'p',nid
c$$$c     open(unit=luout(5),file=zefn,form="formatted")
c$$$      write(zefn,'(a17,i5.5,a1,i1)') 'profiles/viscprof',istep,'p',nid
c$$$      open(unit=luout(6),file=zefn,form="formatted")
c$$$c     endif
c$$$     do e=1,nelt
c$$$        l=0
c$$$        do k=1,nz1
c$$$        do j=1,ny1
c$$$        do i=1,nx1
c$$$           l=l+1
c$$$           s=(xm1(i,j,k,e)-diaph1)/time_cmt
c$$$           CALL SAMPLE(PMstar, UM, S, rhohere, uhere, prshere)
c$$$c profile writer not parallelized correctly
c$$$            if (abs(ym1(i,j,k,e)).lt.1.0e-6) then
c$$$                  write(luout(1),'(3e25.16)')
c$$$     >               xm1(i,j,k,e),u(i,j,k,1,e),rhohere
c$$$                  write(luout(2),'(3e25.16)')
c$$$     >               xm1(i,j,k,e),vx(i,j,k,e),uhere
c$$$                  write(luout(3),'(2e25.16)')
c$$$     >               xm1(i,j,k,e),vy(i,j,k,e)
c$$$                  write(luout(4),'(3e25.16)') xm1(i,j,k,e),T(i,j,k,e,1),
c$$$     >               prshere/(gmaref-1.0)/rhohere/cvgref
c$$$                  write(luout(5),'(3e25.16)')
c$$$     >               xm1(i,j,k,e),u(i,j,k,5,e),energy
c$$$                  write(luout(6),'(6e25.16)') ! x,visco,diffk,wavevisc,resvisc
c$$$c    >               xm1(i,j,k,e),vdiff(i,j,k,e,imu),vdiff(i,j,k,e,inus)
c$$$     >               xm1(i,j,k,e),vdiff(i,j,k,e,inus)
c$$$c    >              ,t(i,j,k,e,3),res2(i,j,k,e,1),res2(i,j,k,e,2)
c$$$     >              ,t(i,j,k,e,3),res2(i,j,k,e,2)
c$$$            endif
c$$$c profile writer not parallelized correctly
c$$$         enddo
c$$$         enddo
c$$$         enddo
c$$$      enddo
c$$$      do i=1,6
c$$$         close(luout(i))
c$$$      enddo

c     call out_fld_nek ! need restart condition
c-----------------------------------------------------------------------
c JH030317
c adding output fields for diagnostic purposes. optional, but perhaps should
c be flagged by values in uservp (Navier-Stokes vs EVM GP vs both)
         call copy(t(1,1,1,1,2),vdiff(1,1,1,1,imu),n)
         call cmult(t(1,1,1,1,2),2.0,n)
         call invcol2(t(1,1,1,1,2),vtrans(1,1,1,1,irho),n)
c                  t(:,:,:,:,3)=wavevisc already
         call copy(t(1,1,1,1,4),vdiff(1,1,1,1,inus),n)
         call copy(t(1,1,1,1,5),res2,n) ! residual viscosity
         call copy(t(1,1,1,1,6),res2(1,1,1,1,2),n) ! raw residual
      endif ! ifoutfld

      return
      end
c-----------------------------------------------------------------------

      subroutine my_full_restart
      include 'SIZE'
      include 'TOTAL'
      character*80 s80(2)

      call blank(s80,2*80)
      s80(1) ='rs6pvort0.f00001'
      s80(2) ='rs6pvort0.f00001'
      call full_restart(s80,2) 

      iosave = iostep           ! Trigger save based on iostep
      call full_restart_save(iosave)

      return
      end

c-----------------------------------------------------------------------

      subroutine userbc (ix,iy,iz,iside,eg)
      include 'SIZE'
      include 'TSTEP'
      include 'NEKUSE'
      include 'INPUT'
      include 'TORO'
      include 'CMTDATA'
      include 'GEOM' ! not sure if this is a good idea.
      include 'PERFECTGAS'
      real nx,ny,nz  ! bite me it's fun
      integer e

c     e = gllel(eg)
      molarmass=molmass
      phi=1.0
      cp=cpgref
      cv=cvgref
      uy=0.0
      uz=0.0
      if (x .lt. diaph1) then
         ux=ul
         pres=pl
         rho=dl
      else
         ux=ur
         pres=pright
         rho=dr
      endif
      temp=MixtPerf_T_DPR(rho,pres,rgasref)
      asnd=MixtPerf_C_GRT(gmaref,rgasref,temp)
      
      return
      end

c-----------------------------------------------------------------------

      subroutine useric (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'TORO'
      include 'PERFECTGAS'
      include 'CMTDATA'
      integer e,eg, eqnum

c JH073114 Toro e1rpex provides SUBROUTINE SAMPLE and is crudely grafted
c          to the end of this .usr file.
      e=gllel(eg)
      molarmass=molmass

      if (x.gt.diaph1) then
         ux=ur
         pres=pright
         rho=dr
      else
         ux=ul
         pres=pl
         rho=dl
      endif
      temp=MixtPerf_T_DPR(rho,pres,rgasref)

      uy = 0.
      uz = 0.
      phi = 1.0
      cp=cpgref
      cv=cvgref

      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      include 'CMTTIMERS'
      include 'CMTBCDATA'
      include 'PERFECTGAS'
      molmass    = 8314.3
      muref      = 0.0
      coeflambda = -2.0/3.0
      suthcoef   = 1.0
      reftemp    = 1.0
      prlam      = 0.72
      pinfty     = 1.0
      gmaref     = 1.4
      rgasref    = MixtPerf_R_M(molmass,dum)
      cvgref     = rgasref/(gmaref-1.0)
      cpgref     = MixtPerf_Cp_CvR(cvgref,rgasref)
      gmaref     = MixtPerf_G_CpR(cpgref,rgasref) 

c-----------------------------------------------------------------------
c JH030317
c adding output fields for diagnostic purposes. optional, but perhaps should
c be flagged by values in uservp (Navier-Stokes vs EVM GP vs both)
c
c ldimt>=3 because we need ldimt1>=4
c vdiff(:,imu) = mu, first viscosity coeff, acting on symmetric strain rate sij
c vdiff(:,iknd) = kappa, thermal conductivity
c vdiff(:,ilam) = lambda, second viscosity coeff, acting on dilatational strain
c vdiff(:,inus) = nu_s, artificial mass diffusivity in GP fluxes
c
c Of course, these coefficients are not applied to corresponding ifield like in
c vanilla nek5000. likewise, vtrans doesn't correspond directly to vdiff, either.
c So T+1 array gets used for other stuff. Document things
c here and in userchk until they become standard

      nbc = 0      ! No changes in boundary conditions required
      do i=1,ldimt
         call add_temp(f2tbc,nbc,1)
      enddo

      igeom = 2
      call setup_topo
      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat2
      include 'SIZE'
      include 'TOTAL'
      include 'TORO'
      include 'CMTBCDATA'
      include 'CMTDATA'
      include 'PERFECTGAS'   

c until bx initialization explicitly depends on IFMHD
      ifldmhd=9999
c until bx initialization explicitly depends on IFMHD

      outflsub=.true.
      IFCNTFILT=.false.
      ifrestart=.false.
c JH080714 Now with parameter space to sweep through
      open(unit=81,file="riemann.inp",form="formatted")
      read (81,*) domlen
      read (81,*) diaph1
      read (81,*) gmaref
      read (81,*) dl
      read (81,*) ul
      read (81,*) pl
      read (81,*) dr
      read (81,*) ur
      read (81,*) pright
      close(81)

      molmass=8314.3
      rgasref    = MixtPerf_R_M(molmass,dum)
      cvgref     = rgasref/(gmaref-1.0)
      cpgref     = MixtPerf_Cp_CvR(cvgref,rgasref)
      gmaref     = MixtPerf_G_CpR(cpgref,rgasref) 

      c_max=0.3  ! should be 0.5, really
      c_sub_e=40.0

      call e1rpex(domlen,diaph1,gmaref,dl,ul,pl,dr,ur,
     >            pright,1.0)

      return
      end
c-----------------------------------------------------------------------
      subroutine cmt_userEOS(ix,iy,iz,eg)
      include 'SIZE'
      include 'NEKUSE'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'PERFECTGAS'
      integer e,eg

      cp=cpgref
      cv=cvgref
      temp=e_internal/cv
      asnd=MixtPerf_C_GRT(gmaref,rgasref,temp)
      pres=MixtPerf_P_DRT(rho,rgasref,temp)
      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat3
      return
      end
c-----------------------------------------------------------------------
      subroutine e1rpex(DOMin,DIAPHin,GAMMAin,DLin,ULin,PLin,DRin,URin,
     $                  PRin,PSCALEin)
c----------------------------------------------------------------------*
c                                                                      *
C     Exact Riemann Solver for the Time-Dependent                      *
C     One Dimensional Euler Equations                                  *
c                                                                      *
C     Name of program: HE-E1RPEX                                       *
c                                                                      *
C     Purpose: to solve the Riemann problem exactly,                   *
C              for the time dependent one dimensional                  *
C              Euler equations for an ideal gas                        *
c                                                                      *
C     Input  file: e1rpex.ini                                          *
C     Output file: e1rpex.out (exact solution)                         *
c                                                                      *
C     Programer: E. F. Toro                                            *
c                                                                      *
C     Last revision: 31st May 1999                                     *
c                                                                      *
C     Theory is found in Ref. 1, Chapt. 4 and in original              *
C     references therein                                               *
c                                                                      *
C     1. Toro, E. F., "Riemann Solvers and Numerical                   *
C                      Methods for Fluid Dynamics"                     *
C                      Springer-Verlag, 1997                           *
C                      Second Edition, 1999                            *
c                                                                      *
C     This program is part of                                          *
c                                                                      *
C     NUMERICA                                                         *
C     A Library of Source Codes for Teaching,                          *
C     Research and Applications,                                       *
C     by E. F. Toro                                                    *
C     Published by NUMERITEK LTD, 1999                                 *
C     Website: www.numeritek.com                                       *
c                                                                      *
c----------------------------------------------------------------------*
c
      include 'TORO'
c
C     Declaration of variables:
c
      INTEGER I, CELLS
c
c
C     Input variables
c
C     DOMLEN   : Domain length
C     DIAPH1   : Position of diaphragm 1
C     CELLS    : Number of computing cells
C     GAMMA    : Ratio of specific heats
C     TIMEOU   : Output time
C     DL       : Initial density  on left state
C     UL       : Initial velocity on left state
C     PL       : Initial pressure on left state
C     DR       : Initial density  on right state
C     UR       : Initial velocity on right state
C     PR       : Initial pressure on right state
C     PSCALE   : Normalising constant
c
c     Initial data and parameters are now arguments

           DOMLEN=DOMin
           DIAPH1=DIAPHin
           GAMMA =GAMMAin
           DL    =DLin
           UL    =ULin
           PL    =PLin
           DR    =DRin
           UR    =URin
           PRight=PRin
           PSCALE=PSCALEin

C     Compute gamma related constants
c
      G1 = (GAMMA - 1.0)/(2.0*GAMMA)
      G2 = (GAMMA + 1.0)/(2.0*GAMMA)
      G3 = 2.0*GAMMA/(GAMMA - 1.0)
      G4 = 2.0/(GAMMA - 1.0)
      G5 = 2.0/(GAMMA + 1.0)
      G6 = (GAMMA - 1.0)/(GAMMA + 1.0)
      G7 = (GAMMA - 1.0)/2.0
      G8 = GAMMA - 1.0
c
C     Compute sound speeds
c
      CL = SQRT(GAMMA*PL/DL)
      CR = SQRT(GAMMA*PRight/DR)
c
C     The pressure positivity condition is tested for
c
      IF(G4*(CL+CR).LE.(UR-UL))THEN
c
C        The initial data is such that vacuum is generated.
C        Program stopped.
c
         WRITE(6,*)
         WRITE(6,*)'***Vacuum is generated by data***'
         WRITE(6,*)'***Program stopped***'
         WRITE(6,*)
c
         call exitt
      ENDIF
c
C     Exact solution for pressure and velocity in star
C     region is found
c
      CALL STARPU(PMstar, UM, PSCALE)
c
      return
      end
c
c----------------------------------------------------------------------*
c
      SUBROUTINE STARPU(P, U, PSCALE)
c
c     IMPLICIT NONE
c
C     Purpose: to compute the solution for pressure and
C              velocity in the Star Region
c
C     Declaration of variables
c
      INTEGER I, NRITER
c
      REAL    DL, UL, PL, CL, DR, UR, PRight, CR,
     &        CHANGE, FL, FLD, FR, FRD, P, POLD, PSTART,
     &        TOLPRE, U, UDIFF, PSCALE
c
      COMMON /STATES/ DL, UL, PL, CL, DR, UR, PRight, CR
      DATA TOLPRE, NRITER/1.0E-06, 20/
c
C     Guessed value PSTART is computed
c
      CALL GUESSP(PSTART)
c
      POLD  = PSTART
      UDIFF = UR - UL
c
      WRITE(6,*)'----------------------------------------'
      WRITE(6,*)'   Iteration number      Change  '
      WRITE(6,*)'----------------------------------------'
c
      DO 10 I = 1, NRITER
c
         CALL PREFUN(FL, FLD, POLD, DL, PL, CL)
         CALL PREFUN(FR, FRD, POLD, DR, PRight, CR)
         P      = POLD - (FL + FR + UDIFF)/(FLD + FRD)
         CHANGE = 2.0*ABS((P - POLD)/(P + POLD))
         WRITE(6, 30)I, CHANGE
         IF(CHANGE.LE.TOLPRE)GOTO 20
         IF(P.LT.0.0)P = TOLPRE
         POLD  = P
c
 10   CONTINUE
c
      WRITE(6,*)'Divergence in Newton-Raphson iteration'
c
 20   CONTINUE
c
C     Compute velocity in Star Region
c
      U = 0.5*(UL + UR + FR - FL)
c
      WRITE(6,*)'---------------------------------------'
      WRITE(6,*)'   Pressure        Velocity'
      WRITE(6,*)'---------------------------------------'
      WRITE(6,40)P/PSCALE, U
      WRITE(6,*)'---------------------------------------'
c
 30   FORMAT(5X, I5,15X, F12.7)
 40   FORMAT(2(F14.6, 5X))
c
      END
c
c----------------------------------------------------------------------*
c
      SUBROUTINE GUESSP(PMstar)
c
C     Purpose: to provide a guessed value for pressure
C              PM in the Star Region. The choice is made
C              according to adaptive Riemann solver using
C              the PVRS, TRRS and TSRS approximate
C              Riemann solvers. See Sect. 9.5 of Chapt. 9
C              of Ref. 1
c
c     IMPLICIT NONE
c
C     Declaration of variables
c
      REAL    DL, UL, PL, CL, DR, UR, PRight, CR,
     &        GAMMA, G1, G2, G3, G4, G5, G6, G7, G8,
     &        CUP, GEL, GER, PMstar, PMAX, PMIN, PPV, PQ,
     &        PTL, PTR, QMAX, QUSER, UM
c
      COMMON /GAMMAS/ GAMMA, G1, G2, G3, G4, G5, G6, G7, G8
      COMMON /STATES/ DL, UL, PL, CL, DR, UR, PRight, CR
c
      QUSER = 2.0
c
C     Compute guess pressure from PVRS Riemann solver
c
      CUP  = 0.25*(DL + DR)*(CL + CR)
      PPV  = 0.5*(PL + PRight) + 0.5*(UL - UR)*CUP
      PPV  = MAX(0.0, PPV)
      PMIN = MIN(PL,  PRight)
      PMAX = MAX(PL,  PRight)
      QMAX = PMAX/PMIN
c
      IF(QMAX.LE.QUSER.AND.
     & (PMIN.LE.PPV.AND.PPV.LE.PMAX))THEN
c
C        Select PVRS Riemann solver
c
         PMstar = PPV
      ELSE
         IF(PPV.LT.PMIN)THEN
c
C           Select Two-Rarefaction Riemann solver
c
            PQ  = (PL/PRight)**G1
            UM  = (PQ*UL/CL + UR/CR +
     &            G4*(PQ - 1.0))/(PQ/CL + 1.0/CR)
            PTL = 1.0 + G7*(UL - UM)/CL
            PTR = 1.0 + G7*(UM - UR)/CR
            PMstar  = 0.5*(PL*PTL**G3 + PRight*PTR**G3)
         ELSE
c
C           Select Two-Shock Riemann solver with
C           PVRS as estimate
c
            GEL = SQRT((G5/DL)/(G6*PL + PPV))
            GER = SQRT((G5/DR)/(G6*PRight + PPV))
            PMstar=(GEL*PL+GER*PRight-(UR-UL))/(GEL+GER)
         ENDIF
      ENDIF
c
      END
c
c----------------------------------------------------------------------*
c
      SUBROUTINE PREFUN(F,FD,P,DK,PK,CK)
c
C     Purpose: to evaluate the pressure functions
C              FL and FR in exact Riemann solver
C              and their first derivatives
c
c     IMPLICIT NONE
c
C     Declaration of variables
c
      REAL    AK, BK, CK, DK, F, FD, P, PK, PRATIO, QRT,
     &        GAMMA, G1, G2, G3, G4, G5, G6, G7, G8
c
      COMMON /GAMMAS/ GAMMA, G1, G2, G3, G4, G5, G6, G7, G8
c
      IF(P.LE.PK)THEN
c
C        Rarefaction wave
c
         PRATIO = P/PK
         F    = G4*CK*(PRATIO**G1 - 1.0)
         FD   = (1.0/(DK*CK))*PRATIO**(-G2)
      ELSE
c
C        Shock wave
c
         AK  = G5/DK
         BK  = G6*PK
         QRT = SQRT(AK/(BK + P))
         F   = (P - PK)*QRT
         FD  = (1.0 - 0.5*(P - PK)/(BK + P))*QRT
      ENDIF
c
      END
c
c----------------------------------------------------------------------*
c
      SUBROUTINE SAMPLE(PMstar, UM, S, D, U, P)
c
C     Purpose: to sample the solution throughout the wave
C              pattern. Pressure PM and velocity UM in the
C              Star Region are known. Sampling is performed
C              in terms of the 'speed' S = X/T. Sampled
C              values are D, U, P
c
C     Input variables : PMstar, UM, S, /GAMMAS/, /STATES/
C     Output variables: D, U, P
c
c     IMPLICIT NONE
c
C     Declaration of variables
c
      REAL    DL, UL, PL, CL, DR, UR, PRight, CR,
     &        GAMMA, G1, G2, G3, G4, G5, G6, G7, G8,
     &        C, CML, CMR, D, P, PMstar, PML, PMR,  S,
     &        SHL, SHR, SL, SR, STL, STR, U, UM
c
      COMMON /GAMMAS/ GAMMA, G1, G2, G3, G4, G5, G6, G7, G8
      COMMON /STATES/ DL, UL, PL, CL, DR, UR, PRight, CR

      IF(S.LE.UM)THEN
c
C        Sampling point lies to the left of the contact
C        discontinuity
c
         IF(PMstar.LE.PL)THEN
c
C           Left rarefaction
c
            SHL = UL - CL
c
            IF(S.LE.SHL)THEN
c
C              Sampled point is left data state
c
               D = DL
               U = UL
               P = PL
            ELSE
               CML = CL*(PMstar/PL)**G1
               STL = UM - CML
c
               IF(S.GT.STL)THEN
c
C                 Sampled point is Star Left state
c
                  D = DL*(PMstar/PL)**(1.0/GAMMA)
                  U = UM
                  P = PMstar
               ELSE
c
C                 Sampled point is inside left fan
c
                  U = G5*(CL + G7*UL + S)
                  C = G5*(CL + G7*(UL - S))
                  D = DL*(C/CL)**G4
                  P = PL*(C/CL)**G3
               ENDIF
            ENDIF
         ELSE
c
C           Left shock
c
            PML = PMstar/PL
            SL  = UL - CL*SQRT(G2*PML + G1)
c
            IF(S.LE.SL)THEN
c
C              Sampled point is left data state
c
               D = DL
               U = UL
               P = PL
c
            ELSE
c
C              Sampled point is Star Left state
c
               D = DL*(PML + G6)/(PML*G6 + 1.0)
               U = UM
               P = PMstar
            ENDIF
         ENDIF
      ELSE
c
C        Sampling point lies to the right of the contact
C        discontinuity
c
         IF(PMstar.GT.PRight)THEN
c
C           Right shock
c
            PMR = PMstar/PRight
            SR  = UR + CR*SQRT(G2*PMR + G1)
c
            IF(S.GE.SR)THEN
c
C              Sampled point is right data state
c
               D = DR
               U = UR
               P = PRight
            ELSE
c
C              Sampled point is Star Right state
c
               D = DR*(PMR + G6)/(PMR*G6 + 1.0)
               U = UM
               P = PMstar
            ENDIF
         ELSE
c
C           Right rarefaction
c
            SHR = UR + CR
c
            IF(S.GE.SHR)THEN
c
C              Sampled point is right data state
c
               D = DR
               U = UR
               P = PRight
            ELSE
               CMR = CR*(PMstar/PRight)**G1
               STR = UM + CMR
c
               IF(S.LE.STR)THEN
c
C                 Sampled point is Star Right state
c
                  D = DR*(PMstar/PRight)**(1.0/GAMMA)
                  U = UM
                  P = PMstar
               ELSE
c
C                 Sampled point is inside left fan
c
                  U = G5*(-CR + G7*UR + S)
                  C = G5*(CR - G7*(UR - S))
                  D = DR*(C/CR)**G4
                  P = PRight*(C/CR)**G3
               ENDIF
            ENDIF
         ENDIF
      ENDIF
c
      END
c
c----------------------------------------------------------------------c
c
      subroutine cmt_usrflt(rmult)
      include 'SIZE'
      real rmult(lx1)
      real alpfilt
      integer sfilt, kut
      real eta, etac
      call rone(rmult,lx1)
      alpfilt=36.0 ! H&W 5.3
      kut=lx1/2
      sfilt=8
      etac=real(kut)/real(nx1)
      do i=kut,nx1
         eta=real(i)/real(nx1)
         rmult(i)=exp(-alpfilt*((eta-etac)/(1.0-etac))**sfilt)
      enddo
      return
      end

c automatically added by makenek
      subroutine usrsetvert(glo_num,nel,nx,ny,nz) ! to modify glo_num
      integer*8 glo_num(1)

      return
      end

c automatically added by makenek
      subroutine userqtl

      call userqtl_scig

      return
      end

c automatically added by makenek
      subroutine cmt_userflux ! user defined flux
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'CMTDATA'
      real fluxout(lx1*lz1)
      return
      end
c
c automatically added by makenek:
c -------------------------------------------------------
      subroutine place_particles_user ! used for particles
      return
      end
c
      subroutine usr_particles_f_user(i) ! used for particles
      return
      end
c -------------------------------------------------------

