C> @file drive1_cmt.f high-level driver for CMT-nek
C> \defgroup convhvol Volume integral for inviscid fluxes
C> \defgroup bcond Surface integrals due to boundary conditions
C> \defgroup diffhvol Volume integral for viscous fluxes
C> \defgroup vfjac Jacobians for viscous fluxes
C> \defgroup isurf Inviscid surface terms
C> \defgroup vsurf Viscous surface terms
C> \defgroup faceops utility functions for manipulating face data
C> Branch from subroutine nek_advance in core/drive1.f
C> Advance CMT-nek one time step within nek5000 time loop
      subroutine cmt_nek_advance
c     Solve the Euler equations

      include 'SIZE'
      include 'INPUT'
      include 'MASS'
      include 'TSTEP'
      include 'SOLN'
      include 'GEOM'
      include 'CTIMER'
      include 'CMTDATA'
      include 'CMTTIMERS'
      
      integer e,eq
      character*32 dumchars

      ftime_dum = dnekclock()
      nxyz1=lx1*ly1*lz1
      n = nxyz1*lelt*toteq
      nfldpart = ldim*npart
      if(istep.eq.1) then
         call cmt_ics
         if (ifrestart) then
            time_cmt=time
         else
            time_cmt=0.0 !time !0.0 ! until we can get settime to behave
         endif


         call cmt_flow_ics
         call init_cmt_timers
         call userchk ! need more ifdefs
         call compute_mesh_h(meshh,xm1,ym1,zm1)
         call compute_grid_h(gridh,xm1,ym1,zm1)
         call compute_primitive_vars ! get good mu
         call entropy_viscosity      ! for high diffno
         call compute_transport_props! at t=0

c        if(nid.eq.15) then
!            call printVdiff("before")
c        endif

         if(nid.eq.0) then !c15
            !call printMeshh("before")
            !call printXm1("before")
            !call printYm1("before")
            !call printZm1("before")
            call usr_particles_init_gpu()
         endif

      endif

      nstage = 3
      do stage=1,nstage

         rhst_dum = dnekclock()
         if(nid.eq.0) then !c15
             !print *,"cmt_nek_advance before to rhs_and_dt_gpu",nid
             if (stage.eq.1) call gpu_copy()
             
             call compute_rhs_and_dt_gpu
         else
              if (stage.eq.1) call copy(res3(1,1,1,1,1),U(1,1,1,1,1),n)
            ! print *,"cmt_nek_advance before to rhs_and_dt",nid
              call compute_rhs_and_dt
         endif
         timefortimestep = dnekclock() - rhst_dum
         rhst = rhst + timefortimestep
!         call nekgsync()
!         print *,"cmt_nek_advance after rhs_and_dt_gpu time is",rhst,
!     >      timefortimestep,stage,istep,nelt,lelt,lx1,lxd,nid
       
c particle equations of motion are solved (also includes forcing)
c In future this subroutine may compute the back effect of particles
c on the fluid and suitably modify the residue computed by 
c compute_rhs_dt for the 5 conserved variables
         !call usr_particles_solver

! JH111815 soon....
! JH082316 someday...maybe?
!        do eq=1,toteq
!           call fbinvert(res1(1,1,1,1,eq))
!        enddo
!         print *,"cmt_nek_advance after usr_particles_solver",nid
         if(nid.eq.0) then !c15
c           write(6,*) "debug tcoef:", tcoef(1,1),tcoef(1,2)
c     $,tcoef(1,3),tcoef(2,1),tcoef(2,2),tcoef(2,3)
c     $,tcoef(3,1),tcoef(3,2), tcoef(3,3)  
            call update_u_gpu
c            stop
         else 
         do e=1,nelt
            do eq=1,toteq
            do i=1,nxyz1
c multiply u with bm1 as res has been multiplied by bm1 in compute_rhs
               u(i,1,1,eq,e) = bm1(i,1,1,e)*tcoef(1,stage)
     >                     *res3(i,1,1,eq,e)+bm1(i,1,1,e)*
     >                     tcoef(2,stage)*u(i,1,1,eq,e)-
     >                     tcoef(3,stage)*res1(i,1,1,e,eq)
c              u(i,1,1,eq,e) = bm1(i,1,1,e)*u(i,1,1,eq,e) - DT *
c    >                        (c1*res1(i,1,1,e,eq) + c2*res2(i,1,1,e,eq)
c    >                       + c3*res3(i,1,1,e,eq))
c-----------------------------------------------------------------------
! JH111815 in fact, I'd like to redo the time marching stuff above and
!          have an fbinvert call for res1
               u(i,1,1,eq,e) = u(i,1,1,eq,e)/bm1(i,1,1,e)
c-----------------------------------------------------------------------
            enddo
            enddo
         enddo
         endif
!         print *,"cmt_nek_advance after 3 nested  for loop",nid
  
c     if(nid.eq.15) then
!      do i = 1, nelt
!        do j = 1, toteq
!           do k = 1, lz1
!              do m = 1, ly1
!              do n = 1,lx1
!              write(6,*) "uarray222", n, m, k, j, i, u(n,m,k,j,i)
!              enddo
!              enddo
!           enddo
!        enddo
!      enddo
!      stop
c     endif
      
      enddo

!      print *,"cmt_nek_advance after stage for loop",nid
      if(nid.eq.0) then !c15
          call compute_primitive_vars_gpu
          call copy_t_vtrans_gpu
      else

      call compute_primitive_vars ! for next time step? Not sure anymore
      call copy(t(1,1,1,1,2),vtrans(1,1,1,1,irho),nxyz1*nelt)
      endif
      ftime = ftime + dnekclock() - ftime_dum

!      if (mod(istep,iostep).eq.0.or.istep.eq.1 .or. istep.eq.2)then
!         call out_fld_nek
!         call mass_balance(if3d)
!         !dump out particle information. 
!c        call usr_particles_io(istep)
!         !following to output vx/y/z/t/pr to see the diff between cpu and gpu
!         !if(nid.eq.15) then
!         !   call printVx('debug')
!         !   call printVy('debug')
!         !   call printVz('debug')
!         !   call printTArray('debug')
!         !   call printPr('debug')
!         !endif
!      end if

!     call print_cmt_timers ! NOT NOW
!      print *,"cmt_nek_advance End for istep and nid ",istep,nid
 101  format(4(2x,e18.9))

      return
      end

c-----------------------------------------------------------------------

C> Compute right-hand-side of the semidiscrete conservation law
C> Store it in res1
      subroutine compute_rhs_and_dt
      include 'SIZE'
      include 'TOTAL'
      include 'DG'
      include 'CMTDATA'
      include 'CTIMER'

      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelt,
     >                   heresize=nqq*3*lfq,! guarantees transpose of Q+ fits
     >                   hdsize=toteq*3*lfq) ! might not need ldim
! not sure if viscous surface fluxes can live here yet
      common /CMTSURFLX/ flux(heresize),graduf(hdsize)
      real graduf

      integer e,eq
      real wkj(lx1+lxd)
      character*32  dumchars

      call compute_mesh_h(meshh,xm1,ym1,zm1)
      call compute_grid_h(gridh,xm1,ym1,zm1)

      if (lxd.gt.lx1) then
         call set_dealias_face
      else
         call set_alias_rx(istep)
      endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!     call set_dealias_rx ! done in set_convect_cons,
! JH113015                ! now called from compute_primitive_variables

!     filter the conservative variables before start of each
!     time step
!     if(IFFLTR)  call filter_cmtvar(IFCNTFILT)
!        primitive vars = rho, u, v, w, p, T, phi_g

      !only for testing. delete later.

      call compute_primitive_vars
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Compute_primitive_vars time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!-----------------------------------------------------------------------
! JH072914 We can really only proceed with dt once we have current
!          primitive variables. Only then can we compute CFL and/or dt.
!-----------------------------------------------------------------------
      if(stage.eq.1) then
         call setdtcmt
         call set_tstep_coef
      endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Setdtcmt time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      call entropy_viscosity ! accessed through uservp. computes
                             ! entropy residual and max wave speed
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Entropy_viscosity time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      call compute_transport_props ! everything inside rk stage
!     call smoothing(vdiff(1,1,1,1,imu)) ! still done in usr file
! you have GOT to figure out where phig goes!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Compute_transport_props time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      ntot = lx1*ly1*lz1*lelt*toteq
      call rzero(res1,ntot)
      call rzero(flux,heresize)
      call rzero(graduf,hdsize)

!     !Total_eqs = 5 (we will set this up so that it can be a user 
!     !defined value. 5 will be its default value)
!     !eq = 1 -------- Mass balance
!     !eq = 2 -------- x  momentum 
!     !eq = 3 -------- y  momentum 
!     !eq = 4 -------- z  momentum 
!     !eq = 5 -------- Energy Equation 

C> Restrict via \f$\mathbf{E}\f$ to get primitive and conserved variables
C> on interior faces \f$\mathbf{U}^-\f$ and neighbor faces
C> \f$\mathbf{U}^+\f$; store in CMTSURFLX
      call fluxes_full_field
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Fluxes_full_field time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

C> res1+=\f$\oint \mathbf{H}^{c\ast}\cdot\mathbf{n}dA\f$ on face points
      nstate=nqq
      nfq=lx1*lz1*2*ldim*nelt
      iwm =1
      iwp =iwm+nstate*nfq
      iflx=iwp+nstate*nfq
      do eq=1,toteq
         ieq=(eq-1)*ndg_face+iflx
         call surface_integral_full(res1(1,1,1,1,eq),flux(ieq))
      enddo
      dumchars='after_inviscid'
!     call dumpresidue(dumchars,999)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Surface_integral_full time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

               !                   -
      iuj=iflx ! overwritten with U -{{U}}
!-----------------------------------------------------------------------
!                          /     1  T \
! JH082316 imqqtu computes | I - -QQ  | U for all 5 conserved variables
!                          \     2    /
! which I now make the following be'neon-billboarded assumption:
!***********************************************************************
! ASSUME CONSERVED VARS U1 THROUGH U5 ARE CONTIGUOUSLY STORED
! SEQUENTIALLY IN /CMTSURFLX/ i.e. that iu2=iu1+1, etc.
! CMTDATA BETTA REFLECT THIS!!!
!***********************************************************************
C> res1+=\f$\int_{\Gamma} \{\{\mathbf{A}^{\intercal}\nabla v\}\} \cdot \left[\mathbf{U}\right] dA\f$
      ium=(iu1-1)*nfq+iwm
      iup=(iu1-1)*nfq+iwp
      call   imqqtu(flux(iuj),flux(ium),flux(iup))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Imqqtu time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      call   imqqtu_dirichlet(flux(iuj),flux(iwm),flux(iwp))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Imqqtu_dirichlet time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      call igtu_cmt(flux(iwm),flux(iuj),graduf) ! [[u]].{{gradv}}
      dumchars='after_igtu'
!     call dumpresidue(dumchars,999)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Igtu_cmt time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

C> res1+=\f$\int \left(\nabla v\right) \cdot \left(\mathbf{H}^c+\mathbf{H}^d\right)dV\f$ 
C> for each equation (inner), one element at a time (outer)
!      print *,"***in rhs_dt functino value of nelt is ",nelt
      do e=1,nelt
!       print *,'printing nelt',e
!-----------------------------------------------------------------------
! JH082216 Since the dawn of CMT-nek we have called this particular loop
!***********************************************************************
!*         "THE" ELEMENT LOOP                                          *
!***********************************************************************
!          since it does several operations, mostly for volume integrals,
!          for all equations, one element at a time. If we get memory
!          under control and GPUs really need to act on gigabytes all
!          at once, then this and its dependents can still have their
!          loop order flipped and things like totalh declared for
!          15 full fields or more.
!-----------------------------------------------------------------------
! Get user defined forcing from userf defined in usr file
         call cmtusrf(e)
         call compute_gradients(e) ! gradU
         do eq=1,toteq
            call convective_cmt(e,eq)        ! convh & totalh -> res1
            call    viscous_cmt(e,eq) ! diffh -> half_iku_cmt -> res1
                                             !       |
                                             !       -> diffh2graduf
! Compute the forcing term in each of the 5 eqs
            call compute_forcing(e,eq)
         enddo
      enddo
      dumchars='after_elm'
!     call dumpresidue(dumchars,999)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Compute_forcing total time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

C> res1+=\f$\int_{\Gamma} \{\{\mathbf{A}\nabla \mathbf{U}\}\} \cdot \left[v\right] dA\f$
      call igu_cmt(flux(iwp),graduf,flux(iwm))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Igu_cmt time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      do eq=1,toteq
         ieq=(eq-1)*ndg_face+iwp
!Finally add viscous surface flux functions of derivatives to res1.
         call surface_integral_full(res1(1,1,1,1,eq),flux(ieq))
      enddo
      dumchars='end_of_rhs'
!     call dumpresidue(dumchars,999)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.1) then
         end1 = dnekclock()
         write(6,*) "CPU Surface_integral_full second time", end1 -start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      return
      end
!-----------------------------------------------------------------------
C> Compute coefficients for Runge-Kutta stages \cite{TVDRK}
      subroutine set_tstep_coef

      real tcoef(3,3),dt_cmt,time_cmt
      COMMON /TIMESTEPCOEF/ tcoef,dt_cmt,time_cmt

      tcoef(1,1) = 0.0
      tcoef(2,1) = 1.0 
      tcoef(3,1) = dt_cmt
      tcoef(1,2) = 3.0/4.0
      tcoef(2,2) = 1.0/4.0 
      tcoef(3,2) = dt_cmt/4.0 
      tcoef(1,3) = 1.0/3.0
      tcoef(2,3) = 2.0/3.0 
      tcoef(3,3) = dt_cmt*2.0/3.0 

      return
      end
!-----------------------------------------------------------------------

      subroutine cmt_flow_ics
      include 'SIZE'
      include 'CMTDATA'
      include 'SOLN'

      integer e
      nxyz1 = lx1*ly1*lz1
      n     = nxyz1*lelt*toteq
      if (ifrestart)then
         do e=1,nelt
            call copy(U(1,1,1,2,e),vx(1,1,1,e),nxyz1) 
            call copy(U(1,1,1,3,e),vy(1,1,1,e),nxyz1) 
            call copy(U(1,1,1,4,e),vz(1,1,1,e),nxyz1) 
            call copy(U(1,1,1,5,e),t(1,1,1,e,1),nxyz1) 
            call copy(U(1,1,1,1,e),pr(1,1,1,e),nxyz1) 
         enddo
         call copy(tlag(1,1,1,1,1,2),t(1,1,1,1,2),nxyz1*nelt) ! s_{n-1}
         call copy(tlag(1,1,1,1,2,1),t(1,1,1,1,3),nxyz1*nelt) ! s_n
      endif
      call rzero(res1,n)
!     call copy(res2,t(1,1,1,1,5),n) ! art visc hardcoding. old entropy resid
      call rzero(res2,n) ! Actually,...
      return
      end
!-----------------------------------------------------------------------

      subroutine print_cmt_timers
      include 'SIZE'
      include 'CMTTIMERS'
      include 'TSTEP'
      include 'PARALLEL'

c we need our own IO features. Until then we use the default nek routines
      if ((mod(istep,flio_freq).eq.0.and.istep.gt.0)
     $                               .or.istep.eq.nstep)then
         dmtime1 = ftime/istep
         dtime_ = glsum(dmtime1,1)
         if(nio.eq.0) write(6,*) 'fluid rhs compute time(Avg)  '
     $               ,dtime_/np
      endif
      return 
      end
!-----------------------------------------------------------------------

      subroutine init_cmt_timers
      include 'CMTTIMERS'

      rhst    = 0.00
      ftime   = 0.00

      return
      end
!-----------------------------------------------------------------------