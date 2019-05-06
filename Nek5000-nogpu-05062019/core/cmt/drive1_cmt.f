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
         !if(nid.eq.15) then
         !    print *,"cmt_nek_advance before to init_gpu",nid
         !    call usr_particles_init_gpu()
         !endif
         !if(nid.eq.15) then
         !   call printMeshh("before")
         !   call printXm1("before")
         !   call printYm1("before")
         !   call printZm1("before")
         !endif

      endif
c     if(nid.eq.0) then
c       write(6,*) "where where where", istep
c     endif

      nstage = 3
      do stage=1,nstage

         rhst_dum = dnekclock()
         if (stage.eq.1) call copy(res3(1,1,1,1,1),U(1,1,1,1,1),n)
            !if(nid.eq.15) call printRes3('firstCopy')

         call compute_rhs_and_dt
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
         if(nid.eq.15) then
            call printUarray('last')
         endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         !endif
         !print *,"cmt_nek_advance after rhs_and_dt_gpu",nid
         rhst = rhst + dnekclock() - rhst_dum
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
         !print *,"cmt_nek_advance after usr_particles_solver",nid
         do e=1,nelt
            do eq=1,toteq
            do i=1,nxyz1
c multiply u with bm1 as res has been multiplied by bm1 in compute_rhs
c              if(nid.eq.15 .and. e.eq.1 .and. 
c    $     eq.eq.2 .and. i.eq.11) then
c                write(6,*) "debug u here", bm1(i,1,1,e), tcoef(1,stage)
c    $ ,res3(i,1,1,eq,e), tcoef(2,stage), u(i,1,1,eq,e), tcoef(3,stage)
c    $ , res1(i,1,1,e,eq), bm1(i,1,1,e)*tcoef(1,stage)*res3(i,1,1,eq,e)
c    $ , bm1(i,1,1,e)*tcoef(2,stage)*u(i,1,1,eq,e), 
c    $ tcoef(3,stage)*res1(i,1,1,e,eq)
c              endif

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
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
         if(nid.eq.15) then
            write(6,*) "debug tcoef222222:", step, stage, tcoef(1,1),
     >  tcoef(1,2)
     > ,tcoef(1,3),tcoef(2,1),tcoef(2,2),tcoef(2,3),tcoef(3,1)
     > ,tcoef(3,2), tcoef(3,3)
            call printUarray('afterUpdateU')
            call printRes1('afterUpdateU')
            call printRes3('afterUpdateU')
            call printBm1('afterUpdateU')
c           stop
         endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         !print *,"cmt_nek_advance after 3 nested  for loop",nid
      enddo

      !print *,"cmt_nek_advance after stage for loop",nid
      call compute_primitive_vars ! for next time step? Not sure anymore
      call copy(t(1,1,1,1,2),vtrans(1,1,1,1,irho),nxyz1*nelt)
      ftime = ftime + dnekclock() - ftime_dum
c     if(nid.eq.15 .and. mod(istep,iostep) .eq. 0) then
c     print *,"1cmt_nek_advance after stage for loop",nid
c        call printUarray("fffinal")
c     endif

      if (mod(istep,iostep).eq.0.or.istep.eq.1 .or. istep.eq.2)then
         call out_fld_nek
         call mass_balance(if3d)
         ! dump out particle information. 
c        call usr_particles_io(istep)
c        if(nid.eq.15) then
c           call printVx('debug')
c           call printVy('debug')
c           call printVz('debug')
c           call printTArray('debug')
c           call printPr('debug')
c        endif
      end if

!     call print_cmt_timers ! NOT NOW
!     print *,"cmt_nek_advance End for istep and nid ",istep,nid
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
      real*8 start, end1

#ifdef DEBUG
      if(nid.eq.15) then
        call printVdiff("begin")
        call printTArray("begin")
      endif
#endif
      call compute_mesh_h(meshh,xm1,ym1,zm1)
      call compute_grid_h(gridh,xm1,ym1,zm1)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
c     if(nid.eq.15) then
c      call printUnx("afterCompMeshh")
c      call printXm1("afterCompMeshh")
c      call printYm1("afterCompMeshh")
c      call printZm1("afterCompMeshh")
c      call printMeshh("afterCompMeshh")
c      call printGridh("afterCompMeshh")
c     endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      if (lxd.gt.lx1) then
         call set_dealias_face
c        if(nid.eq.15) then
c           call printUnx("afterDealias")
c        endif
      else
         call set_alias_rx(istep)
      endif

!     call set_dealias_rx ! done in set_convect_cons,
! JH113015                ! now called from compute_primitive_variables

!     filter the conservative variables before start of each
!     time step
!     if(IFFLTR)  call filter_cmtvar(IFCNTFILT)
!        primitive vars = rho, u, v, w, p, T, phi_g

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef MTIME
      if(nid.eq.15) then
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      call compute_primitive_vars

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !if(nid.eq.15) then
      !   call printVxyzd("afterComp")
      !endif
#ifdef DEBUG
      if(nid.eq.15) then
        call printUarray('afterPrimitiveVars')
        call printVx('afterPrimitiveVars')
        call printVy('afterPrimitiveVars')
        call printVz('afterPrimitiveVars')
        call printCsound('afterPrimitiveVars')
        call printUnx("afterPrimitiveVars")
        call printVdiff("afterPrimitiveVars")
        call printTArray("afterPrimitiveVars")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
         end1 = dnekclock()
         write(6,*) "CPU Compute_primitive_vars time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

c     if(nid.eq.15) then
c       print *,"$$$ drive1_cmt.f compute_primitive_vars check", nid
c       do i=1,10
c           print *, 'vx, vy, vz, vtrans,t,pr,csound,vxd,vyd,vzd', i,
c    >       vx(i),vy(i),vz(i),vtrans(i),t(i),pr(i),csound(i),vxd(i),
c    >       vyd(i),vzd(i)
c       enddo
c     endif
!-----------------------------------------------------------------------
! JH072914 We can really only proceed with dt once we have current
!          primitive variables. Only then can we compute CFL and/or dt.
!-----------------------------------------------------------------------
      if(stage.eq.1) then
         call setdtcmt
         call set_tstep_coef
      endif

#ifdef DEBUG
      if(nid.eq.15) then
         call printVtrans("beforeEntropy")
         call printTlag("beforeEntropy")
         call printRes2("beforeEntropy")
         call printTArray("beforeEntropy")
         call printVdiff("beforeEntropy")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
         end1 = dnekclock()
         write(6,*) "CPU Setdtcmt time", end1 - start
         start = dnekclock()
      endif
#endif
      call entropy_viscosity ! accessed through uservp. computes
                             ! entropy residual and max wave speed
#ifdef DEBUG
      if(nid.eq.15) then
         call printVdiff("afterEntropy")
         call printRes2("afterEntropy")
         call printTlag("afterEntropy")
         call printUnx("afterEntropy")
         call printTArray("afterEntropy")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
         end1 = dnekclock()
         write(6,*) "CPU Entropy_viscosity time", end1 - start
         start = dnekclock()
      endif
#endif

      call compute_transport_props ! everything inside rk stage
!     call smoothing(vdiff(1,1,1,1,imu)) ! still done in usr file
! you have GOT to figure out where phig goes!!!!

#ifdef DEBUG
      if(nid.eq.15) then
         call printVdiff("afterTransportProps")
         call printUnx("afterTransportProps")
         call printTArray("afterTransportProps")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
         end1 = dnekclock()
         write(6,*) "CPU Compute_transport_props time", end1 - start
         start = dnekclock()
      endif
#endif

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
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
      if(nid.eq.15) then
         call printFatface('beforefluxes')
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      call fluxes_full_field
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
      if(nid.eq.15) then
         call printRes1('afterfluxes')
         call printFatface('afterfluxes')
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
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
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
      if(nid.eq.15) then
         call printRes1('afterfirstintegral')
         call printFatface("afterfirstintegral")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
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
      ium=(iu1-1)*nfq+iwm !iu1=14
      iup=(iu1-1)*nfq+iwp
      !if(nid.eq.15) then
      !   write(6,*) "debug ium, iup", iu1, ium, iup, iwm, iwp
      !endif
      call   imqqtu(flux(iuj),flux(ium),flux(iup))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
      if(nid.eq.15) then
         call printRes1('afterImqqtu')
         call printFatface("afterImqqtu")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
         end1 = dnekclock()
         write(6,*) "CPU Imqqtu time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      call   imqqtu_dirichlet(flux(iuj),flux(iwm),flux(iwp))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
      if(nid.eq.15) then
         call printRes1('afterImqqtud')
         call printFatface("afterImqqtud")
         call printGraduf("afterImqqtud")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
         end1 = dnekclock()
         write(6,*) "CPU Imqqtu_dirichlet time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      call igtu_cmt(flux(iwm),flux(iuj),graduf) ! [[u]].{{gradv}}
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
      if(nid.eq.15) then
         call printRes1('afterIgtu')
         call printFatface("afterIgtu")
         call printGraduf("afterIgtu")
         call printUarray("afterIgtu")
         call printPhig("afterIgtu")
         call printVxyzd("afterIgtu")
         call printRx("afterIgtu")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
         end1 = dnekclock()
         write(6,*) "CPU Igtu_cmt time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      dumchars='after_igtu'
!     call dumpresidue(dumchars,999)

C> res1+=\f$\int \left(\nabla v\right) \cdot \left(\mathbf{H}^c+\mathbf{H}^d\right)dV\f$ 
C> for each equation (inner), one element at a time (outer)
      !print *,"***in rhs_dt functino value of nelt is ",nelt
      do e=1,nelt
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



         !comment for debug by Kk 02/13
         ! uncomment later
          do eq=1,toteq
             call convective_cmt(e,eq)        ! convh & totalh -> res1
             call    viscous_cmt(e,eq) ! diffh -> half_iku_cmt -> res1
         !                                    !       |
         !                                    !       -> diffh2graduf
! Compute the forcing term in each of the 5 eqs
             call compute_forcing(e,eq)
          enddo
      enddo
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
      if(nid.eq.15) then
         !gradu is local variable, output in compute_gradients
         !call printRes1('afterConvective')
         !call printRes1('afterViscousCMT')
         !call printRes1('afterCompGrad')
         call printRes1('afterCompForcing')
c        call printUarray('afterCompGrad')
c        call printPhig('afterCompGrad')
c        call printDxm1('afterCompGrad')
c        call printRxm1('afterCompGrad')
c        call printJacmi('afterCompGrad')
c        call printDgDgt("afterIgtu")
         call printFatface("afterCompForcing")
         !call printGraduf("afterCompForcing")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
         end1 = dnekclock()
         write(6,*) "CPU Compute_forcing total time", end1 - start
         start = dnekclock()
      endif
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      dumchars='after_elm'
!     call dumpresidue(dumchars,999)

C> res1+=\f$\int_{\Gamma} \{\{\mathbf{A}\nabla \mathbf{U}\}\} \cdot \left[v\right] dA\f$
      call igu_cmt(flux(iwp),graduf,flux(iwm))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
      if(nid.eq.15) then
         call printRes1('afterIguCMT')
         call printFatface("afterIguCMT")
         call printGraduf("afterIguCMT")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
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
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ifdef DEBUG
      if(nid.eq.15) then
         call printRes1('last')
         call printUarray('last')
         call printVdiff("last")
         call printTArray("last")
      endif
#endif
#ifdef MTIME
      if(nid.eq.15) then
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
      subroutine printUarray(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'TSTEP'

      character (len=*), intent(in):: atype
      real   lpm_xerange(2,3,lelt)
      common /lpm_elementrange/ lpm_xerange

      integer isprint, i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'
      isprint = 0
         do i = 1, nelt
            write(x1, fmt) i
            write(x2, fmt) istep
            write(x3, fmt) stage
          OPEN(UNIT=9999+i,FILE='uarray.id.'//trim(x1)//'.' 
     >       //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >      trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
              do j=1, toteq
                do k=1, lz1
                   do n=1, ly1
                       do m=1, lx1
            WRITE(UNIT=9999+i, FMT=*) m, n, k, j, u(m,n,k,j,i)
                       enddo
                   enddo
                enddo
             enddo
            CLOSE(UNIT=9999+i)
        enddo
      end

!---------------------------------------------------------------------
!---------------------------------------------------------------------
      subroutine printRes1(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA' !include res2
      include 'SOLN'
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

!     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      do i = 1, nelt
!        geid = lglel(i)
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=12299+i,FILE='res1.id.'//trim(x1)//'.' 
     >         //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >        trim(atype),FORM="FORMATTED", 
     >        STATUS="REPLACE",ACTION="WRITE")
         do nn=1, toteq
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=12299+i, FMT = *) m,k,j, nn 
     >  ,res1(m, k, j, i, nn)
         enddo
         enddo
         enddo
         enddo
         CLOSE(UNIT=12299+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printRes3(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA' !include res2
      include 'SOLN'
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=14299+i,FILE='res3.id.'//trim(x1)//'.' 
     >        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >        trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, toteq
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=14299+i, FMT = *) m,k,j, nn 
     > ,res3(m, k, j, nn, i)
         enddo
         enddo
         enddo
         enddo
         CLOSE(UNIT=14299+i)
      enddo
      end


!---------------------------------------------------------------------
      subroutine printBm1(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'MASS' !include bm1
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

!     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      do i = 1, nelt
         write(x1, fmt) i
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=13299+i,FILE='bm1.id.'//trim(x1)//'.' 
     >        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >       trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=13299+i, FMT = *) m,k,j 
     > ,bm1(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=13299+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printFatface(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA' !include res2
      include 'SOLN'
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelt, 
     >                  heresize=nqq*3*lfq, ! guarantees transpose of Q+ fits
     >                  hdsize=toteq*3*lfq) ! might not need ldim
      common /CMTSURFLX/ flux(heresize),graduf(hdsize)
      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3, x4

      fmt = '(I4.4)'
      lfn = lx1*lz1*2*ldim*nelt
      do j = 1, 3
      do i = 1, nqq
         write(x1, fmt) i !geid
         write(x2, fmt) j !geid
         write(x3, fmt) istep
         write(x4, fmt) stage
         OPEN(UNIT=12399+i,FILE='fatface.nqq.'//trim(x1)//'.iwp.' 
     >       //trim(x2) 
     >       //'.step.'//trim(x3)//'.stage.'//trim(x4)//'.'// 
     >        trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, lfn
          WRITE(UNIT=12399+i, FMT = *) nn 
     > ,flux((j-1)*nqq*lfn + (i-1)*lfn + nn)
c         WRITE(UNIT=12399+i, FMT = "(I10, ES30.13)") nn 
c    > ,flux((j-1)*nqq*lfn + (i-1)*lfn + nn)
         enddo
         CLOSE(UNIT=12399+i)
      enddo
      enddo
      end
!---------------------------------------------------------------------
      subroutine printArea(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include area
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=13399+i,FILE='area.id.'//trim(x1)//'.' 
     >        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >       trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, 6
         do k=1, lz1
         do m=1, lx1
          WRITE(UNIT=13399+i, FMT = *) m,k,j 
     > ,area(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=13399+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printWghtc(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'DG' !include wghtc
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=13499,FILE='wghtc.id.'!//trim(x1)//'.' 
     >        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >       trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do k=1, lz1
         do m=1, lx1
          WRITE(UNIT=13499, FMT = *) m,k 
     > ,wghtc((k-1)*lx1+m)
         enddo
         enddo
         do k=1, lzd
         do m=1, lxd
          WRITE(UNIT=13499, FMT = *) m,k 
     > ,wghtf((k-1)*lxd+m)
         enddo
         enddo
         CLOSE(UNIT=13499)
      end
!---------------------------------------------------------------------
      subroutine printVx(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'SOLN' !include vx
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'
      do i = 1, nelt
!        geid = lglel(i)
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=10799+i,FILE='vx.id.'//trim(x1)//'.'
     $        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'
     $      //trim(atype),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=10799+i, FMT = *) m,k,j
     $ ,vx(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=10799+i)
      enddo
      end
c---------------------------------------------------------------------
      subroutine printVy(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'SOLN' !include vx
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2,x3

      fmt = '(I4.4)'
      do i = 1, nelt
c        geid = lglel(i)
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=10899+i,FILE='vy.id.'//trim(x1)//'.'
     $        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'
     $    //trim(atype),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=10899+i, FMT = *) m,k,j
     $ ,vy(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=10899+i)
      enddo
      end

c---------------------------------------------------------------------
      subroutine printVz(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'SOLN' !include vx
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2,x3

      fmt = '(I4.4)'

c     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      do i = 1, nelt
c        geid = lglel(i)
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=10999+i,FILE='vz.id.'//trim(x1)//'.'
     $        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'
     $   //trim(atype),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=10999+i, FMT = *) m,k,j
     $ ,vz(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=10999+i)
      enddo
      end

c---------------------------------------------------------------------
      subroutine printCsound(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'SOLN' !include vx
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'
      do i = 1, nelt
c        geid = lglel(i)
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=11999+i,FILE='csound.id.'//trim(x1)//'.'
     $        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'
     $    //trim(atype),FORM="FORMATTED",
     $        STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=11999+i, FMT = *) m,k,j
     $ ,csound(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=11999+i)
      enddo
      end
!---------------------------------------------------------------------
      subroutine printVtrans(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'SOLN' !include vdiff
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         OPEN(UNIT=12199+i,FILE='vtrans.id.'//trim(x1)//'.' 
     >       //'step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, ldimt1
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=12199+i, FMT = *) m,k,j, nn 
     > ,vtrans(m, k, j, i, nn)
         enddo
         enddo
         enddo
         enddo
         CLOSE(UNIT=12199+i)
      enddo
      end
!---------------------------------------------------------------------
      subroutine printVdiff(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'SOLN' !include vdiff
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

!     if(istep .eq. 2 .or.istep .eq. 1) then ! .and. isprint .eq. 1) then
      do i = 1, nelt
!        geid = lglel(i)
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=12099+i,FILE='vdiff.id.'//trim(x1)//'.' 
     >       //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'
     >       //trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, ldimt1
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=12099+i, FMT = *) m,k,j, nn 
     > ,vdiff(m, k, j, i, nn)
         enddo
         enddo
         enddo
         enddo
         CLOSE(UNIT=12099+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printRes2(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA' !include res2
      include 'SOLN'
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=12299+i,FILE='res2.id.'//trim(x1)//'.'  
     >       //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'
     >       //trim(atype),FORM="FORMATTED",  
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, toteq
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=12299+i, FMT = *) m,k,j, nn  
     > ,res2(m, k, j, i, nn)
         enddo
         enddo
         enddo
         enddo
         CLOSE(UNIT=12299+i)
      enddo
      end


!---------------------------------------------------------------------
      subroutine printTlag(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'SOLN' !include tlag
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, j, k, m,tt, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=13199+i,FILE='tlag.id.'//trim(x1)//'.' 
     >       //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'
     >       //trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, ldimt1
         do tt=1, lorder-1
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=13199+i, FMT = *) m,k,j, tt, nn 
     > ,tlag(m, k, j, i, tt, nn)
         enddo
         enddo
         enddo
         enddo
         enddo
         CLOSE(UNIT=13199+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printTArray(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'SOLN' !include t
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=14199+i,FILE='t.id.'//trim(x1)//'.' 
     >       //'step.'//trim(x2)//'.stage.'//trim(x3)//"."
     >       //trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, ldimt
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=14199+i, FMT = *) m,k,j, nn 
     > ,t(m, k, j, i, nn)
         enddo
         enddo
         enddo
         enddo
         CLOSE(UNIT=14199+i)
      enddo
      end


!---------------------------------------------------------------------
      subroutine printMeshh(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA' !include meshh
      include 'SOLN' !include t
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2

      fmt = '(I4.4)'

      write(x2, fmt) istep
      OPEN(UNIT=14299,FILE='meshh.' 
     >//'step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >STATUS="REPLACE",ACTION="WRITE")
      do i = 1, nelt
          WRITE(UNIT=14299, FMT = *) i 
     > , meshh(i)
      enddo
      CLOSE(UNIT=14299)
      end

!---------------------------------------------------------------------
      subroutine printXm1(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include xm1
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         OPEN(UNIT=14199+i,FILE='xm1.id.'//trim(x1)//'.' 
     >       //'step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=14199+i, FMT = *) m,k,j 
     > ,xm1(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=14199+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printYm1(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include xm1
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         OPEN(UNIT=14299+i,FILE='ym1.id.'//trim(x1)//'.' 
     >       //'step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=14299+i, FMT = *) m,k,j 
     > ,ym1(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=14299+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printZm1(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include xm1
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         OPEN(UNIT=14199+i,FILE='zm1.id.'//trim(x1)//'.' 
     >       //'step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=14199+i, FMT = *) m,k,j 
     > ,zm1(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=14199+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printGridh(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA' !include gridh
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         OPEN(UNIT=14399+i,FILE='gridh.id.'//trim(x1)//'.' 
     >       //'step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=14399+i, FMT = *) m,k,j 
     > ,gridh(m+ (k-1)*lx1+(j-1)*lx1*ly1, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=14399+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printCbc(atype)
      include 'SIZE'
      include 'INPUT' !include cbc
      include 'PARALLEL'
      include 'CMTDATA' 
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         OPEN(UNIT=14399+i,FILE='cbc.id.'//trim(x1)//'.' 
     >       //'step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=0, ldimt1
         do k=1, 6
          WRITE(UNIT=14399+i, FMT = *) k,j 
     > , cbc(k, i, j)
         enddo
         enddo
         CLOSE(UNIT=14399+i)
      enddo
      end
!---------------------------------------------------------------------
      subroutine printUnx(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include unx
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelt, 
     >                  heresize=nqq*3*lfq, ! guarantees transpose of Q+ fits
     >                  hdsize=toteq*3*lfq) ! might not need ldim
      common /CMTSURFLX/ flux(heresize),graduf(hdsize)
      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         OPEN(UNIT=12499+i,FILE='unx.'//trim(x1) 
     >       //'.step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, 6
         do j=1, lz1
         do k=1, lx1
          WRITE(UNIT=12499+i, FMT = *) k, j, nn 
     > , unx(k, j, nn, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=12499+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printUnz(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include unz
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelt, 
     >                  heresize=nqq*3*lfq, ! guarantees transpose of Q+ fits
     >                  hdsize=toteq*3*lfq) ! might not need ldim
      common /CMTSURFLX/ flux(heresize),graduf(hdsize)
      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         OPEN(UNIT=12699+i,FILE='unz.'//trim(x1) 
     >       //'.step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, 6
         do j=1, lz1
         do k=1, lx1
          WRITE(UNIT=12699+i, FMT = *) k, j, nn 
     > , unz(k, j, nn, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=12699+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printUny(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include uny
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelt, 
     >                  heresize=nqq*3*lfq, ! guarantees transpose of Q+ fits
     >                  hdsize=toteq*3*lfq) ! might not need ldim
      common /CMTSURFLX/ flux(heresize),graduf(hdsize)
      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         OPEN(UNIT=12599+i,FILE='uny.'//trim(x1) 
     >       //'.step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, 6
         do j=1, lz1
         do k=1, lx1
          WRITE(UNIT=12599+i, FMT = *) k, j, nn 
     > , uny(k, j, nn, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=12599+i)
      enddo
      end

c--------------------------------------------------------------------
      subroutine printJglt(atype)
      include 'SIZE'
      !include 'INPUT'
      !include 'PARALLEL'
      !include 'CMTDATA'
      !include 'TSTEP'

      character (len=*), intent(in):: atype
      parameter (ldg=lxd**3,lwkd=4*lxd*lxd)
      common /dgrad/ d(ldg),dt(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

         OPEN(UNIT=12599,FILE='jglt' 
     >       //'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do k=1, ldg
          WRITE(UNIT=12599, FMT = *) k  
     > , jgl(k), jgt(k)
         enddo
         CLOSE(UNIT=12599)
      end

!---------------------------------------------------------------------
      subroutine printGraduf(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA' !include res2
      include 'SOLN'
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer lfq,heresize,hdsize
      parameter (lfq=lx1*lz1*2*ldim*lelt, 
     >                  heresize=nqq*3*lfq, ! guarantees transpose of Q+ fits
     >                  hdsize=toteq*3*lfq) ! might not need ldim
      common /CMTSURFLX/ flux(heresize),graduf(hdsize)
      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3, x4

      fmt = '(I4.4)'
      lfn = lx1*lz1*2*ldim*nelt
      do j = 1, 3
      do i = 1, toteq
         write(x1, fmt) i !geid
         write(x2, fmt) j !geid
         write(x3, fmt) istep
         write(x4, fmt) stage
         OPEN(UNIT=12499+i,FILE='graduf.eq.'//trim(x1)//'.iwp.' 
     >       //trim(x2) 
     >       //'.step.'//trim(x3)//'.stage.'//trim(x4)//'.'// 
     >        trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do nn=1, lfn
          WRITE(UNIT=12499+i, FMT = *) nn 
     > ,graduf((j-1)*toteq*lfn + (i-1)*lfn + nn)
c         WRITE(UNIT=12399+i, FMT = "(I10, ES30.13)") nn 
c    > ,flux((j-1)*nqq*lfn + (i-1)*lfn + nn)
         enddo
         CLOSE(UNIT=12499+i)
      enddo
      enddo
      end

!---------------------------------------------------------------------
      subroutine printDxm1(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'DXYZ' !include dxm1 dxtm1
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

         write(x2, fmt) istep
         OPEN(UNIT=12699+i,FILE='dxm1.' 
     >       //'.step.'//trim(x2)//'.'//trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lx1
         do k=1, lx1
          WRITE(UNIT=12699+i, FMT = *) k, j 
     > , dxm1(k, j), dxtm1(k,j)
         enddo
         enddo
         CLOSE(UNIT=12699+i)
      end

!---------------------------------------------------------------------
      subroutine printPhig(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'MASS' !include bm1
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'
      do i = 1, nelt
         write(x1, fmt) i
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=13399+i,FILE='phig.id.'//trim(x1)//'.' 
     >        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >       trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=13399+i, FMT = *) m,k,j 
     > ,phig(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=13399+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printRxm1(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include rxm1
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'
      do i = 1, nelt
         write(x1, fmt) i
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=13499+i,FILE='rxm1.id.'//trim(x1)//'.' 
     >        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >       trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz1
         do k=1, ly1
         do m=1, lx1
          WRITE(UNIT=13499+i, FMT = *) m,k,j 
     > ,rxm1(m, k, j, i), rym1(m, k, j, i), rzm1(m, k, j, i)
     > ,sxm1(m, k, j, i), sym1(m, k, j, i), szm1(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=13499+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printJacmi(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include rxm1
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'
      do i = 1, nelt
         write(x1, fmt) i
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=13599+i,FILE='jacmi.id.'//trim(x1)//'.' 
     >        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >       trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lx1*ly1*lz1
          WRITE(UNIT=13599+i, FMT = *) j 
     > ,jacmi(j, i)
         enddo
         CLOSE(UNIT=13599+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printVxyzd(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'DEALIAS' !include vxd, vyd, vzd
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'
      do i = 1, nelt
         write(x1, fmt) i
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=13699+i,FILE='vxyzd.id.'//trim(x1)//'.' 
     >        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >       trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lzd
         do k=1, lyd
         do m=1, lxd
          WRITE(UNIT=13699+i, FMT = *) m,k,j 
     > ,vxd(m, k, j, i), vyd(m, k, j, i), vzd(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=13699+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printRx(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include rx
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'
      do i = 1, nelt
         write(x1, fmt) i
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=13699+i,FILE='rx.id.'//trim(x1)//'.' 
     >        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >       trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do k=1, ldim*ldim
         do j=1, lxd*lyd*lzd
          WRITE(UNIT=13699+i, FMT = *) j,k 
     > ,rx(j, k, i)
         enddo
         enddo
         CLOSE(UNIT=13699+i)
      enddo
      end

!---------------------------------------------------------------------
      subroutine printDgDgt(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'GEOM' !include rx
      include 'TSTEP'

      parameter (ldd=lxd*lyd*lzd)
      parameter (ldg=lxd**3,lwkd=2*ldg)
      common /ctmp1/ ur(ldd),us(ldd),ut(ldd),ju(ldd),ud(ldd),tu(ldd)
      real ju
      common /dgrad/ d(ldg),dt1(ldg),dg(ldg),dgt(ldg),jgl(ldg),jgt(ldg)
     $             , wkd(lwkd)
      real jgl,jgt

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=13699+i,FILE='dgdgt'//'.' 
     >        //'step.'//trim(x2)//'.stage.'//trim(x3)//'.'// 
     >       trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, ldg
          WRITE(UNIT=13699+i, FMT = *) j 
     > ,dg(j), dgt(j)
         enddo
         CLOSE(UNIT=13699+i)
      end

!---------------------------------------------------------------------
      subroutine printPr(atype)
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'CMTDATA'
      include 'SOLN' !include pr
      include 'TSTEP'

      character (len=*), intent(in):: atype

      integer i, geid, pn
      character (len=8):: fmt !format descriptor
      character(5) x1, x2, x3

      fmt = '(I4.4)'

      do i = 1, nelt
         write(x1, fmt) i !geid
         write(x2, fmt) istep
         write(x3, fmt) stage
         OPEN(UNIT=16199+i,FILE='pr.id.'//trim(x1)//'.'  
     >       //'step.'//trim(x2)//'.stage.'//trim(x3)//'.' 
     >       //trim(atype),FORM="FORMATTED", 
     >       STATUS="REPLACE",ACTION="WRITE")
         do j=1, lz2
         do k=1, ly2
         do m=1, lx2
          WRITE(UNIT=16199+i, FMT = *) m,k,j 
     > ,pr(m, k, j, i)
         enddo
         enddo
         enddo
         CLOSE(UNIT=16199+i)
      enddo
      end
