c----------------------------------------------------------------------
      subroutine place_particles_user
c
c     Set particles initial coordinates, diameter, and velocity
c
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      include 'CMTPART'

c     ------------------------------------------------------------
c     USER SET BOUNDS TO DISTRIBUTION PARTICLES IN
c     ------------------------------------------------------------
      rxbo(1,1) = xdrange(1,1) + rspl**(1./3.)*dp(2)     ! X-Left
      rxbo(2,1) = xdrange(2,1) - rspl**(1./3.)*dp(2)     ! X-Right
      rxbo(1,2) = xdrange(1,2) + rspl**(1./3.)*dp(2)     ! Y-Left
      rxbo(2,2) = xdrange(2,2) - rspl**(1./3.)*dp(2)     ! Y-Right
      rxbo(1,3) = xdrange(1,3) + rspl**(1./3.)*dp(2)     ! Z-Left
      rxbo(2,3) = xdrange(2,3) - rspl**(1./3.)*dp(2)     ! Z-Right

      if (ipart_restartr .eq. 0) then

c        -----------------------------------------------------------
c        USER PLACE X,Y,Z PARTICLES COORDS IF DISTRIBUTING PARTICLES
c        -----------------------------------------------------------
         x_part(0) = unif_random(rxbo(1,1),rxbo(2,1)) ! X-COORD
         x_part(1) = unif_random(rxbo(1,2),rxbo(2,2)) ! Y-COORD
         x_part(2) = unif_random(rxbo(1,3),rxbo(2,3)) ! Z-COORD
        
c        -------------------------------------------------------
c        USER SET PARTICLE DIAMETERS IF DISTRIBUTING PARTICLES
c        -------------------------------------------------------
         d_part    = unif_random(dp(1),dp(2))              ! UR
c        d_part    = unif_random_norm(dp(1),dp(2),STD_DEV) ! URN

      endif

c     ---------------------------------------------------------
c     USER SET PARTICLE INITIAL VELOCITY (COMMENT IF NOT WANTED
c     FOR RESTARTING)
c     ---------------------------------------------------------
c     v_part(0) = 0. ! X-VEL
c     v_part(1) = 0. ! Y-VEL
c     v_part(2) = 0. ! Z-VEL

      return
      end
c-----------------------------------------------------------------------
      subroutine usr_particles_f_user(i)
c
c     user/body forces (total force = mass * acceleration)
c
      include 'SIZE'
      include 'TOTAL'
      include 'CMTDATA'
      include 'CMTPART'

      parameter(rgrav = 9.8)

c     --------------------
c     USER SET BODY FORCES 
c     --------------------
      f_part(0) = 0.0
      f_part(1) = 0.0
      f_part(2) = 0.0

      return
      end
c-----------------------------------------------------------------------
      subroutine uservp (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'   ! this is not
      include 'CMTDATA' ! the best idea
      include 'NEKUSE'
      integer e,eg

      e = gllel(eg)

      mu=rho*res2(ix,iy,iz,e,1) ! finite c_E;
      nu_s=0.75*mu/rho

      mu=0.5*mu 
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
      include 'CMTDATA'

      integer e,eg

      e = gllel(eg)

      call userf_particles(ix,iy,iz,e,ffx,ffy,ffz,qvol)

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
      include 'CMTDATA'
      integer  e,f

      ifxyo=.true.
      if (istep.gt.1) ifxyo=.false.

      if(mod(istep,iostep).eq.0) then
         call compute_primitive_vars
      endif
      return
      end
c-----------------------------------------------------------------------
      subroutine userbc (ix,iy,iz,iside,eg)
      include 'SIZE'
      include 'TSTEP'
      include 'NEKUSE'
      include 'INPUT'
      include 'CMTDATA'
      include 'GEOM' 

      molarmass=molmass

      return
      end

c-----------------------------------------------------------------------

      subroutine useric (ix,iy,iz,eg)
      include 'SIZE'
      include 'TOTAL'
      include 'NEKUSE'
      include 'PERFECTGAS'
      include 'CMTDATA'
      integer e,eg, eqnum

      e=gllel(eg)
      molarmass=molmass

!     force uniform
      ux = 0.01
      uy = 0.
      uz = 0.
      pres = 101000.
      rho = param(1)

      cp=cpgref
      cv=cvgref
      temp=MixtPerf_T_DPR(rho,pres,rgasref)
      asnd=MixtPerf_C_GRT(gmaref,rgasref,temp)

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

      return
      end
c-----------------------------------------------------------------------
      subroutine usrdat2
      include 'SIZE'
      include 'TOTAL'
      include 'CMTBCDATA'
      include 'CMTDATA'
      include 'PERFECTGAS'   

      outflsub=.false.
      IFCNTFILT=.false.
      ifrestart=.false.
      ifsip=.false.
      gasmodel = 1

      molmass=29.0
      muref=0.0
      gmaref     = 1.4
      coeflambda=-2.0/3.0
      suthcoef=1.0
      prlam = 0.72
      rgasref    = MixtPerf_R_M(molmass,dum)
      cvgref     = rgasref/(gmaref-1.0)
      cpgref     = MixtPerf_Cp_CvR(cvgref,rgasref)
      gmaref     = MixtPerf_G_CpR(cpgref,rgasref) 

      c_max=0.5     
      c_sub_e=10.

      return
      end
!-----------------------------------------------------------------------
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
!-----------------------------------------------------------------------
      subroutine usrdat3
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
      subroutine cmt_usrflt(rmult) ! user defined filter
      include 'SIZE'
      real rmult(lx1)
      call rone(rmult,lx1)
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
