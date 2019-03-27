!
!     Include file to dimension static arrays
!     and to set some hardwired run-time parameters
!
      integer ldim,lx1,lxd,lx2,lx1m,lelg,lp,lelt,ldimt,if3dgpu
      integer ax1,ax2,lpx1,lpx2,lpelt,lbx1,lbx2,lbelt
      integer toteq
      integer lcvx1,lcvelt
      integer lelx,lely,lelz,mxprev,lgmres,lorder,lhis
      integer maxobj,maxmbr,lpert,nsessmax,nmaxl,nfldmax,nmaxcom

      ! GENERAL
      parameter (if3dgpu=0)        ! if3d is true for gpu version
      parameter (ldim=2)        ! domain dimension
      parameter (lx1=6)        ! polynomial order; in 2D set lz1=1
      parameter (lxd=12)        ! polynomial order for over-integration 
      parameter (lx2=lx1)       ! polynomial order for pressure
      parameter (lx1m=1)        ! polynomial order mesh solver; =1 no mesh motion

      parameter (lelg=500)     ! max total number of elements
      parameter (lp=125)       ! max number of MPI ranks
      parameter (lelt=125)       ! max number of elements per MPI rank 
      parameter (ldimt=6)       ! max number of auxiliary fields (temperature + scalars)

      ! OPTIONAL
      parameter (lelx=1,lely=1,lelz=1)          ! global tensor mesh dimensions
      parameter (ax1=1,ax2=1)                   ! averages
      parameter (lbx1=1,lbx2=1,lbelt=1)         ! mhd
      parameter (lpx1=1,lpx2=1,lpelt=1,lpert=1) ! linear stability
      parameter (toteq=5)                       ! cmt
      parameter (lcvx1=1,lcvelt=1)              ! cvode

      parameter (mxprev=1,lgmres=1)             ! projection + Krylov space dimension
      parameter (lorder=3)                      ! upper limit for order in time
      parameter (lhis=100)                      ! max intp points per MPI rank
      parameter (maxobj=4,maxmbr=lelt*6)        ! max number number of objects
      parameter (nsessmax=1,nmaxl=1,nfldmax=1,nmaxcom=1) ! multimesh parameters

      parameter (llpart=400) ! max number of particles per mpi rank

      ! INTERNALS
      include 'SIZE.inc'

! automatically added by makenek
      integer ldimt_proj
      parameter(ldimt_proj = 1) ! max auxiliary fields residual projection 
