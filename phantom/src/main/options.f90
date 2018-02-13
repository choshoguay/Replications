!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2017 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://users.monash.edu.au/~dprice/phantom                               !
!--------------------------------------------------------------------------!
!+
!  MODULE: options
!
!  DESCRIPTION:
!  Sets default values of input parameters
!  these are overwritten by reading from the input file or
!  by setting them in the setup routine
!
!  REFERENCES: None
!
!  OWNER: Daniel Price
!
!  $Id$
!
!  RUNTIME PARAMETERS: None
!
!  DEPENDENCIES: dim, eos, kernel, part, timestep, viscosity
!+
!--------------------------------------------------------------------------
module options
 use eos, only:ieos ! so that this is available via options
 implicit none
 character(len=80), parameter, public :: &  ! module version
    modid="$Id$"
!
! these are parameters which may be changed by the user
! but which at present are set as parameters
!
 real, public :: avdecayconst
!
! these are parameters which may be changed by the user
! and read from the input file
!
 integer, public :: nfulldump,nmaxdumps,iexternalforce
 real, public :: tolh,damp,tolv
 real(kind=4), public :: twallmax, dtwallmax

! artificial viscosity, thermal conductivity, resistivity

 real, public :: alpha,alphau,beta
 real, public :: alphamax
 real, public :: alphaB, etamhd, psidecayfac, overcleanfac
 integer, public :: ishock_heating,ipdv_heating,icooling,iresistive_heating

 public :: set_default_options
 public :: ieos

 private

contains

subroutine set_default_options
 use timestep,  only:C_cour,C_force,C_cool,tmax,dtmax,nmax,nout,restartonshortest
 use part,      only:hfact,Bextx,Bexty,Bextz,mhd,maxalpha
 use viscosity, only:set_defaults_viscosity
 use dim,       only:maxp,maxvxyzu,nalpha
 use kernel,    only:hfact_default

 C_cour = 0.3
 C_force = 0.25
 C_cool = 0.05
 tolv = 1.e-2
 tmax = 10.0
 dtmax = 1.0

 nmax = -1
 nout = -1
 nmaxdumps = -1
 twallmax = 0.0  ! maximum wall time for run, in seconds
 dtwallmax = 0.0 ! maximum wall time between full dumps, in seconds
 nfulldump = 10  ! frequency of writing full dumps
 hfact = hfact_default     ! smoothing length in units of average particle spacing
 Bextx = 0.      ! external magnetic field
 Bexty = 0.
 Bextz = 0.
 tolh = 1.e-4    ! tolerance on h iterations
 iexternalforce = 0  ! external forces
 damp = 0.       ! damping of velocities

 ! equation of state
 if (maxvxyzu==4) then
    ieos = 2
    ishock_heating = 1
    ipdv_heating = 1
    iresistive_heating = 1
 else
    ieos = 1
    ishock_heating = 1
    ipdv_heating = 1
    iresistive_heating = 1
 endif
 icooling = 0

 ! artificial viscosity
 if (maxalpha==maxp) then
    if (nalpha >= 2) then
       alpha = 0. ! Cullen-Dehnen switch
    else
       alpha = 0.1 ! Morris-Monaghan switch
    endif
 else
    alpha = 1.
 endif
 alphamax = 1.0

 ! artificial thermal conductivity
 alphau = 1.

 ! artificial resistivity (MHD only)
 alphaB = 1.
 etamhd = 0.0
 psidecayfac = 1.0  ! psi decay factor (MHD only)
 overcleanfac = 1.0  ! factor to increase signal velocity for (only) time steps and psi cleaning
 beta = 2.0      ! beta viscosity term
 avdecayconst = 0.1  ! decay time constant for viscosity switches
 restartonshortest = .false. ! whether or not to restart with all parts on shortest step

 call set_defaults_viscosity

end subroutine set_default_options

end module options
