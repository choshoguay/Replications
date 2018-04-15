!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2017 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://users.monash.edu.au/~dprice/phantom                               !
!--------------------------------------------------------------------------!
!+
!  MODULE: setup
!
!  DESCRIPTION:
!   Setup of dust settling problem from PL15
!
!  REFERENCES: Price & Laibe (2015), MNRAS 451, 5332
!
!  OWNER: Daniel Price
!
!  $Id$
!
!  RUNTIME PARAMETERS: None
!
!  DEPENDENCIES: boundary, dim, dust, externalforces, io, mpiutils,
!    options, part, physcon, prompting, setup_params, timestep, unifdis,
!    units
!+
!--------------------------------------------------------------------------
module setup
 implicit none
 public :: setpart

 private

contains

!----------------------------------------------------------------
!+
!  setup for uniform particle distributions
!+
!----------------------------------------------------------------
subroutine setpart(id,npart,npartoftype,xyzh,massoftype,vxyzu,polyk,gamma,hfact,time,fileprefix)
 use setup_params,   only:npart_total,rhozero
 use io,             only:master
 use unifdis,        only:set_unifdis
 use boundary,       only:set_boundary,xmin,xmax,zmin,zmax,dxbound,dzbound
 use mpiutils,       only:bcast_mpi
 use part,           only:labeltype,set_particle_type,igas,dustfrac
 use physcon,        only:pi,au,solarm
 use dim,            only:maxvxyzu,use_dust,use_dustfrac,maxp
 use prompting,      only:prompt
 use externalforces, only:mass1,Rdisc,iext_discgravity
 use options,        only:iexternalforce
 use timestep,       only:dtmax,tmax
 use units,          only:set_units,udist
 use dust,           only:init_drag,grainsizecgs,grainsize,graindens,get_ts
 integer,           intent(in)    :: id
 integer,           intent(inout) :: npart
 integer,           intent(out)   :: npartoftype(:)
 real,              intent(out)   :: xyzh(:,:)
 real,              intent(out)   :: vxyzu(:,:)
 real,              intent(out)   :: massoftype(:)
 real,              intent(out)   :: polyk,gamma,hfact
 real,              intent(inout) :: time
 character(len=20), intent(in)    :: fileprefix
 real :: totmass,deltax,dz,length
 integer :: i,iregime,ierr
 integer :: itype,ntypes,npartx
 integer :: npart_previous
 real    :: H0,HonR,omega,ts
 real    :: xmini,xmaxi,ymaxdisc,cs,dtg,t_orb
!
! default options
!
 npartx = 32
 if (id==master) then
    itype = 1
    print "(/,a,/)",'  >>> Setting up particles for dust settling test <<<'
    call prompt(' enter number of '//trim(labeltype(itype))//' particles in x ',npartx,8,maxp/144)
 endif
 call bcast_mpi(npartx)
 rhozero = 1.e-3
 if (id==master) call prompt('enter '//trim(labeltype(itype))//&
                      ' midplane density (gives particle mass)',rhozero,0.)
 call bcast_mpi(rhozero)
 dtg = 0.
 if (use_dustfrac) then
    if (id==master) call prompt('enter dust-to-gas ratio',dtg,0.,1.)
    call bcast_mpi(dtg)
 endif
 call set_units(dist=10.*au,mass=solarm,G=1.)
!
! general parameters
!
 HonR  = 0.05
 Rdisc = 5.
 mass1 = 1.
 H0    = HonR*Rdisc
 omega = sqrt(mass1/Rdisc**3)
 t_orb = 2.*pi/omega
 cs    = H0*omega
 time   = 0.
 iexternalforce = iext_discgravity
 dtmax = 0.1*t_orb
 tmax  = 15.*t_orb
!
! equation of state
!
 if (maxvxyzu >= 4) then
    gamma = 5./3.
 else
    gamma  = 1.
    polyk  = cs**2
 endif
!
! get stopping time information
!
 grainsizecgs = 0.1
 print*,' grain size in cgs = ',0.1
 call init_drag(ierr)
 call get_ts(1,grainsize,graindens,rhozero,0.0*rhozero,cs,0.,ts,iregime)
 print*,' ts * Omega for 1mm grains = ',ts*omega
!
! boundaries
!
 xmini = -0.25
 xmaxi =  0.25
 length = xmaxi - xmini
 deltax = length/npartx
 dz = 2.*sqrt(6.)/npartx
 !deltay = fac*deltax*sqrt(0.75)
 call set_boundary(xmini,xmaxi,-10.*H0,10.*H0,-dz,dz)

 npart = 0
 npart_total = 0
 npartoftype(:) = 0

 ntypes = 1
 if (use_dust .and. .not.use_dustfrac) ntypes = 2

 overtypes: do itype=1,ntypes
    !
    ! get total mass from integration of density profile
    !
    ymaxdisc = 3.*H0
    totmass = 2.*rhozero*sqrt(0.5*pi)*H0*erf(ymaxdisc/(sqrt(2.)*H0))*dxbound*dzbound

    npart_previous = npart

    call set_unifdis('closepacked',id,master,xmin,xmax,-ymaxdisc,ymaxdisc,zmin,zmax,deltax, &
                      hfact,npart,xyzh,nptot=npart_total,rhofunc=rhofunc,dir=2)

    !--set which type of particle it is
    do i=npart_previous+1,npart
       call set_particle_type(i,itype)

       vxyzu(:,i) = 0.
!
!--set internal energy if necessary
!
       if (maxvxyzu >= 4) then
          if (gamma > 1.) then
             vxyzu(4,i) = cs**2/(gamma-1.)
          else
             vxyzu(4,i) = 1.5*cs**2
          endif
       endif
!
!--one fluid dust: set dust fraction on gas particles
!
       if (use_dustfrac) then
          if (itype==igas) then
             dustfrac(i) = dtg/(1. + dtg)
          else
             dustfrac(i) = 0.
          endif
       endif
    enddo

    npartoftype(itype) = npart - npart_previous
    if (id==master) print*,' npart = ',npart,npart_total

    massoftype(itype) = totmass/npartoftype(itype)*(1. + dtg)
    if (id==master) print*,' particle mass = ',massoftype(itype)

 enddo overtypes

contains

real function rhofunc(x)
 real, intent(in) :: x

 rhofunc = exp(-0.5*(x/H0)**2)

end function rhofunc

end subroutine setpart

end module setup
