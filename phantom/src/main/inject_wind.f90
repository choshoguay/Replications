!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2017 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://users.monash.edu.au/~dprice/phantom                               !
!--------------------------------------------------------------------------!
!+
!  MODULE: inject
!
!  DESCRIPTION:
!  Handles wind injection
!
!  REFERENCES: None
!
!  OWNER: Stéven Toupin
!
!  $Id$
!
!  RUNTIME PARAMETERS:
!    ihandled_spheres -- handle inner spheres of the wind (integer)
!    iwind_resolution -- resolution of the wind -- DO NOT CHANGE DURING SIMULATION --
!    shift_spheres    -- shift the spheres of the wind
!    wind_gamma       -- polytropic indice of the wind
!    wind_mu          -- mean molecular weight (g/mol)
!    wind_sphdist     -- distance between spheres / neighbours -- DO NOT CHANGE DURING SIMULATION --
!    wind_temperature -- initial temperature of the wind (Kelvin)
!
!  DEPENDENCIES: icosahedron, infile_utils, io, part, partinject, physcon,
!    units
!+
!--------------------------------------------------------------------------
module inject
  use physcon, only: au, solarm
  implicit none
  character(len=*), parameter, public :: inject_type = 'wind'

  public :: inject_particles, write_options_inject, read_options_inject
!
!--runtime settings for this module
!

! Read from input file
  real, public ::    wind_velocity = 35.d5
  real, public ::    wind_mass_rate = 7.65d-7
  real, public ::    wind_temperature = 3000.
  real, public ::    wind_gamma = 5./3.
  real, public ::    wind_mu = 1.26
  integer, public :: iwind_resolution = 4
  real, public ::    wind_sphdist = 1.
  real, public ::    shift_spheres = 2.
  integer, public :: ihandled_spheres = 2
  real, public ::    wind_injection_radius = 1.7*au
  real, public ::    wind_extrapolation = 0.
  real, public ::    central_star_mass = 1.*solarm
  real, public ::    central_star_radius = 1.*au
  integer, public ::    icompanion_star = 0
  real, public ::    companion_star_mass
  real, public ::    companion_star_radius
  real, public ::    semi_major_axis
  real, public ::    eccentricity

! Calculated from the previous parameters
  real, public ::    mass_of_particles, mass_of_spheres, time_between_spheres, neighbour_distance
  integer, public :: particles_per_sphere

  private

  logical, parameter :: wind_verbose = .false.
  integer, parameter :: wind_emitting_sink = 1
  real :: geodesic_R(0:19,3,3), geodesic_v(0:11,3), u_to_temperature_ratio

contains

!-----------------------------------------------------------------------
!+
!  Initialize reusable variables
!+
!-----------------------------------------------------------------------
subroutine wind_init()
  use physcon,     only: Rg
  use physcon,     only: solarm, years
  use icosahedron, only: compute_matrices, compute_corners

  real, parameter :: phi = (sqrt(5.)+1.)/2. ! Golden ratio

  u_to_temperature_ratio = Rg/(wind_mu*(wind_gamma-1.))
  particles_per_sphere = 20 * (2*iwind_resolution*(iwind_resolution-1)) + 12
  neighbour_distance = 2./((2.*iwind_resolution-1.)*sqrt(sqrt(5.)*phi))
  mass_of_particles = wind_sphdist * neighbour_distance * wind_injection_radius * wind_mass_rate &
                    / (particles_per_sphere * wind_velocity)
  mass_of_spheres = mass_of_particles * particles_per_sphere
  time_between_spheres = mass_of_spheres / wind_mass_rate
  call compute_matrices(geodesic_R)
  call compute_corners(geodesic_v)
end subroutine

!-----------------------------------------------------------------------
!+
!  Main routine handling wind injection.
!+
!-----------------------------------------------------------------------
subroutine inject_particles(time_u,dtlast_u,xyzh,vxyzu,xyzmh_ptmass,vxyz_ptmass,npart,npartoftype)
  use io,      only: iprint
  use units,   only: utime, umass
  use physcon, only: au
  real,    intent(in)    :: time_u, dtlast_u
  real,    intent(inout) :: xyzh(:,:), vxyzu(:,:), xyzmh_ptmass(:,:), vxyz_ptmass(:,:)
  integer, intent(inout) :: npart
  integer, intent(inout) :: npartoftype(:)

  integer :: outer_sphere, inner_sphere, inner_handled_sphere, i
  real :: time, dtlast, local_time, r, v, u, rho, e, mass_lost

  time = time_u * utime
  dtlast = dtlast_u * utime
  outer_sphere = floor((time-dtlast)/time_between_spheres) + 1
  inner_sphere = floor(time/time_between_spheres)
  inner_handled_sphere = inner_sphere + ihandled_spheres
  do i=inner_sphere+ihandled_spheres,outer_sphere,-1
    local_time = time - (i-shift_spheres) * time_between_spheres
    call compute_sphere_properties(local_time, r, v, u, rho, e)
    if (wind_verbose) then
      write(iprint,*) '* Sphere:'
      write(iprint,*) '   Number: ', i
      write(iprint,*) '   Local Time: ', local_time
      write(iprint,*) '   Radius: ', r, '(', r/au, ' au)'
      write(iprint,*) '   Expansion velocity: ', v, '(', v/1.d5, ' km/s)'
      write(iprint,*) '   Temperature: ', u / u_to_temperature_ratio, ' K'
      write(iprint,*) '   Density: ', rho, ' (g/cm³)'
      write(iprint,*) '   Constant e: ', e, ' (J/g)'
      write(iprint,*) ''
    endif
    if (i > inner_sphere) then
      call inject_geodesic_sphere(i, (ihandled_spheres-i+inner_sphere)*particles_per_sphere+1, r, v, u, rho, &
        npart, npartoftype, xyzh, vxyzu) ! handled sphere
    else
      call inject_geodesic_sphere(i, npart+1, r, v, u, rho, npart, npartoftype, xyzh, vxyzu) ! injected sphere
    endif
  enddo
  mass_lost = mass_of_spheres * (inner_sphere-outer_sphere+1)
  xyzmh_ptmass(4,wind_emitting_sink) = xyzmh_ptmass(4,wind_emitting_sink) - mass_lost/umass

end subroutine inject_particles

!-----------------------------------------------------------------------
!+
!  Time derivative of r and v, for Runge-Kutta iterations
!+
!-----------------------------------------------------------------------
subroutine drv_dt(rv, drv, GM)
  use physcon, only: Rg
  real, intent(in) :: rv(2), GM
  real, intent(out) :: drv(2)

  real :: r, v, dr_dt, r2, T, vs2, dv_dr, dv_dt

  r = rv(1)
  v = rv(2)
  dr_dt = v
  r2 = r*r
  T = wind_temperature * (wind_injection_radius**2 * wind_velocity / (r2 * v))**(wind_gamma-1.)
  vs2 = wind_gamma * Rg * T / wind_mu
  dv_dr = (-GM/r2+2.*vs2/r)/(v-vs2/v)
  dv_dt = dv_dr * v

  drv(1) = dr_dt
  drv(2) = dv_dt
end subroutine

!-----------------------------------------------------------------------
!+
!  Compute the radius, velocity and temperature of a sphere in function of its local time
!+
!-----------------------------------------------------------------------
subroutine compute_sphere_properties(local_time, r, v, u, rho, e)
  use part,    only: nptmass, xyzmh_ptmass
  use physcon, only: pi, Gg
  use units,   only: umass
  real, intent(in) :: local_time
  real, intent(out) :: r, v, u, rho, e

  real :: GM
  real :: dt, rv(2), k1(2), k2(2), k3(2), k4(2), T
  integer, parameter :: N = 10000
  integer :: i

  dt = local_time / N
  rv(1) = wind_injection_radius
  rv(2) = wind_velocity
  if (nptmass == 0) then
    GM = 0.
  else
    GM = Gg * xyzmh_ptmass(4,wind_emitting_sink) * umass
  endif
  ! Runge-Kutta iterations
  do i=1,N
    call drv_dt(rv,          k1, GM)
    call drv_dt(rv+dt/2.*k1, k2, GM)
    call drv_dt(rv+dt/2.*k2, k3, GM)
    call drv_dt(rv+dt*k3,    k4, GM)
    rv = rv + dt/6. * (k1 + 2.*k2 + 2.*k3 + k4)
  enddo
  r = rv(1)
  v = rv(2)
  T = wind_temperature * (wind_injection_radius**2 * wind_velocity / (r**2 * v))**(wind_gamma-1.)
  u = T * u_to_temperature_ratio
  rho = wind_mass_rate / (4.*pi*r**2*v)
  e = .5*v**2 - GM/r + wind_gamma*u
end subroutine


!-----------------------------------------------------------------------
!+
!  Inject a quasi-spherical distribution of particles.
!+
!-----------------------------------------------------------------------
subroutine inject_geodesic_sphere(sphere_number, first_particle, r, v, u, rho, npart, npartoftype, xyzh, vxyzu)
  use icosahedron, only: pixel2vector
  use units,       only: udist, utime, umass
  use partinject,  only: add_or_update_particle
  use part,        only: igas, hrho, xyzmh_ptmass, vxyz_ptmass, nptmass
  integer, intent(in) :: sphere_number, first_particle
  real, intent(in) :: r, v, u, rho
  integer, intent(inout) :: npart, npartoftype(:)
  real, intent(inout) :: xyzh(:,:), vxyzu(:,:)

  real :: rotation_angles(3), r_sim, v_sim, u_sim, h_sim
  real :: radial_unit_vector(3), rotmat(3,3), radial_unit_vector_rotated(3)
  real :: particle_position(3), particle_velocity(3)
  integer :: j

  select case (iwind_resolution)
    case(1)
      rotation_angles = (/ 1.28693610288783, 2.97863087745917, 1.03952835451832 /)
    case(2)
      rotation_angles = (/ 1.22718722289660, 2.58239466067315, 1.05360422660344 /)
    case(3)
      rotation_angles = (/ 0.235711384317490, 3.10477287368657, 2.20440220924383 /)
    case(4)
      rotation_angles = (/ 3.05231445647236, 0.397072776282339, 2.27500616856518 /)
    case(5)
      rotation_angles = (/ 0.137429597545199, 1.99860670500403, 1.71609391574493 /)
    case(6)
      rotation_angles = (/ 2.90443293496604, 1.77939686318657, 1.04113050588920 /)
    case(10)
      rotation_angles = (/ 2.40913070927068, 1.91721010369865, 0.899557511636617 /)
    case(15)
      rotation_angles = (/ 1.95605828396746, 0.110825898718538, 1.91174856362170 /)
    case default
      rotation_angles = (/ 1.28693610288783, 2.97863087745917, 1.03952835451832 /)
  end select
  rotation_angles = rotation_angles * sphere_number

  ! Quantities in simulation units
  r_sim = r / udist
  v_sim = v / (udist/utime)
  u_sim = u / (udist**2/utime**2)
  h_sim = hrho(rho / (umass/udist**3))

  call make_rotation_matrix(rotation_angles, rotmat)
  do j=0,particles_per_sphere-1
    call pixel2vector(j, iwind_resolution, geodesic_R, geodesic_v, radial_unit_vector)
    radial_unit_vector_rotated(1) = radial_unit_vector(1)*rotmat(1,1) &
                                  + radial_unit_vector(2)*rotmat(1,2) &
                                  + radial_unit_vector(3)*rotmat(1,3)
    radial_unit_vector_rotated(2) = radial_unit_vector(1)*rotmat(2,1) &
                                  + radial_unit_vector(2)*rotmat(2,2) &
                                  + radial_unit_vector(3)*rotmat(2,3)
    radial_unit_vector_rotated(3) = radial_unit_vector(1)*rotmat(3,1) &
                                  + radial_unit_vector(2)*rotmat(3,2) &
                                  + radial_unit_vector(3)*rotmat(3,3)
    radial_unit_vector = radial_unit_vector_rotated !/ sqrt(dot_product(radial_unit_vector_rotated, radial_unit_vector_rotated))
    particle_position = r_sim*radial_unit_vector
    particle_velocity = v_sim*radial_unit_vector
    if (nptmass > 0) then
      particle_position = particle_position + xyzmh_ptmass(1:3,wind_emitting_sink)
      particle_velocity = particle_velocity + vxyz_ptmass(1:3,wind_emitting_sink)
    endif
    call add_or_update_particle(igas, particle_position, particle_velocity, h_sim, u_sim, first_particle+j, &
         npart,npartoftype,xyzh,vxyzu)
  enddo
end subroutine

!-----------------------------------------------------------------------
!+
!  Make a 3x3 rotation matrix from three angles.
!+
!-----------------------------------------------------------------------
subroutine make_rotation_matrix(rotation_angles, rot_m)
  real, intent(in)  :: rotation_angles(3)
  real, intent(out) :: rot_m(3,3)

  real :: angle_x, angle_y, angle_z
  real :: c_x, s_x, c_y, s_y, c_z, s_z

  angle_x = rotation_angles(1)
  angle_y = rotation_angles(2)
  angle_z = rotation_angles(3)

  c_x = cos(angle_x)
  s_x = sin(angle_x)
  c_y = cos(angle_y)
  s_y = sin(angle_y)
  c_z = cos(angle_z)
  s_z = sin(angle_z)

  rot_m(1,1) = c_y*c_z
  rot_m(1,2) = -c_y*s_z
  rot_m(1,3) = -s_y
  rot_m(2,1) = -s_x*s_y*c_z + c_x*s_z
  rot_m(2,2) = s_x*s_y*s_z + c_x*c_z
  rot_m(2,3) = -s_x*c_y
  rot_m(3,1) = c_x*s_y*c_z + s_x*s_z
  rot_m(3,2) = -c_x*s_y*s_z + s_x*c_z
  rot_m(3,3) = c_x*c_y
end subroutine

!-----------------------------------------------------------------------
!+
!  Writes input options to the input file
!+
!-----------------------------------------------------------------------
subroutine write_options_inject(iunit)
  use physcon,      only: au, solarm, years
  use infile_utils, only: write_inopt
  integer, intent(in) :: iunit

  call write_inopt(wind_velocity,'wind_velocity', &
      'velocity at which wind is injected (cm/s) -- DO NOT CHANGE DURING SIMULATION --',iunit)
  call write_inopt(wind_mass_rate/(solarm/years),'wind_mass_rate', &
      'wind mass per unit time (Msun/yr) -- DO NOT CHANGE DURING SIMULATION --',iunit)
  call write_inopt(wind_temperature,'wind_temperature','initial temperature of the wind (Kelvin)',iunit)
  call write_inopt(wind_mu,'wind_mu','mean molecular weight (g/mol)',iunit)
  call write_inopt(wind_gamma,'wind_gamma','polytropic indice of the wind',iunit)
  call write_inopt(iwind_resolution,'iwind_resolution','resolution of the wind -- DO NOT CHANGE DURING SIMULATION --',iunit)
  call write_inopt(wind_sphdist,'wind_sphdist','distance between spheres / neighbours -- DO NOT CHANGE DURING SIMULATION --',iunit)
  call write_inopt(shift_spheres,'shift_spheres','shift the spheres of the wind',iunit)
  call write_inopt(ihandled_spheres,'ihandled_spheres','handle inner spheres of the wind (integer)',iunit)
  call write_inopt(wind_injection_radius/au,'wind_inject_radius', &
      'radius of injection of the wind (au) -- DO NOT CHANGE DURING SIMULATION --',iunit)
  call write_inopt(central_star_mass/solarm,'central_star_mass', &
      'mass of the central star (Msun)',iunit)
  call write_inopt(central_star_radius/au,'central_star_radius', &
      'radius of the central star (au)',iunit)
  call write_inopt(icompanion_star,'icompanion_star', &
      'set to 1 for a binary system',iunit)
  if (icompanion_star > 0) then
    call write_inopt(companion_star_mass/solarm,'companion_star_mass', &
      'mass of the companion star (Msun)',iunit)
    call write_inopt(companion_star_radius/au,'companion_star_radius', &
      'radius of the companion star (au)',iunit)
    call write_inopt(semi_major_axis/au,'semi_major_axis', &
      'semi-major axis of the binary system (au)',iunit)
    call write_inopt(eccentricity,'eccentricity', &
      'eccentricity of the binary system',iunit)
  endif
end subroutine

!-----------------------------------------------------------------------
!+
!  Reads input options from the input file.
!+
!-----------------------------------------------------------------------
subroutine read_options_inject(name,valstring,imatch,igotall,ierr)
  use physcon, only: au, solarm, years
  use io,      only: fatal, error, warning
  character(len=*), intent(in)  :: name,valstring
  logical, intent(out) :: imatch,igotall
  integer,intent(out) :: ierr

  integer, save :: ngot = 0
  integer :: noptions
  character(len=30), parameter :: label = 'read_options_inject'

  imatch  = .true.
  igotall = .false.
  select case(trim(name))
  case('wind_velocity')
    read(valstring,*,iostat=ierr) wind_velocity
    ngot = ngot + 1
    if (wind_velocity < 0.)    call fatal(label,'invalid setting for wind_velocity (<0)')
    if (wind_velocity > 1.e10) call error(label,'wind_velocity is huge!!!')
  case('wind_mass_rate')
    read(valstring,*,iostat=ierr) wind_mass_rate
    wind_mass_rate = wind_mass_rate * (solarm/years)
    ngot = ngot + 1
    if (wind_mass_rate < 0.)    call fatal(label,'invalid setting for wind_mass_rate (<0)')
  case('wind_temperature')
    read(valstring,*,iostat=ierr) wind_temperature
    ngot = ngot + 1
    if (wind_temperature < 0.)    call fatal(label,'invalid setting for wind_temperature (<0)')
  case('wind_mu')
    read(valstring,*,iostat=ierr) wind_mu
    ngot = ngot + 1
    if (wind_mu < 0.)    call fatal(label,'invalid setting for wind_mu (<0)')
  case('wind_gamma')
    read(valstring,*,iostat=ierr) wind_gamma
    ngot = ngot + 1
    if (wind_gamma < 0.)    call fatal(label,'invalid setting for wind_gamma (<0)')
  case('iwind_resolution')
    read(valstring,*,iostat=ierr) iwind_resolution
    ngot = ngot + 1
    if (iwind_resolution < 1) call fatal(label,'iwind_resolution must be bigger than zero')
  case('wind_sphdist')
    read(valstring,*,iostat=ierr) wind_sphdist
    ngot = ngot + 1
    if (wind_sphdist <= 0.) call fatal(label,'wind_sphdist must be >=0')
  case('shift_spheres')
    read(valstring,*,iostat=ierr) shift_spheres
    ngot = ngot + 1
  case('ihandled_spheres')
    read(valstring,*,iostat=ierr) ihandled_spheres
    ngot = ngot + 1
    if (ihandled_spheres <= 0) call fatal(label,'ihandled_spheres must be > 0')
  case('wind_inject_radius')
    read(valstring,*,iostat=ierr) wind_injection_radius
    wind_injection_radius = wind_injection_radius * au
    ngot = ngot + 1
    if (wind_injection_radius < 0.) call fatal(label,'invalid setting for wind_inject_radius (<0)')
  case('central_star_mass')
    read(valstring,*,iostat=ierr) central_star_mass
    central_star_mass = central_star_mass * solarm
    ngot = ngot + 1
    if (central_star_mass <= 0.) call fatal(label,'invalid setting for central_star_mass (<=0)')
  case('central_star_radius')
    read(valstring,*,iostat=ierr) central_star_radius
    central_star_radius = central_star_radius * au
    ngot = ngot + 1
    if (central_star_radius < 0.) call fatal(label,'invalid setting for central_star_radius (<0)')
  case('icompanion_star')
    read(valstring,*,iostat=ierr) icompanion_star
    ngot = ngot + 1
  case('companion_star_mass')
    read(valstring,*,iostat=ierr) companion_star_mass
    companion_star_mass = companion_star_mass * solarm
    ngot = ngot + 1
    if (companion_star_mass <= 0.) call fatal(label,'invalid setting for companion_star_mass (<=0)')
  case('companion_star_radius')
    read(valstring,*,iostat=ierr) companion_star_radius
    companion_star_radius = companion_star_radius * au
    ngot = ngot + 1
    if (companion_star_radius < 0.) call fatal(label,'invalid setting for companion_star_radius (<0)')
  case('semi_major_axis')
    read(valstring,*,iostat=ierr) semi_major_axis
    semi_major_axis = semi_major_axis * au
    ngot = ngot + 1
    if (semi_major_axis < 0.) call fatal(label,'invalid setting for semi_major_axis (<0)')
  case('eccentricity')
    read(valstring,*,iostat=ierr) eccentricity
    ngot = ngot + 1
    if (eccentricity < 0.) call fatal(label,'invalid setting for eccentricity (<0)')
  case default
    imatch = .false.
  end select

  noptions = 13
  if (icompanion_star > 0) then
    noptions = noptions + 4
  endif
  igotall = (ngot >= noptions)

  call wind_init()
end subroutine

end module inject
