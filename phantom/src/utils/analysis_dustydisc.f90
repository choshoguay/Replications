!--------------------------------------------------------------------------!
! The Phantom Smoothed Particle Hydrodynamics code, by Daniel Price et al. !
! Copyright (c) 2007-2017 The Authors (see AUTHORS)                        !
! See LICENCE file for usage and distribution conditions                   !
! http://users.monash.edu.au/~dprice/phantom                               !
!--------------------------------------------------------------------------!
!+
!  MODULE: analysis
!
!  DESCRIPTION:
!  Analysis routine for dustydisc
!
!  REFERENCES: None
!
!  OWNER: Giovanni Dipierro
!
!  $Id$
!
!  RUNTIME PARAMETERS: None
!
!  DEPENDENCIES: dim, infile_utils, io, part, physcon
!+
!--------------------------------------------------------------------------
module analysis
 implicit none
 character(len=20), parameter, public :: analysistype = 'dustydisc'
 public :: do_analysis

 integer, parameter :: nr = 300
 real,dimension(nr) :: twist,twistprev

 private

contains

subroutine do_analysis(dumpfile,numfile,xyzh,vxyz,pgasmass,npart,time,iunit)
 use dim,     only:use_dustfrac
 use io,      only:fatal
 use physcon, only:pi
 use part,    only:iphase,npartoftype,igas,idust,massoftype,labeltype,dustfrac
 character(len=*), intent(in) :: dumpfile
 real,             intent(in) :: xyzh(:,:),vxyz(:,:)
 real,             intent(in) :: pgasmass,time
 integer,          intent(in) :: npart,iunit,numfile
 character(len=9) :: output
 character(len=20) :: filename
 character(len=20) :: discprefix
 integer :: i,ii,ierr,iline,ninbin(nr),ninbindust(nr),iwarp
 real :: R_in,R_out,R_warp,H_R,p_index,q_index,M_star,M_disc,R_c,R_cdust
 real :: R_in_dust,R_out_dust,R_warp_dust,H_R_dust,p_index_dust,M_star_dust,M_disc_dust
 real :: G,rmin,rmax,dr,cs0,angx,angy,angz,ri,area
 real :: angtot,Ltot,tilt,dtwist
 real :: Li(3)
 real :: rad(nr),Lx(nr),Ly(nr),Lz(nr),h_smooth(nr),sigma(nr),cs(nr),H(nr),omega(nr)
 real :: zsettlgas(npartoftype(igas),nr),hgas(nr),meanzgas(nr),dustfracsum(nr),dust_fraction(nr)
 real :: unitlx(nr),unitly(nr),unitlz(nr),tp(nr)
 real :: sigmadust(nr),zsettldust(npartoftype(idust),nr),hdust(nr),meanzdust(nr)
 real :: psi_x,psi_y,psi_z,psi

 integer, parameter :: iparams = 10
 integer, parameter :: iprec   = 24
 logical :: do_precession,ifile

! Print the analysis being done
 write(*,'("Performing analysis type ",A)') analysistype
 write(*,'("Input file name is ",A)') dumpfile

 write(output,"(a4,i5.5)") 'angm',numfile
 write(*,'("Output file name is ",A)') output

 if (use_dustfrac) write(*,'("one-fluid model")')

! Assuming G=1
 write(*,*)
 write(*,'("ASSUMING G==1")')
 G = 1.0

 iline = index(dumpfile,'_')
 discprefix = dumpfile(1:iline-1)
 inquire(file=trim(discprefix)//'.discparams', exist=ifile)
 if (ifile) then
    call read_discparams(trim(discprefix)//'.discparams',R_in,R_out,R_warp,H_R,p_index,R_c,q_index,M_star,M_disc,iparams,ierr)
    if (ierr /= 0) call fatal('analysis','could not open/read .discparams file')
 else
    call read_discparams('discparams.list',R_in,R_out,R_warp,H_R,p_index,R_c,q_index,M_star,M_disc,iparams,ierr)
    if (ierr /= 0) call fatal('analysis','could not open/read discparams.list')
 endif

 ! Print out the parameters of gas disc
 write(*,*)
 write(*,'("Gas disc parameters are:")')
 write(*,*) 'R_in    = ',R_in
 write(*,*) 'R_out   = ',R_out
 write(*,*) 'H_R     = ',H_R
 if(R_warp/=0.) write(*,*) 'Rwarp     = ',R_warp
 write(*,*) 'p_index = ',p_index
 if(R_c/=0.) write(*,*) 'R_c     = ',R_c
 write(*,*) 'q_index = ',q_index
 write(*,*) 'M_star  = ',M_star
 write(*,*) 'M_disc  = ',M_disc
 write(*,*)
 write(*,*)

 inquire(file=trim(discprefix)//'-'//trim(labeltype(idust))//'.discparams', exist=ifile)
 if (ifile) then
    call read_discparams(trim(discprefix)//'-'//trim(labeltype(idust))//'.discparams',&
    R_in_dust,R_out_dust,R_warp_dust,H_R_dust,p_index_dust,R_cdust,q_index,M_star_dust,M_disc_dust,iparams,ierr)
    if (ierr /= 0) call fatal('analysis','could not open/read -'//trim(labeltype(idust))//' .discparams file')

 ! Print out the parameters of dust disc
 write(*,*)
 write(*,'("Dust disc parameters are:")')
 write(*,*) 'R_in    = ',R_in_dust
 write(*,*) 'R_out   = ',R_out_dust
 write(*,*) 'H_R     = ',H_R_dust
 if(R_warp_dust/=0.) write(*,*) 'Rwarp     = ',R_warp_dust
 write(*,*) 'p_index = ',p_index_dust
 if(R_cdust/=0.) write(*,*) 'R_c     = ',R_cdust
 write(*,*) 'M_disc  = ',M_disc_dust
 write(*,*)
 write(*,*)
 endif

! Setup rmin and rmax for the analysis
if (npartoftype(idust) > 0) then
   rmin = min(R_in,R_in_dust)
   rmax = max(R_out,R_out_dust)
else
   rmin = R_in
   rmax = R_out
endif


! Set up the radius array
 dr = (rmax-rmin)/real(nr-1)
 do i=1,nr
    rad(i)=rmin + real(i-1)*dr
 enddo

! Initialise arrays to zero
 ninbin(:)=0
 lx(:)=0.0
 ly(:)=0.0
 lz(:)=0.0
 h_smooth(:)=0.0
 sigma(:)=0.0
 dustfracsum(:)=0.0
 dust_fraction(:)=0.0
 sigmadust(:)=0.0
 ninbindust(:)=0
 hgas(:)=0.0
 hdust(:)=0.0
 meanzgas(:)=0.0
 meanzdust(:)=0.0
 zsettlgas(:,:)=0.0
 zsettldust(:,:)=0.0

! Set up cs0: cs = cs0 * R^-q
 cs0 = H_R * sqrt(G*M_star) * R_in**(q_index-0.5)
! And thus the sound speed array
 do i=1,nr
    cs(i) = cs0 * rad(i)**(-q_index)
    omega(i) = sqrt(G*M_star/rad(i)**3)
 enddo

! and thus the disc scale height
 do i=1,nr
    H(i) = cs(i)/omega(i)
 enddo

 angx = 0.0
 angy = 0.0
 angz = 0.0

 if (R_warp/=0.)then
    iwarp=3
 else
    iwarp=2
 endif

! Loop over gas particles putting properties into the correct bin
 do i = 1,npart
    if (xyzh(4,i)  >  tiny(xyzh)) then ! IF ACTIVE
       ri = sqrt(dot_product(xyzh(1:iwarp,i),xyzh(1:iwarp,i)))
       ii = int((ri-rad(1))/dr + 1)

       if (ii > nr) cycle
       if (ii < 1)  cycle

       area = (pi*((rad(ii)+dr/2.)**2-(rad(ii)- dr/2.)**2))

       if(iphase(i)==igas)then

          sigma(ii) = sigma(ii) + pgasmass/area

          Li(1) = pgasmass*(xyzh(2,i)*vxyz(3,i)-xyzh(3,i)*vxyz(2,i))
          Li(2) = pgasmass*(xyzh(3,i)*vxyz(1,i)-xyzh(1,i)*vxyz(3,i))
          Li(3) = pgasmass*(xyzh(1,i)*vxyz(2,i)-xyzh(2,i)*vxyz(1,i))

          Lx(ii)=Lx(ii)+Li(1)
          Ly(ii)=Ly(ii)+Li(2)
          Lz(ii)=Lz(ii)+Li(3)

          h_smooth(ii) = h_smooth(ii) + xyzh(4,i)

          ninbin(ii) = ninbin(ii) + 1
          zsettlgas(ninbin(ii),ii)=xyzh(3,i)
          if (use_dustfrac) dustfracsum(ii) = dustfracsum(ii) + dustfrac(i)

       elseif(iphase(i)==idust) then
          sigmadust(ii) = sigmadust(ii) + massoftype(iphase(i))/area

          ninbindust(ii) = ninbindust(ii) + 1
          zsettldust(ninbindust(ii),ii)=xyzh(3,i)
       endif

    elseif (xyzh(4,i) < -tiny(xyzh) .and. iphase(i)==igas) then !ACCRETED
       angx = angx + pgasmass*(xyzh(2,i)*vxyz(3,i) - xyzh(3,i)*vxyz(2,i))
       angy = angy + pgasmass*(xyzh(3,i)*vxyz(1,i) - xyzh(1,i)*vxyz(3,i))
       angz = angz + pgasmass*(xyzh(1,i)*vxyz(2,i) - xyzh(2,i)*vxyz(1,i))

    endif
 enddo

 ! Computing Hgas and Hdust
 do i=1,nr
 if(ninbin(i)>1)then
    meanzgas(i)=sum(zsettlgas(1:ninbin(i),i))/real(ninbin(i))
    hgas(i)=sqrt(sum(((zsettlgas(1:ninbin(i),i)-meanzgas(i))**2)/(real(ninbin(i)-1))))
    if (use_dustfrac) dust_fraction(i)=dustfracsum(i)/real(ninbin(i))
 endif
 if(ninbindust(i)>1)then
    meanzdust(i)=sum(zsettldust(1:ninbindust(i),i))/real(ninbindust(i))
    hdust(i)=sqrt(sum(((zsettldust(1:ninbindust(i),i)-meanzdust(i))**2)/(real(ninbindust(i)-1))))
 endif
 enddo


! Print angular momentum of accreted particles
 angtot = sqrt(angx*angx + angy*angy + angz*angz)
 print*,' angular momentum of accreted particles = ',angtot


! Convert total angular momentum into a unit vector, and average h_smooth
 do i = 1,nr
    Ltot = sqrt(Lx(i)*Lx(i) + Ly(i)*Ly(i) + Lz(i)*Lz(i))

    unitlx(i) = Lx(i)/Ltot
    unitly(i) = Ly(i)/Ltot
    unitlz(i) = Lz(i)/Ltot

    if (ninbin(i) > 0) h_smooth(i) = h_smooth(i)/ninbin(i)
 enddo

! Now loop over rings to calculate required quantities
 do i = 1, nr
    if(ninbin(i)==0) then
       lx(i)=0.0
       ly(i)=0.0
       lz(i)=0.0
       sigma(i)=0.0
       h_smooth(i) = 0.0
    else
       h_smooth(i) = h_smooth(i)/H(i)
    endif
 enddo

 open(iunit,file=output)
 write(iunit,'("# Analysis data at t = ",es20.12)') time
 if (npartoftype(idust)==0)then
    if (use_dustfrac) then
       write(iunit,"('#',12(1x,'[',i2.2,1x,a11,']',2x))") &
          1,'radius', &
          2,'sigma', &
          3,'sigmadust', &
          4,'<h>/H', &
          5,'lx', &
          6,'ly', &
          7,'lz', &
          8,'tilt', &
          9,'twist', &
          10,'psi', &
          11,'H/R init', &
          12,'H/R'
    else
       write(iunit,"('#',11(1x,'[',i2.2,1x,a11,']',2x))") &
          1,'radius', &
          2,'sigma', &
          3,'<h>/H', &
          4,'lx', &
          5,'ly', &
          6,'lz', &
          7,'tilt', &
          8,'twist', &
          9,'psi', &
          10,'H/R init', &
          11,'H/R'
    endif
  else
  write(iunit,"('#',13(1x,'[',i2.2,1x,a11,']',2x))") &
       1,'radius', &
       2,'sigma', &
       3,'sigmadust', &
       4,'<h>/H', &
       5,'lx', &
       6,'ly', &
       7,'lz', &
       8,'tilt', &
       9,'twist', &
       10,'psi', &
       11,'H/R init', &
       12,'H/R', &
       13,'Hdust/R'
  endif

 do_precession = .false.

 do i=1,nr
    if(i /= 1.and.i /= nr) then
       psi_x=(unitlx(i+1)-unitlx(i-1))/(rad(i+1)-rad(i-1))
       psi_y=(unitly(i+1)-unitly(i-1))/(rad(i+1)-rad(i-1))
       psi_z=(unitlz(i+1)-unitlz(i-1))/(rad(i+1)-rad(i-1))
       psi=sqrt(psi_x**2 + psi_y**2 + psi_z**2)*rad(i)
    else
       psi=0.
    endif

    if (ninbin(i) > 0) then
       tilt  = acos(unitlz(i))
       twist(i) = atan2(unitly(i),unitlx(i))
        if (i==1 .or. time==0.0) then
          twistprev(i) = 0.0
        endif
        ! Taking into account negative twist
        if (twist(i) < 0) then
          twistprev(i) = 2.*pi + twist(i)
        else
          twistprev(i) = twist(i) !cumulative twist
        endif
    else
       tilt = 0.0
       twist = 0.0
       dtwist = 0.0
    endif

! Calculate the precession time
    if (twist(i) > tiny(twist(i))) then
      tp(i) = time*2.*pi/twist(i)
    else
      tp(i) = 0.0
    endif


if (npartoftype(idust)==0)then
   if (ninbin(i) > 0) then
      if (use_dustfrac) then
         write(iunit,'(12(es18.10,1X))') rad(i),sigma(i)*(1.-dust_fraction(i)),sigma(i)*dust_fraction(i),h_smooth(i),&
                                       unitlx(i),unitly(i),unitlz(i),tilt,twist(i),psi,H(i)/rad(i),hgas(i)/rad(i)
      else
         write(iunit,'(11(es18.10,1X))') rad(i),sigma(i),h_smooth(i),unitlx(i),unitly(i),unitlz(i),&
                                       tilt,twist(i),psi,H(i)/rad(i),hgas(i)/rad(i)
      endif
   endif
else
   if (ninbin(i) > 0 .OR. ninbindust(i) > 0) then
       write(iunit,'(13(es18.10,1X))') rad(i),sigma(i),sigmadust(i),h_smooth(i),unitlx(i),unitly(i),unitlz(i),&
                                       tilt,twist(i),psi,H(i)/rad(i),hgas(i)/rad(i),hdust(i)/rad(i)
    endif
endif

! Printing time and twist for each radius bin
    if (do_precession) then
       write(filename,"(a,i3.3)")"precess",i
       if (numfile==0) then
          open(unit=iprec,file=filename,status="replace")
          write(iprec,'("# tilt and twist with time for r = ",es18.10)') rad(i)
          write(iprec,"('#',6(1x,'[',i2.2,1x,a11,']',2x))") &
               1,'rad', &
               2,'time', &
               3,'tilt', &
               4,'twist', &
               5,'tot twist', &
               6,'tp'
       else
          open(unit=iprec,file=filename,status="old",position="append")
       endif
       write(iprec,'(6(es18.10,1X))') rad(i),time,tilt,twist(i),twistprev(i),tp(i)
       close(unit=iprec)
    endif

 enddo

 close(iunit)

end subroutine do_analysis

!----------------------------------------------------------------
!+
!  Read disc information from discparams.list file
!+
!----------------------------------------------------------------
subroutine read_discparams(filename,R_in,R_out,R_warp,H_R,p_index,R_c,q_index,M_star,M_disc,iunit,ierr)
 use infile_utils, only:open_db_from_file,inopts,read_inopt,close_db
 character(len=*), intent(in)  :: filename
 real,             intent(out) :: R_in,R_out,R_warp,H_R,p_index,q_index,M_star,M_disc,R_c
 integer,          intent(in)  :: iunit
 integer,          intent(out) :: ierr
 type(inopts), allocatable :: db(:)

! Read in parameters from the file discparams.list
 call open_db_from_file(db,filename,iunit,ierr)
 if (ierr /= 0) return
 call read_inopt(R_in,'R_in',db,ierr)
 if (ierr /= 0) return
 call read_inopt(R_out,'R_out',db,ierr)
 if (ierr /= 0) return
 call read_inopt(R_warp,'R_warp',db,ierr)
 call read_inopt(H_R,'H_R',db,ierr)
 if (ierr /= 0) return
 call read_inopt(p_index,'p_index',db,ierr)
 if (ierr /= 0) return
 call read_inopt(R_c,'R_c',db,ierr)
 call read_inopt(q_index,'q_index',db,ierr)
 if (ierr /= 0) return
 call read_inopt(M_star,'M_star',db,ierr)
 if (ierr /= 0) return
 call read_inopt(M_disc,'M_disc',db,ierr)
 if (ierr /= 0) return
 call close_db(db)

end subroutine read_discparams

end module analysis

