"""Test fixtures with inline Fortran samples."""

import pytest


SAMPLE_FORTRAN_F = """\
*> \\brief DTEST computes a test value
*>
*> \\verbatim
*> DTEST computes a simple test operation for unit testing.
*> \\endverbatim
      SUBROUTINE DTEST( N, X, INCX, Y, INCY )
*
*     .. Scalar Arguments ..
      INTEGER            INCX, INCY, N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   X( * ), Y( * )
*     ..
*     .. External Subroutines ..
      EXTERNAL           DSCAL, DCOPY
*     ..
      CALL DSCAL( N, 2.0D0, X, INCX )
      CALL DCOPY( N, X, INCX, Y, INCY )
*
      RETURN
      END
"""

SAMPLE_FORTRAN_F90 = """\
!> \\brief CTEST generates a test rotation
!>
!> CTEST generates a simple test for unit testing.
subroutine ctest( n, x, y )
   implicit none
   integer, intent(in) :: n
   real, intent(inout) :: x(n), y(n)
   integer :: i
   do i = 1, n
      x(i) = x(i) + y(i)
   end do
end subroutine ctest
"""

SAMPLE_LARGE_FORTRAN = "*> Large test file\n" + "      X = X + 1\n" * 5000

SAMPLE_BLOCK_DATA = """\
      BLOCK DATA INIT
      INTEGER N
      COMMON /COEFF/ N
      DATA N /10/
      END
"""

SAMPLE_PROGRAM = """\
      PROGRAM MAIN
      PRINT *, 'Hello'
      END
"""

SAMPLE_FIXED_FORM_MODULE = """\
      MODULE FOO
      CONTAINS
      SUBROUTINE BAR
      END SUBROUTINE
      END MODULE
"""


@pytest.fixture
def sample_block_data():
    return SAMPLE_BLOCK_DATA


@pytest.fixture
def sample_program():
    return SAMPLE_PROGRAM


@pytest.fixture
def sample_fixed_form_module():
    return SAMPLE_FIXED_FORM_MODULE


@pytest.fixture
def sample_f_code():
    return SAMPLE_FORTRAN_F


@pytest.fixture
def sample_f90_code():
    return SAMPLE_FORTRAN_F90


@pytest.fixture
def sample_large_code():
    return SAMPLE_LARGE_FORTRAN
