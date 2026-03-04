"""Test fixtures with inline Fortran samples."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


# --- Factory fixture: temporary Fortran files ---

@pytest.fixture
def tmp_fortran_file():
    """Factory fixture that creates temp files with given suffix/content and auto-cleans."""
    created: list[Path] = []

    def _create(content: str | bytes, suffix: str = ".f") -> Path:
        if isinstance(content, str):
            content = content.encode()
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.write(content)
        f.close()
        p = Path(f.name)
        created.append(p)
        return p

    yield _create

    for p in created:
        p.unlink(missing_ok=True)


# --- Embed cache cleanup fixture ---

@pytest.fixture
def clear_embed_cache():
    """Clear the embed_query LRU cache before and after test."""
    import app.services.embeddings as emb_mod
    emb_mod._embed_cache.clear()
    yield
    emb_mod._embed_cache.clear()


# --- Generation test helpers ---

@pytest.fixture
def mock_gen_settings():
    """Patch get_settings and _get_generation_client for generation tests.

    Yields (settings, mock_client) so tests can configure responses.
    """
    with patch("app.services.generation.get_settings") as ms, \
         patch("app.services.generation._get_generation_client") as mc:
        settings = MagicMock()
        settings.CHAT_MODEL = "gpt-4o-mini"
        ms.return_value = settings
        mock_client = AsyncMock()
        mc.return_value = mock_client
        yield settings, mock_client


@pytest.fixture
def mock_gemini_gen_settings():
    """Patch get_settings and _get_gemini_client for Gemini generation tests.

    Yields (settings, mock_client) so tests can configure responses.
    The client's async methods are pre-configured as AsyncMock to prevent
    TypeError when tests iterate with ``async for``.
    """
    with patch("app.services.generation.get_settings") as ms, \
         patch("app.services.generation._get_gemini_client") as mc:
        settings = MagicMock()
        settings.CHAT_MODEL = "gemini-2.0-flash"
        ms.return_value = settings
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock()
        mock_client.aio.models.generate_content_stream = AsyncMock()
        mc.return_value = mock_client
        yield settings, mock_client


def make_async_iter(*chunks):
    """Create an async iterator from mock stream chunks."""
    async def _iter():
        for c in chunks:
            yield c
    return _iter()


# --- Retrieval test fixtures ---

@pytest.fixture
def retrieval_mocks():
    """Patch embed_query, async_search_by_name, async_search for retrieval tests.

    Yields (mock_embed, mock_search_by_name, mock_search).
    """
    with patch("app.services.retrieval.embed_query", new_callable=AsyncMock) as me, \
         patch("app.services.retrieval.async_search_by_name", new_callable=AsyncMock) as msn, \
         patch("app.services.retrieval.async_search", new_callable=AsyncMock) as ms:
        me.return_value = [0.1] * 1536
        yield me, msn, ms


