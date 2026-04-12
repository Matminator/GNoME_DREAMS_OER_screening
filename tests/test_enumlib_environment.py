"""
Environment integration test for enumlib + magnetic enumeration.

These tests call MagneticStructureEnumerator WITHOUT mocking, so they
require the enumlib binaries (enum.x, makestr.x / multienum.x) to be
on PATH. They are intended to verify that the HPC environment is
correctly set up before running production workflows.

Run with:
    pytest tests/test_enumlib_environment.py -v

These tests are NOT marked 'mini' and are excluded from the normal
fast test suite. Run them explicitly on the HPC to confirm the
environment is ready.
"""

import shutil
import subprocess

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator

from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    enumerate_magnetic_structures,
)


# ---------------------------------------------------------------------------
# Helper: skip if enumlib binaries are not on PATH
# ---------------------------------------------------------------------------

def _enumlib_available() -> bool:
    """Return True if the required enumlib executables are on PATH."""
    required = ["enum.x", "makestr.x"]
    alternatives = [["multienum.x", "makeStr.py"]]  # accepted alternatives
    has_enum = shutil.which("enum.x") is not None or shutil.which("multienum.x") is not None
    has_make = shutil.which("makestr.x") is not None or shutil.which("makeStr.py") is not None
    return has_enum and has_make


needs_enumlib = pytest.mark.skipif(
    not _enumlib_available(),
    reason="enumlib binaries (enum.x / makestr.x) not found on PATH — "
           "install via: conda install -c conda-forge enumlib",
)


# ---------------------------------------------------------------------------
# Binary presence checks (always run — no skip)
# ---------------------------------------------------------------------------

def test_enum_x_is_on_path():
    """enum.x or multienum.x must be on PATH."""
    found = shutil.which("enum.x") or shutil.which("multienum.x")
    assert found is not None, (
        "Neither enum.x nor multienum.x found on PATH.\n"
        "Fix: conda install -c conda-forge enumlib\n"
        "  or: pmg config --install enumlib  (Linux/macOS only)"
    )


def test_makestr_x_is_on_path():
    """makestr.x or makeStr.py must be on PATH."""
    found = shutil.which("makestr.x") or shutil.which("makeStr.py")
    assert found is not None, (
        "Neither makestr.x nor makeStr.py found on PATH.\n"
        "Fix: conda install -c conda-forge enumlib"
    )


# ---------------------------------------------------------------------------
# MagneticStructureEnumerator smoke tests (skip if enumlib missing)
# ---------------------------------------------------------------------------

@needs_enumlib
def test_enumerator_runs_without_error_on_fe_bcc():
    """MagneticStructureEnumerator must complete without raising on Fe BCC."""
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    # If this raises, enumlib is broken or not on PATH
    enum = MagneticStructureEnumerator(struct, automatic=True)
    assert len(enum.ordered_structures) > 0, "Enumerator returned no orderings"


@needs_enumlib
def test_enumerator_reduces_conventional_cell_to_primitive():
    """
    MagneticStructureEnumerator internally converts to primitive, so a
    2-atom conventional BCC cell input returns 1-atom primitive structures.
    This documents the known behaviour that enumerate_magnetic_structures
    works around by pre-converting to primitive before calling the enumerator.
    """
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    enum = MagneticStructureEnumerator(struct, automatic=True)
    # Enumerator reduces to primitive — results have 1 atom, not 2
    assert any(len(s) < len(struct) for s in enum.ordered_structures), (
        "Expected enumerator to return primitive (smaller) structures for "
        "a conventional BCC cell input"
    )


# ---------------------------------------------------------------------------
# enumerate_magnetic_structures integration tests (skip if enumlib missing)
# ---------------------------------------------------------------------------

@needs_enumlib
def test_enumerate_magnetic_structures_returns_at_least_one_config():
    """enumerate_magnetic_structures must return at least one config."""
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    results = enumerate_magnetic_structures(struct, n_max=4)
    assert len(results) >= 1


@needs_enumlib
def test_enumerate_magnetic_structures_fm_is_first_for_fe_bcc():
    """
    For Fe BCC the FM ordering has the highest symmetry and must be
    returned at index 0 after the symmetry-based sort.
    FM = all magmoms same sign and nonzero.

    Uses the conventional 2-atom BCC cell — enumerate_magnetic_structures
    pre-converts to primitive (1 atom) before calling the enumerator,
    so the filter correctly keeps the 1-atom FM result.
    """
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    results = enumerate_magnetic_structures(struct, n_max=4)

    first_magmom = results[0]["magmom"]
    nonzero = [m for m in first_magmom if abs(m) > 0.1]
    assert len(nonzero) > 0, (
        f"First config is NM — expected FM. All results: "
        f"{[r['ordering'] for r in results]}"
    )
    assert all(m > 0 for m in nonzero) or all(m < 0 for m in nonzero), (
        f"First config is not FM: magmom={first_magmom}"
    )


@needs_enumlib
def test_enumerate_magnetic_structures_respects_n_max():
    """Result length must not exceed n_max."""
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    for n in [1, 2, 4]:
        results = enumerate_magnetic_structures(struct, n_max=n)
        assert len(results) <= n, f"n_max={n} but got {len(results)} configs"


@needs_enumlib
def test_enumerate_magnetic_structures_all_configs_have_correct_atom_count():
    """
    Every returned config's magmom list must match the length of the
    returned structure (which may be primitive, not the input cell).
    """
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    results = enumerate_magnetic_structures(struct, n_max=4)
    for i, cfg in enumerate(results):
        n_struct = len(cfg["structure"])
        assert len(cfg["magmom"]) == n_struct, (
            f"Config {i}: magmom length {len(cfg['magmom'])} != "
            f"structure length {n_struct}"
        )
