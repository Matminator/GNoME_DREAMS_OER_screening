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

pytestmark = pytest.mark.filterwarnings(
    "ignore:Not all sites have property wyckoff:UserWarning",
    "ignore:Elements in POSCAR would be overwritten:UserWarning",
)


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


@needs_enumlib
def test_fe_primitive_cell_returns_fm_as_first_config():
    """
    Fe BCC primitive cell (1 atom). Fe is a well-known ferromagnet so the
    first result must be FM: all magmoms the same sign and nonzero.
    The primitive cell has 1 atom so there is only one possible ordering.
    """
    a = 2.87
    lat = Lattice([[a/2, a/2, -a/2],
                   [-a/2, a/2,  a/2],
                   [a/2, -a/2,  a/2]])
    struct = Structure(lat, ["Fe"], [[0, 0, 0]])

    results = enumerate_magnetic_structures(struct, n_max=4)

    # Results must be a list with one dict entry:
    assert isinstance(results, list), f"Expected list output, got {type(results)}"
    assert len(results) == 1, f"Expected 1 result for 1-atom cell, got {len(results)}"
    assert isinstance(results[0], dict), f"Expected dict entry, got {type(results[0])}"

    config_dict = results[0]
    magmoms = config_dict["magmom"]
    assert len(magmoms) == 1, f"Expected magmom list of length 1, got {len(magmoms)}"

    # Check that the single magmom is close to 5 (Fe's default value in pymatgen)
    mag = magmoms[0]
    assert abs(mag) == pytest.approx(5.0, abs=0.1), f"Expected magmom ~5 for Fe, got {mag}"

    # Ordering label must be 'fm'
    assert config_dict["ordering"] == "fm", (
        f"Expected ordering='fm', got '{config_dict['ordering']}'"
    )

    # Returned structure must have the same atom count as the input (no cell expansion)
    assert len(config_dict["structure"]) == len(struct), (
        f"Expected 1-atom structure, got {len(config_dict['structure'])} atoms"
    )
    
    
@needs_enumlib
def test_fe_conventional_cell_returns_multiple_orderings():
    """
    Fe BCC conventional cell (2 atoms). With 2 atoms the enumerator can
    produce both FM [5, 5] and AFM [5, -5]. The AFM ordering comes first
    because the body-center symmetry (translation + spin flip) gives it
    more symmetry operations than FM for this specific geometry.
    We verify that at least one FM and one AFM config are present.
    """
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    results = enumerate_magnetic_structures(struct, n_max=4)

    def is_fm(magmom):
        nz = [m for m in magmom if abs(m) > 0.1]
        return len(nz) > 0 and (all(m > 0 for m in nz) or all(m < 0 for m in nz))

    def is_afm(magmom):
        nz = [m for m in magmom if abs(m) > 0.1]
        return len(nz) > 0 and any(m > 0 for m in nz) and any(m < 0 for m in nz)

    # FM must appear within the first 2 results — it should never be buried
    # at the bottom since it is always physically relevant
    assert any(is_fm(r["magmom"]) for r in results[:2]), (
        f"FM not in first 2 results: {[r['magmom'] for r in results]}"
    )
    assert any(is_afm(r["magmom"]) for r in results), "No AFM config found"
