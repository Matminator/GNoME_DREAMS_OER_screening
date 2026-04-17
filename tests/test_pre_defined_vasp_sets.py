import math

import pytest
from ase import Atoms
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor

from gnome_dreams_oer_screening import (
    RPBE_relax_bulk_set, RPBE_relax_surface_set,
)


def test_RPBE_relax_bulk_set():

    atoms = Atoms('Cu', positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True)
    structure = AseAtomsAdaptor.get_structure(atoms)

    bulk_set = RPBE_relax_bulk_set(structure)
    incar = bulk_set.CONFIG['INCAR']
    assert incar['GGA'] == 'RP'
    assert incar['ISIF'] == 3
    assert incar['EDIFFG'] == -0.05
    assert incar['ENCUT'] == 520
    assert incar['ISMEAR'] == 0
    assert incar['SIGMA'] == 0.1, "Expected SIGMA of .1, got {}".format(incar['SIGMA'])


def test_bulk_set_magmom_written_from_site_properties():
    """INCAR MAGMOM must match structure.site_properties['magmom'].

    job_handler passes enumerate_magnetic_structures output (which stores magmoms
    in site_properties) directly to RPBE_relax_bulk_set.
    VaspInputSet.incar (ISPIN=2) auto-populates MAGMOM from site_properties.
    """
    structure = Structure(
        Lattice.cubic(2.87),
        ["Fe", "Fe"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    structure.add_site_property("magmom", [5.0, -5.0])

    bulk_set = RPBE_relax_bulk_set(structure)
    incar = bulk_set.incar  # PyMatGen property — includes MAGMOM from site_properties

    assert "MAGMOM" in incar, (
        "INCAR must contain MAGMOM when site_properties['magmom'] is set and ISPIN=2"
    )
    magmoms = incar["MAGMOM"]
    assert len(magmoms) == 2
    assert magmoms[0] == pytest.approx(5.0, abs=1e-6)
    assert magmoms[1] == pytest.approx(-5.0, abs=1e-6)


def test_bulk_kpoints_density_formula():
    """Bulk k-point mesh uses ceil(50/a) * ceil(50/b) * ceil(50/c).

    For a cubic 3.0 Å cell: ceil(50/3.0) = 17 in each direction.
    """
    atoms = Atoms("Fe", positions=[[0, 0, 0]], cell=[3.0, 3.0, 3.0], pbc=True)
    structure = AseAtomsAdaptor.get_structure(atoms)

    bulk_set = RPBE_relax_bulk_set(structure)
    kpts = bulk_set.CONFIG["KPOINTS"]

    expected = math.ceil(50 / 3.0)  # = 17
    assert tuple(kpts.kpts[0]) == (expected, expected, expected), (
        f"Expected ({expected},{expected},{expected}), got {kpts.kpts[0]}"
    )


def test_surface_kpoints_one_in_vacuum_direction():
    """Surface KPOINTS must have exactly 1 k-point in the c (vacuum) direction.

    Γ-sampling along the non-periodic axis is mandatory for slab DFT.
    heuristic_compute_kpoints hardcodes 1 in c; this test confirms that
    invariant cannot silently regress to ceil(30/c).
    Also verifies in-plane sampling is non-trivially dense (not accidental Γ-only).
    """
    atoms = Atoms("Ti", positions=[[0, 0, 0]], cell=[3.0, 3.0, 20.0], pbc=True)
    structure = AseAtomsAdaptor.get_structure(atoms)

    surface_set = RPBE_relax_surface_set(structure)
    kpts = surface_set.CONFIG["KPOINTS"]

    k_a, k_b, k_c = kpts.kpts[0]
    assert k_c == 1, (
        f"Surface KPOINTS must have 1 k-point in the vacuum (c) direction, got {k_c}"
    )
    assert k_a == math.ceil(30 / 3.0), f"Expected {math.ceil(30/3.0)} in a, got {k_a}"
    assert k_b == math.ceil(30 / 3.0), f"Expected {math.ceil(30/3.0)} in b, got {k_b}"