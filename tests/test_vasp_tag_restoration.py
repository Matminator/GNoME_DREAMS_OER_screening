"""
Tests that ASE tags required by the O/OH adsorption pipeline are correctly
restored after a VASP round-trip (POSCAR and vasprun.xml do not preserve
ASE tags — they are zeroed on read-back).

These tests exercise the pattern used in job_handler.py:
    atoms.set_tags(ad_study.get_atoms_for_O_calc().get_tags())
    ad_study.set_relaxed_O_calc(atoms, ...)

No enumlib, POTCARs, or SLURM required.
"""
import numpy as np
import pytest
from ase.build import fcc111
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor

from gnome_dreams_oer_screening import (
    RPBE_relax_bulk_set,
    RPBE_relax_surface_set,
    TestRelaxSet,
)
from gnome_dreams_oer_screening.oer.oer_adsorption_site_study import (
    OER_adsorption_site_study,
)
from gnome_dreams_oer_screening.oer.atoms_sanity_checks import (
    StructureConsistencyError,
    are_systems_equal,
)


@pytest.fixture
def ontop_study():
    """Minimal on-top OER_adsorption_site_study for tag-restoration tests.

    Uses a 2x2x3 FCC Al(111) slab. The topmost atom is tagged as the
    adsorption site (tag=3); O is placed 2 A directly above it.
    """
    slab = fcc111('Al', size=(2, 2, 3), vacuum=8.0)
    slab.set_tags([0] * len(slab))  # clear fcc111 layer tags (1/2/3 per layer); need unique tag=3
    top_idx = int(np.argmax(slab.get_positions()[:, 2]))
    slab[top_idx].tag = 3  # mark as adsorption site
    site_pos = slab[top_idx].position.copy()
    o_pos = site_pos + np.array([0.0, 0.0, 2.0])
    return OER_adsorption_site_study(
        init_surface=slab,
        init_add_position=o_pos,
        site_index=0,
        surface_normal=np.array([0.0, 0.0, 1.0]),
        init=("on-top", -100.0),
    )


@pytest.mark.mini
def test_O_calc_atoms_carry_nonzero_tags(ontop_study):
    """atoms_for_O_calc must carry tag=3 on the site atom and tag=4 on adsorbed O."""
    ref = ontop_study.get_atoms_for_O_calc()
    tags = ref.get_tags()
    assert any(t != 0 for t in tags), (
        "atoms_for_O_calc must contain at least one atom with tag != 0"
    )
    o_tags = [ref[i].tag for i in range(len(ref)) if ref[i].symbol == "O"]
    assert 4 in o_tags, "Adsorbed O atom must carry tag=4"
    al_tags = [ref[i].tag for i in range(len(ref)) if ref[i].symbol == "Al"]
    assert 3 in al_tags, "Site Al atom must carry tag=3"


@pytest.mark.mini
def test_tag_stripped_atoms_rejected_then_accepted(ontop_study):
    """
    Simulates a VASP round-trip by zeroing all tags, then demonstrates:
      1. Without tag restoration -> StructureConsistencyError is raised.
      2. After restoring tags from the reference (as job_handler will do)
         -> set_relaxed_O_calc accepts the structure.
    """
    ref = ontop_study.get_atoms_for_O_calc()

    # --- Part 1: stripped tags must be rejected --------------------------------
    stripped = ref.copy()
    stripped.set_tags([0] * len(stripped))  # simulate POSCAR round-trip

    with pytest.raises(StructureConsistencyError, match="tags"):
        ontop_study.set_relaxed_O_calc(
            stripped, energy=-150.0, require_magmom=False
        )
    # study is unmodified (StructureConsistencyError raised before any mutation)
    assert ontop_study.relaxed_O_atoms is None

    # --- Part 2: restore tags (job_handler pattern) -> accepted ---------------
    restored = ref.copy()
    restored.set_tags([0] * len(restored))  # strip (simulate VASP)
    restored.set_tags(ref.get_tags())       # restore from reference

    success, msg = ontop_study.set_relaxed_O_calc(
        restored, energy=-150.0, require_magmom=False
    )
    assert success is True, f"Expected acceptance after tag restoration; got: {msg}"


@pytest.mark.mini
@pytest.mark.parametrize("vasp_set_cls", [
    RPBE_relax_bulk_set,
    RPBE_relax_surface_set,
    TestRelaxSet,
])
def test_poscar_roundtrip_preserves_atom_order(tmp_path, ontop_study, vasp_set_cls):
    """Atom ordering must survive a POSCAR write/read cycle for all three VaspInputSets.

    sort_structure=False in vasp_calculation.py prevents pymatgen from
    alphabetically resorting species. A reordering bug would surface as a
    symbol mismatch in are_systems_equal, causing set_relaxed_O_calc to
    raise StructureConsistencyError in production.

    potcar_spec=True writes a lightweight POTCAR.spec file (no actual
    POTCAR binary needed), so this test runs without PMG_VASP_PSP_DIR.
    """
    ref = ontop_study.get_atoms_for_O_calc()
    structure = AseAtomsAdaptor.get_structure(ref)

    # Write POSCAR via the VaspInputSet (mirrors vasp_calculation.py).
    vasp_set = vasp_set_cls(structure=structure, sort_structure=False)
    vasp_set.write_input(tmp_path, potcar_spec=True)

    # Read back with ASE (mirrors _default_fake_read_vasp_results / read_vasp_results).
    readback = ase_read(str(tmp_path / "POSCAR"))

    # Restore tags (the fix applied by job_handler).
    readback.set_tags(ref.get_tags())

    equal, msg = are_systems_equal(readback, ref)
    assert equal, (
        f"{vasp_set_cls.__name__}: POSCAR round-trip must not reorder atoms: {msg}"
    )
