from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from gnome_dreams_oer_screening import OER_catalyst_study

def test_setup_1():

    atoms = Atoms('Fe', positions=[[0, 0, 0]], cell=[2,2,2], pbc=True)

    oer_study =OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True
    )

    assert oer_study.get_bulk_atoms() ==  atoms
    assert oer_study.get_bulk_structure() == \
        AseAtomsAdaptor.get_structure(atoms)
    energies = oer_study.get_gas_and_energy_corrections()
    assert energies['H2_gas_free_energy'] == -2
    assert energies['H2O_gas_free_energy'] == -4


def test_setup_2():

    atoms = Atoms('Ti', positions=[[0, 0, 0]], cell=[2,2,2], pbc=True)

    oer_study =OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4, 
        use_init_as_relaxed_bulk=True
    )

    atoms_to_relax = oer_study.get_init_atoms()
    oer_study.set_bulk_structure(atoms_to_relax)

    assert oer_study.get_bulk_atoms() ==  atoms
    assert oer_study.get_bulk_structure() == \
        AseAtomsAdaptor.get_structure(atoms)
    energies = oer_study.get_gas_and_energy_corrections()
    assert energies['H2_gas_free_energy'] == -2
    assert energies['H2O_gas_free_energy'] == -4
