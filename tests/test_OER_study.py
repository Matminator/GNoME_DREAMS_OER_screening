from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from GNoME_DREAMS_OER_screening.OER.OER_study import OER_catalyst_study

def test_setup():

    atoms = Atoms('Fe', positions=[[0, 0, 0]], cell=[2,2,2], pbc=True)

    oer_stody =OER_catalyst_study(
        atoms, H2_gas_free_energy = -2, H2O_gas_free_energy = -4
    )

    assert oer_stody.get_bulk_structure() ==  atoms
    energies = oer_stody.get_gas_and_energy_corrections()
    assert energies['H2_gas_free_energy'] == -2
    assert energies['H2O_gas_free_energy'] == -4

    
