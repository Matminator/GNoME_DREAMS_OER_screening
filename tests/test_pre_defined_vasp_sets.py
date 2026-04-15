from ase import Atoms
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