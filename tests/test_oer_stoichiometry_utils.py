import numpy as np

from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from gnome_dreams_oer_screening.oer.oer_study_utils import (
    get_stoichiometry_dict, get_stoichiometry_diff, 
    eval_stoichiometry_diff
)

def test_get_stoichiometry_dict_small():

    atoms = Atoms('Cu', positions=[[0, 0, 0]])

    dict = get_stoichiometry_dict(atoms)
    assert 'Cu' in dict.keys()
    assert np.isclose(dict['Cu'], 1.0)

def test_get_stoichiometry_dict_binary():

    atoms = Atoms('Cu2O', positions=[[0, 0, 0], [1, 0, 0],
                                     [0.5, 0.5, 0.5]])

    dict = get_stoichiometry_dict(atoms)
    assert 'Cu' in dict.keys()
    assert 'O' in dict.keys()
    assert np.isclose(dict['Cu'], 2/3)
    assert np.isclose(dict['O'], 1/3)

def test_get_stoichiometry_diff_1():
    dict1 = {'Cu': 0.5, 'O': 0.5}
    dict2 = {'Cu': 0.5, 'O': 0.4, 'H': 0.1}

    diff_dict = get_stoichiometry_diff(dict1, dict2)
    assert np.isclose(diff_dict['Cu'], 0.0)
    assert np.isclose(diff_dict['O'], 0.1)
    # Negative value, to indicate that H is missing in dict1:
    assert np.isclose(diff_dict['H'], -0.1)

def test_get_stoichiometry_diff_2():
    dict1 = {'Cu': 0.5, 'O': 0.5}
    dict2 = {'Fe': 0.5, 'O': 0.5, 'H': 0}

    diff_dict = get_stoichiometry_diff(dict1, dict2)
    # Negative value, to indicate that Cu is missing in dict1:
    assert np.isclose(diff_dict['Cu'], -0.5)
    assert np.isclose(diff_dict['O'], 0.0)
    assert np.isclose(diff_dict['H'], 0.0)

def test_eval_stoichiometry_diff_1():
    atoms_1 = Atoms('Cu2O', positions=[[0, 0, 0], [1, 0, 0],
                                     [0.5, 0.5, 0.5]])
    atoms_2 = Atoms('Cu2O1H1', positions=[[0, 0, 0], [1, 0, 0],
                                     [0.5, 0.5, 0.5], [0.6, 0.6, 0.6]])

    result = eval_stoichiometry_diff(atoms_1, atoms_2)
    assert result is False

def test_eval_stoichiometry_diff_2():
    atoms_1 = Atoms('Cu2O', positions=[[0,0,0], [0,0,0], [0,0,0]],
                     cell =[2, 2, 2])
    atoms_2 = Atoms('Cu2O1H1', positions=[[0,0,0], [0,0,0], [0,0,0],
                                           [0,0,0]])
    
    structure_1 = AseAtomsAdaptor.get_structure(atoms_1)
    result = eval_stoichiometry_diff(structure_1, atoms_2)
    assert result is False

def test_eval_stoichiometry_diff_3():
    atoms_1 = Atoms('Cu10O')
    atoms_2 = Atoms('Cu9O1H1')

    result = eval_stoichiometry_diff(atoms_1, atoms_2,
                                     all_species_present=False)
    assert result is True

def test_eval_stoichiometry_diff_4():
    atoms_1 = Atoms('Cu10O')
    atoms_2 = Atoms('Cu9O1H1')

    result = eval_stoichiometry_diff(atoms_1, atoms_2,
                                     all_species_present=True)
    assert result is False

def test_eval_stoichiometry_diff_5():
    atoms_1 = Atoms('Cu10O')
    atoms_2 = Atoms('Cu9O1H1')

    result = eval_stoichiometry_diff(atoms_1, atoms_2,
                                     all_species_present=True,
                                     ignore_species=['H']
                                     )
    assert result is True

def test_eval_stoichiometry_diff_6():
    atoms_1 = Atoms('Cu10O')
    atoms_2 = Atoms('Cu9O2H1')

    result = eval_stoichiometry_diff(atoms_1, atoms_2,
                                     all_species_present=True,
                                     max_diff_dict={'O': 0.0001},
                                     ignore_species=['H']
                                     )
    assert result is False

def test_eval_stoichiometry_diff_7():
    atoms_1 = Atoms('Cu10O')
    atoms_2 = Atoms('Cu9O1H1')

    result = eval_stoichiometry_diff(atoms_1, atoms_2,
                                     all_species_present=False,
                                     max_diff_dict={'O': 0.0001},
                                     ignore_species=['H']
                                     )
    assert result is True

