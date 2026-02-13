import numpy as np
import pandas as pd

from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from pymatgen.core.surface import Slab

from gnome_dreams_oer_screening.oer.oer_study_utils import (
    get_atoms_and_normal_from_input, get_atom_projections_on_normal
)

def determine_surface_plane(input_surface: Atoms | Slab,
                            normal_vector: np.ndarray | None = None,
                            top_layer_tol: float = 1.5
                            ) -> Atoms:

    # Make sure that the surfaces is on the form of an ASE Atoms 
    # object and an np.array contaning the normal vector:
    atoms, surface_normal = get_atoms_and_normal_from_input(
                                        input_surface, normal_vector
                                        )
    
    # Determining top metal (non-O) layer:
    metal_indices = [i for i, atom in enumerate(atoms) 
                        if atom.symbol != 'O']
    projections = get_atom_projections_on_normal(atoms, surface_normal)
    top_metal = np.max(projections[metal_indices])
    
    top_metal_indices = [i for i in metal_indices if projections[i]
                            >= top_metal - top_layer_tol]
    
    for atom in atoms:
        if atom.index in top_metal_indices:
            atom.tag = 1
        else:
            atom.tag = 0

    return atoms


def determine_adsorption_sites(input_surface: Slab,
                            normal_vector: np.ndarray | None = None,
                            top_layer_tol: float = 1.5,
                            ) -> pd.DataFrame:

    atoms, surface_normal = get_atoms_and_normal_from_input(
                                        input_surface, normal_vector)
    atoms = determine_surface_plane(input_surface, surface_normal, 
                                    top_layer_tol)
    top_metals_indices = [i for i, atom in enumerate(atoms) 
                            if atom.tag == 1]
    atom_projections = get_atom_projections_on_normal(atoms, 
                                                      surface_normal)
    top_layer_projection = np.mean(atom_projections[top_metals_indices])

    df = pd.DataFrame(columns=['site type', 'ad site element', 
                            'ad site neighboring elements', 
                            'reduced coordination', 'position'])

    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False,
                        bothways=True, skin=1e-4)
    nl.update(atoms)
    
    for atom in atoms:
        if atom.tag != 1 and atom.symbol != 'O':
            continue

        indices, offsets = nl.get_neighbors(atom.index)
        neighboring_elements = [atoms[i].symbol for i in indices 
                                if atoms[i].symbol != 'O']


        if atom.tag == 1:

            element = atom.symbol
            top_metal_pos = atom.position

            init_bond_length = natural_cutoffs(Atoms(element))[0]
            init_bond_length += natural_cutoffs(Atoms('O'))[0]
            ad_site_pos = surface_normal * init_bond_length
            ad_site_pos += top_metal_pos
            
            df = pd.concat([
                df,
                pd.DataFrame({
                    'site type': ['on-top'],
                    'ad site element': [element],
                    'ad site neighboring elements': [neighboring_elements],
                    'reduced coordination': 'N/A',
                    'position': [ad_site_pos],

                })
            ], ignore_index=True)

        if atom.symbol == 'O':
            if atom_projections[atom.index] >= top_layer_projection:
                reduced_coordination = input_surface.site_properties\
                        ['bulk-custom-coord'][atom.index]\
                        - len(neighboring_elements)  
                
                if reduced_coordination > 0:
                    atom.tag = 2
                    df = pd.concat([
                        df,
                        pd.DataFrame({
                            'site type': ['lattice O'],
                            'ad site element': ['N/A'],
                            'ad site neighboring elements':
                                [neighboring_elements],
                            'reduced coordination':
                                [reduced_coordination],
                            'position': [atom.position],
                        })
                    ], ignore_index=True)

    for energy_col in ['G_approx(O)', 'G(O)', 'G_approx(OH)', 
                        'G(OH)', 'G_approx(OOH)', 'G(OOH)']:
        df[energy_col] = None


    
    # <<<<<---- NOTE: NEED TO DETERMINE IF ADSORBATE SITES ARE TOO CLOSE
        
    return df, atoms  