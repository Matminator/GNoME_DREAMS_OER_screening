import numpy as np
import pandas as pd

from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from pymatgen.core.surface import Slab, Structure

from gnome_dreams_oer_screening.oer.oer_study_utils import (
    get_atoms_and_normal_from_input, get_atom_projections_on_normal,
    determine_coordination_numbers
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

def determine_adsorption_sites(
                input_surface: Slab,
                normal_vector: np.ndarray | None = None,
                top_layer_tol: float = 1.5,
                only_reduced_coord_O_sotes: bool = True
                            ) -> pd.DataFrame:

    # Obtain the atoms object, given the Slab input:
    atoms, surface_normal = get_atoms_and_normal_from_input(
                                        input_surface, normal_vector)
    
    # Determine the surface plane, set - any atoms in the top: tag = 1:
    atoms = determine_surface_plane(input_surface, surface_normal, 
                                    top_layer_tol)
    # Get indices of the top metal atoms:
    top_metals_indices = [i for i, atom in enumerate(atoms) 
                            if atom.tag == 1]
    
    # Get the "height" (projections) of all atoms in the slab:
    atom_projections = get_atom_projections_on_normal(atoms, 
                                                      surface_normal)
    # Get the mean "height" of the top metal layer, to be used for 
    # determining which O atoms may be considered as adsorption sites:
    top_layer_projection = np.mean(atom_projections[top_metals_indices])

    # Initialize the dataframe listing all adsorption sites + info:
    df = pd.DataFrame(columns=['site type', 'ad site element', 
                            'ad site neighboring elements', 
                            'reduced coordination', 'position'])

    # Regular neighbor list for determining coordination of O atoms:
    nl = NeighborList(natural_cutoffs(atoms), self_interaction=False,
                        bothways=True, skin=1e-4)
    nl.update(atoms)
    # Extended neighbor list for determining all neighboring element,
    # and there distances to the relevant site:
    nl_extended = NeighborList(np.array(natural_cutoffs(atoms))*1.6,
                    self_interaction=False, bothways=True, skin=1e-4)
    nl_extended.update(atoms)
    
    for atom in atoms:
        if atom.tag != 1 and atom.symbol != 'O':
            continue
        if atom.symbol == 'O':
            if atom_projections[atom.index] < top_layer_projection -0.5:
                continue

        indices, offsets = nl_extended.get_neighbors(atom.index)
        closest_neighbors = []
        for i in indices:
            neighbor_atom = atoms[i].symbol
            distance = atoms.get_distance(atom.index, i, mic=True)
            closest_neighbors.append((neighbor_atom, distance))

        # Sort the closest neighbors by distance (closest first):
        closest_neighbors.sort(key=lambda x: x[1], reverse=False)

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
                    'ad site neighboring elements': [closest_neighbors],
                    'reduced coordination': 'N/A',
                    'position': [ad_site_pos],

                })
            ], ignore_index=True)

        if atom.symbol == 'O':

            indices, offsets = nl.get_neighbors(atom.index)

            reduced_coord = input_surface.site_properties\
                    ['bulk-default-coord'][atom.index]\
                    - len(indices)  
            
            if reduced_coord > 0 or not only_reduced_coord_O_sotes:
                atom.tag = 2
                df = pd.concat([
                    df,
                    pd.DataFrame({
                        'site type': ['lattice O'],
                        'ad site element': ['N/A'],
                        'ad site neighboring elements':
                            [closest_neighbors],
                        'reduced coordination':
                            [reduced_coord],
                        'position': [atom.position],
                    })
                ], ignore_index=True)

    for energy_col in ['G_approx(O)', 'G(O)', 'G_approx(OH)', 
                        'G(OH)', 'G_approx(OOH)', 'G(OOH)']:
        df[energy_col] = None
        
    return df, atoms  