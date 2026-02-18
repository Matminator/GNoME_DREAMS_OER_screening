import numpy as np
import pandas as pd
from ase import Atoms

    
class OER_adsorption_site_study:
    """
    Class for managing placement of OER intermediates on a given 
    adsorption site of a clean surface, and storage of the energies.
    Is intended to be initialized through the OER_surface_study class,
    via the 'initialize_adsorption_site_study' method.

    Args:
        init_surface: Atoms
            The clean surface on which the adsorbate is to be placed.
        init_add_position: np.ndarray
            The initial position (3D coordinates) for placing the O 
            adsorbate.
        site_index: int
            The index of the adsorption site on the surface (determined 
            by the OER_surface_study class).
        init: tuple(str, float)
            A tuple containing the type of adsorption site and the 
            initial energy of the surface.
    """

    def __init__(self, init_surface: Atoms, 
                 init_add_position: np.ndarray, 
                 site_index: int,
                 init: tuple[str, float]
                 ):
        
        init_surface = init_surface.copy()

        self.init_surface = init_surface
        self.init_add_position = init_add_position
        self.site_index = site_index
        self.site_type = init[0]

        energies_dict = {
            'clean_surface_energy': None,
            'O_adsorbed_energy': None,
            'OH_adsorbed_energy': None,
            'OOH_adsorbed_energy': None # NOT IMPLEMENTED (used for anything) !!!
        }
        if init[0] == 'on-top':
            energies_dict['clean_surface_energy'] = init[1]
            self.atoms_for_O_calc = self._place_O_adsorbate()
        elif init[0] == 'lattice O':
            energies_dict['O_adsorbed_energy'] = init[1]
            self.atoms_for_O_calc = self._remove_O_adsorbate()
        else:
            raise ValueError("Invalied adsorbtion site type:", init[0])
        
        self.energies_dict = energies_dict
        self.relaxed_O_atoms = None
        self.relaxed_OH_atoms = None
        
    def get_atoms_for_O_calc(self) -> Atoms:
        return self.atoms_for_O_calc.copy()
    
    def set_relaxed_O_calc(self, atoms: Atoms,
                                energy: float | None = None) -> None:
      
        if energy is not None:
            E = energy
        else:
            try:
                E = atoms.get_potential_energy()
            except:
                raise ValueError("Energy not provided and could not " \
                                 "be obtained from the relaxed atoms"
                                 "object.")
            
        self.relaxed_O_atoms = atoms.copy()

        if self.site_type == 'on-top':
            self.energies_dict['O_adsorbed_energy'] = E
        elif self.site_type == 'lattice O':
            self.energies_dict['clean_surface_energy'] = E
        else:
            raise ValueError("Invalied adsorbtion site type")       
            
    def get_adsorption_site_energies(self) -> dict[str, float | None]:
        return self.energies_dict.copy()
    
    def _remove_O_adsorbate(self) -> Atoms:
        atoms_copy = self.init_surface.copy()
        removed_atom = False
        for atom in atoms_copy:
            if atom.tag == 3: # Adsorbtion site tag
                atoms_copy.pop(atom.index)
                removed_atom = True
                break
        if not removed_atom:
            raise RuntimeError('Could not identify the lattice O site')
        return atoms_copy

    def _place_O_adsorbate(self) -> Atoms:
        ad_atom = Atoms('O', positions=[self.init_add_position],
                        tags = [4])  # Tag 4 for adsorbate
        system = self.init_surface.copy()
        system += ad_atom
        return system
    

