import numpy as np
import pandas as pd
from ase import Atoms

    
class OER_adsorption_site_study:
    
    def __init__(self, clean_surface: Atoms, 
                 init_add_position: np.ndarray, site_index: int
                 ):
        
        for atom in clean_surface:
            atom.tag = 0  # Resetting tags to zero
        
        self.clean_surface = clean_surface
        self.init_add_position = init_add_position
        self.site_index = site_index
        
        self.init_surface_with_O = self._place_O_adsorbate()
        self.relaxed_surface_with_O = None
        
        self.adsorbed_O_energy = None
        self.adsorbed_OH_energy = None
        self.adsorbed_OOH_energy = None

        """
        TO BE ADDED:
        Sould include methods for adding ZPEs and entropy corrections
        to the adsorbed species energies. 
        """
        

    def get_initial_surface_with_O(self) -> Atoms:
        return self.init_surface_with_O.copy()
    
    def set_relaxed_surface_with_O(self, relaxed_surface_with_O: Atoms,
                                energy: float | None = None) -> None:
        self.relaxed_surface_with_O = relaxed_surface_with_O
        if energy is not None:
            self.adsorbed_O_energy = energy
        else:
            try:
                self.adsorbed_O_energy = \
                      relaxed_surface_with_O.get_potential_energy()
            except:
                raise ValueError("Energy not provided and could not " \
                                 "be obtained from the relaxed atoms"
                                 "object.")
            
    def get_adsorption_site_energies(self) -> dict[str, float | None]:
        return {
            "O_adsorbed_energy": self.adsorbed_O_energy,
            "OH_adsorbed_energy": self.adsorbed_OH_energy,
            "OOH_adsorbed_energy": self.adsorbed_OOH_energy
        }

    def _place_O_adsorbate(self) -> Atoms:
        ad_atom = Atoms('O', positions=[self.init_add_position],
                        tags = [1])  # Tag 1 for adsorbate
        system = self.clean_surface.copy()
        system += ad_atom
        return system
    

