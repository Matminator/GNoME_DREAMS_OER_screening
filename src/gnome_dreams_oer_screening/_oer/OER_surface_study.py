import numpy as np
import pandas as pd
import warnings
from ase import Atoms
from ase.visualize import view
from ase.neighborlist import natural_cutoffs #, NeighborList
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab

from _oer.OER_study_utils import get_atoms_and_normal_from_input
from _oer.OER_adsorption_site_study import OER_adsorption_site_study

class OER_surface_study:

    def __init__(self, 
                 OER_catalyst_study,
                 input_surface: Atoms | Slab,
                 normal_vector: np.ndarray | None = None,
                 fixed_bottom_half: bool = True, vacuum = 8
                 ):
    
        # Make sure that the surfaces is on the form of an ASE Atoms 
        # object and an np.array contaning the normal vector:
        base_surface, surface_normal = get_atoms_and_normal_from_input(
                                            input_surface, normal_vector
                                            )
        assert np.isclose(
            np.linalg.norm(surface_normal), 1.0, atol=1e-5
            ), "Surface normal vector must be normalized."
        self.surface_normal = surface_normal
        self.OER_catalyst_study = OER_catalyst_study
        
        # Add vacuum to and center the surface:
        base_surface.center(vacuum=vacuum, axis=2)
        base_surface.wrap()
        if fixed_bottom_half:
            pos = base_surface.get_positions()
            vertical_pos = pos @ surface_normal
            z_center = np.mean(vertical_pos) # - .1  SLIGHT OFFSET !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            fixed_indices = [i for i, z in enumerate(vertical_pos) 
                             if z < z_center]
            base_surface.set_constraint(
                FixAtoms(indices=fixed_indices)
            )

        self.base_surface_non_relaxed = base_surface
        self.add_sites_df_non_relaxed = None
        self.base_surface_relaxed = None
        self.add_sites_df_relaxed = None

        self.relaxed_surface_add_sites_studies = {}
        self.non_relaxed_surface_add_sites_studies = {}

    def set_relaxed_base_surface(self, surface: Atoms, 
                                 energy: float | None = None) -> None:
        self.base_surface_relaxed = surface
        if energy is not None:
            self.relaxed_surface_energy = energy
        else:
            try:
                self.relaxed_surface_energy = \
                    surface.get_potential_energy()
            except:
                raise ValueError("Energy not provided and could not " \
                                 "be obtained from the relaxed atoms")

    def determine_adsorption_sites(self, top_layer_tol: float = 1.5,
                                  use_relaxed_surface: bool = True):
        if use_relaxed_surface and self.add_sites_df_relaxed is not None:
            raise ValueError("Adsorption sites for the relaxed " \
            "surface have already been determined And may not be " \
            "overwritten.")
        if not use_relaxed_surface and self.add_sites_df_non_relaxed is not None:
            raise ValueError("Adsorption sites for the non-relaxed" \
            " surface have already been determined And may not be " \
            "overwritten.")

        if use_relaxed_surface and self.base_surface_relaxed is None:
            raise ValueError("No relaxed surface has been provided " \
            "yet. It needs to be set by calling " \
            "'set_relaxed_base_surface' method.")
        
        if use_relaxed_surface:
            surface = self.base_surface_relaxed.copy()
            self.add_sites_df_relaxed = self._determine_adsorption_sites(
                surface, top_layer_tol)
        else:
            surface = self.base_surface_non_relaxed.copy()
            self.add_sites_df_non_relaxed = self._determine_adsorption_sites(
                surface, top_layer_tol)
    
    def _determine_adsorption_sites(self, surface: Atoms,
                                    top_layer_tol: float = 1.5):

        # Determining top metal (non-O) layer:
        metal_indices = [i for i, atom in enumerate(surface) 
                         if atom.symbol != 'O']
        metal_pos = surface.get_positions()
        metal_vertical_pos = metal_pos @ self.surface_normal
        top_metals = np.max(metal_vertical_pos[metal_indices])
        top_metal_indices = [i for i in metal_indices 
                             if metal_vertical_pos[i] >= 
                             top_metals - top_layer_tol]
        
        df = pd.DataFrame(columns=['site type', 'add site element', 
                                   'position'])
        for top_metal_index in top_metal_indices:
            element = surface[top_metal_index].symbol
            top_metal_pos = metal_pos[top_metal_index]

            init_bond_length = natural_cutoffs(Atoms(element))[0]
            init_bond_length += natural_cutoffs(Atoms('O'))[0]
            ad_site_pos = self.surface_normal * init_bond_length
            ad_site_pos += top_metal_pos
            
            df = pd.concat([
                df,
                pd.DataFrame({
                    'site type': ['on-top'],
                    'add site element': [element],
                    'position': [ad_site_pos],
                })
            ], ignore_index=True)

        for energy_col in ['G(O)', 'G(OH)', 'G(OOH)']:
            df[energy_col] = None
            
        return df        
        
    def get_base_surface(self, relaxed_surface: bool = True,
                     include_adsorption_sites: bool = False,
                     use_adsorbate: str = "X"):
        if relaxed_surface and self.base_surface_relaxed is None:
            raise ValueError("No relaxed surface has been provided " \
                             "yet. It needs to be set by calling " \
                                "'set_relaxed_base_surface' method.")
        if relaxed_surface:
            surface = self.base_surface_relaxed.copy()
        else:
            surface = self.base_surface_non_relaxed.copy()
            
        if include_adsorption_sites:
            if relaxed_surface:
                if self.add_sites_df_relaxed is None:
                    raise ValueError("adsorbtion sites have not been " \
                    "determined yet")
                df = self.add_sites_df_relaxed.copy()
            else:
                if self.add_sites_df_non_relaxed is None:
                    raise ValueError("adsorbtion sites have not been " \
                    "determined yet")
                df = self.add_sites_df_non_relaxed.copy()

        # Adding adsorption sites if requested:
        if include_adsorption_sites:
            for _, row in df.iterrows():
                position = row['position']
                ad_atom = Atoms(use_adsorbate, positions=[position])
                surface += ad_atom

        return surface

    def view_base_surface(self, relaxed_surface: bool = True,
                     include_adsorption_sites: bool = False,
                     use_adsorbate: str = "X"):
        surface = self.get_base_surface(
            relaxed_surface, include_adsorption_sites, use_adsorbate
        )
        view(surface, viewer='x3d')

    def initialize_adsorption_site_study(self, index: int,
                                         relaxed_surface: bool = True
                                         ):
        if relaxed_surface:
            if self.add_sites_df_relaxed is None:
                raise ValueError("No adsorption sites have been " \
                "determined for the relaxed surface yet. They " \
                "need to be set by calling " \
                "'determine_adsorption_sites'")
            if index in self.relaxed_surface_add_sites_studies:
                raise ValueError("Adsorption site study for index " \
                f"{index} on the relaxed surface has already " \
                "been initialized.")
            
            self.relaxed_surface_add_sites_studies[index] = \
                OER_adsorption_site_study(
                    self.base_surface_relaxed.copy(),
                    self.add_sites_df_relaxed.loc[index, 'position'],
                    index
                )

        if not relaxed_surface:
            if self.add_sites_df_non_relaxed is None:
                raise ValueError("No adsorption sites have been " \
                "determined for the non-relaxed surface yet. " \
                "They need to be set by calling " \
                "'determine_adsorption_sites'")
            if index in self.non_relaxed_surface_add_sites_studies:
                raise ValueError("Adsorption site study for index " \
                f"{index} on the non-relaxed surface has already " \
                "been initialized.")

            warnings.warn("\n Warning: For calculating adsorption " \
            "energies it is recommended to use relaxed surfaces only.",
            UserWarning)
            
            self.non_relaxed_surface_add_sites_studies[index] = \
                OER_adsorption_site_study(
                self.base_surface_non_relaxed.copy(),
                self.add_sites_df_non_relaxed.loc[index, 'position'],
                index
                )
            
    def update_adsorption_energies(self):
        if self.add_sites_df_relaxed is None:
            raise ValueError("No adsorption sites have been " \
            "determined for the relaxed surface yet. They " \
            "need to be set by calling " \
            "'determine_adsorption_sites'")
        
        df = self.add_sites_df_relaxed
        studies_dict = self.relaxed_surface_add_sites_studies
        default_energies_and_corrections = \
            self.OER_catalyst_study.get_gas_and_energy_corrections()
        E_pure = self.relaxed_surface_energy
        
        for index, row in df.iterrows():
            study = studies_dict.get(index, None)
            if study is None:
                continue
            energies_dict = study.get_adsorption_site_energies()

            for species in ['O', 'OH', 'OOH']:
                energy_key = f"{species}_adsorbed_energy"
                E = energies_dict.get(energy_key, None)
                if E is not None:
                    G, _ = self._compute_adsorption_free_energies(
                        species, energies_dict,
                        default_energies_and_corrections
                    )
                else:
                    G = None

                df.loc[index, f"G({species})"] = G
                    
            
    def _compute_adsorption_free_energies(self, 
                    species: str, energies_dict,
                    default_energies_and_corrections: dict[str, float]):
        
        
        E_pure = self.relaxed_surface_energy
        E_add = energies_dict.get(f"{species}_adsorbed_energy")
        E_adsorbed = E_add - E_pure 
        corr = default_energies_and_corrections.get(
            f"{species}_ad_correction")
        G_adsorbed = E_adsorbed + corr

        G_H = default_energies_and_corrections.get('H2_gas_free_energy') / 2
        G_H2O = default_energies_and_corrections.get('H2O_gas_free_energy')

        if species == 'O':
            G = G_adsorbed + 2*G_H - G_H2O
            
        return G, None  # Placeholder for now

            
    def get_adsorption_site_studies_dict(self, 
                            relaxed_surface: bool = True
                            ) -> dict[int, OER_adsorption_site_study]:
        if relaxed_surface:
            return self.relaxed_surface_add_sites_studies
        else:
            return self.non_relaxed_surface_add_sites_studies

    def get_adsorption_sites_df(self, relaxed_surface: bool = True
                                ) -> pd.DataFrame:
        if relaxed_surface:
            if self.add_sites_df_relaxed is None:
                raise ValueError("No relaxed adsorption sites have" \
                " been determined yet. They need to be set by " \
                "calling 'determine_adsorption_sites' method with " \
                "use_relaxed_surface=True.")
            df = self.add_sites_df_relaxed.copy()
        else:
            df = self.add_sites_df_non_relaxed.copy()
        return df

    def get_non_relaxed_surface(self) -> Atoms:
        return self.base_surface_non_relaxed.copy()

    def get_relaxed_surface(self) -> Atoms | None:
        if self.base_surface_relaxed is None:
            raise ValueError("No relaxed surface has been provided " \
                             "yet. It needs to be set by calling " \
                                "'set_relaxed_base_surface' method.")
        return self.base_surface_relaxed.copy()


