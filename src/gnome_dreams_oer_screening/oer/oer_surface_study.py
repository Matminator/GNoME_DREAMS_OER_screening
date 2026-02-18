import numpy as np
from numpy.typing import NDArray
import pandas as pd
import warnings
from ase import Atoms
from ase.visualize import view
from ase.neighborlist import natural_cutoffs #, NeighborList
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab

from gnome_dreams_oer_screening.oer.oer_study_utils import (
    get_atoms_and_normal_from_input)
from gnome_dreams_oer_screening.oer.ad_site_determination import (
    determine_adsorption_sites)
from gnome_dreams_oer_screening.oer.oer_adsorption_site_study import (
    OER_adsorption_site_study)

class OER_surface_study:
    """
    Class for determening surface adsorption sites. This class is 
    intented to be initialized through an OER_catalyst_study object, via
    the method 'initialize_OER_surface_studies' method. This class 
    handles the determination and storage of surface adsorption sites.
    For later computation of adsorption energies for OER intermediates 
    it is highly recommended to set a relaxed surface structure first 
    and use adsorption sites determined on the relaxed surface. This
    must be done through the 'set_relaxed_base_surface' method (the 
    'get_non_relaxed_surface' method may be used to obtain the
    non-relaxed surface structure for this purpose).
    To compute adsorption energies of OER intermediates for a given 
    site, the 'initialize_adsorption_site_study' method should be
    called. This initializes an OER_adsorption_site_study object for the
    given site index. The OER_adsorption_site_study object then handles
    the placement of the adsorbate and storage of the adsorbate
    energies. After having stored the relvant energies in the
    OER_adsorption_site_study object, the 'update_adsorption_energies'
    method may be called to compute adsorption energies. These may then 
    be retrieved through the 'get_adsorption_sites_df' method, yielding 
    a pandas DataFrame with the adsorption sites and adsorbate energies.

    Args:
        OER_catalyst_study: OER_catalyst_study
            The OER_catalyst_study object to which this surface study
            belongs.
        input_surface: Atoms | Slab
            The surface structure to be studied, either as an ASE Atoms
            object or a Pymatgen Slab object.
        normal_vector: np.ndarray | None
            The normal vector of the surface. If None, the normal vector
            will be inferred from the input_surface object (if
            possible).
        fixed_bottom_half: bool
            Whether to fix the bottom half of the surface. Default is 
            True.
        vacuum: float
            The amount of vacuum (in Ångstrom) to add to the surface.
            Default is 14 Ångstrom.
    """
    

    def __init__(self, 
                 OER_catalyst_study,
                 input_surface: Slab,
                 normal_vector: np.ndarray | None = None,
                 fixed_bottom_half: bool = True, vacuum: float = 14
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

        # Save original slab - to store site atributes:
        self.original_slab = input_surface.copy()

        # Initialize properties list:
        self.site_properties = [{} for _ in range(len(input_surface))]
        input_site_properties = input_surface.site_properties
        self._add_site_data('bulk-default-coord', 
                           input_site_properties['bulk-default-coord']
                           )
        
        # Add vacuum to and center the surface:
        base_surface.center(vacuum=vacuum, axis=2)
        base_surface.wrap()
        if fixed_bottom_half:
            pos = base_surface.get_positions()
            vertical_pos = pos @ surface_normal
            z_center = np.median(vertical_pos)
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

    def get_non_relaxed_surface(self) -> Atoms:
        return self.base_surface_non_relaxed.copy()

    def get_relaxed_surface(self) -> Atoms | None:
        if self.base_surface_relaxed is None:
            raise ValueError("No relaxed surface has been provided " \
                             "yet. It needs to be set by calling " \
                                "'set_relaxed_base_surface' method.")
        return self.base_surface_relaxed.copy()

    def set_relaxed_base_surface(self, surface: Atoms, 
                                 energy: float | None = None) -> None:
        
        if len(surface) != len(self.base_surface_non_relaxed):
            raise ValueError("This is not the correct relaxed surface" \
                "The provided surface has a different number of atoms" \
                f" than the non-relaxed surface, namly {len(surface)}" \
                f" instead of {len(self.base_surface_non_relaxed)}.")
        
        for i in range(len(surface)):
            if surface[i].symbol != self.base_surface_non_relaxed[i].symbol:
                raise ValueError("The provided surfaces is either not "\
                    "the correct relaxed surface or the order of the " \
                    "atoms is different. BOTH ARE CRITICAL ISSUES.")

        if energy is not None:
            self.relaxed_surface_energy = energy
        else:
            try:
                self.relaxed_surface_energy = \
                    surface.get_potential_energy()
            except:
                raise ValueError("Energy not provided and could not " \
                                 "be obtained from the relaxed atoms")
            
        self.base_surface_relaxed = surface.copy()
            
    def _add_site_data(self, 
            property_name: str,
            properties: list,
            indices: list[int] | NDArray[np.int_] | None = None
            ) -> None:
        
        if indices is None:
            indices = range(len(self.original_slab))

        for i, prop in zip(indices, properties):
            self.site_properties[i][property_name] = prop


    def determine_adsorption_sites(self, 
            top_layer_tol: float = 1.5,
            only_reduced_coord_O_sites: bool = False
            ) -> None:
        
        if self.add_sites_df_relaxed is not None:
            raise ValueError("Adsorption sites for the relaxed " \
            "surface have already been determined And may not be " \
            "overwritten.")
        if self.base_surface_relaxed is None:
            raise ValueError("No relaxed surface has been provided " \
            "yet. It needs to be set by calling " \
            "'set_relaxed_base_surface' method.")
        
        structrue = AseAtomsAdaptor.get_structure(
            self.base_surface_relaxed)
        
        structrue.add_site_property('bulk-default-coord',
                                    self.original_slab.site_properties\
                                    ['bulk-default-coord'])
    
        df, atoms = determine_adsorption_sites(
            structrue,
            normal_vector = self.surface_normal,
            top_layer_tol = top_layer_tol,
            only_reduced_coord_O_sites = only_reduced_coord_O_sites
            )
        
        df.insert(loc=0, column='Site index', value=df.index)
        
        tags = [atom.tag for atom in atoms]
        for atom, tag in zip(self.base_surface_relaxed, tags):
            atom.tag = tag
        self._add_site_data('tag', tags)
        self.add_sites_df_relaxed = df

    def initialize_adsorption_site_study(self, index: int) -> None:
        if self.add_sites_df_relaxed is None:
            raise ValueError("No adsorption sites have been " \
            "determined for the relaxed surface yet. They " \
            "need to be set by calling 'determine_adsorption_sites'")
        if index in self.relaxed_surface_add_sites_studies:
            raise ValueError("Adsorption site study for index " \
            f"{index} on the relaxed surface has already been " \
            "initialized.")
        
        site_row = self.add_sites_df_relaxed[
            self.add_sites_df_relaxed['Site index'] == index
        ]
        pos = site_row['position'].values[0]
        site_type = site_row['site type'].values[0]
        if site_type not in ['on-top', 'lattice O']:
            raise ValueError(f"Unknown site type: {site_type}"\
                "No methoded are implemented for this type")
        
        atoms = self.base_surface_relaxed.copy()
        atoms[int(site_row['atom_index'].values[0])].tag = 3 # ad-site tag
        
        self.relaxed_surface_add_sites_studies[index] = \
            OER_adsorption_site_study(atoms, pos, index,
                init = (site_type, self.relaxed_surface_energy))
        
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
        
        for index, row in df.iterrows():

            # Test if the adsorption site study for this index:
            study = studies_dict.get(index, None)
            if study is None:
                continue

            # Test if enough energies are available:
            energies_dict = study.get_adsorption_site_energies()
            num_none_Es = 0
            for energy in energies_dict.values():
                if energy is None:
                    num_none_Es += 1
                if num_none_Es > 2: # there only exist 1 energy
                    continue
            
            for species in ['O', 'OH']:
                energy_key = f"{species}_adsorbed_energy"
                E = energies_dict.get(energy_key, None)
                if E is not None:
                    G_approx, G = self._compute_adsorption_free_energies(
                        species, energies_dict,
                        default_energies_and_corrections
                    )
                else:
                    G_approx = None

                df.loc[index, f"G_approx({species})"] = G_approx

            
    def _compute_adsorption_free_energies(self, 
                    species: str, energies_dict,
                    default_energies_and_corrections: dict[str, float]):
        
        E_pure = energies_dict.get('clean_surface_energy')
        E_add = energies_dict.get(f"{species}_adsorbed_energy")
        E_adsorbed = E_add - E_pure 
        corr = default_energies_and_corrections.get(
            f"{species}_ad_correction")
        G_adsorbed_approx = E_adsorbed + corr

        G_H = default_energies_and_corrections['H2_gas_free_energy'] / 2
        G_H2O = default_energies_and_corrections['H2O_gas_free_energy']

        if species == 'O':
            G_approx = G_adsorbed_approx + 2*G_H - G_H2O
        if species == 'OH':
            G_approx = G_adsorbed_approx + G_H - G_H2O
            
        return G_approx, None  # Placeholder for now

            
    def get_adsorption_site_studies_dict(self, 
                            relaxed_surface: bool = True
                            ) -> dict[int, OER_adsorption_site_study]:
        if relaxed_surface:
            return self.relaxed_surface_add_sites_studies
        else:
            return self.non_relaxed_surface_add_sites_studies

    def get_adsorption_sites_df(self) -> pd.DataFrame:
        if self.add_sites_df_relaxed is None:
            return None
        return self.add_sites_df_relaxed.copy()
