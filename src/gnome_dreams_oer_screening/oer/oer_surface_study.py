import numpy as np
from numpy.typing import NDArray
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab

from gnome_dreams_oer_screening.oer.oer_study_utils import (
    get_atoms_and_normal_from_input)
from gnome_dreams_oer_screening.oer.ad_site_determination import (
    determine_adsorption_sites)
from gnome_dreams_oer_screening.oer.oer_adsorption_site_study import (
    OER_adsorption_site_study)
from gnome_dreams_oer_screening.oer.atoms_sanity_checks import (
    are_all_atoms_connected,
    are_systems_equal,
    StructureConsistencyError,
)


class SurfaceStudyLockedError(RuntimeError):
    """Raised when a locked surface study blocks further analysis."""


class OER_surface_study:
    """
    Class for determining surface adsorption sites. This class is 
    intended to be initialized through an OER_catalyst_study object, via
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
    energies. After having stored the relevant energies in the
    OER_adsorption_site_study object, the 'update_adsorption_energies'
    method may be called to compute adsorption energies. These may then 
    be retrieved through the 'get_adsorption_sites_df' method, yielding 
    a pandas DataFrame with the adsorption sites and adsorbate energies.

    Args:
        OER_catalyst_study: OER_catalyst_study
            The OER_catalyst_study object to which this surface study
            belongs.
        input_surface: Slab
            The surface structure to be studied as a Pymatgen Slab
            object.
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

        # Make sure that the surface is on the form of an ASE Atoms 
        # object and an np.array containing the normal vector:
        base_surface, surface_normal = get_atoms_and_normal_from_input(
                                            input_surface, normal_vector
                                            )

        if not np.isclose(np.linalg.norm(surface_normal), 1.0,
                            atol=1e-5):
            raise ValueError("Surface normal vector must be " \
                "normalized.")
        self.surface_normal = surface_normal
        self.OER_catalyst_study = OER_catalyst_study

        # Save original slab - to store site attributes:
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
        self.relaxed_surface_energy = None
        self.add_sites_df_relaxed = None
        
        # This lock prevents further calculations (add-site studies)
        # if a non-continuous relaxed surface has been provided.
        self.study_locked = False

        self.relaxed_surface_add_sites_studies = {}
        self.non_relaxed_surface_add_sites_studies = {}

    def get_non_relaxed_surface(self) -> Atoms:
        return self.base_surface_non_relaxed.copy()

    def get_relaxed_surface(self) -> Atoms | None:
        out = None
        if self.base_surface_relaxed is not None:
            out = self.base_surface_relaxed.copy()
        return out

    def set_relaxed_base_surface(self, surface: Atoms, 
                                 energy: float | None = None,
                                 ) -> bool:
        """
        Set the relaxed surface structure for this surface study. This 
        is for later use in the determination of adsorption sites and
        adsorption energies.

        Args:
        surface: Atoms
        energy: float | None
            The potential energy of the relaxed surface. If None,
            the energy will be obtained from the relaxed surface
            structure (if possible). Default is None.

        Returns:
        bool
            Whether the provided relaxed surface is a single connected
            system. If False, the surface study is locked until a valid
            relaxed surface is provided.
        """

        if self.base_surface_relaxed is not None:
            raise ValueError("A relaxed surface has already been "
            "provided for this surface study. Providing a new relaxed "\
            "surface at this stage would break the validity of any " \
            "downstream analysis (e.g. adsorption sites and energies)")
        
        # Test if the provided surface is consistent with the 
        # non-relaxed surface:
        equal_results = are_systems_equal(surface, 
                                        self.base_surface_non_relaxed)
        if not equal_results[0]:
            raise StructureConsistencyError(
                "The provided relaxed surface is not "
                "consistent with the non-relaxed surface due to: " + \
                f"{equal_results[1]}"
            )        
        
        # Test if the provided surface is a single connected system:
        if not are_all_atoms_connected(surface):
            self.study_locked = True

        else:
            if energy is not None:
                self.relaxed_surface_energy = energy
            else:
                try:
                    self.relaxed_surface_energy = \
                        surface.get_potential_energy()
                except Exception as exc:
                    raise ValueError("Energy not provided and could " \
                        "not be obtained from the relaxed atoms") from exc
                
            self.study_locked = False
            self.base_surface_relaxed = surface.copy()

        # Return bool on whether the surface is continuous or not. 
        # Significant for determining if further computations/analysis
        # should be carried out for this surface study:
        return not self.study_locked
            
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
        
        if self.study_locked:
            raise SurfaceStudyLockedError(
                "Adsorption sites cannot be determined because this "
                "surface study is locked due to a previously provided "
                "non-continuous relaxed surface."
            )
        if self.add_sites_df_relaxed is not None:
            raise ValueError("Adsorption sites for the relaxed " \
            "surface have already been determined and may not be " \
            "overwritten.")
        if self.base_surface_relaxed is None:
            raise ValueError("No relaxed surface has been provided " \
            "yet. It needs to be set by calling " \
            "'set_relaxed_base_surface' method.")
        
        structure = AseAtomsAdaptor.get_structure(
            self.base_surface_relaxed)
        
        # This is needed for determining under-coordinated sites:
        structure.add_site_property('bulk-default-coord',
                                    self.original_slab.site_properties\
                                    ['bulk-default-coord'])
    
        df, atoms = determine_adsorption_sites(
            structure,
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
        if index not in self.add_sites_df_relaxed['Site index'].values:
            raise ValueError(
                f"No adsorption site with index {index} exists for "
                "this surface study."
            )
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
                " No methods are implemented for this type")
        
        atoms = self.base_surface_relaxed.copy()
        # Add tag 3 to the adsorption site atom:
        atoms[int(site_row['atom_index'].values[0])].tag = 3
        
        self.relaxed_surface_add_sites_studies[index] = \
            OER_adsorption_site_study(atoms, pos, index,
                self.surface_normal,
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

            energies_dict = study.get_adsorption_site_energies()
            E_pure = energies_dict.get("clean_surface_energy", None)
            
            for species in ['O', 'OH']:
                energy_key = f"{species}_adsorbed_energy"
                E_add = energies_dict.get(energy_key, None)
                if E_pure is not None and E_add is not None:
                    G, _ = self._compute_adsorption_free_energies(
                        species, energies_dict,
                        default_energies_and_corrections
                    )
                else:
                    G = None

                df.loc[index, f"G({species})"] = G

        self._compute_overpotentials()
            
    def _compute_adsorption_free_energies(self, 
                    species: str, energies_dict,
                    default_energies_and_corrections: dict[str, float]):
        
        E_pure = energies_dict.get('clean_surface_energy')
        E_add = energies_dict.get(f"{species}_adsorbed_energy")
        E_adsorbed = E_add - E_pure 
        corr = default_energies_and_corrections.get(
            f"{species}_ad_correction")
        G_adsorbed = E_adsorbed + corr

        G_H = default_energies_and_corrections['H2_gas_free_energy'] / 2
        G_H2O = default_energies_and_corrections['H2O_gas_free_energy']

        if species == 'O':
            G = G_adsorbed + 2*G_H - G_H2O
        elif species == 'OH':
            G = G_adsorbed + G_H - G_H2O
        else:
            raise ValueError(
                f"Unsupported species for adsorption free energy: "
                f"{species}"
            )
            
        return G, None  # Placeholder for now

    def _compute_overpotentials(self) -> None:
        df = self.get_adsorption_sites_df()
        if df is None:
            return
        
        for index, row in df.iterrows():
            G_O = row.get("G(O)", None)
            G_OH = row.get("G(OH)", None)
            if G_O is None:
                continue
            if G_OH is None:
                # Computing the overpotential, assuming an ideal 
                # energies of the OOH and OH intermediates based the O 
                # energy:
                ideal_overpot = np.abs(G_O - 2.46)/2
                ideal_overpot = float(np.round(ideal_overpot, 2))
                df.loc[index, "ideal overpotential"] = ideal_overpot
            else:
                # Computing the overpotential, assuming an ideal energy
                # of the OOH intermediate based on the known energies:
                G_H2O_OH_step = G_OH 
                G_OH_O_step = G_O - G_OH
                G_O_O2_step = (4.92 - G_O)/2
                ideal_overpot = max(G_H2O_OH_step, G_OH_O_step, 
                                    G_O_O2_step) - 1.23
                ideal_overpot = float(np.round(ideal_overpot, 2))
                df.loc[index, "ideal overpotential"] = ideal_overpot

                # Computing the overpotential, applying the set OH-OOH 
                # scaling relation:
                b = self.OER_catalyst_study.OH_OOH_scaling_intercept
                G_OOH_from_scaling = G_OH + b
                # set value in df for reference:
                df.loc[index, "G(OOH) from scaling relation"] = \
                    G_OOH_from_scaling

                G_O_OOH_step = G_OOH_from_scaling - G_O
                G_OOH_O2_step = 4.92 - G_OOH_from_scaling
                scaling_overpot = max(G_H2O_OH_step, G_OH_O_step, 
                                    G_O_OOH_step, G_OOH_O2_step) - 1.23
                scaling_overpot = float(np.round(scaling_overpot, 2))
                df.loc[index,
                        "overpotential from OH-OOH scaling relation"] =\
                    scaling_overpot

        self.add_sites_df_relaxed = df


    def get_adsorption_site_studies_dict(self, 
                            relaxed_surface: bool = True
                            ) -> dict[int, OER_adsorption_site_study]:
        if relaxed_surface:
            return self.relaxed_surface_add_sites_studies
        else:
            return self.non_relaxed_surface_add_sites_studies

    def get_adsorption_sites_df(self) -> pd.DataFrame | None:
        if self.add_sites_df_relaxed is None:
            return None
        return self.add_sites_df_relaxed.copy()
