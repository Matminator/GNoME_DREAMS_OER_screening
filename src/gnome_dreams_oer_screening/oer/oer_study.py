import time

import numpy as np
import pandas as pd

from typing import Literal
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab, Structure
from pymatgen.core.surface import (
    get_symmetrically_distinct_miller_indices, SlabGenerator)

from gnome_dreams_oer_screening.oer.oer_study_utils import (
    eval_stoichiometry_diff,get_stoichiometry_diff, 
    get_stoichiometry_dict, compute_slab_thickness,
    get_top_half_atom_projections, determine_coordination_numbers
    )
from gnome_dreams_oer_screening.oer.oer_surface_study import (
    OER_surface_study
)
from gnome_dreams_oer_screening.oer.ad_site_determination import (
    determine_adsorption_sites
)

from gnome_dreams_oer_screening.oer.atoms_sanity_checks import (
    assert_elements_subset,
)

# Suppress specific warnings from pymatgen about too many sites removed.
# These warnings can occur during slab generation and are handled by
# the code logic, so they can be safely ignored.
import warnings
warnings.filterwarnings(
    "ignore",
    message="Too many sites removed, please use a larger slab",
    category=UserWarning,
)


class OER_catalyst_study:
    """
    The core class for ranking and generating surfaces from a bulk
    structure for OER catalyst screening. All
    data related to computation of adsorption energies is stored in this
    class. Surfaces are initialized as OER_surface_study instances, via 
    the method 'initialize_oer_surface_study', and stored in a 
    dictionary, with miller indices as keys. Surface structures with 
    adsorbates are managed in the OER_surface_study instances. Gas phase
    free energies of H2O and H2 are stored in this class, in addition 
    to initial corrections (correction = ZPE + Cp - TS) to be applied to
    adsorption energies.

    Args:
        init_atoms (Atoms): the unrelaxed bulk structure of the 
        catalyst.
        H2O_gas_free_energy (float): Free energy of H2O in gas phase.
        H2_gas_free_energy (float): Free energy of H2 in gas phase.
        default_O_ad_correction (float, optional): Default correction
        made to the DFT O* adsorption energy.
        default_OH_ad_correction (float, optional): Default correction
        made to the DFT OH* adsorption energy.
        default_OOH_ad_correction (float, optional): Default correction
        made to the DFT OOH* adsorption energy.
        use_init_as_relaxed_bulk (bool): If True, the init structure is used
        directly as the relaxed bulk, bypassing the magnetic enumeration
        pipeline entirely. Intended for testing, notebooks, and non-magnetic
        materials where running the full magnetic batch (enumerate → store →
        finalize) is not required. Sets _mag_bulk_batch_finalized = True so
        finalize_mag_bulk_batch will raise if accidentally called afterwards.
    """

    def __init__(self, init_atoms: Atoms,
                 H2O_gas_free_energy: float,
                 H2_gas_free_energy: float,
                 default_O_ad_correction: float = 0.070,
                 default_OH_ad_correction: float = 0.361,
                 default_OOH_ad_correction: float = 0.431,
                 OH_OOH_scaling_intercept: float = 3.2,
                 use_init_as_relaxed_bulk: bool = False,
                 ):

         # Avoid mutations outside of this class:
        init_atoms = init_atoms.copy()
        for atom in init_atoms:
            atom.tag = 0 # Ensure all tags are set to 0

        self.init_atoms = init_atoms
        self.init_structure = AseAtomsAdaptor.get_structure(init_atoms)
        self.bulk_atoms = None
        self.bulk_structure = None
            
        self.species_energies = {
            'H2O_gas_free_energy': H2O_gas_free_energy,
            'H2_gas_free_energy': H2_gas_free_energy,
            'O_ad_correction': default_O_ad_correction,
            'OH_ad_correction': default_OH_ad_correction,
            'OOH_ad_correction': default_OOH_ad_correction,
        }

        self.max_miller = None
        self.distinct_millers = None
        self.termination_ranking_df = None
        self.terminations = {}
        self.applied_ranking_method = None
        self.oer_surface_studies = {}
        # self.failed_slab_generation = False
        self.OH_OOH_scaling_intercept = OH_OOH_scaling_intercept

        self._mag_bulk_candidates: dict[int, tuple] = {}  # {idx: (atoms, energy, magmom)}
        self._mag_bulk_batch_finalized: bool = False
        self._bulk_magmom: list[float] | None = None
        self._mag_bulk_configs: list[dict] | None = None

        if use_init_as_relaxed_bulk:
            self._set_relaxed_structure(self.init_atoms)
            self._mag_bulk_batch_finalized = True

    def get_init_structure(self) -> Structure:
        return self.init_structure.copy()
    
    def get_init_atoms(self) -> Atoms:
        return self.init_atoms.copy()
    
    def _set_relaxed_structure(self, relaxed_bulk: Atoms | Structure
                           ) -> None:
            
        # Do not allow resetting the relaxed bulk structure if surfaces 
        # have already been generated:
        if len(self.terminations) > 0:
            raise RuntimeError("Relaxed bulk structure cannot be set " \
                "after surfaces have been generated. Once surfaces "
                "have been generated, the relaxed bulk structure " \
                "You will need to create a new OER_catalyst_study " \
                "instance if you want to use a different relaxed bulk" \
                " structure")
        
        # Type check:        
        if isinstance(relaxed_bulk, Atoms):
            relaxed_bulk_atoms = relaxed_bulk
            relaxed_bulk_structure = AseAtomsAdaptor.get_structure(
                relaxed_bulk
            )
        elif isinstance(relaxed_bulk, Structure):
            relaxed_bulk_atoms = AseAtomsAdaptor.get_atoms(relaxed_bulk)
            relaxed_bulk_structure = relaxed_bulk
        else:
            raise ValueError("relaxed_bulk must be an ASE Atoms " \
                "object or a Pymatgen Structure object.")
        
        # Make a copy, to avoide mutations outside of this class
        relaxed_bulk = relaxed_bulk.copy() 
        
        # Allow primitive-cell representations of the same material.  Only require
        # that the relaxed bulk contains no elements absent from the initial
        # structure — atom count may differ if the enumerator reduced the cell.
        assert_elements_subset(
            relaxed_bulk_atoms, self.init_atoms, label="relaxed_bulk"
        )

        self.bulk_structure = relaxed_bulk_structure
        self.bulk_atoms = relaxed_bulk_atoms

    def set_relaxed_structure(self, atoms: Atoms) -> None:
        """Bypass for direct bulk setting — used in tests and notebooks.
        Marks the magnetic batch as finalized so finalize_mag_bulk_batch
        is not expected to run. Does not raise if already set."""
        self._set_relaxed_structure(atoms)
        self._mag_bulk_batch_finalized = True

    def set_energies(self,
            H2O_gas_free_energy: float | None = None,
            H2_gas_free_energy: float | None = None,
            O_ad_correction: float | None = None,
            OH_ad_correction: float | None = None,
            OOH_ad_correction: float | None = None,
            ) -> None:
        if H2O_gas_free_energy is not None:
            self.species_energies['H2O_gas_free_energy'] = \
            H2O_gas_free_energy
        if H2_gas_free_energy is not None:
            self.species_energies['H2_gas_free_energy'] = \
                H2_gas_free_energy
        if O_ad_correction is not None:
            self.species_energies['O_ad_correction'] = \
                O_ad_correction
        if OH_ad_correction is not None:
            self.species_energies['OH_ad_correction'] = \
                OH_ad_correction
        if OOH_ad_correction is not None:
            self.species_energies['OOH_ad_correction'] = \
                OOH_ad_correction

    def get_gas_and_energy_corrections(self) -> dict[str, float]:
        return self.species_energies.copy()

    def get_bulk_atoms(self) -> Atoms | None:
        if self.bulk_atoms is None:
            return None
        else:
            return self.bulk_atoms.copy()
    
    def get_bulk_structure(self) -> Structure | None:
        if self.bulk_structure is None:
            return None
        else:
            return self.bulk_structure.copy()

    def store_mag_bulk_candidate(self, idx: int, atoms: Atoms,
                                  energy: float, magmom: list[float]) -> None:
        """Store a completed magnetic bulk result keyed by config index.

        Raises ValueError if magmom length does not match atom count.
        Raises StructureConsistencyError if atoms are inconsistent with
        the initial structure (wrong element order, wrong count, etc.).
        """
        if len(magmom) != len(atoms):
            raise ValueError(
                f"magmom length ({len(magmom)}) does not match "
                f"number of atoms ({len(atoms)}) for candidate {idx}."
            )
        # Allow primitive-cell representations: only require that the candidate
        # contains no elements absent from the initial structure.  The enumerator
        # may return a cell with fewer atoms (e.g. FM BCC Fe collapses to the
        # 1-atom primitive cell), which is valid as long as no new species appear.
        assert_elements_subset(
            atoms, self.init_atoms, label=f"Candidate {idx}"
        )
        self._mag_bulk_candidates[idx] = (atoms, energy, magmom)

    def is_mag_bulk_batch_finalized(self) -> bool:
        return self._mag_bulk_batch_finalized


    # Returns False when all bulk relaxations failed (no candidates stored).
    # All process rows are already in terminal state in the process log —
    # that is the correct and sufficient failure signal.
    def finalize_mag_bulk_batch(self) -> bool:
        """
        Pick the lowest-energy-per-atom magnetic candidate, set it as the
        relaxed bulk, and store its magmoms as site properties so that
        SlabGenerator propagates them to slabs.
        Returns True on success, False if no completed candidates exist.
        Raises RuntimeError if called after already finalized (indicates a
        bug in the calling code).
        """
        if self._mag_bulk_batch_finalized:
            raise RuntimeError(
                "finalize_mag_bulk_batch called after batch already finalized."
            )
        if not self._mag_bulk_candidates:
            return False

        best_idx = min(
            self._mag_bulk_candidates,
            key=lambda i: (
                self._mag_bulk_candidates[i][1] / len(self._mag_bulk_candidates[i][0])
            ),
        )
        best_atoms, best_energy, best_magmom = self._mag_bulk_candidates[best_idx]

        if np.isnan(best_energy):
            raise ValueError(
                f"Winning bulk candidate {best_idx} has NaN energy. "
                "Check VASP convergence for this candidate."
            )
        if np.any(np.isnan(best_magmom)):
            raise ValueError(
                f"Winning bulk candidate {best_idx} has NaN magnetic moments. "
                "Check VASP convergence for this candidate."
            )

        # Calls the existing validated setter (checks atom count, symbols, tags):
        self._set_relaxed_structure(best_atoms)

        # _set_relaxed_structure converts via AseAtomsAdaptor which drops magmom.
        # Set site properties explicitly on the now-assigned bulk_structure:
        for site, m in zip(self.bulk_structure.sites, best_magmom):
            site.properties["magmom"] = m

        self._bulk_magmom = list(best_magmom)
        self._mag_bulk_batch_finalized = True
        return True

    def get_bulk_magmom(self) -> list[float] | None:
        """Per-atom magnetic moments for the winning bulk configuration.
        Relaxed VASP moments in production; initial enumerator moments in test mode."""
        return list(self._bulk_magmom) if self._bulk_magmom is not None else None


    def store_mag_bulk_configs(self, configs: list[dict]) -> None:
        """Store the enumerated initial magnetic configurations for this study.
        Called once from enter_process. Raises RuntimeError if configs are already
        stored — re-storing indicates a duplicate enter_process call."""
        if getattr(self, '_mag_bulk_configs', None) is not None:
            raise RuntimeError(
                "Magnetic bulk configs are already stored for this study. "
                "store_mag_bulk_configs must only be called once per study."
            )
        self._mag_bulk_configs: list[dict] = configs

    def get_mag_bulk_configs(self) -> list[dict] | None:
        """Return the stored initial magnetic configs, or None if not yet stored."""
        return getattr(self, '_mag_bulk_configs', None)

    def get_distinct_millers(self) -> list[tuple[int, int, int]] | None:
        if self.distinct_millers is None:
            return None
        else:
            return self.distinct_millers.copy()
    
    def get_terminations_dict(self) -> dict[int, Slab]:
        return self.terminations.copy()
    
    def _set_bulk_coordination(self, method: str) -> None:
        """
        Compute and store coordination numbers for bulk atoms, applying
        the specified method. The method SHOULD be the same as that used
        for surface slabs, to ensure consistency!
        
        Args:
            method (str): Method to compute coordination numbers.
        """
        # Compute coordination numbers for bulk atoms, using the method
        coord_numbers = determine_coordination_numbers(
            self.bulk_atoms, method)
        
        if method != 'all':
            coord_numbers_default = determine_coordination_numbers(
                self.bulk_atoms, method='all')
        else:
            coord_numbers_default = coord_numbers.copy()

        # Store coordination numbers as a site property in the bulk 
        # structure:
        self.bulk_structure.add_site_property("bulk-custom-coord", 
                                              coord_numbers)
        self.bulk_structure.add_site_property("bulk-default-coord",
                                               coord_numbers_default)

    def _build_surfaces(self, 
            miller_indices: tuple[int, int, int],
            stoichiometry_tolerance: float = 0.1,
            max_diff_dict: dict | None = None,
            ignore_stoichiometry_of_species: list[str] | None = None,
            all_species_present: bool = True,
            symmetrize: bool = True,
            select_closest_O_stoichiometry: bool = True,
            min_slab_thickness: float = 9,
            max_slab_thickness: float = 18,
            min_atoms: int = 20, 
            max_atoms: int = 120,
            vacuum: float = 1,
            max_layers: int = 6,
            timeout: int = 5*60,  # Default timeout of 5 minutes
            **kwargs
            ) -> list[Slab]:

        # if self.failed_slab_generation:
        #     raise RuntimeError("Previous slab generation attempt " \
        #     "failed due to timeout. Slab generation will not be " \
        #     "attempted again. It is not possible to generate " \
        #     "surfaces for this bulk structure.")
        
        if max_layers < 3:
            raise ValueError("max_layers must be at least 3 to allow"
                " for reasonable slab generation attempts.")

        if max_diff_dict is None:
            max_diff_dict = {}
        if ignore_stoichiometry_of_species is None:
            ignore_stoichiometry_of_species = []

        # Log time, to be used for timeout handling:
        start_time = time.time()

        num_layers = 2
        while True:
            time_elapsed = time.time() - start_time
            if num_layers > max_layers or time_elapsed > timeout:
                slabs = []
                break

            sg = SlabGenerator(
            initial_structure = self.bulk_structure.copy(),
            miller_index = miller_indices,
            min_slab_size = num_layers,
            min_vacuum_size = vacuum,
            lll_reduce = True,
            in_unit_planes = True,
            center_slab = True,
            primitive = True,
            )
            slabs = sg.get_slabs(symmetrize = symmetrize)
            
            # Check if any slabs were generated, if not, retry:

            if len(slabs) == 0:
                num_layers += 1
                continue

            # Test if slabs are thick enough, if not, retry:
            thicknesses = [compute_slab_thickness(slab) for slab 
                        in slabs]
            if np.min(thicknesses) < min_slab_thickness:
                num_layers += 1
                continue
            
            # Test if slabs have enough atoms, if not, retry:
            num_atoms = [len(slab) for slab in slabs]
            if np.min(num_atoms) < min_atoms:
                num_layers += 1
                continue

            if not all([eval_stoichiometry_diff(
                    slab, self.bulk_structure,
                    max_diff = stoichiometry_tolerance,
                    max_diff_dict = max_diff_dict,
                    ignore_species = ignore_stoichiometry_of_species,
                    all_species_present = all_species_present
                    ) for slab in slabs]):
                num_layers += 1
                continue
            
            break  # Surfaces were generated successfully; exit loop.

        # Removing slabs that contain too many atoms or are too thick.
        # FUTURE-NOTE: could consider a more refined approach in the future.
        slabs = [slab for slab in slabs if len(slab) <= max_atoms]
        slabs = [slab for slab in slabs if 
                    compute_slab_thickness(slab)
                    <= max_slab_thickness]
        
        # Select slab with closest O stoichiometry to bulk, if enabled:
        if select_closest_O_stoichiometry:
            # 1. Find minimum O-stoichiometry difference:
            min_O_stoichiometry_diff = float('inf')
            for slab in slabs:
                dict_1 = get_stoichiometry_dict(self.bulk_structure)
                dict_2 = get_stoichiometry_dict(slab)
                diff_dict = get_stoichiometry_diff(dict_1, dict_2)
                O_diff = abs(diff_dict.get('O', 0.0))
                if O_diff < min_O_stoichiometry_diff:
                    min_O_stoichiometry_diff = O_diff

            # 2. Select slabs with closest O-stoichiometry difference:
            selected_slabs = []
            for slab in slabs:
                dict_1 = get_stoichiometry_dict(self.bulk_structure)
                dict_2 = get_stoichiometry_dict(slab)
                diff_dict = get_stoichiometry_diff(dict_1, dict_2)
                O_diff = abs(diff_dict.get('O', 0.0))
                if np.abs(O_diff) \
                    < 0.01/len(slab) + min_O_stoichiometry_diff:
                    selected_slabs.append(slab)

            slabs = selected_slabs              

        return slabs        

    def identify_distinct_surfaces(self, max_miller = 1):
        self.max_miller = max_miller
        distinct_millers = get_symmetrically_distinct_miller_indices(
            self.bulk_structure, max_miller)
        self.distinct_millers = distinct_millers
        if self.termination_ranking_df is None:
            self.termination_ranking_df = pd.DataFrame(columns=[
                "Miller indices", "Termination index", 'Score', 
                'Normalized score', "Surface atoms:"])
        
    def predict_most_likely_surfaces(self, 
        max_miller: int = 1,
        method: Literal['all', 'disregard-coord-of-O',
                         'O-coordination'] = 'all',
        stoichiometry_tolerance: float = 0.2,
        max_diff_dict: dict | None = None,
        ignore_stoichiometry_of_species: list[str] | None = None,
        all_species_present: bool = True,
        symmetrize: bool = True,
        select_closest_O_stoichiometry: bool = True,
        min_slab_thickness: float = 9,
        max_slab_thickness: float = 18,
        min_atoms: int = 20, 
        max_atoms: int = 120,
        vacuum: float = 1,
        max_layers: int = 6,
        timeout: int = 5*60,
        ) -> None:

        if self.bulk_atoms is None or self.bulk_structure is None:
            raise RuntimeError(
                "Cannot rank terminations because the relaxed bulk "
                "structure is not set. Use "
                "`store_mag_bulk_candidate` + `finalize_mag_bulk_batch` "
                "to set the relaxed bulk through the magnetic workflow."
            )

        if max_diff_dict is None:
            max_diff_dict = {}
        if ignore_stoichiometry_of_species is None:
            ignore_stoichiometry_of_species = []
        
        # Build method dict for later reference:
        new_ranking_method = {
            'method': method,
            'stoichiometry_tolerance': stoichiometry_tolerance,
            'max_diff_dict': max_diff_dict,
            'ignore_stoichiometry_of_species': 
                ignore_stoichiometry_of_species,
            'all_species_present': all_species_present,
            'symmetrize': symmetrize,
            'select_closest_O_stoichiometry': 
                select_closest_O_stoichiometry,
            'min_slab_thickness': min_slab_thickness,
            'max_slab_thickness': max_slab_thickness,
            'min_atoms': min_atoms,
            'max_atoms': max_atoms,
            'vacuum': vacuum,
            'max_layers': max_layers,
        }

        if self.applied_ranking_method is not None: # <<< -------------- CONSIDER BUILDING CODE ALLOWING FOR THIS
            raise RuntimeError("Surfaces have already been ranked. " \
                "Re-ranking is not supported at this time.")

        if max_miller < 1:
            raise ValueError("max_miller must be at least 1.")
        
        # <<< ---------------------------------------------------------- THIS CHECK DOES NOT WORK AS INTENDED
        # if self.max_miller >= max_miller :
        #     raise ValueError("ERROR: Surfaces have "
        #         f"already been identified up to max_miller "
        #         f"{self.max_miller}, which is >= the requested "
        #         f"max_miller {max_miller}. Re-ranking is not supported."
        #     )
        
        if self.max_miller is not None \
            and self.max_miller != max_miller: # <<< ------------------- CONSIDER BUILDING CODE ALLOWING FOR THIS
            raise RuntimeError("Current implementation does not support "
                    "re-ranking with a different max_miller value.") 

        # Ensure bulk coordination numbers are set:
        self._set_bulk_coordination(method)
        
        # Ensure distinct miller-indices are identified:
        if self.distinct_millers is None:
            self.identify_distinct_surfaces(max_miller = max_miller)

        termination_ranking_df = self.termination_ranking_df
        for miller_indices in self.distinct_millers:

            slabs = self._build_surfaces(miller_indices, 
                                         timeout = timeout,
                                         **new_ranking_method)
            if len(slabs) == 0:
                N_ts = len(termination_ranking_df)
                termination_ranking_df.loc[N_ts] = {
                    "Miller indices": miller_indices,
                    "Termination index": N_ts,
                    "Score": float('nan'),
                    "Normalized score": 0,
                    "Surface atoms:": []
                }
            else:                
                for slab in slabs:       
                    atoms = AseAtomsAdaptor.get_atoms(slab)

                    # Calculate the surface area:
                    slab_normal = slab.normal
                    cell = atoms.get_cell()
                    a_1 = cell[0]
                    a_2 = cell[1]
                    tol = 1e-5
                    if np.abs(np.dot(slab_normal, a_1)) > tol or \
                          np.abs(np.dot(slab_normal, a_2)) > tol:
                        raise RuntimeError("Failed to find vectors in plane " \
                                           "of the surface.")
                    A_surface = np.linalg.norm(np.cross(cell[0], 
                                                        cell[1]))
                    
                    # Determine indices of the top half of atoms in the 
                    # surface:
                    top_half_indices = \
                        get_top_half_atom_projections(
                            atoms, slab.normal)

                    # Get the corresponding bulk coordination numbers
                    # for the atoms in the top half of the slab (i.e.,
                    # what the coordination numbers would have been in
                    # the bulk):
                    corresponding_bulk_coords = np.array( \
                        slab.site_properties.get('bulk-custom-coord'))
                    corresponding_bulk_coords = \
                        corresponding_bulk_coords[top_half_indices]

                    # Get coordination numbers for top half of slab:
                    coords_numbers = np.array( \
                    determine_coordination_numbers(
                        atoms, method))[top_half_indices]
                
                    corrd_diff = np.sum(coords_numbers - 
                                        corresponding_bulk_coords)
                    
                    # Score, equal to coordination difference per area:
                    score = corrd_diff/A_surface

                    # Determine top-layer adsorption-site elements:
                    ad_sites_df, _ = determine_adsorption_sites(slab)
                    surface_atoms = [
                        element
                        for element in
                        ad_sites_df['ad site element'].tolist()
                        if element != 'N/A'
                    ]

                    N_ts = len(termination_ranking_df)
                    termination_ranking_df.loc[N_ts] = {
                        "Miller indices": miller_indices,
                        "Termination index": N_ts,
                        "Score": score,
                        "Normalized score": 0,
                        "Surface atoms:": surface_atoms
                    }
                    self.terminations[N_ts] = slab
    
        self.termination_ranking_df = termination_ranking_df
        self.applied_ranking_method = method

    def normalize_termination_scores(self) -> None:

        df = self.termination_ranking_df
        scores = df['Score']

        min_score = scores.min(skipna=True)
        max_score = scores.max(skipna=True)

        if pd.isna(min_score) or pd.isna(max_score):
            df['Normalized score'] = float('nan')
        elif np.isclose(max_score, min_score):
            # all non-NaN values are identical
            df['Normalized score'] = 1.0
        else:
            df['Normalized score'] = np.round(
                (scores - min_score) / (max_score - min_score), 6)
            
        for i, row in df.iterrows():
            if pd.isna(row['Score']):
                df.at[i, 'Normalized score'] = 0.0

        self.termination_ranking_df = df

    def get_termination_rankings(self, normalize_scores = True
                                     ) -> pd.DataFrame | None:
        out = None
        if self.termination_ranking_df is not None:
            if normalize_scores:
                self.normalize_termination_scores()
            out = self.termination_ranking_df.copy()
        return out
    
    def _get_init_adsorption_sites(self, 
            termination_index: int,
            top_layer_tol: float = 1.5,
            only_reduced_coord_O_sites: bool = False
            ) -> tuple[pd.DataFrame, Atoms]:
        
        if termination_index not in self.terminations:
            raise ValueError(
                "Invalid termination_index: "
                f"{termination_index}. No generated termination exists "
                "for this index."
            )
        
        return determine_adsorption_sites(
            self.terminations[termination_index],
            top_layer_tol=top_layer_tol,
            only_reduced_coord_O_sites = only_reduced_coord_O_sites
        )
    
    def get_init_adsorption_sites_atoms(self, 
            termination_index: int,
            top_layer_tol: float = 1.5,
            only_reduced_coord_O_sites: bool = False
            ) -> Atoms:
        _, atoms = self._get_init_adsorption_sites(
            termination_index=termination_index,
            top_layer_tol=top_layer_tol,
            only_reduced_coord_O_sites = only_reduced_coord_O_sites
        )
        return atoms

    def get_init_adsorption_sites_df(self, 
            termination_index: int,
            top_layer_tol: float = 1.5,
            only_reduced_coord_O_sites: bool = False
            ) -> pd.DataFrame:
        df, _ = self._get_init_adsorption_sites(
            termination_index=termination_index,
            top_layer_tol=top_layer_tol,
            only_reduced_coord_O_sites = only_reduced_coord_O_sites
        )
        return df
    
    def initialize_oer_surface_study(self, 
            termination_index: int,
            # miller_indices: tuple[int, int, int] | None = None, 
            fixed_bottom_half: bool = True, 
            vacuum = 12,
            force = False) -> None:
        
        if termination_index in self.oer_surface_studies and not force:
            raise ValueError("An OER_surface_study has already been "\
                "initialized for the provided termination_index. "\
                "Use force=True to overwrite the existing study."
                "!!! DOING THIS IS HIGHLY DISCOURAGED !!!")

        if termination_index not in self.terminations:
            raise ValueError(
                "Invalid termination_index: "
                f"{termination_index}. No generated termination exists "
                "for this index."
            )

        slab = self.terminations[termination_index]
        oer_surface_study = OER_surface_study(self, slab,
                fixed_bottom_half = fixed_bottom_half, vacuum = vacuum)
        
        self.oer_surface_studies[termination_index] = oer_surface_study

    def get_surface_studies(self
            ) -> dict[int, OER_surface_study]:
        return self.oer_surface_studies
    
