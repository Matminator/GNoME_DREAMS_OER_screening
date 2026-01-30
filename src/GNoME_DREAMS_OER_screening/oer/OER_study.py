import sys
sys.path.append("/nfs/turbo/coe-venkvis/ziqiw-turbo/material_agent/GNoME_DREAMS_OER_screening/src/GNoME_DREAMS_OER_screening")

import numpy as np
import pandas as pd

from typing import Literal
from collections import Counter
from ase import Atoms
from ase.neighborlist import natural_cutoffs
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab, Structure
from pymatgen.core import Composition
from pymatgen.core.surface import (
    Slab, get_symmetrically_distinct_miller_indices, SlabGenerator)
from ase.neighborlist import NeighborList, natural_cutoffs

from gnome_dreams_oer_screening.oer.oer_study_utils import (
    get_atoms_and_normal_from_input, eval_stoichiometry_diff
    )
from gnome_dreams_oer_screening.oer.oer_surface_study import (
    OER_surface_study
)


class OER_catalyst_study:
    """
    The core class for managing for ranking and generating surfaces from
    a bulk structure, for the purpose of OER catalyst screening. All 
    data related to computation of adsorption energies is stored in this
    class. Surfaces are initialized as OER_surface_study instances, via 
    the method 'initialize_OER_surface_studies', and stored in a 
    dictionary, with miller indices as keys. Surface structures with 
    adsorbates are managed in the OER_surface_study instances. Gas phase
    free energies of H2O, H2, O2 are stored in this class, in addition 
    to initial corrections (correction = ZPE + Cp - TS) to be applied to
    adsorption energies.

    Args:
        bulk (Atoms): Bulk structure of the catalyst material.
        H2O_gas_free_energy (float): Free energy of H2O in gas phase.
        H2_gas_free_energy (float): Free energy of H2 in gas phase.
        O2_gas_free_energy (float, optional): Free energy of O2 in gas
        phase.
        default_O_ad_correction (float, optional): Default correction
        made to the DFT O* adsorption energy.
        default_OH_ad_correction (float, optional): Default correction
        made to the DFT OH* adsorption energy.
        default_OOH_ad_correction (float, optional): Default correction
        made to the DFT OOH* adsorption energy.
        bulk_energy (float, optional): Energy of the bulk structure.
    """


    def __init__(self, bulk = Atoms,    
                 H2O_gas_free_energy = float,
                 H2_gas_free_energy = float,
                 O2_gas_free_energy: float = None,
                 default_O_ad_correction: float = 0.070,
                 default_OH_ad_correction: float = 0.361,
                 default_OOH_ad_correction: float = 0.431,
                 bulk_energy: float = None,
                 ):
        
        self.bulk_structure = AseAtomsAdaptor.get_structure(bulk)
        self.bulk_atoms = bulk
        if bulk_energy is None:
            try:
                self.bulk_energy = bulk.get_potential_energy()
            except:
                self.bulk_energy = None
        else:
            self.bulk_energy = bulk_energy

        self.species_energies = {
            'H2O_gas_free_energy': H2O_gas_free_energy,
            'H2_gas_free_energy': H2_gas_free_energy,
            'O2_gas_free_energy': O2_gas_free_energy,
            'O_ad_correction': default_O_ad_correction,
            'OH_ad_correction': default_OH_ad_correction,
            'OOH_ad_correction': default_OOH_ad_correction,
        }

        self.max_miller = None
        self.distinct_millers = None
        self.df_millers_ranking = None
        self.OER_surface_studies = {}

        # --- symmetrize slabs when generating? ---
        # Currently always unknown if True or False is better. False 
        # is faster and more robust, but True might give better slabs...
        self.symmetrize_slabs = False 

    def set_energies(self,
            H2O_gas_free_energy: float = None,
            H2_gas_free_energy: float = None,
            O2_gas_free_energy: float = None,
            O_ad_correction: float = None,
            OH_ad_correction: float = None,
            OOH_ad_correction: float = None,
            ) -> None:
        if H2O_gas_free_energy is not None:
            self.species_energies['H2O_gas_free_energy'] = \ 
            H2O_gas_free_energy
        if H2_gas_free_energy is not None:
            self.species_energies['H2_gas_free_energy'] = \
                H2_gas_free_energy
        if O2_gas_free_energy is not None:
            self.species_energies['O2_gas_free_energy'] = \
                O2_gas_free_energy
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

    def get_bulk_atoms(self) -> Atoms:
        return self.bulk_atoms.copy()
    
    def get_bulk_structure(self) -> Structure:
        return self.bulk_structure.copy()

    def _determine_coordination_numbers(self, atoms: Atoms,
                method: Literal['all', 'non-O-coordination', 
                'O-coordination']) -> list[int]:
        """
        Determine coordination numbers for all atoms in the given ASE
        Atoms object, using the specified method. The method "all" gives
        gives the corrdination numbers for all atoms given any neighbor,
        "non-O-coordination" gives the coordination numbers for all 
        atoms excluding oxygen (sets oxygen coordination numbers to 
        zero). "O-coordination" gives the coordination numbers for all
        atoms considering only oxygen neighbors (also sets oxygen 
        coordination numbers to zero).
        Args:
            atoms (Atoms): ASE Atoms object representing the structure.
            method (str): Method to compute coordination numbers.
                          Currently only 'coordination' is implemented.
        Returns:
            list[int]: List of coordination numbers for each atom.
        """
        
        atoms_copy = atoms.copy()
        atoms_copy.set_pbc(True)

        # Build neighbor list
        cutoffs = natural_cutoffs(atoms_copy)
        nl = NeighborList(cutoffs, self_interaction=False, 
                          bothways=True, skin = 0.3)
        nl.update(atoms_copy)
        coord_nums = []
        for i, atom in enumerate(atoms_copy):
            # if method == 'non-O-coordination' and atom.symbol == 'O':
            #     continue
            indices, offsets = nl.get_neighbors(i)
            cn = len(indices)
            coord_nums.append(cn)
                
        return coord_nums
    
    def _set_bulk_coordination(self, method: str) -> None:
        """
        Compute and store coordination numbers for bulk atoms, applying
        the specified method. The method SHOULD be the same as that used
        for surface slabs, to ensure consistency!
        
        Args:
            method (str): Method to compute coordination numbers.
        """
        # Compute coordination numbers for bulk atoms, using the method
        coord_numbers = self._determine_coordination_numbers(
            self.bulk_atoms, method)
        # Store coordination numbers as a site property in the bulk 
        # structure:
        self.bulk_structure.add_site_property("bulk-custom-coord", 
                                              coord_numbers)

    def _compare_to_bulk_stoichiometry(self, atoms: Atoms,
                compare_O_only_stoichiometry: bool = False) -> bool:
        
        slab_symbols = [atom.symbol for atom in atoms]
        slab_counts = Counter(slab_symbols)
        slab_comp = Composition(dict(slab_counts))
        slab_reduced, slab_rfactor = slab_comp.get_reduced_composition_and_factor()

        bulk_symbols = [atom.symbol for atom in self.bulk_atoms]
        bulk_counts = Counter(bulk_symbols)
        bulk_comp = Composition(dict(bulk_counts))
        bulk_reduced, bulk_rfactor = bulk_comp.get_reduced_composition_and_factor()

        if compare_O_only_stoichiometry:
            slab_num_O = slab_reduced.get("O", 0)
            bulk_num_O = bulk_reduced.get("O", 0)
            result = slab_num_O == bulk_num_O
        else:
            result = slab_comp.reduced_formula == bulk_comp.reduced_formula

        return result

    def identify_distinct_surfaces(self, max_miller = 1):
        self.max_miller = max_miller
        distinct_millers = get_symmetrically_distinct_miller_indices(
            self.bulk_structure, max_miller)
        df = pd.DataFrame()
        df['miller_indices'] = distinct_millers
        self.distinct_millers = distinct_millers
        self.df_millers_ranking = df

    def add_scores_to_df(self, all_scores):
        df = self.df_millers_ranking
        df['Scores'] = None
        df['Scores'] = df['Scores'].astype(object)
        # Adding raw scores to df:
        max_scores = []

        for miller, scores in all_scores.items():
            # Find the matching index
            idx_list = df.index[df['miller_indices'] == miller].tolist()
            if not idx_list:
                continue
            idx = idx_list[0]

            # Store the array in the cell
            df.at[idx, 'Scores'] = scores           # treat scores as a single object
            df.at[idx, 'Max Score'] = np.max(scores) # numeric is fine
        
        df['Max Score'] = pd.to_numeric(df['Max Score'], 
                                        errors='coerce')
        # adding normalized scores to df:
        try:
            scores_values = df['Max Score'].values
            min_score = np.nanmin(scores_values)
            max_score = np.nanmax(scores_values)
            normalized_scores = (scores_values - min_score)
            normalized_scores /= (max_score - min_score)
            df['Normalized Score'] = normalized_scores
        except:
            df['Normalized Score'] = float('nan')
        self.df_millers_ranking = df

    def get_df_with_surface_rankings(self) -> pd.DataFrame:
        return self.df_millers_ranking.copy()
        
    def predict_most_likely_surfaces(self, 
        method: Literal['coordination', 'non-O-coordination'],
        stoichiometry: Literal['stoichiometric', 'O-stoichiometric',
                                 'any'],
        min_slab_size = 4, max_slab_size = 10, max_miller: int = 1
        ):

        # Ensure bulk coordination numbers are set, if needed by method:
        if 'coordination' in method:
            self._set_bulk_coordination(method)
        
        # If distinct millers have not been identified yet, do it now:
        if self.distinct_millers is None:
            self.identify_distinct_surfaces(max_miller = max_miller)
        elif self.max_miller != max_miller:
            self.identify_distinct_surfaces(max_miller = max_miller)
            print("OER-study-too-WARNING: max_miller changed from "
                  f"{self.max_miller} to {max_miller}. ")

        slabs_for_millers = {}
        dev_slabs_for_millers = {}
        for miller_indices in self.distinct_millers:
            slab_size = min_slab_size
            while True:
                sg = SlabGenerator(
                    initial_structure = self.bulk_structure,
                    miller_index = miller_indices,
                    min_slab_size = slab_size,
                    min_vacuum_size = 4,
                    lll_reduce = True,
                    in_unit_planes = True,
                    center_slab = True,
                    primitive = True,
                )
                slabs = sg.get_slabs(symmetrize = self.symmetrize_slabs)
                if len(slabs) > 0 or slab_size >= max_slab_size:
                    break
                else:
                    slab_size = max_slab_size
                
            slabs_for_millers[miller_indices] = slabs
            dev_slabs_for_millers[miller_indices] = (sg, slabs)

        # Saving, only for debugging purposes:
        # self.slabs_for_millers = slabs_for_millers 
        self.dev_slabs_for_millers = dev_slabs_for_millers
            
        # ---------------------------------------------------------------------------
        # --- CODE USED BY PREVIOUS IMPLEMENTATION, TO BE DELETED IN THE FUTURE <--------------------
        # self.slabs_for_millers = slabs_for_millers # ----------------- SEEMS TO BE UNUSED? REMOVE? <-------
        # slabs_as_list_of_atoms = {
        #     miller: [AseAtomsAdaptor.get_atoms(slab) for slab in slabs]
        #     for miller, slabs in slabs_for_millers.items()
        # }
        # self.slabs_as_list_of_atoms = slabs_as_list_of_atoms

        # bulk_coordination = self._determine_coordination_numbers(
        #     self.bulk_atoms, method
        # )
        # mean_bulk_coordination = np.mean(bulk_coordination)

        # scores = {}
        # for miller, atoms_list in slabs_as_list_of_atoms.items():
        #     score = None # Default score if no slabs pass the filters
        #     if len(atoms_list) != 0:
        #         # Computing the surface area of the slab cell:
        #         cell = atoms_list[0].get_cell()
        #         A_surface = np.linalg.norm(np.cross(cell[0], cell[1]))

        #         for atoms in atoms_list:       
        #             if stoichiometry == 'stoichiometric':
        #                 if not self._compare_to_bulk_stoichiometry(
        #                     atoms):
        #                     continue
        #             elif stoichiometry == 'O-stoichiometric':
        #                 if not self._compare_to_bulk_stoichiometry(
        #                     atoms,
        #                     compare_O_only_stoichiometry = True
        #                     ):
        #                     continue
                        
        #             coordination_numbers = \
        #             self._determine_coordination_numbers(atoms, method)
                    
        #             N_surface = len(coordination_numbers)
        #             mean_slab_coordination = np.mean(
        #                 coordination_numbers)

        #             K = N_surface / A_surface / 2  # dividing by 2 to get per surface side
        #             coordination_diff = K * (mean_slab_coordination 
        #                                     - mean_bulk_coordination)
                    
        #             if (score is None) or (coordination_diff > score):
        #                 score = coordination_diff
                
        #     scores[miller] = score
        # ---------------------------------------------------------------------------

        all_scores = {}
        for miller, slabs in slabs_for_millers.items():
            scores = None # Default score if no slabs pass the filters
            if len(slabs) != 0:
                
                for slab in slabs:       
                    atoms = AseAtomsAdaptor.get_atoms(slab)

                    if stoichiometry == 'stoichiometric':
                        if not self._compare_to_bulk_stoichiometry(
                            atoms):
                            continue
                    elif stoichiometry == 'O-stoichiometric':
                        if not self._compare_to_bulk_stoichiometry(
                            atoms,
                            compare_O_only_stoichiometry = True
                            ):
                            continue
                    cell = atoms.get_cell()
                    A_surface = np.linalg.norm(np.cross(cell[0], 
                                                        cell[1]))
                    
                    # Determine indices of the top half of atoms in the 
                    # surface:
                    top_half_indices = \
                        self.get_top_half_atom_projections(
                            atoms, slab.normal)

                    corresponding_bulk_coords = np.array( \
                        slab.site_properties.get('bulk-custom-coord'))
                    corresponding_bulk_coords = \
                        corresponding_bulk_coords[top_half_indices]
                    
                    # Consider positive values. Negative values are set
                    # indicate that the site (atom) should be ignored:
                    N_non_zero = np.count_nonzero(
                        corresponding_bulk_coords > -0.5)
                    # SHOULD THIS BE USED (N_non_zero) ?? <<<--------------------------------------------------------------------------
                    
                    # Get coordination numbers for top half of slab:
                    coordination_numbers = np.array( \
                    self._determine_coordination_numbers(
                        atoms, method))[top_half_indices]
                
                    corrd_diff = np.sum(coordination_numbers - 
                                        corresponding_bulk_coords)
                    
                    # Norm the corrd difference:
                    corrd_diff = corrd_diff/A_surface
                    
                    if scores is None:
                        scores = []
                    scores.append(float(corrd_diff))

                all_scores[miller] = scores

        self.add_scores_to_df(all_scores)


# CURRENT WORK IN PROGRESS BELOW <-----------------------------------------------------------------------------
    def build_surfaces(self, 
            miller_indices: tuple[int, int, int],
            stoichiometric_variation: float = 0.1,
            min_slab_thickness: float = 8,
            max_slab_thickness: float = 14,
            min_atoms = 20, 
            max_atoms: int = 100,
            fixed_bottom_half: bool = True,
            vacuum: float = 12,
            force = False
            ) -> Atoms | None:
        """
        TEMP DOC
        """
        
        if miller_indices in self.OER_surface_studies and not force:
            raise ValueError("OER surface studies for the given "
        "miller indices already exist. Use force=True to overwrite.")

        accepted_slabs = []
        normal_vectors = []
        stop_flag = False
        slab_thickness = min_slab_thickness
        while True:
            sg = SlabGenerator(
            initial_structure = self.bulk_structure,
            miller_index = miller_indices,
            min_slab_size = slab_thickness,
            min_vacuum_size = 2,
            lll_reduce = True,
            in_unit_planes = False,
            center_slab = True,
            primitive = True,
            )
            slabs = sg.get_slabs(symmetrize = self.symmetrize_slabs)

        
        return None
# -------------------------------------------------------------------------------------------------------------------


    def initialize_OER_surface_studies(self, miller_indices: 
            tuple[int, int, int], 
            stoichiometric = False,
            min_slab_thickness: float = 8,
            max_slab_thickness: float = 14,
            min_atoms = 20, max_atoms: int = 100,
            fixed_bottom_half: bool = True, vacuum = 8,
            force = False) -> None:
        
        if miller_indices in self.OER_surface_studies and not force:
            raise ValueError("OER surface studies for the given "
        "miller indices already exist. Use force=True to overwrite.")
        
        accepted_slabs = []
        normal_vectors = []
        stop_flag = False
        slab_thickness = min_slab_thickness
        while True:
            sg = SlabGenerator(
            initial_structure = self.bulk_structure,
            miller_index = miller_indices,
            min_slab_size = slab_thickness,
            min_vacuum_size = 2,
            lll_reduce = True,
            in_unit_planes = False,
            center_slab = True,
            primitive = True,
            )
            slabs = sg.get_slabs(symmetrize = self.symmetrize_slabs)
            
            if len(slabs) > 0:
                
                for slab in slabs:
                    normal = slab.normal
                    atoms = AseAtomsAdaptor.get_atoms(slab)
                    if len(atoms) < min_atoms:
                        continue
                    if len(atoms) > max_atoms:
                        stop_flag = True
                        continue
                    if stoichiometric:
                        if not self._compare_to_bulk_stoichiometry(
                                                                atoms):
                            continue

                    compute_slab_thickness = self.compute_slab_thickness(
                        slab)
                    if slab_thickness <= min_slab_thickness:
                        continue
                    if slab_thickness > max_slab_thickness:
                        stop_flag = True
                        continue

                    accepted_slabs.append(atoms)
                    normal_vectors.append(normal)
                
                if len(accepted_slabs) > 0:
                    stop_flag = True

            elif slab_thickness >= max_slab_thickness:
                return None
            
            if len(accepted_slabs) == 0:
                slab_thickness += 1

            if stop_flag:
                break

        oer_surface_studies = []
        for atoms, normal in zip(accepted_slabs, normal_vectors):
            oer_surface_study = OER_surface_study(self, atoms, normal, 
                                                  fixed_bottom_half, 
                                                  vacuum)
            oer_surface_studies.append(oer_surface_study)

        oer_surface_studies = tuple(oer_surface_studies)
        self.OER_surface_studies[miller_indices] = oer_surface_studies

    def get_surface_studies(self
            ) -> dict[tuple[int, int, int], tuple[OER_surface_study]]:
        return self.OER_surface_studies

    def get_atom_projections_on_normal(self, input: Atoms | Slab,
                               normal_vector: np.ndarray | None = None
                               ) -> float:
        
        atoms, normal_vector = get_atoms_and_normal_from_input(input,
                                                         normal_vector)
        
        pos = atoms.get_positions()
        return np.dot(pos, normal_vector)

    def compute_slab_thickness(self, input: Atoms | Slab, 
                               normal_vector: np.ndarray | None = None
                               ) -> float:
        projections = self.get_atom_projections_on_normal(
            input, normal_vector)
    
        slab_thickness = projections.max() - projections.min()
        return slab_thickness
    

    def get_top_half_atom_projections(self, input: Atoms | Slab,
            normal_vector: np.ndarray | None = None,
            ) -> np.ndarray:
        
        projections = self.get_atom_projections_on_normal(
            input, normal_vector)
        
        median_projection = np.median(projections)
        indices = np.where(projections >= \
                            median_projection)[0].astype(int)
        
        return indices