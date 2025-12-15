import numpy as np
import pandas as pd

from typing import Literal
from collections import Counter
from ase import Atoms
from ase.visualize import view
from ase.neighborlist import natural_cutoffs #, NeighborList
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.surface import Slab
from pymatgen.core import Composition
from pymatgen.core.surface import (
    Slab, get_symmetrically_distinct_miller_indices, SlabGenerator)
from ase.neighborlist import NeighborList, natural_cutoffs

from GNoME_DREAMS_OER_screening.OER.OER_study_utils import get_atoms_and_normal_from_input
from GNoME_DREAMS_OER_screening.OER.OER_surface_study import OER_surface_study


class OER_catalyst_study:

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

        self.max_miller = None
        self.distinct_millers = None
        self.df_millers_ranking = None

        self.OER_surface_studies = {}

        self.species_energies = {
            'H2O_gas_free_energy': H2O_gas_free_energy,
            'H2_gas_free_energy': H2_gas_free_energy,
            'O2_gas_free_energy': O2_gas_free_energy,
            'O_ad_correction': default_O_ad_correction,
            'OH_ad_correction': default_OH_ad_correction,
            'OOH_ad_correction': default_OOH_ad_correction,
        }

        # --- symmetrize slabs when generating? ---
        # Currently always unknown if True or False is better. False 
        # is faster and more robust, but True might give better slabs...
        self.symmetrize_slabs = False 

    def set_gas_energies(self,
            H2O_gas_free_energy: float = None,
            H2_gas_free_energy: float = None,
            O2_gas_free_energy: float = None
            ) -> None:
        if H2O_gas_free_energy is not None:
            self.species_energies['H2O_gas_free_energy'] = H2O_gas_free_energy
        if H2_gas_free_energy is not None:
            self.species_energies['H2_gas_free_energy'] = H2_gas_free_energy
        if O2_gas_free_energy is not None:
            self.species_energies['O2_gas_free_energy'] = O2_gas_free_energy
    
    def set_default_adsorption_corrections(self,
            O_ad_correction: float = None,
            OH_ad_correction: float = None,
            OOH_ad_correction: float = None,
            ) -> None:
        if O_ad_correction is not None:
            self.species_energies['O_ad_correction'] = O_ad_correction
        if OH_ad_correction is not None:
            self.species_energies['OH_ad_correction'] = OH_ad_correction
        if OOH_ad_correction is not None:
            self.species_energies['OOH_ad_correction'] = OOH_ad_correction

    def get_gas_and_energy_corrections(self) -> dict[str, float]:
        return self.species_energies.copy()

    def _determine_coordination_numbers(self, atoms: Atoms,
                method: Literal['coordination', 'non-O-coordination']
                = 'coordination') -> list[int]:
        
        # Build neighbor list
        cutoffs = natural_cutoffs(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, 
                          bothways=True)
        nl.update(atoms)
        coord_nums = []
        for i, atom in enumerate(atoms):
            if method == 'non-O-coordination' and atom.symbol == 'O':
                continue
            indices, offsets = nl.get_neighbors(i)
            cn = len(indices)
            coord_nums.append(cn)
                
        return coord_nums
    
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
        distinct_millers = get_symmetrically_distinct_miller_indices(self.bulk_structure, max_miller)
        df = pd.DataFrame()
        df['miller_indices'] = distinct_millers
        self.distinct_millers = distinct_millers
        self.df_millers_ranking = df

    def add_scores_to_df(self):
        df = self.df_millers_ranking
        # Adding raw scores to df:
        for miller, score in self.scores.items():
            df.loc[
                df['miller_indices'] == miller,
                'Score'
            ] = score
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        # adding normalized scores to df:
        try:
            scores_values = df['Score'].values
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
        min_slab_size = 4, max_slab_size = 10
        ):
        
        if self.distinct_millers is None:
            raise ValueError("You must first identify surfaces by" \
                             " calling 'identify_distinct_surfaces'" \
                             " method.")

        slabs_for_millers = {}
        for miller_indices in self.distinct_millers:
            slab_size = min_slab_size
            while True:
                sg = SlabGenerator(
                    initial_structure = self.bulk_structure,
                    miller_index = miller_indices,
                    min_slab_size = slab_size,
                    min_vacuum_size = 2,
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
            
        self.slabs_for_millers = slabs_for_millers
        slabs_as_list_of_atoms = {
            miller: [AseAtomsAdaptor.get_atoms(slab) 
                     for slab in slabs]
            for miller, slabs in slabs_for_millers.items()
        }
        self.slabs_as_list_of_atoms = slabs_as_list_of_atoms

        bulk_coordination = self._determine_coordination_numbers(
            self.bulk_atoms, method
        )
        mean_bulk_coordination = np.mean(bulk_coordination)

        scores = {}
        for miller, atoms_list in slabs_as_list_of_atoms.items():
            score = None # Default score if no slabs pass the filters
            if len(atoms_list) != 0:
                # Computing the surface area of the slab cell:
                cell = atoms_list[0].get_cell()
                A_surface = np.linalg.norm(np.cross(cell[0], cell[1]))

                for atoms in atoms_list:       
                    if stoichiometry == 'stoichiometric':
                        if not self._compare_to_bulk_stoichiometry(atoms):
                            continue
                    elif stoichiometry == 'O-stoichiometric':
                        if not self._compare_to_bulk_stoichiometry(
                            atoms,
                            compare_O_only_stoichiometry = True
                            ):
                            continue
                        
                    coordination_numbers = self._determine_coordination_numbers(
                        atoms, method
                        )
                    
                    N_surface = len(coordination_numbers)
                    mean_slab_coordination = np.mean(coordination_numbers)

                    K = N_surface / A_surface / 2  # dividing by 2 to get per surface side
                    coordination_diff = K * (mean_slab_coordination 
                                            - mean_bulk_coordination)
                    
                    if (score is None) or (coordination_diff > score):
                        score = coordination_diff
                
            scores[miller] = score

        self.scores = scores
        self.add_scores_to_df()

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

    def compute_slab_thickness(self, input: Atoms | Slab, 
                               normal_vector: np.ndarray | None = None
                               ) -> float:
        
        atoms, normal_vector = get_atoms_and_normal_from_input(input,
                                                         normal_vector)
        
        pos = atoms.get_positions()
        projections = np.dot(pos, normal_vector)
        slab_thickness = projections.max() - projections.min()
        return slab_thickness


