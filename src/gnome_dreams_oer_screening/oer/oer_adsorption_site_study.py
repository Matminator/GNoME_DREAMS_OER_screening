import site

import numpy as np
import pandas as pd
from ase import Atoms, atoms
from ase.geometry import find_mic

from gnome_dreams_oer_screening.oer.atoms_sanity_checks import (
    are_all_atoms_connected,
    are_systems_equal,
    StructureConsistencyError,
    )

    
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
                 surface_normal: np.ndarray,
                 init: tuple[str, float]
                 ):
        
        init_surface = init_surface.copy()

        self.init_surface = init_surface
        self.init_add_position = init_add_position
        self.site_index = site_index
        self.site_type = init[0]
        self.surface_normal = surface_normal

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
        self.OH_init_atoms = None
        self.relaxed_OH_atoms = None
        self.accepted_relaxed_OH_pairs = []
        self.study_lock = False
        self.lock_reason = None
        
    def get_atoms_for_O_calc(self) -> Atoms:
        return self.atoms_for_O_calc.copy()

    def _get_adsorbed_O_index(self) -> int:
        if self.site_type == "on-top":
            atoms = self.relaxed_O_atoms # O added for the O relaxation
            target_tag = 4
        elif self.site_type == "lattice O":
            atoms = self.init_surface # O already present in the init
            target_tag = 3
        else:
            raise ValueError(f"Unsupported adsorption site type: "
                                                    f"{self.site_type}")

        O_indices = [
            i
            for i, atom in enumerate(atoms)
            if atom.symbol == "O" and atom.tag == target_tag
        ]
        if len(O_indices) != 1:
            raise ValueError(
                "Could not identify a unique adsorbed O atom "
                f"with tag={target_tag}."
            )

        return int(O_indices[0])

    def is_adsorbate_near_site(
        self,
        atoms: Atoms,
        max_site_drift: float = 1.5,
    ) -> bool:
        """
        Test if the adsorbed O atom has drifted too far from the initial
        adsorption position for the site study.
        """

        if max_site_drift <= 0:
            raise ValueError("max_site_drift must be positive.")

        O_index = self._get_adsorbed_O_index()
        displacement = atoms[O_index].position - np.asarray(
            self.init_add_position, dtype=float 
        )

        displacement, _ = find_mic(
            displacement[None, :], cell=atoms.cell, pbc=atoms.pbc
        )
        displacement = displacement[0]

        drift_distance = float(np.linalg.norm(displacement))
        return drift_distance <= max_site_drift

    def is_OH_adsorbate_intact(
        self,
        atoms: Atoms,
        max_OH_distance: float = 1.5,
    ) -> bool:
        """
        Test if O and H in an OH adsorbate remain bonded.
        """

        O_index = self._get_adsorbed_O_index()
        H_indices = [
            i
            for i, atom in enumerate(atoms)
            if atom.symbol == "H" and atom.tag == 4
        ]
        if len(H_indices) != 1:
            raise ValueError(
                "Could not identify a unique OH adsorbate H atom with" \
                " tag=4."
            )
        H_index = int(H_indices[0])

        d_OH = float(atoms.get_distance(O_index, H_index, mic=True))
        return d_OH <= max_OH_distance
    
    def set_relaxed_O_calc(self, atoms: Atoms,
                                energy: float | None = None) -> None:
        
        # Test provided relaxed_bulk structure matches the initial bulk:
        equal_results = are_systems_equal(atoms,
                                          self.get_atoms_for_O_calc())
        if not equal_results[0]:
            raise StructureConsistencyError(
                "The provided relaxed_bulk structure does not match "
                "the initial bulk structure. Details: "
                f"{equal_results[1]}"
            )
      
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
            raise ValueError("Invalid adsorption site type")       
            
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
    
    def get_OH_adsorbed_atoms(self) -> list[Atoms]:
        """
        Build three initial OH-adsorbed structures from the relaxed
        O-adsorbed structure by placing H near the adsorbed O atom.

        Returns:
            list[Atoms]: Three candidate OH initialization structures.
        """

        if self.OH_init_atoms is not None:
            return self.OH_init_atoms.copy()

        if self.relaxed_O_atoms is None and self.site_type == 'on-top':
            raise RuntimeError("Relaxed O-adsorbed structure has not" \
                " been set yet. For the 'on-top' site type, this is " \
                "required for OH initializations.")

        if self.site_type == 'on-top':
            # Use the relaxed O-adsorbed structure, as start for OH:
            target_tag = 4
            atoms_with_O = self.relaxed_O_atoms.copy()
        elif self.site_type == 'lattice O':
            # Use the lattice O, as start for OH:
            target_tag = 3
            atoms_with_O = self.init_surface.copy()
        else:
            raise ValueError("Invalid adsorption site type:", 
                             self.site_type)

        ads_O_indices = [
            i for i, atom in enumerate(atoms_with_O)
            if atom.symbol == 'O' and atom.tag == target_tag
        ]
        if len(ads_O_indices) != 1:
            raise RuntimeError("Could not identify a unique adsorbed" \
                f" O atom with tag={target_tag} for site type " \
                f"'{self.site_type}'.")
        
        ads_O_index = ads_O_indices[0]

        ads_O_pos = atoms_with_O[ads_O_index].position.copy()

        # Place H relative to the provided surface normal.
        base_direction = np.array(self.surface_normal, dtype=float)
        norm = np.linalg.norm(base_direction)
        if norm < 1e-8:
            raise RuntimeError("Surface normal has near-zero norm and" \
                " cannot be used for OH initialization.")
        base_direction = base_direction / norm

        # Build orthonormal basis around the O-surface direction.
        ref = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(base_direction, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])

        perp_1 = np.cross(base_direction, ref)
        perp_1 = perp_1 / np.linalg.norm(perp_1)
        perp_2 = np.cross(base_direction, perp_1)
        perp_2 = perp_2 / np.linalg.norm(perp_2)

        oh_bond_length = 0.98
        tilt_angle = np.deg2rad(35.0)
        phis = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]

        OH_init_atoms= []
        for phi in phis:
            lateral = np.cos(phi) * perp_1 + np.sin(phi) * perp_2
            H_direction = np.cos(tilt_angle) * base_direction + \
                np.sin(tilt_angle) * lateral
            H_direction = H_direction / np.linalg.norm(H_direction)
            H_pos = ads_O_pos + oh_bond_length * H_direction

            oh_atoms = atoms_with_O.copy()
            oh_atoms += Atoms('H', positions=[H_pos], tags=[4])
            OH_init_atoms.append(oh_atoms)

        self.OH_init_atoms = OH_init_atoms

        return OH_init_atoms

    def set_relaxed_OH_calc(
        self, 
        atoms_list: Atoms | list[Atoms],
        energies: list[float] | None = None
        ) -> bool:
        """
        Validate and store relaxed OH structures for this site study.

        Accepted relaxed structures are accumulated across calls. A
        structure is accepted only if it is system-consistent with the
        initialized OH setup, fully connected, still near the target
        adsorption site, and has an intact OH adsorbate.

        The method updates `self.relaxed_OH_atoms` and
        `self.energies_dict['OH_adsorbed_energy']` to the lowest-energy
        accepted structure seen so far (including previous calls).

        Args:
            atoms_list: Atoms | list[Atoms]
                One or more relaxed OH structures to evaluate.
            energies: list[float] | None
                Energies matching `atoms_list` order. If None, energies
                are read from each Atoms object.

        Returns:
            bool
                True if at least one structure from this call was
                accepted and stored; otherwise False.

        Raises:
            RuntimeError:
                If required prior setup (e.g. on-top relaxed O or OH
                initialization structures) is missing.
            ValueError:
                If inputs are malformed or energies cannot be obtained.
            StructureConsistencyError:
                If any provided structure does not match the expected
                OH system composition/order/tags.
        """

        if self.relaxed_O_atoms is None and self.site_type == 'on-top':
            raise RuntimeError(
                "Relaxed O-adsorbed structure has not been set yet. "
                "This DFT calculation must be run and set before "
                "setting relaxed OH structures for an \"on-top\" site."
            )

        # Ensure correct format of atoms_list:
        if isinstance(atoms_list, Atoms):
            atoms_list = [atoms_list.copy()]
        else:
            atoms_list = [atoms.copy() for atoms in atoms_list]

        # --- Basic checks on the provided structures and energies: ----
        if len(atoms_list) == 0:
            raise ValueError(
                "atoms_list must contain at least one structure."
            )

        if self.OH_init_atoms is None:
            raise RuntimeError(
                "Initial OH structures have not been set yet."
            )

        if energies is not None:
            if len(energies) != len(atoms_list):
                raise ValueError(
                    "If provided, energies must have the same length " \
                    "as atoms_list."
                )
            structure_energy_pairs = [
                (atoms, float(E))
                for atoms, E in zip(atoms_list, energies)
            ]
        else:
            structure_energy_pairs = []
            for atoms in atoms_list:
                try:
                    E = float(atoms.get_potential_energy())
                except:
                    raise ValueError(
                        "Energy not provided and could not be "
                        "obtained from the relaxed atoms object."
                    )
                structure_energy_pairs.append((atoms, E))

        # --- run baseline sanity checks on atoms, in atoms_list: ------
        for atoms, _ in structure_energy_pairs:
            equal_res = are_systems_equal(atoms, self.OH_init_atoms[0])
            if not equal_res[0]:
                raise StructureConsistencyError("The provided " \
                    "relaxed_OH structure is not a relaxed version of" \
                    " the initial OH structure. Details: "
                    + equal_res[1])
            
        # --- Only keep accepted OH structures for this site study: ----
        accepted_pairs = []
        for atoms, E in structure_energy_pairs:
            if not are_all_atoms_connected(atoms):
                continue
            if not self.is_adsorbate_near_site(atoms):
                continue
            if not self.is_OH_adsorbate_intact(atoms):
                continue
            accepted_pairs.append((atoms.copy(), E))

        if len(accepted_pairs) == 0:
            return False

        self.accepted_relaxed_OH_pairs.extend(accepted_pairs)
        best_atoms, best_energy = min(
            self.accepted_relaxed_OH_pairs,
            key=lambda pair: pair[1],
        )
        self.relaxed_OH_atoms = best_atoms.copy()
        self.energies_dict['OH_adsorbed_energy'] = best_energy

        return True
        
