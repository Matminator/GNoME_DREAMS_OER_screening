
import os
from pathlib import Path
from time import time
import numpy as np
from typing import Literal

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import Trajectory
from pymatgen.core.surface import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Vasprun, Incar as VaspIncar

from custodian.custodian import Custodian
from custodian.vasp.jobs import VaspJob
from custodian.vasp.handlers import (
    VaspErrorHandler,
    MeshSymmetryErrorHandler,
    UnconvergedErrorHandler,
    NonConvergingErrorHandler,
    PotimErrorHandler,
    PositiveEnergyErrorHandler,
    FrozenJobErrorHandler,
    WalltimeHandler,
)

from gnome_dreams_oer_screening.vasp.pre_defined_vasp_sets import (
    RPBE_relax_bulk_set, RPBE_relax_surface_set, TestRelaxSet)

BULK_RELAXATION_HANDLERS = [
    VaspErrorHandler(),           # master: 40+ runtime errors
    MeshSymmetryErrorHandler(),   # k-mesh/Bravais lattice mismatch (bulk symmetry)
    UnconvergedErrorHandler(),    # overall convergence fallback; ALGO ladder
    NonConvergingErrorHandler(),  # persistent NELM failures; ALGO + mixing fix
    PotimErrorHandler(),          # large ionic steps → reduce POTIM or switch IBRION
    PositiveEnergyErrorHandler(), # pathological positive energy → ALGO fix
    FrozenJobErrorHandler(),      # stalled HPC job → ALGO or SYMPREC change
    WalltimeHandler(),            # graceful HPC wall-time shutdown via STOPCAR
]

SURFACE_RELAXATION_HANDLERS = [
    VaspErrorHandler(),           # master: 40+ runtime errors
    UnconvergedErrorHandler(),    # overall convergence fallback; ALGO ladder
    NonConvergingErrorHandler(),  # persistent NELM failures; ALGO + mixing fix
    PotimErrorHandler(),          # large ionic steps → reduce POTIM or switch IBRION
    FrozenJobErrorHandler(),      # stalled HPC job → ALGO or SYMPREC change
    WalltimeHandler(),            # graceful HPC wall-time shutdown via STOPCAR
]


def run_vasp_via_custodian(
                directory,
                handlers=None,
                vasp_cmd=["mpiexec", "vasp_std"],
                auto_npar=True,
                max_errors=20,
                ):
    """
    Run a VASP calculation in a specified directory using Custodian.

    Parameters:
    directory (str): The directory where the VASP calculation will be 
    run.
    handlers (list): A list of Custodian error handlers to use.
    vasp_cmd (list): The command to run VASP.
    auto_npar (bool): Whether to automatically adjust NPAR.
    max_errors (int): Maximum number of errors to tolerate by Custodian
    before terminating the calculation.
    """

    if handlers is None:
        handlers = BULK_RELAXATION_HANDLERS

    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not" \
                                 " exist.")

    # save current working directory
    cwd = os.getcwd()

    # change to the VASP directory
    os.chdir(directory)

    # instantiate VASP job
    job = VaspJob(vasp_cmd=vasp_cmd, auto_npar=auto_npar)

    # run with custodian
    c = Custodian(handlers=handlers, jobs=[job],
                  max_errors=max_errors)
    c.run()

    # change back to the original directory
    os.chdir(cwd)


def run_surface_vasp_two_stage(
        directory,
        handlers=None,
        use_dipole_correction=True,
        vasp_cmd=["mpiexec", "vasp_std"],
        auto_npar=True,
        max_errors=20,
        ):
    """Two-stage surface VASP: SCF pre-convergence, then full ionic relaxation with dipole correction.

    Stage 1 — SCF only (NSW=0, no ionic steps), no dipole correction.
    Stabilises the wavefunction before introducing the dipole sawtooth
    potential, which can otherwise slow or destabilise SCF convergence
    (VASP recommendation; see vasp.at/tutorials/latest/surface/part2).

    Stage 2 — Full ionic relaxation with LDIPOL=True, IDIPOL=3, DIPOL set
    to the structure centre of mass. Forces are dipole-corrected throughout
    the relaxation, giving physically correct geometry and energy
    (consistent with pymatgen MVLSlabSet auto_dipole=True).

    Both stages run in the same Slurm job. Stage 2 restarts from the Stage 1
    WAVECAR (ISTART=1), so the extra SCF cost of Stage 1 is small.

    If use_dipole_correction=False the function falls back to a single
    uncorrected ionic relaxation (equivalent to run_vasp_via_custodian).
    """
    if handlers is None:
        handlers = SURFACE_RELAXATION_HANDLERS

    directory = Path(directory)

    if not use_dipole_correction:
        run_vasp_via_custodian(directory, handlers, vasp_cmd, auto_npar, max_errors)
        return

    # --- Stage 1: SCF pre-convergence without dipole correction ---
    incar = VaspIncar.from_file(directory / "INCAR")
    orig_nsw    = incar.get("NSW",    100)
    orig_ibrion = incar.get("IBRION", 2)
    incar["NSW"]    = 0   # no ionic steps
    incar["IBRION"] = -1  # static
    incar.write_file(directory / "INCAR")

    run_vasp_via_custodian(directory, handlers, vasp_cmd, auto_npar, max_errors)

    # --- Stage 2: full ionic relaxation with dipole correction ---
    struct = Structure.from_file(directory / "POSCAR")
    weights = [site.species.weight for site in struct]
    dipol = np.average(struct.frac_coords, weights=weights, axis=0).tolist()

    incar = VaspIncar.from_file(directory / "INCAR")
    incar["NSW"]    = orig_nsw
    incar["IBRION"] = orig_ibrion
    incar["ISTART"] = 1      # restart from Stage 1 WAVECAR
    incar["LDIPOL"] = True
    incar["IDIPOL"] = 3
    incar["DIPOL"]  = dipol
    incar["AMIN"]   = 0.01   # dampen charge oscillations with LDIPOL active
    incar.write_file(directory / "INCAR")

    run_vasp_via_custodian(directory, handlers, vasp_cmd, auto_npar, max_errors)


def read_vasp_results(directory, return_only_final=True) -> list[Atoms]:
    """Read VASP results using pymatgen, then return ASE Atoms objects.

    Uses pymatgen's Vasprun exclusively rather than ASE's reader because
    ASE silently omits magnetic moments when LORBIT≥10 (VASP writes
    <array name="magnetization"> with per-orbital columns; ASE only handles
    <varray name="magnetization">). Pymatgen handles both formats and exposes
    per-atom total magmoms via ionic_steps[i]["magnetization"].

    Parameters:
    directory (str): The directory where the VASP calculation was run.
    return_only_final (bool): Whether to return only the final structure
    or all intermediate structures.

    Returns:
    list[Atoms]: ASE Atoms objects with a SinglePointCalculator carrying
    energy and magnetic moments.
    """
    directory = Path(directory)

    vr = Vasprun(directory / "vasprun.xml")
    if not vr.converged_ionic:
        raise RuntimeError(
            f"VASP ionic relaxation not converged in {directory}. "
            "Check if wall time was exceeded."
        )

    def _step_to_atoms(structure, energy):
        magmoms = structure.site_properties.get("magmom")
        if magmoms is None:
            raise RuntimeError(
                f"No per-atom magnetization found in {directory}/vasprun.xml. "
                "Ensure LORBIT is set in INCAR for ISPIN=2 calculations."
            )
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, magmoms=magmoms)
        return atoms

    if return_only_final:
        return [_step_to_atoms(vr.final_structure, vr.final_energy)]

    return [
        _step_to_atoms(step["structure"], step["electronic_steps"][-1]["e_wo_entrp"])
        for step in vr.ionic_steps
    ]

def clean_up_vasp_directory(
    directory: str, files_to_remove: list[str] = None,
    purge_all_files: bool = False
    ) -> None:
    
    dir_path =  Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if purge_all_files:
        targets = list(dir_path.iterdir())
    else:
        if files_to_remove is None:
            files_to_remove = [
                "POTCAR", "WAVECAR", "CHGCAR", "DOSCAR", "PROCAR", 
                'DOSCAR', 'CHG', 'POTCAR.orig', 'OUTCAR'
                ]
        # Collect all files that match any of the specified substrings
        targets = [
            file for file in dir_path.iterdir()
            if any(token in file.name for token in files_to_remove)
        ]

    for file in targets:
        try:
            if file.is_file():
                file.unlink()
        except Exception as e:
            pass


def save_fake_results_to_traj(directory: Path, atoms: Atoms):
    # Expose magmoms via a SinglePointCalculator so that
    # atoms.get_magnetic_moments() works when the traj is read back.
    # Uses initial_magmoms from the pymatgen structure conversion if
    # available (set by enumerate_magnetic_structures via
    # site_properties["magmom"]), otherwise falls back to zeros.
    # This ensures the real magmom extraction pipeline is exercised in
    # test/slurm_test mode without a VASP run.
    if atoms.has('initial_magmoms'):
        magmoms = atoms.get_initial_magnetic_moments()
    else:
        magmoms = [0.0] * len(atoms)
    atoms.calc = SinglePointCalculator(atoms, magmoms=magmoms)
    out_traj = Trajectory(directory / "fake_result.traj", "w")
    out_traj.write(atoms)
    out_traj.close()
    

class calculational_setup():

    def __init__(self, 
                 mode: Literal['dummy_test', 'test', 'production_test', 
                    'production'] = 'production',
                 bulk_set = RPBE_relax_bulk_set,
                 surface_set = RPBE_relax_surface_set
                 ) -> None:
        self.mode = mode

        # Setup of test modes:
        if mode == 'production_test':
            self.bulk_set = TestRelaxSet
            self.surface_set = TestRelaxSet
        else:
            self.bulk_set = bulk_set
            self.surface_set = surface_set

    def bulk_relaxation(self, structure: Structure, directory: Path
                        ) -> None:
        vasp_set = self.bulk_set(structure = structure, 
                                 sort_structure = False)
        vasp_set.write_input(directory, potcar_spec = False if 
                             'production' in self.mode else True)
        
        if 'test' in self.mode:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            fake_energy = len(atoms)
            atoms[0].charge = fake_energy  # just to store energy
            save_fake_results_to_traj(directory, atoms)
        
    def surface_relaxation(self, structure: Structure, directory: Path
                           ) -> None:
        vasp_set = self.surface_set(structure = structure, 
                                    sort_structure = False)
        vasp_set.write_input(directory, potcar_spec = False if 
                             'production' in self.mode else True)
        
        if 'test' in self.mode:
            atoms = AseAtomsAdaptor.get_atoms(structure)

            K_O = (-14.217) - 0.070 - (-6.77)
            K_OH = 0.361 + (-6.77)/2 - (-14.217)
            x = 2.46 + K_O
            # Choose the OH offset so the test-mode free energy is
            # centered near G(OH)=1.23 eV for an OH system, given that
            # OH adds one O and one H relative to the clean surface.
            y = 2*x + K_OH - 1.23
            
            if 'dummy' in self.mode:
                random_factor = 0
            else:
                random_factor = np.random.normal(loc = 0, scale = 1)

            # Fake energy calculation:
            if 'H' in atoms.symbols:
                fake_energy = x * len(atoms) - y + random_factor
            else:
                fake_energy = x * len(atoms) + random_factor

            atoms[0].charge = fake_energy  # just to store energy
            save_fake_results_to_traj(directory, atoms)


    



"""
NOTE FOR FUTURE:

Initial structure
       |
       v
Static SCF (ISMEAR=0, SIGMA=0.05)
       |
       v
Handlers check:
   - LargeSigmaHandler
   - IncorrectSmearingHandler
       |
       v
Determine electronic type:
   - Metal -> ISMEAR=1/2, SIGMA=0.2
   - Insulator -> ISMEAR=-5
       |
       v
Relaxation calculation (NSW>0)

"""
