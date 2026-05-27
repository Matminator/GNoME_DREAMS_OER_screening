
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
from pymatgen.io.vasp import Vasprun, Outcar, Incar as VaspIncar

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

    Stage 1 — Cold-start SCF only (NSW=0), no dipole correction. ALGO=Normal,
    NELMDL=-12, AMIX=0.2 for cold-start stability.

    Stage 2 — Full ionic relaxation with LDIPOL=True, IDIPOL=3, DIPOL set to
    the structure centre of mass. Always built from the original pre-Stage-1
    INCAR so no Custodian corrections from Stage 1 carry over.
      - If Stage 1 converged: warm restart (ISTART=1), original ALGO.
      - If Stage 1 failed:    cold start (ISTART=0), ALGO=Normal, NELMDL=-12,
                              AMIX=0.2 enforced throughout.

    If use_dipole_correction=False the function falls back to a single
    uncorrected ionic relaxation (equivalent to run_vasp_via_custodian).
    """
    if handlers is None:
        handlers = SURFACE_RELAXATION_HANDLERS

    directory = Path(directory)

    if not use_dipole_correction:
        run_vasp_via_custodian(directory, handlers, vasp_cmd, auto_npar, max_errors)
        return

    # --- Stage 1: cold-start SCF pre-convergence without dipole correction ---
    incar = VaspIncar.from_file(directory / "INCAR")
    orig_incar  = dict(incar)            # clean snapshot — Stage 2 base
    orig_nsw    = incar.get("NSW",    100)
    orig_ibrion = incar.get("IBRION", 2)
    orig_algo   = incar.get("ALGO",   "Fast")
    incar["NSW"]    = 0        # no ionic steps
    incar["IBRION"] = -1       # static
    incar["NELMDL"] = -12      # delay mixing onset for cold-start stability
    incar["ALGO"]   = "Normal" # blocked Davidson; more stable than RMM-DIIS on cold start
    incar["AMIX"]   = 0.2      # conservative mixing for cold start
    incar.write_file(directory / "INCAR")

    # Stage 1 gets fewer retries: the ALGO ladder has ~2 unique strategies,
    # so 5 errors is enough. Beyond that Custodian just cycles with bad settings.
    try:
        run_vasp_via_custodian(directory, handlers, vasp_cmd, auto_npar,
                               max_errors=5)
    except Exception:
        pass  # Stage 2 convergence check determines the path

    # Determine Stage 2 strategy based on Stage 1 outcome
    try:
        vr = Vasprun(directory / "vasprun.xml")
        stage1_converged = vr.converged_electronic
    except Exception:
        stage1_converged = False

    # --- Stage 2: full ionic relaxation with dipole correction ---
    # Start from the clean original INCAR — Stage 1 Custodian corrections are
    # private to Stage 1 and must not carry over to Stage 2.
    struct = Structure.from_file(directory / "POSCAR")
    weights = [site.species.weight for site in struct]
    dipol = np.average(struct.frac_coords, weights=weights, axis=0).tolist()

    incar = VaspIncar(orig_incar)
    incar["NSW"]    = orig_nsw
    incar["IBRION"] = orig_ibrion
    incar["LDIPOL"] = True
    incar["IDIPOL"] = 3
    incar["DIPOL"]  = dipol
    incar["AMIN"]   = 0.01     # dampen charge oscillations with LDIPOL active

    if stage1_converged:
        incar["ISTART"] = 1        # warm restart from Stage 1 WAVECAR
        incar["ALGO"]   = orig_algo
    else:
        incar["ISTART"] = 0        # cold start — Stage 1 WAVECAR unreliable
        incar["ALGO"]   = "Normal"
        incar["NELMDL"] = -12      # delay mixing onset
        incar["AMIX"]   = 0.2      # conservative mixing enforced throughout

    incar.write_file(directory / "INCAR")

    run_vasp_via_custodian(directory, handlers, vasp_cmd, auto_npar, max_errors)


def read_vasp_results(directory, return_only_final=True) -> list[Atoms]:
    """Read VASP results using pymatgen, then return ASE Atoms objects.

    Energy and structure are read from vasprun.xml via pymatgen's Vasprun.
    Per-atom magnetic moments are read from OUTCAR via pymatgen's Outcar.

    Requires LORBIT=11 in INCAR. LORBIT=11 always writes the
    "magnetization (x)" block to OUTCAR for every ionic relaxation —
    including non-magnetic materials (e.g. TiO2) where all moments are zero.
    This is more reliable than reading from vasprun.xml, where VASP omits
    <varray name="magnetization"> when moments converge to zero.

    Per-step magmoms are only available for the final step (OUTCAR records
    only the last magnetization block). When return_only_final=False,
    intermediate steps carry zero magmoms; only the final Atoms has real magmoms.

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
    if not vr.converged_electronic:
        raise RuntimeError(
            f"VASP electronic SCF not converged in {directory}. "
            f"NELM={vr.incar.get('NELM', 60)} was reached without satisfying EDIFF."
        )

    outcar = Outcar(directory / "OUTCAR")
    # outcar.magnetization: tuple[dict[str, float]], one dict per atom,
    # keys "s", "p", "d", "tot" — from the last magnetization block in OUTCAR.
    final_magmoms = [site["tot"] for site in outcar.magnetization]

    def _step_to_atoms(step, energy, magmoms):
        atoms = AseAtomsAdaptor.get_atoms(step["structure"])
        atoms.calc = SinglePointCalculator(atoms, energy=energy, magmoms=magmoms)
        return atoms

    n_atoms = len(vr.ionic_steps[-1]["structure"])

    if return_only_final:
        return [_step_to_atoms(vr.ionic_steps[-1], vr.final_energy, final_magmoms)]

    # Intermediate steps: magmoms not available from OUTCAR per-step.
    result = [
        _step_to_atoms(step, step["e_0_energy"], [0.0] * n_atoms)
        for step in vr.ionic_steps[:-1]
    ]
    result.append(_step_to_atoms(vr.ionic_steps[-1], vr.final_energy, final_magmoms))
    return result

# Minimal set of files retained after a completed VASP run.
# Everything else is deleted to keep inode counts manageable at production scale.
_VASP_KEEP_FILES = {
    "INCAR",           # settings used
    "KPOINTS",         # k-point mesh
    "POSCAR",          # input structure
    "CONTCAR",         # final relaxed structure
    "OSZICAR",         # convergence summary
    "vasprun.xml",     # machine-readable output
    "OUTCAR",          # full log; kept for post-hoc debugging
    "vasp.out",        # stdout; kept for post-hoc debugging
    "custodian.json",  # Custodian error/correction log; needed to diagnose convergence failures
}


def clean_up_vasp_directory(
    directory: str,
    files_to_remove: list[str] = None,
    purge_all_files: bool = False,
) -> None:

    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    if purge_all_files:
        targets = [f for f in dir_path.iterdir() if f.is_file()]
    elif files_to_remove is not None:
        # Explicit remove-list: delete files whose names contain any token.
        targets = [
            f for f in dir_path.iterdir()
            if f.is_file() and any(token in f.name for token in files_to_remove)
        ]
    else:
        # Default keep-list mode: delete everything not in _VASP_KEEP_FILES.
        targets = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.name not in _VASP_KEEP_FILES
        ]

    for f in targets:
        try:
            f.unlink()
        except Exception:
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
