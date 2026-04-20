"""
H2O gas-phase reference energy calculation (two-step RPBE/VASP).

Step 1: Geometry relaxation at ENCUT=520 eV (RPBE_relax_bulk_set base)
Step 2: Single-point at ENCUT=400 eV (RPBE_relax_surface_set base) on
        the relaxed geometry — matches the cutoff used in adsorption
        calculations for consistent OER thermochemistry.

Run from the repo root on HPC:
    python gas_ref_calculations/run_H2O_dft.py

Copy the printed energy into OER_catalyst_study as H2O_gas_free_energy.
"""

import numpy as np
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Kpoints as VaspKpoints

from gnome_dreams_oer_screening.vasp.pre_defined_vasp_sets import (
    RPBE_relax_bulk_set, RPBE_relax_surface_set)
from gnome_dreams_oer_screening.vasp.vasp_calculation import (
    run_vasp_via_custodian, read_vasp_results, SURFACE_RELAXATION_HANDLERS)

# --- Molecule geometry -------------------------------------------------------
box = 15.0          # Å — 15 Å vacuum box, standard for GGA molecule-in-box
d_OH = 0.96         # Å, O-H bond length
angle = 104.45      # degrees, H-O-H angle

half = np.radians(angle / 2)
h1 = [ d_OH * np.sin(half), d_OH * np.cos(half), 0.0]
h2 = [-d_OH * np.sin(half), d_OH * np.cos(half), 0.0]

atoms = Atoms(
    symbols=['O', 'H', 'H'],
    positions=[[0.0, 0.0, 0.0], h1, h2],
    cell=[box, box, box],
    pbc=True,
)
atoms.center()

structure = AseAtomsAdaptor.get_structure(atoms)

GAMMA_KPOINTS = VaspKpoints.gamma_automatic((1, 1, 1))

# --- Step 1: geometry relaxation at ENCUT=520 --------------------------------
relax_dir = "vasp_H2O_relax"

relax_set = RPBE_relax_bulk_set(
    structure,
    user_incar_settings={
        "ISIF": 2,      # ions only — bulk default ISIF=3 would relax the cell
        "ISPIN": 2,     # consistent with slab runs; converges to 0 moment for H2O
        "SIGMA": 0.001,
        "EDIFF": 1e-6,
        "LWAVE": False,
        "LDAU": False,
        "NSW": 100,
    }
)
relax_set.CONFIG["KPOINTS"] = GAMMA_KPOINTS
relax_set.write_input(relax_dir)

print(f"Running Step 1: geometry relaxation (ENCUT=520) in {relax_dir}/", flush=True)
run_vasp_via_custodian(relax_dir, handlers=SURFACE_RELAXATION_HANDLERS)
print("Step 1 done.", flush=True)

# --- Step 2: single-point at ENCUT=400 on relaxed geometry ------------------
sp_dir = "vasp_H2O_sp"

relaxed_structure = Structure.from_file(f"{relax_dir}/CONTCAR")

sp_set = RPBE_relax_surface_set(
    relaxed_structure,
    user_incar_settings={
        "NSW": 0,       # static calculation
        "IBRION": -1,
        "ISPIN": 2,
        "SIGMA": 0.001,
        "EDIFF": 1e-6,
        "LWAVE": False,
        "LDAU": False,
    }
)
sp_set.CONFIG["KPOINTS"] = GAMMA_KPOINTS
sp_set.write_input(sp_dir)

print(f"Running Step 2: single-point (ENCUT=400) in {sp_dir}/", flush=True)
run_vasp_via_custodian(sp_dir, handlers=SURFACE_RELAXATION_HANDLERS)
print("Step 2 done.", flush=True)

# --- Extract and report energy -----------------------------------------------
E_H2O = read_vasp_results(sp_dir)[-1].get_potential_energy()
print(f"\nH2O DFT total energy (ENCUT=400): {E_H2O:.6f} eV")
print(f"Use as: H2O_gas_free_energy = {E_H2O:.6f}")
