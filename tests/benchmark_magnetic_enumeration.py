"""
Benchmark: max_cell_size=1 vs pymatgen default for MagneticStructureEnumerator.

Compares wall-clock time for structures where pymatgen's auto-scaling
sets max_cell_size > 1 (structures with few magnetic sites).

Run on HPC where enumlib is available:
    python tests/benchmark_magnetic_enumeration.py
"""

import time
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator

from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    enumerate_magnetic_structures,
)


def _make_structure(formula: str, n_atoms: int) -> Structure:
    """Build a simple BCC-like Fe structure with n_atoms in the cell."""
    a = 2.87
    if n_atoms == 1:
        lat = Lattice([[a/2, a/2, -a/2], [-a/2, a/2, a/2], [a/2, -a/2, a/2]])
        return Structure(lat, ["Fe"], [[0, 0, 0]])
    if n_atoms == 2:
        return Structure(Lattice.cubic(a), ["Fe", "Fe"], [[0,0,0],[0.5,0.5,0.5]])
    raise ValueError(n_atoms)


def _time_enumerator(structure, max_cell_size, label: str) -> float:
    """Time MagneticStructureEnumerator directly with a given max_cell_size."""
    kwargs = {"max_cell_size": max_cell_size} if max_cell_size is not None else {}
    t0 = time.perf_counter()
    try:
        MagneticStructureEnumerator(
            structure,
            automatic=True,
            transformation_kwargs=kwargs or None,
        )
    except ValueError:
        pass
    elapsed = time.perf_counter() - t0
    print(f"  {label:45s}  {elapsed:.2f} s")
    return elapsed


def _time_enumerate_magnetic_structures(structure, label: str) -> float:
    """Time the full enumerate_magnetic_structures (uses max_cell_size=1)."""
    t0 = time.perf_counter()
    results = enumerate_magnetic_structures(structure, n_max=4)
    elapsed = time.perf_counter() - t0
    print(f"  {label:45s}  {elapsed:.2f} s  ({len(results)} configs returned)")
    return elapsed


def _auto_max_cell_size(structure) -> int:
    """Reproduce pymatgen's auto-scaling formula to show what the default would be."""
    from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
    analyzer = CollinearMagneticStructureAnalyzer(structure, make_primitive=False)
    mag_species = [sp for sp in analyzer.types_of_magnetic_specie]
    n_mag = sum(1 for site in structure if site.specie in mag_species)
    return max(1, int(4 / n_mag)) if n_mag > 0 else 1


structures = {
    "Fe BCC primitive (1 atom)":   _make_structure("Fe", 1),
    "Fe BCC conventional (2 atoms)": _make_structure("Fe", 2),
}

print("=" * 70)
print("Magnetic enumeration benchmark")
print("Compares default max_cell_size vs explicit max_cell_size=1")
print("=" * 70)

for name, struct in structures.items():
    auto = _auto_max_cell_size(struct)
    print(f"\nStructure: {name}  (pymatgen default max_cell_size={auto})")
    print("-" * 70)
    t_default = _time_enumerator(struct, max_cell_size=None,
                                 label=f"default (max_cell_size={auto})")
    t_new     = _time_enumerator(struct, max_cell_size=1,
                                 label="max_cell_size=1")
    t_wrapper = _time_enumerate_magnetic_structures(struct,
                                 label="enumerate_magnetic_structures()")

    if t_default > 0:
        print(f"  Speedup (default → max_cell_size=1): {t_default / t_new:.1f}x")

print("\nDone.")
