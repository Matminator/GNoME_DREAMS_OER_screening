"""
Benchmark: max_cell_size=1 vs pymatgen default for MagneticStructureEnumerator.

Always runs:
  - Fe BCC primitive  (1 atom)   : default AND new (max_cell_size=1)
  - Fe BCC conventional (2 atoms): default AND new (max_cell_size=1)
  - VNiTeO6 (18 atoms)           : new only (fast)

Optional (WARNING — may take 30+ minutes):
  - VNiTeO6 through pymatgen default: enabled with --also-time-vniteo6-default

Usage
-----
    python tests/benchmark_magnetic_enumeration.py
    python tests/benchmark_magnetic_enumeration.py --also-time-vniteo6-default

Requires enumlib binaries (enum.x, makestr.x) on PATH.
"""

import argparse
import pathlib
import time

from pymatgen.analysis.magnetism.analyzer import (
    CollinearMagneticStructureAnalyzer,
    MagneticStructureEnumerator,
)
from pymatgen.core import Lattice, Structure

from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    enumerate_magnetic_structures,
)

TESTS_DIR = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_max_cell_size(structure: Structure) -> int:
    """Reproduce pymatgen's auto-scaling: max(1, int(4 / n_mag_sites))."""
    analyzer = CollinearMagneticStructureAnalyzer(structure, make_primitive=False)
    mag_species = set(analyzer.types_of_magnetic_specie)
    n_mag = sum(1 for site in structure if site.specie in mag_species)
    return max(1, int(4 / n_mag)) if n_mag > 0 else 1


def _time_enumerator(structure: Structure, max_cell_size, label: str) -> float:
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
    print(f"  {label:55s}  {elapsed:7.2f} s")
    return elapsed


def _time_wrapper(structure: Structure) -> float:
    t0 = time.perf_counter()
    results = enumerate_magnetic_structures(structure, n_max=4)
    elapsed = time.perf_counter() - t0
    print(f"  {'enumerate_magnetic_structures() [our wrapper]':55s}  {elapsed:7.2f} s"
          f"  ({len(results)} configs returned)")
    return elapsed


def _run_structure(name: str, structure: Structure,
                   run_default: bool, run_new: bool) -> None:
    auto = _auto_max_cell_size(structure)
    print(f"\nStructure: {name}")
    print(f"  pymatgen default max_cell_size = {auto}")
    print("-" * 72)

    t_default = None
    if run_default:
        t_default = _time_enumerator(
            structure, max_cell_size=None,
            label=f"pymatgen default (max_cell_size={auto})")

    t_new = None
    if run_new:
        t_new = _time_enumerator(
            structure, max_cell_size=1,
            label="new: max_cell_size=1 (via MagneticStructureEnumerator)")
        _time_wrapper(structure)

    if t_default is not None and t_new is not None and t_new > 0:
        print(f"  Speedup (default → new): {t_default / t_new:.1f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark magnetic structure enumeration speed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "By default the benchmark is safe to run.\n"
            "Add --also-time-vniteo6-default to measure the slow pymatgen\n"
            "default on VNiTeO6 — WARNING: this may take 30+ minutes."
        ),
    )
    parser.add_argument(
        "--also-time-vniteo6-default",
        action="store_true",
        help=(
            "WARNING: also run VNiTeO6 through pymatgen's default enumeration. "
            "This is the known slow path and may take 30+ minutes. "
            "Only use this if you want to measure how bad the hang was."
        ),
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Magnetic enumeration benchmark")
    print("=" * 72)

    # Fe structures: always run both methods
    _run_structure(
        "Fe BCC primitive  (1 atom)",
        Structure(
            Lattice([[2.87/2, 2.87/2, -2.87/2],
                     [-2.87/2, 2.87/2, 2.87/2],
                     [2.87/2, -2.87/2, 2.87/2]]),
            ["Fe"], [[0, 0, 0]],
        ),
        run_default=True,
        run_new=True,
    )

    _run_structure(
        "Fe BCC conventional (2 atoms)",
        Structure(Lattice.cubic(2.87), ["Fe", "Fe"],
                  [[0, 0, 0], [0.5, 0.5, 0.5]]),
        run_default=True,
        run_new=True,
    )

    # VNiTeO6: new method always, default only if explicitly requested
    if args.also_time_vniteo6_default:
        print("\n*** --also-time-vniteo6-default is ON — this may take 30+ min ***")

    _run_structure(
        "VNiTeO6  (18 atoms, V+Ni magnetic)",
        Structure.from_file(TESTS_DIR / "POSCAR_VniTeO6"),
        run_default=args.also_time_vniteo6_default,
        run_new=True,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
