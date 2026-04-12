import spglib
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator


def _count_magnetic_symmetry_ops(mag_structure) -> int:
    """
    Count symmetry operations with spin included as a site coloring,
    replicating the sorting criterion from Horton et al. (2019).

    Each (element, spin_sign) pair is assigned a unique integer atom type
    before passing to spglib. This means Fe↑ and Fe↓ are treated as
    distinct species: a FM structure (all same spin sign) retains the full
    crystallographic symmetry, while an AFM structure (mixed signs) has
    fewer operations because swapping ↑ and ↓ sites is no longer allowed.

    symprec=0.1 Å is a generous tolerance to handle the slight distortions
    that arise from magnetic ordering in pymatgen-generated structures.

    Returns 1 on any failure so the candidate is sorted last rather than
    raising an exception during enumeration.
    """
    type_map: dict[tuple, int] = {}
    numbers: list[int] = []
    for site in mag_structure.sites:
        m = site.properties.get("magmom", 0)
        elem = site.specie.symbol
        spin_sign = 1 if m > 0.1 else (-1 if m < -0.1 else 0)
        key = (elem, spin_sign)
        if key not in type_map:
            type_map[key] = len(type_map) + 1
        numbers.append(type_map[key])

    cell = (
        mag_structure.lattice.matrix,
        [site.frac_coords for site in mag_structure.sites],
        numbers,
    )
    try:
        dataset = spglib.get_symmetry_dataset(cell, symprec=0.1)
        return len(dataset.rotations) if dataset is not None else 1
    except Exception:
        return 1


def enumerate_magnetic_structures(structure, n_max: int = 4) -> list[dict]:
    """
    Enumerate distinct collinear magnetic orderings for a structure and
    return the most symmetrically prioritised ones up to n_max.

    Each returned dict has the form:
        {"structure": Structure, "magmom": list[float], "ordering": str}

    Ordering and filtering rules:
    - The input is reduced to its primitive cell before enumeration.
      This prevents the enumerator from returning sub-cell structures
      (pymatgen internally converts to primitive, so passing a conventional
      cell causes all results to have fewer atoms than the input, which
      the supercell filter would then discard — leaving only NM fallback).
    - Orderings that require a supercell (atom count > primitive cell) are
      discarded so that only same-cell orderings are returned.
    - Remaining candidates are sorted by number of symmetry operations with
      spin as site coloring (Horton et al. 2019): FM always lands first
      (highest symmetry), followed by the most symmetric AFM, etc.
    - The first n_max candidates are returned.
    - If enumeration yields nothing (non-magnetic material, enumlib
      unavailable), a single non-magnetic fallback entry is returned so
      the pipeline always has at least one bulk job to submit.

    Note: MagneticStructureEnumerator.ordered_structures is NOT guaranteed
    to be symmetry-sorted in current pymatgen — the sort is applied here
    explicitly to replicate the paper's enumeration index definition.
    """
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    primitive = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
    n_input = len(primitive)
    candidates: list[dict] = []

    try:
        enumerator = MagneticStructureEnumerator(primitive, automatic=True)
        for mag_struct, origin in zip(
            enumerator.ordered_structures, enumerator.ordered_structure_origins
        ):
            if len(mag_struct) != n_input:
                continue  # discard supercell orderings
            magmoms = [site.properties.get("magmom", 0) for site in mag_struct.sites]
            candidates.append({
                "structure": mag_struct,
                "magmom": magmoms,
                "ordering": str(origin),
            })
    except Exception:
        candidates = []

    if not candidates:
        # Non-magnetic material or enumlib unavailable — one NM fallback job
        return [{
            "structure": primitive,
            "magmom": [0.0] * n_input,
            "ordering": "NM",
        }]

    # Sort descending by symmetry operation count (spin as site coloring).
    # FM lands first (highest symmetry); most symmetric AFM next; etc.
    # Equal symmetry counts preserve the enumerator's original order.
    candidates.sort(
        key=lambda c: _count_magnetic_symmetry_ops(c["structure"]),
        reverse=True,
    )
    return candidates[:n_max]


def _extract_mag_config_index(process_note: object) -> int | None:
    """Parse 'mag_config_index=N' from a processNote string. Returns None if absent."""
    if not isinstance(process_note, str):
        return None
    token = "mag_config_index="
    if token not in process_note:
        return None
    tail = process_note.split(token, 1)[1].split("|", 1)[0].strip()
    try:
        return int(tail)
    except (TypeError, ValueError):
        return None
