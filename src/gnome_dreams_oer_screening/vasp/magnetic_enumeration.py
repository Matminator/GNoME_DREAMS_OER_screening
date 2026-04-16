import spglib
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator


def _count_magnetic_symmetry_ops(mag_structure) -> int:
    """
    Count symmetry operations with spin included as a site coloring,
    replicating the sorting criterion from Horton et al. (2019).

    Each (element, spin_sign) pair is assigned a unique integer atom type
    before passing to spglib. This means Fe↑ and Fe↓ are treated as
    distinct species when counting symmetry operations.

    In general FM structures retain higher crystallographic symmetry than AFM
    structures, so FM typically receives a higher count and sorts first.
    However, this is NOT guaranteed: for certain geometries an AFM ordering
    can match or exceed the FM count. For example, the BCC Fe conventional
    cell (2 atoms) places the AFM ordering first because the body-centring
    translation combined with a spin flip is a valid magnetic symmetry
    operation, giving the AFM structure as many rotations as the FM one —
    and Python's stable sort then preserves the enumerator's original order,
    which happens to put AFM before FM for that geometry.

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


def enumerate_magnetic_structures(structure, n_max: int = 4
                                  ) -> list[dict]:
    """
    Enumerate distinct collinear magnetic orderings for a structure and
    return the most symmetrically prioritised ones up to n_max.

    Each returned dict has the form:
        {"structure": Structure, "magmom": list[float], "ordering": str}

    The returned Structure always has magmoms set as site_properties["magmom"],
    so passing it directly to RPBE_relax_bulk_set (which has ISPIN=2) causes
    VaspInputSet to write the MAGMOM INCAR tag automatically — no separate
    magnetic VASP set class is needed.

    Ordering and filtering rules:
    - The input structure is used as-is (no cell conversion).
      Input structures from the GNoME database are already primitive cells.
    - Orderings that require a supercell (atom count > input) are discarded.
      Same-size and smaller (primitive) structures are both accepted.
      The winner is selected by energy per atom in finalize_mag_bulk_batch.
    - Remaining candidates are sorted by number of symmetry operations with
      spin as site coloring (Horton et al. 2019). FM is typically first
      because it retains the full crystallographic symmetry, but this is
      not guaranteed: an AFM ordering can tie or beat FM when the geometry
      contains a translation or rotation that is simultaneously a valid
      spin-flip operation. In that case Python's stable sort preserves the
      enumerator's original order for tied candidates, and either FM or AFM
      may appear at index 0.
    - The first n_max candidates are returned.
    - If enumeration yields nothing (non-magnetic material, enumlib
      unavailable), a single non-magnetic fallback entry is returned so
      the pipeline always has at least one bulk job to submit.

    Note: MagneticStructureEnumerator.ordered_structures is NOT guaranteed
    to be symmetry-sorted in current pymatgen — the sort is applied here
    explicitly to replicate the paper's enumeration index definition.
    """
    n_input = len(structure)
    candidates: list[dict] = []

    try:
        enumerator = MagneticStructureEnumerator(structure, automatic=True)
    except ValueError as e:
        if "Not detected as magnetic" in str(e):
            enumerator = None # non-magnetic elements -> FM fallback
        else:
            raise e

    if enumerator is not None:
        for mag_struct, origin in zip(
            enumerator.ordered_structures,
            enumerator.ordered_structure_origins
        ):
            if len(mag_struct) != n_input:
                continue  # discard supercell orderings
            # MagneticStructureEnumerator stores spin in site.specie.spin, not
            # in site.properties["magmom"]. Write it back into site properties
            # so that (a) _count_magnetic_symmetry_ops reads the correct values
            # and (b) VaspInputSet generates the MAGMOM INCAR tag automatically.
            magmoms = [
                float(site.specie.spin) if site.specie.spin is not None else 0.0
                for site in mag_struct.sites
            ]
            for site, m in zip(mag_struct.sites, magmoms):
                site.properties["magmom"] = m
            candidates.append({
                "structure": mag_struct,
                "magmom": magmoms,
                "ordering": str(origin),
            })

    if not candidates:
        # Non-magnetic material — return a copy with magmoms set as site
        # properties so the structure is self-consistent with the magmom list
        # and VaspInputSet writes the correct MAGMOM tag.
        nm_magmom = [0.6] * n_input
        nm_struct = structure.copy()
        for site, m in zip(nm_struct.sites, nm_magmom):
            site.properties["magmom"] = m
        return [{
            "structure": nm_struct,
            "magmom": nm_magmom,
            "ordering": "NM",
        }]

    # Sort descending by symmetry operation count (spin as site coloring).
    # FM is typically first (highest symmetry), but an AFM ordering with an
    # equal count will precede FM when the enumerator's original order puts
    # it there — Python's stable sort preserves ties.
    candidates.sort(
        key=lambda c: _count_magnetic_symmetry_ops(c["structure"]),
        reverse=True,
    )
    return candidates[:n_max]


def extract_mag_config_index(process_note: object) -> int | None:
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
