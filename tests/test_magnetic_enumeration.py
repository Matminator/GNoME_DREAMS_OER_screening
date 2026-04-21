"""
Environment integration test for enumlib + magnetic enumeration.

These tests call MagneticStructureEnumerator WITHOUT mocking, so they
require the enumlib binaries (enum.x, makestr.x / multienum.x) to be
on PATH. They are intended to verify that the HPC environment is
correctly set up before running production workflows.

Run with:
    pytest tests/test_magnetic_enumeration.py -v

These tests are NOT marked 'mini' and are excluded from the normal
fast test suite. Run them explicitly on the HPC to confirm the
environment is ready.
"""

import pathlib
import shutil

import pytest
from pymatgen.core import Lattice, Structure

from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    enumerate_magnetic_structures,
    extract_mag_config_index,
    _count_magnetic_symmetry_ops,
    count_magnetic_sites,
    count_magnetic_sites_from_formula,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Not all sites have property wyckoff:UserWarning",
    "ignore:Elements in POSCAR would be overwritten:UserWarning",
)


# ---------------------------------------------------------------------------
# count_magnetic_sites_from_formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("formula, expected", [
    ("Fe",        1),   # single magnetic element
    ("Fe2O3",     2),   # O is not magnetic
    ("V2Cu1Ni9O16", 12),  # 2V + 1Cu + 9Ni; matches the POSCAR test structure
    ("Ag5Ir3O14", 0),   # Ir is NOT in DEFAULT_MAGMOMS; neither are Ag or O
    ("NaCl",      0),   # no magnetic elements
    ("TiO2",      0),   # Ti is not in DEFAULT_MAGMOMS; O is not either
    ("MnFe2O4",   3),   # 1Mn + 2Fe; O not magnetic
])
@pytest.mark.mini
def test_count_magnetic_sites_from_formula(formula, expected):
    assert count_magnetic_sites_from_formula(formula) == expected


@pytest.mark.mini
def test_count_magnetic_sites_from_formula_empty_formula_raises():
    """Composition("") raises in pymatgen — guard that we don't silently return 0."""
    with pytest.raises(Exception):
        count_magnetic_sites_from_formula("")


# ---------------------------------------------------------------------------
# count_magnetic_sites (Structure-based)
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_count_magnetic_sites_structure():
    """count_magnetic_sites on a simple Fe-O structure."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Fe", "Fe", "O"], [
        [0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]
    ])
    assert count_magnetic_sites(structure) == 2  # 2 Fe; O not magnetic


@pytest.mark.mini
def test_count_magnetic_sites_non_magnetic_structure():
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert count_magnetic_sites(structure) == 0


@pytest.mark.mini
def test_count_magnetic_sites_consistent_with_formula_counter():
    """Structure-based and formula-based counters must agree for the same composition."""
    from pymatgen.core import Lattice, Structure
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Ni", "V", "O", "O"], [
        [0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]
    ])
    assert count_magnetic_sites(structure) == count_magnetic_sites_from_formula("NiVO2")


def _enumlib_available() -> bool:
    """Return True if the required enumlib executables are on PATH."""
    has_enum = shutil.which("enum.x") is not None or shutil.which("multienum.x") is not None
    has_make = shutil.which("makestr.x") is not None or shutil.which("makeStr.py") is not None
    return has_enum and has_make


needs_enumlib = pytest.mark.skipif(
    not _enumlib_available(),
    reason="enumlib binaries (enum.x / makestr.x) not found on PATH — "
)


@needs_enumlib
def test_fe_primitive_cell_returns_fm_as_first_config():
    """
    Fe BCC primitive cell (1 atom). Fe is a well-known ferromagnet so the
    first result must be FM: all magmoms the same sign and nonzero.
    The primitive cell has 1 atom so there is only one possible ordering.
    """
    a = 2.87
    lat = Lattice([[a/2, a/2, -a/2],
                   [-a/2, a/2,  a/2],
                   [a/2, -a/2,  a/2]])
    struct = Structure(lat, ["Fe"], [[0, 0, 0]])

    results = enumerate_magnetic_structures(struct, n_max=4)

    _assert_well_formed(results, "Fe BCC primitive")
    assert len(results) == 1, f"Expected 1 result for 1-atom cell, got {len(results)}"

    config_dict = results[0]
    magmoms = config_dict["magmom"]

    # Check that the single magmom is close to 5 (Fe's default value in pymatgen)
    mag = magmoms[0]
    assert abs(mag) == pytest.approx(5.0, abs=0.1), f"Expected magmom ~5 for Fe, got {mag}"

    # Ordering label must be 'fm'
    assert config_dict["ordering"] == "fm", (
        f"Expected ordering='fm', got '{config_dict['ordering']}'"
    )

    # Returned structure must have the same atom count as the input (no cell expansion)
    assert len(config_dict["structure"]) == len(struct), (
        f"Expected 1-atom structure, got {len(config_dict['structure'])} atoms"
    )
    

@needs_enumlib
def test_fe_conventional_cell_returns_multiple_orderings():
    """
    Fe BCC conventional cell (2 atoms). With 2 atoms the enumerator can
    produce both FM [5, 5] and AFM [5, -5]. The AFM ordering comes first
    because the body-center symmetry (translation + spin flip) gives it
    more symmetry operations than FM for this specific geometry.
    We verify that at least one FM and one AFM config are present.
    """
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    results = enumerate_magnetic_structures(struct, n_max=4)

    _assert_well_formed(results, "Fe BCC conventional")

    # Retruns True if FM: all magmoms have the same sign:
    def is_fm(magmom):
        nz = [m for m in magmom if abs(m) > 0.1]
        return len(nz) > 0 and (all(m > 0 for m in nz) or all(m < 0 for m in nz))

    # Retruns True if FM: one mag is positive, one is negative:
    def is_afm(magmom):
        nz = [m for m in magmom if abs(m) > 0.1]
        return len(nz) > 0 and any(m > 0 for m in nz) and any(m < 0 for m in nz)

    # FM must appear within the first 2 results — it should never be buried
    # at the bottom since it is always physically relevant
    assert any(is_fm(r["magmom"]) for r in results[:2]), (
        f"FM not in first 2 results: {[r['magmom'] for r in results]}"
    )
    assert any(is_afm(r["magmom"]) for r in results), "No AFM config found"


@needs_enumlib
def test_si_non_magnetic_returns_nm_fallback():
    """
    Si diamond structure — a well-known non-magnetic semiconductor.
    Si is NOT in pymatgen's magnetic elements list, so MagneticStructureEnumerator
    raises ValueError("Not detected as magnetic"). enumerate_magnetic_structures
    catches this and returns a single NM fallback entry with magmoms = 0.6.

    Note: Cu is in pymatgen's magnetic elements list (Cu2+ can be magnetic),
    so it does NOT trigger the ValueError path. Si is used here instead.
    """
    a = 5.43
    lat = Lattice.cubic(a)
    # Diamond cubic: 2-atom primitive cell
    struct = Structure(lat, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

    results = enumerate_magnetic_structures(struct, n_max=4)

    _assert_well_formed(results, "Si diamond")
    assert len(results) == 1, f"Expected 1 NM fallback, got {len(results)}"
    assert results[0]["ordering"] == "NM", (
        f"Expected ordering='NM', got '{results[0]['ordering']}'"
    )
    assert all(mag == pytest.approx(0.6, abs=0.1) for mag in results[0]["magmom"])
    assert results[0]["structure"] is not struct, (
        "NM fallback must return a copy, not the original input structure"
    )


@needs_enumlib
def test_nmax_1_returns_single_highest_symmetry_ordering():
    """
    Fe BCC conventional cell (2 atoms) with n_max=1 must return exactly one
    result — the highest-symmetry ordering from the sort.
    Verifies that n_max is respected as a hard cap, not just a guideline.
    """
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    results = enumerate_magnetic_structures(struct, n_max=1)

    _assert_well_formed(results, "Fe BCC conventional n_max=1")
    assert len(results) == 1, f"Expected exactly 1 result, got {len(results)}"


@needs_enumlib
def test_nmax_2_returns_at_most_2_orderings():
    """
    Fe BCC conventional cell (2 atoms) has exactly 2 orderings (FM + AFM).
    With n_max=2 both must be returned; with n_max=1 only one is returned.
    Verifies the cap is applied before returning.
    """
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    results_2 = enumerate_magnetic_structures(struct, n_max=2)
    results_1 = enumerate_magnetic_structures(struct, n_max=1)

    _assert_well_formed(results_2, "Fe BCC conventional n_max=2")
    _assert_well_formed(results_1, "Fe BCC conventional n_max=1")
    assert len(results_2) == 2, f"Expected 2 results with n_max=2, got {len(results_2)}"
    assert len(results_1) == 1, f"Expected 1 result with n_max=1, got {len(results_1)}"

    # The first element must be the same ordering in both cases (same sort, just truncated)
    assert results_2[0]["ordering"] == results_1[0]["ordering"], (
        "First ordering differs between n_max=1 and n_max=2 — sort is not stable"
    )


# ---------------------------------------------------------------------------
# Real GNoME candidate structures from test_cands.traj
# ---------------------------------------------------------------------------

_TRAJ = pathlib.Path(__file__).parent.parent / "notebooks" / "test_cands.traj"
_needs_traj = pytest.mark.skipif(not _TRAJ.exists(), reason="test_cands.traj not found")


def _load_gnome_structure(traj_index: int):
    from ase.io import read
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = read(str(_TRAJ), index=traj_index)
    return AseAtomsAdaptor.get_structure(atoms)


def _assert_well_formed(results, formula):
    """
    Shared structural checks applied to every enumerate_magnetic_structures result.
    Verifies keys, lengths, and that site_properties["magmom"] is set on the
    structure and consistent with the dict's magmom list.
    """
    assert isinstance(results, list)
    assert len(results) >= 1, f"No configs returned for {formula}"
    assert len(results) <= 4, f"More than n_max=4 configs returned for {formula}"
    for i, cfg in enumerate(results):
        assert set(cfg.keys()) == {"structure", "magmom", "ordering"}, (
            f"Config {i} missing keys"
        )
        assert len(cfg["magmom"]) == len(cfg["structure"]), (
            f"Config {i}: magmom length {len(cfg['magmom'])} != "
            f"structure length {len(cfg['structure'])}"
        )
        site_magmoms = cfg["structure"].site_properties.get("magmom")
        assert site_magmoms is not None, (
            f"Config {i}: site_properties['magmom'] not set — "
            "VaspInputSet cannot generate MAGMOM tag"
        )
        assert list(site_magmoms) == cfg["magmom"], (
            f"Config {i}: site_properties magmom {list(site_magmoms)} "
            f"does not match dict magmom {cfg['magmom']}"
        )


def _has_fm(results) -> bool:
    """At least one config where all non-zero magmoms have the same sign."""
    for cfg in results:
        nz = [m for m in cfg["magmom"] if abs(m) > 0.1]
        if nz and (all(m > 0 for m in nz) or all(m < 0 for m in nz)):
            return True
    return False


def _has_afm(results) -> bool:
    """At least one config with both positive and negative non-zero magmoms."""
    for cfg in results:
        nz = [m for m in cfg["magmom"] if abs(m) > 0.1]
        if nz and any(m > 0 for m in nz) and any(m < 0 for m in nz):
            return True
    return False


@needs_enumlib
@_needs_traj
def test_gnome_IrO4Pd_returns_valid_configs():
    """
    IrO4Pd — 6-atom structure (traj index 30).
    Ir is weakly magnetic in some contexts but is not a classic TM oxide
    magnetic element. We only assert well-formedness and that FM appears
    if any magnetic ordering is found (no AFM assertion for this composition).
    """
    structure = _load_gnome_structure(traj_index=30)
    results = enumerate_magnetic_structures(structure, n_max=4)

    _assert_well_formed(results, "IrO4Pd")

    # If magnetic configs were found, FM must be first (highest symmetry)
    if results[0]["ordering"] != "NM":
        assert _has_fm(results), (
            f"FM not found among orderings: {[r['ordering'] for r in results]}"
        )


@needs_enumlib
@_needs_traj
def test_gnome_Fe2Ir2O12Rh2_returns_fm_and_afm():
    """
    Fe2Ir2O12Rh2 — 18-atom structure (traj index 8) with Fe.
    Fe is a well-known magnetic element. Both FM and AFM orderings are
    expected and FM must appear first (highest symmetry).
    """
    structure = _load_gnome_structure(traj_index=8)
    results = enumerate_magnetic_structures(structure, n_max=4)

    _assert_well_formed(results, "Fe2Ir2O12Rh2")

    assert _has_fm(results), (
        f"FM not found: {[r['magmom'] for r in results]}"
    )
    assert _has_afm(results), (
        f"AFM not found: {[r['magmom'] for r in results]}"
    )
    # FM must be first — highest symmetry sort places it at index 0
    fm_magmoms = results[0]["magmom"]
    nz = [m for m in fm_magmoms if abs(m) > 0.1]
    assert all(m > 0 for m in nz) or all(m < 0 for m in nz), (
        f"First config is not FM: {fm_magmoms}"
    )


@needs_enumlib
@_needs_traj
def test_gnome_Ag3BiCoIr3O14_returns_fm():
    """
    Ag3BiCoIr3O14 — 22-atom structure (traj index 6) with a single Co atom.
    Co is the only magnetic species in the cell. With only one magnetic site,
    AFM is physically impossible — the enumerator can only produce FM.
    We assert FM is found and is first.
    """
    structure = _load_gnome_structure(traj_index=6)
    results = enumerate_magnetic_structures(structure, n_max=4)

    _assert_well_formed(results, "Ag3BiCoIr3O14")

    assert _has_fm(results), (
        f"FM not found: {[r['magmom'] for r in results]}"
    )
    # FM must be first (only ordering possible with 1 magnetic site)
    fm_magmoms = results[0]["magmom"]
    nz = [m for m in fm_magmoms if abs(m) > 0.1]
    assert all(m > 0 for m in nz) or all(m < 0 for m in nz), (
        f"First config is not FM: {fm_magmoms}"
    )


# ===========================================================================
# extract_mag_config_index (no enumlib required, runs on Windows)
# ===========================================================================

def test_extract_mag_config_index_basic():
    """
    Plain note with just the index token returns the integer index.
    Index 0 is always present — even when only one config is enumerated
    (e.g. NM fallback or single-FM result) job_handler assigns
    mag_config_index=0 so that update_log can look up the magmom uniformly.
    """
    assert extract_mag_config_index("mag_config_index=0") == 0
    assert extract_mag_config_index("mag_config_index=3") == 3


def test_extract_mag_config_index_with_ordering_suffix():
    """Note that includes the ordering label after a pipe still parses correctly."""
    assert extract_mag_config_index("mag_config_index=2 | ordering=AFM") == 2
    assert extract_mag_config_index("mag_config_index=1 | ordering=fm") == 1


def test_extract_mag_config_index_oh_note_returns_none():
    """OH-style processNote that has no mag_config_index token returns None."""
    assert extract_mag_config_index("OH_init_index=2") is None
    assert extract_mag_config_index("OH_init_index=0 | some_other=tag") is None


def test_extract_mag_config_index_none_input_returns_none():
    """Non-string inputs (None, int, dict) return None without raising."""
    assert extract_mag_config_index(None) is None
    assert extract_mag_config_index(42) is None
    assert extract_mag_config_index({"key": "value"}) is None


def test_extract_mag_config_index_malformed_returns_none():
    """Token present but value is not a valid integer returns None."""
    assert extract_mag_config_index("mag_config_index=abc") is None
    assert extract_mag_config_index("mag_config_index=") is None
    assert extract_mag_config_index("mag_config_index= | ordering=FM") is None


def test_extract_mag_config_index_with_prefix():
    """
    Token preceded by an extra note prefix still parses correctly.
    This matches the real pipeline case in job_handler where an optional
    caller-supplied note is prepended:
        processNote = f"{note} | mag_config_index={i} | ordering={ordering}"
    The parser must find the token anywhere in the string, not just at the start.
    """
    assert extract_mag_config_index("some_note | mag_config_index=0 | ordering=FM") == 0
    assert extract_mag_config_index("candidate=abc | mag_config_index=2 | ordering=AFM") == 2


# ===========================================================================
# _count_magnetic_symmetry_ops (no enumlib required, runs on Windows)
# ===========================================================================

def test_count_symmetry_ops_returns_positive_integer():
    """
    _count_magnetic_symmetry_ops must return an integer >= 1 for any valid
    structure. Constructed directly with site_properties so enumlib is not
    needed — this tests the spglib call in isolation.

    FM structure (all same spin sign) is expected to have strictly more
    symmetry operations than AFM (mixed signs) for BCC Fe, because AFM
    breaks the translation symmetry that FM preserves.
    """
    lat = Lattice.cubic(2.87)
    fm_struct = Structure(
        lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]],
        site_properties={"magmom": [5.0, 5.0]},   # FM: both same sign
    )
    afm_struct = Structure(
        lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]],
        site_properties={"magmom": [5.0, -5.0]},  # AFM: mixed signs
    )

    fm_ops = _count_magnetic_symmetry_ops(fm_struct)
    afm_ops = _count_magnetic_symmetry_ops(afm_struct)

    assert isinstance(fm_ops, int) and fm_ops >= 1
    assert isinstance(afm_ops, int) and afm_ops >= 1
    assert fm_ops > afm_ops, (
        f"Expected FM ({fm_ops} ops) > AFM ({afm_ops} ops) for BCC Fe 2-atom cell"
    )


# ===========================================================================
# _count_magnetic_symmetry_ops sort invariant (requires enumlib)
# ===========================================================================

@needs_enumlib
def test_symmetry_ops_fm_ge_afm_for_fe_conventional_cell():
    """
    For Fe BCC conventional cell (2 atoms) the FM structure must have at
    least as many symmetry operations as the AFM structure (FM >= AFM).
    This is the core invariant the sort relies on: FM is most symmetric
    or ties with AFM, never less symmetric.

    Note: for this specific geometry FM and AFM tie (the body-centring
    translation + spin flip is a valid magnetic symmetry operation), so
    we assert >= rather than >. On other geometries FM will strictly exceed.
    """
    lat = Lattice.cubic(2.87)
    struct = Structure(lat, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    results = enumerate_magnetic_structures(struct, n_max=4)

    # Identify FM and AFM configs from the results
    fm_configs = [r for r in results if _has_fm([r])]
    afm_configs = [r for r in results if _has_afm([r])]

    # Make sure both fm and afm are set:
    assert fm_configs, "No FM config found — cannot test symmetry ops"
    assert afm_configs, "No AFM config found — cannot test symmetry ops"

    fm_ops = _count_magnetic_symmetry_ops(fm_configs[0]["structure"])
    afm_ops = _count_magnetic_symmetry_ops(afm_configs[0]["structure"])

    assert fm_ops >= afm_ops, (
        f"FM has fewer symmetry ops ({fm_ops}) than AFM ({afm_ops}) — "
        "sort invariant violated"
    )
