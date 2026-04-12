"""
Tests for gnome_dreams_oer_screening.vasp.magnetic_enumeration

Covers:
  - _extract_mag_config_index   (pure parser, no pymatgen needed)
  - enumerate_magnetic_structures  (uses mock to avoid calling enumlib)

TODO (integration-level tests to add later):
  - Run enumerate_magnetic_structures on a real pymatgen Structure for a
    known magnetic material (e.g. Fe bcc) and verify that the FM ordering
    is returned first and that all returned configs have the correct atom count.
  - Verify that _count_magnetic_symmetry_ops gives a higher count for FM
    than for a simple AFM structure on a known crystal.
  - Test that the supercell filter actually drops any enumerator result
    whose atom count exceeds the input atom count.
"""

from unittest.mock import MagicMock, patch

import pytest
from pymatgen.core import Lattice, Structure

from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    _extract_mag_config_index,
    enumerate_magnetic_structures,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_structure(n_atoms: int = 2) -> Structure:
    """Return a minimal cubic Structure with n_atoms Fe atoms."""
    lat = Lattice.cubic(2.87)
    species = ["Fe"] * n_atoms
    coords = [[i / n_atoms, 0, 0] for i in range(n_atoms)]
    return Structure(lat, species, coords)


def _make_mag_site(symbol: str, magmom: float):
    """Return a mock Site-like object with .specie.symbol and .properties."""
    site = MagicMock()
    site.specie.symbol = symbol
    site.properties = {"magmom": magmom}
    site.frac_coords = [0.0, 0.0, 0.0]
    return site


def _make_mag_structure(magmoms: list[float], symbol: str = "Fe") -> MagicMock:
    """
    Return a mock Structure whose .sites list has the given per-atom magmoms.
    len(mock_structure) == len(magmoms).
    """
    struct = MagicMock()
    struct.sites = [_make_mag_site(symbol, m) for m in magmoms]
    struct.__len__ = lambda self: len(magmoms)
    struct.lattice = MagicMock()
    struct.lattice.matrix = [[2.87, 0, 0], [0, 2.87, 0], [0, 0, 2.87]]
    return struct


# ---------------------------------------------------------------------------
# _extract_mag_config_index
# ---------------------------------------------------------------------------

class TestExtractMagConfigIndex:
    # Basic parsing: index is extracted from expected token format
    def test_parses_index_at_start(self):
        assert _extract_mag_config_index("mag_config_index=0") == 0

    def test_parses_index_with_trailing_pipe(self):
        assert _extract_mag_config_index("mag_config_index=2 | ordering=FM") == 2

    def test_parses_index_with_leading_note(self):
        assert _extract_mag_config_index("some note | mag_config_index=3 | x") == 3

    # Should NOT pick up OH_init_index or other tokens
    def test_returns_none_for_oh_note(self):
        assert _extract_mag_config_index("OH_init_index=1") is None

    def test_returns_none_for_missing_token(self):
        assert _extract_mag_config_index("no token here") is None

    def test_returns_none_for_none_input(self):
        assert _extract_mag_config_index(None) is None

    def test_returns_none_for_non_string_input(self):
        assert _extract_mag_config_index(42) is None

    # Malformed values
    def test_returns_none_for_non_integer_value(self):
        assert _extract_mag_config_index("mag_config_index=abc") is None

    def test_returns_none_for_float_value(self):
        assert _extract_mag_config_index("mag_config_index=1.5") is None

    # TODO: decide whether negative indices should be accepted; currently
    # int("−1") succeeds and returns -1. Add a test if we add range validation.


# ---------------------------------------------------------------------------
# enumerate_magnetic_structures
# ---------------------------------------------------------------------------

class TestEnumerateMagneticStructures:
    """
    All tests patch MagneticStructureEnumerator so that enumlib is not
    required to run these tests in CI.
    """

    def _patch_enumerator(self, configs: list[tuple]):
        """
        configs: list of (magmoms, origin_str) — one entry per ordering
                 magmoms is a list of per-atom moments (float).
        """
        n_atoms = len(configs[0][0]) if configs else 2
        ordered_structures = [_make_mag_structure(mm) for mm, _ in configs]
        ordered_origins = [orig for _, orig in configs]

        mock_enum = MagicMock()
        mock_enum.ordered_structures = ordered_structures
        mock_enum.ordered_structure_origins = ordered_origins

        return patch(
            "gnome_dreams_oer_screening.vasp.magnetic_enumeration"
            ".MagneticStructureEnumerator",
            return_value=mock_enum,
        )

    # -- NM fallback --------------------------------------------------------

    def test_returns_nm_fallback_when_enumerator_raises(self):
        """If MagneticStructureEnumerator raises, return one NM entry."""
        structure = _make_structure(2)
        with patch(
            "gnome_dreams_oer_screening.vasp.magnetic_enumeration"
            ".MagneticStructureEnumerator",
            side_effect=RuntimeError("enumlib not found"),
        ):
            result = enumerate_magnetic_structures(structure, n_max=4)

        assert len(result) == 1
        assert result[0]["ordering"] == "NM"
        assert result[0]["magmom"] == [0.0, 0.0]

    def test_returns_nm_fallback_when_enumerator_returns_empty(self):
        """If the enumerator returns zero candidates, return one NM entry."""
        structure = _make_structure(2)
        with self._patch_enumerator([]):
            result = enumerate_magnetic_structures(structure, n_max=4)

        assert len(result) == 1
        assert result[0]["ordering"] == "NM"

    # -- Supercell filter ---------------------------------------------------

    def test_supercell_configs_are_dropped(self):
        """
        Enumerator returns one same-cell config (2 atoms) and one supercell
        config (4 atoms). Only the same-cell config should survive.
        """
        structure = _make_structure(2)

        # Patch enumerator manually so we can mix atom counts
        supercell_struct = _make_mag_structure([2.0, 2.0, -2.0, -2.0])
        same_cell_struct = _make_mag_structure([2.0, -2.0])

        mock_enum = MagicMock()
        mock_enum.ordered_structures = [supercell_struct, same_cell_struct]
        mock_enum.ordered_structure_origins = ["AFM_super", "AFM_same"]

        with patch(
            "gnome_dreams_oer_screening.vasp.magnetic_enumeration"
            ".MagneticStructureEnumerator",
            return_value=mock_enum,
        ):
            result = enumerate_magnetic_structures(structure, n_max=4)

        assert len(result) == 1
        assert result[0]["ordering"] == "AFM_same"

    def test_all_supercell_configs_triggers_nm_fallback(self):
        """If every candidate is a supercell, the NM fallback is returned."""
        structure = _make_structure(2)
        supercell = _make_mag_structure([2.0, 2.0, -2.0, -2.0])  # 4 atoms

        mock_enum = MagicMock()
        mock_enum.ordered_structures = [supercell]
        mock_enum.ordered_structure_origins = ["AFM_super"]

        with patch(
            "gnome_dreams_oer_screening.vasp.magnetic_enumeration"
            ".MagneticStructureEnumerator",
            return_value=mock_enum,
        ):
            result = enumerate_magnetic_structures(structure, n_max=4)

        assert len(result) == 1
        assert result[0]["ordering"] == "NM"

    # -- n_max cap ----------------------------------------------------------

    def test_n_max_caps_number_of_returned_configs(self):
        """Only the first n_max candidates (after sorting) are returned."""
        structure = _make_structure(2)
        # Provide 4 same-cell configs; request only 2
        configs = [
            ([2.0, 2.0], "FM"),
            ([2.0, -2.0], "AFM_1"),
            ([-2.0, 2.0], "AFM_2"),
            ([0.0, 0.0], "NM"),
        ]
        with self._patch_enumerator(configs):
            result = enumerate_magnetic_structures(structure, n_max=2)

        assert len(result) == 2

    def test_n_max_1_returns_single_config(self):
        configs = [([2.0, 2.0], "FM"), ([2.0, -2.0], "AFM")]
        structure = _make_structure(2)
        with self._patch_enumerator(configs):
            result = enumerate_magnetic_structures(structure, n_max=1)
        assert len(result) == 1

    # -- Return-value shape -------------------------------------------------

    def test_returned_dicts_have_required_keys(self):
        structure = _make_structure(2)
        configs = [([2.0, 2.0], "FM")]
        with self._patch_enumerator(configs):
            result = enumerate_magnetic_structures(structure)

        assert set(result[0].keys()) == {"structure", "magmom", "ordering"}

    def test_magmom_length_matches_atom_count(self):
        structure = _make_structure(2)
        configs = [([2.0, -2.0], "AFM")]
        with self._patch_enumerator(configs):
            result = enumerate_magnetic_structures(structure)

        assert len(result[0]["magmom"]) == 2

    # TODO: symmetry-sort tests
    # The current implementation sorts by _count_magnetic_symmetry_ops
    # (descending). A proper integration test on a real Fe structure should
    # verify that FM ends up at index 0. With mocks it is hard to drive
    # spglib, so skip for now and rely on the integration test in Verification
    # section of MAGNETIC_BULK_PLAN.md.
