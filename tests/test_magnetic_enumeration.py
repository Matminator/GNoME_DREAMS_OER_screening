"""
Tests for gnome_dreams_oer_screening.vasp.magnetic_enumeration

Covers:
  - _extract_mag_config_index   (pure parser, no pymatgen needed)
  - enumerate_magnetic_structures  (mocks _get_primitive and
    MagneticStructureEnumerator so enumlib is never called)

TODO (integration-level tests to add later):
  - Run enumerate_magnetic_structures on a real pymatgen Structure for a
    known magnetic material (e.g. Fe bcc) and verify that the FM ordering
    is returned first and that all returned configs have the correct atom count.
  - Verify that _count_magnetic_symmetry_ops gives a higher count for FM
    than for a simple AFM structure on a known crystal.
"""

from unittest.mock import MagicMock, patch

import pytest
from pymatgen.core import Lattice, Structure

from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    _extract_mag_config_index,
    enumerate_magnetic_structures,
)

_MOD = "gnome_dreams_oer_screening.vasp.magnetic_enumeration"


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
    site.specie.spin = magmom
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
    # Basic parsing
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

    def test_returns_none_for_non_integer_value(self):
        assert _extract_mag_config_index("mag_config_index=abc") is None

    def test_returns_none_for_float_value(self):
        assert _extract_mag_config_index("mag_config_index=1.5") is None


# ---------------------------------------------------------------------------
# enumerate_magnetic_structures
# ---------------------------------------------------------------------------

class TestEnumerateMagneticStructures:
    """
    All tests patch both _get_primitive (returns the input unchanged) and
    MagneticStructureEnumerator so that enumlib is never called.

    Patching _get_primitive (a named module-level function) is simpler and
    more reliable than patching SpacegroupAnalyzer directly.
    """

    @staticmethod
    def _patch_primitive(structure):
        """Make _get_primitive a no-op: returns the input structure as-is."""
        return patch(f"{_MOD}._get_primitive", return_value=structure)

    @staticmethod
    def _make_mock_enumerator(configs: list[tuple]):
        """
        Build a mock MagneticStructureEnumerator from a list of
        (magmoms: list[float], origin: str) tuples.
        """
        mock_enum = MagicMock()
        mock_enum.ordered_structures = [
            _make_mag_structure(mm) for mm, _ in configs
        ]
        mock_enum.ordered_structure_origins = [orig for _, orig in configs]
        return mock_enum

    # -- NM fallback --------------------------------------------------------

    def test_returns_nm_fallback_when_enumerator_raises(self):
        """If MagneticStructureEnumerator raises, return one NM entry."""
        structure = _make_structure(2)
        with self._patch_primitive(structure):
            with patch(f"{_MOD}.MagneticStructureEnumerator",
                       side_effect=RuntimeError("enumlib not found")):
                result = enumerate_magnetic_structures(structure, n_max=4)

        assert len(result) == 1
        assert result[0]["ordering"] == "NM"
        assert result[0]["magmom"] == [0.0, 0.0]

    def test_returns_nm_fallback_when_enumerator_returns_empty(self):
        """If the enumerator returns zero candidates, return one NM entry."""
        structure = _make_structure(2)
        mock_enum = self._make_mock_enumerator([])
        with self._patch_primitive(structure):
            with patch(f"{_MOD}.MagneticStructureEnumerator",
                       return_value=mock_enum):
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
        mock_enum = MagicMock()
        mock_enum.ordered_structures = [
            _make_mag_structure([2.0, 2.0, -2.0, -2.0]),  # 4 atoms — supercell
            _make_mag_structure([2.0, -2.0]),              # 2 atoms — keep
        ]
        mock_enum.ordered_structure_origins = ["AFM_super", "AFM_same"]

        with self._patch_primitive(structure):
            with patch(f"{_MOD}.MagneticStructureEnumerator",
                       return_value=mock_enum):
                result = enumerate_magnetic_structures(structure, n_max=4)

        assert len(result) == 1
        assert result[0]["ordering"] == "AFM_same"

    def test_all_supercell_configs_triggers_nm_fallback(self):
        """If every candidate is a supercell, the NM fallback is returned."""
        structure = _make_structure(2)
        mock_enum = MagicMock()
        mock_enum.ordered_structures = [_make_mag_structure([2.0, 2.0, -2.0, -2.0])]
        mock_enum.ordered_structure_origins = ["AFM_super"]

        with self._patch_primitive(structure):
            with patch(f"{_MOD}.MagneticStructureEnumerator",
                       return_value=mock_enum):
                result = enumerate_magnetic_structures(structure, n_max=4)

        assert len(result) == 1
        assert result[0]["ordering"] == "NM"

    # -- n_max cap ----------------------------------------------------------

    def test_n_max_caps_number_of_returned_configs(self):
        """Only n_max candidates are returned."""
        structure = _make_structure(2)
        mock_enum = self._make_mock_enumerator([
            ([2.0, 2.0], "FM"),
            ([2.0, -2.0], "AFM_1"),
            ([-2.0, 2.0], "AFM_2"),
            ([0.0, 0.0], "NM"),
        ])
        with self._patch_primitive(structure):
            with patch(f"{_MOD}.MagneticStructureEnumerator",
                       return_value=mock_enum):
                result = enumerate_magnetic_structures(structure, n_max=2)

        assert len(result) == 2

    def test_n_max_1_returns_single_config(self):
        structure = _make_structure(2)
        mock_enum = self._make_mock_enumerator([
            ([2.0, 2.0], "FM"),
            ([2.0, -2.0], "AFM"),
        ])
        with self._patch_primitive(structure):
            with patch(f"{_MOD}.MagneticStructureEnumerator",
                       return_value=mock_enum):
                result = enumerate_magnetic_structures(structure, n_max=1)

        assert len(result) == 1

    # -- Return-value shape -------------------------------------------------

    def test_returned_dicts_have_required_keys(self):
        structure = _make_structure(2)
        mock_enum = self._make_mock_enumerator([([2.0, 2.0], "FM")])
        with self._patch_primitive(structure):
            with patch(f"{_MOD}.MagneticStructureEnumerator",
                       return_value=mock_enum):
                result = enumerate_magnetic_structures(structure)

        assert set(result[0].keys()) == {"structure", "magmom", "ordering"}

    def test_magmom_length_matches_atom_count(self):
        structure = _make_structure(2)
        mock_enum = self._make_mock_enumerator([([2.0, -2.0], "AFM")])
        with self._patch_primitive(structure):
            with patch(f"{_MOD}.MagneticStructureEnumerator",
                       return_value=mock_enum):
                result = enumerate_magnetic_structures(structure)

        assert len(result[0]["magmom"]) == 2
