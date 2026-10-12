# --- standard library imports ---
import copy
import pathlib
import shutil

import pandas as pd
import numpy as np
import pytest

from ase import Atoms
from ase.io import read

# --- IMPORTS FROM THIS PROJECT ---
import math

from pymatgen.io.ase import AseAtomsAdaptor

from gnome_dreams_oer_screening import (
    OER_catalyst_study,
    OER_adsorption_site_study,
    RPBE_relax_surface_set,
)
from gnome_dreams_oer_screening.oer.oer_study_utils import compute_slab_thickness
from gnome_dreams_oer_screening.vasp.magnetic_enumeration import (
    enumerate_magnetic_structures,
)
from gnome_dreams_oer_screening.oer.oer_surface_study import (
    OER_surface_study,
    SurfaceStudyLockedError,
)
from gnome_dreams_oer_screening.oer.atoms_sanity_checks import (
    StructureConsistencyError,
)


# ---------------------------------------------------------------------------
# Skip markers for environment-dependent tests
# ---------------------------------------------------------------------------

_GNOME_TRAJ = (
    pathlib.Path(__file__).parent.parent / "notebooks" / "test_cands.traj"
)
_needs_traj = pytest.mark.skipif(
    not _GNOME_TRAJ.exists(),
    reason="notebooks/test_cands.traj not found",
)

def _enumlib_available() -> bool:
    has_enum = (shutil.which("enum.x") is not None
                or shutil.which("multienum.x") is not None)
    has_make = (shutil.which("makestr.x") is not None
                or shutil.which("makeStr.py") is not None)
    return has_enum and has_make

_needs_enumlib = pytest.mark.skipif(
    not _enumlib_available(),
    reason="enumlib binaries (enum.x / makestr.x) not found on PATH",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finalize(study):
    """Set the relaxed bulk via the magnetic workflow using the init atoms."""
    init_atoms = study.get_init_atoms()
    study.store_mag_bulk_candidate(0, init_atoms, -1.0, [0.0] * len(init_atoms))
    study.finalize_mag_bulk_batch()


def _finalize_with_magmoms(study, magmoms: list[float]):
    """Finalize bulk using specific magmoms (must match len of init atoms)."""
    init_atoms = study.get_init_atoms()
    study.store_mag_bulk_candidate(0, init_atoms, -1.0, magmoms)
    study.finalize_mag_bulk_batch()


def _load_gnome_atoms(traj_index: int) -> Atoms:
    """Load a structure from the GNoME test trajectory as ASE Atoms."""
    return read(str(_GNOME_TRAJ), index=traj_index)


def _bulk_magmoms_by_element(bulk_structure) -> dict[str, list[float]]:
    """Map element symbol → list of all magmom values for that element in the bulk."""
    result: dict[str, list[float]] = {}
    magmoms = bulk_structure.site_properties.get("magmom", [])
    for site, m in zip(bulk_structure.sites, magmoms):
        result.setdefault(site.species_string, []).append(m)
    return result


def _assert_slab_magmoms_match_bulk(surface_study, bulk_structure) -> None:
    """
    Assert that every slab atom's magmom is one of the bulk values for its element.

    This is the core invariant: SlabGenerator must copy each bulk site's magmom
    to the corresponding surface site — no mixing across elements and no invented values.
    """

    # Verify the magmons are set (the non-relaxed):
    slab_magmom = surface_study.get_slab_magmom()
    assert slab_magmom is not None, (
        "get_slab_magmom() returned None — magmoms not propagated to slab"
    )

    # Magmoms, are of equal length the original slab (num atoms):
    assert len(slab_magmom) == len(surface_study.original_slab)

    bulk_by_elem = _bulk_magmoms_by_element(bulk_structure)
    for i, (site, m) in enumerate(
        zip(surface_study.original_slab.sites, slab_magmom)
    ):
        elem = site.species_string
        bulk_vals = bulk_by_elem.get(elem, [])
        # Guard: every slab element must appear in the bulk — if this fails
        # it is a test-setup error (e.g. bulk magmoms never written), not a
        # propagation bug, so give an explicit message.
        assert bulk_vals, (
            f"Slab atom {i} ({elem}): element not found in bulk magmom map — "
            "check that finalize_mag_bulk_batch wrote site_properties['magmom']"
        )
        # SlabGenerator copies site_properties by reference with no arithmetic,
        # so exact equality holds; 1e-6 tolerance absorbs only float repr noise.
        assert any(abs(m - v) < 1e-6 for v in bulk_vals), (
            f"Slab atom {i} ({elem}): magmom {m:.6f} not found among "
            f"bulk values for {elem}: {[f'{v:.6f}' for v in bulk_vals]}"
        )


class _MockCalc:
    """Minimal ASE calculator stub that returns pre-set magnetic moments."""
    def __init__(self, magmoms):
        self._magmoms = np.array(magmoms)
    def get_magnetic_moments(self, _atoms=None):
        return self._magmoms


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def TiO2_r_bulk():
    dir_path = pathlib.Path(__file__).parent
    TiO2_r_path = dir_path / '../notebooks/TiO2_rutile.cif'
    atoms = read(TiO2_r_path.resolve(), format='cif')
    if atoms is None:
        pytest.fail("Could not read TiO2 CIF")
    oer_study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )
    _finalize(oer_study)
    return atoms, oer_study


@pytest.fixture(scope="module")
def TiO2_r_init_surface_study(TiO2_r_bulk):
    atoms, study = TiO2_r_bulk

    study.predict_most_likely_surfaces(method='O-coordination')

    # From the study object:
    df = study.get_init_adsorption_sites_df(0, only_reduced_coord_O_sites=False)

    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]
    init_surface = surface_study.get_non_relaxed_surface()
    fake_energy = -278.13500000000005  # NOTE: needed for test_adsorbtion_update_adsorption_energies
    # require_magmom=False: init_surface has no VASP calculator (test mode opt-out)
    surface_study.set_relaxed_base_surface(
        init_surface, fake_energy, require_magmom=False
    )
    surface_study.determine_adsorption_sites(only_reduced_coord_O_sites=False)

    # From the surface study object:
    df_relaxed = surface_study.get_adsorption_sites_df()

    return surface_study, df, df_relaxed


# ---------------------------------------------------------------------------
# Adsorption site tests
# ---------------------------------------------------------------------------

def test_adsorption_site_dataframe_has_stable_column_name(
        TiO2_r_init_surface_study):
    surface_study, df_init_ads, df_relaxed_ads = TiO2_r_init_surface_study
    surface_study = copy.deepcopy(surface_study)

    # Test column names are in all the dfs:
    df = surface_study.get_adsorption_sites_df()
    assert isinstance(df, pd.DataFrame)
    assert "ad site element" in df.columns
    assert "add site element" not in df.columns

    assert isinstance(df_init_ads, pd.DataFrame)
    assert "ad site element" in df_init_ads.columns
    assert "add site element" not in df_init_ads.columns

    assert isinstance(df_relaxed_ads, pd.DataFrame)
    assert "ad site element" in df_relaxed_ads.columns
    assert "add site element" not in df_relaxed_ads.columns    


def test_update_adsorption_energies_skips_incomplete_studies(
                TiO2_r_init_surface_study):
    surface_study, _, _ = TiO2_r_init_surface_study
    surface_study = copy.deepcopy(surface_study)

    site_index = int(surface_study.get_adsorption_sites_df()["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(site_index)

    surface_study.update_adsorption_energies()
    row = surface_study.get_adsorption_sites_df().loc[site_index]

    assert pd.isna(row["G(O)"])
    assert pd.isna(row["G(O) deviation"])
    assert pd.isna(row["ideal overpotential"])


def test_adsorbtion_sites_setup(TiO2_r_init_surface_study):
    surface_study, df, df_relaxed = TiO2_r_init_surface_study

    surface_study = copy.deepcopy(surface_study)
    df = copy.deepcopy(df)
    df_relaxed = copy.deepcopy(df_relaxed)

    assert df['reduced coordination'].equals(df_relaxed['reduced coordination'])
    assert df['ad site element'].equals(df_relaxed['ad site element'])
    assert df['site type'].equals(df_relaxed['site type'])



def test_update_adsorption_energies_does_not_compute_overpotential_without_oh(
    TiO2_r_init_surface_study,
):
    surface_study, _, _ = TiO2_r_init_surface_study
    surface_study = copy.deepcopy(surface_study)
    
    study = surface_study.OER_catalyst_study

    df = surface_study.get_adsorption_sites_df()
    on_top_site_index = int(df[df["site type"] == "on-top"]["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(on_top_site_index)

    ad_study = surface_study.get_adsorption_site_studies_dict()[on_top_site_index]
    energies = ad_study.get_adsorption_site_energies()
    clean_energy = energies["clean_surface_energy"]
    assert clean_energy is not None

    target_g_o = 2.46
    constants = study.get_gas_and_energy_corrections()
    e_add = (
        clean_energy
        + target_g_o
        - constants["O_ad_correction"]
        - 2 * (constants["H2_gas_free_energy"] / 2)
        + constants["H2O_gas_free_energy"]
    )
    ad_study.set_relaxed_O_calc(ad_study.get_atoms_for_O_calc(),
                                energy=e_add,
                                require_magmom=False # Test mode
                                )

    surface_study.update_adsorption_energies()
    row = surface_study.get_adsorption_sites_df().loc[on_top_site_index]

    assert row["G(O)"] == pytest.approx(target_g_o, abs=1e-8)
    assert row["G(O) deviation"] == pytest.approx(0.0, abs=1e-8)
    assert pd.isna(row["G(OH)"])
    assert pd.isna(row["G(OH) deviation"])
    assert pd.isna(row["ideal overpotential"])
    assert pd.isna(row["G(OOH) from scaling relation"])
    assert pd.isna(row["G(OOH, scaling) deviation"])
    assert pd.isna(row["overpotential from OH-OOH scaling relation"])


def test_g_oh_formula_reference_value(TiO2_r_init_surface_study):
    """G(OH) formula numerical verification against a hand-computed reference.

    Reference values (TiO2 fixture: H2=-6.77, H2O=-14.217, OH_corr=0.361,
    slab_energy=-278.135):
        G(OH) = ΔE_OH + OH_corr + G_H - G_H₂O
              = ΔE_OH + 0.361 + (-3.385) - (-14.217) = ΔE_OH + 11.193
        For G(OH)=1.23: ΔE_OH = -9.963 → E(slab+OH) = -288.098
    """
    surface_study, _, _ = TiO2_r_init_surface_study
    surface_study = copy.deepcopy(surface_study)

    df = surface_study.get_adsorption_sites_df()
    on_top_idx = int(df[df["site type"] == "on-top"]["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(on_top_idx)
    ad_study = surface_study.get_adsorption_site_studies_dict()[on_top_idx]

    ad_study.set_relaxed_O_calc(
        ad_study.get_atoms_for_O_calc(), energy=-283.192, require_magmom=False
    )
    oh_init_atoms = ad_study.get_OH_adsorbed_atoms()
    ad_study.set_relaxed_OH_calc(oh_init_atoms, energies=[-288.098] * 3)

    surface_study.update_adsorption_energies()
    row = surface_study.get_adsorption_sites_df().loc[on_top_idx]

    assert row["G(OH)"] == pytest.approx(1.23, abs=1e-5)
    assert row["G(OH) deviation"] == pytest.approx(0.0, abs=1e-5)


def test_overpotentials_at_ideal_scaling_point(TiO2_r_init_surface_study):
    """Overpotential formulas verified at the ideal scaling point G(O)=2.46, G(OH)=1.23.

    At ideal scaling all four elementary steps are equal (1.23 eV each), so:
        ideal_overpotential                  = 0.00 V
        G(OOH) from scaling (b=3.2)          = 4.43 eV
        G(OOH, scaling) deviation            = |4.43 - 3.69| = 0.74 eV
        overpotential from scaling relation  = 0.74 V  (step 3 limits: ΔG₃=1.97)
    """
    surface_study, _, _ = TiO2_r_init_surface_study
    surface_study = copy.deepcopy(surface_study)

    df = surface_study.get_adsorption_sites_df()
    on_top_idx = int(df[df["site type"] == "on-top"]["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(on_top_idx)
    ad_study = surface_study.get_adsorption_site_studies_dict()[on_top_idx]

    ad_study.set_relaxed_O_calc(
        ad_study.get_atoms_for_O_calc(), energy=-283.192, require_magmom=False
    )
    oh_init_atoms = ad_study.get_OH_adsorbed_atoms()
    ad_study.set_relaxed_OH_calc(oh_init_atoms, energies=[-288.098] * 3)

    surface_study.update_adsorption_energies()
    row = surface_study.get_adsorption_sites_df().loc[on_top_idx]

    assert row["G(O)"]  == pytest.approx(2.46, abs=1e-5)
    assert row["G(OH)"] == pytest.approx(1.23, abs=1e-5)
    assert row["ideal overpotential"] == pytest.approx(0.00, abs=1e-5)
    assert row["G(OOH) from scaling relation"] == pytest.approx(4.43, abs=1e-5)
    assert row["G(OOH, scaling) deviation"] == pytest.approx(0.74, abs=1e-5)
    assert row["overpotential from OH-OOH scaling relation"] == pytest.approx(0.74, abs=1e-5)


def test_overpotentials_non_ideal_step2_limiting(TiO2_r_init_surface_study):
    """Overpotential formulas at a non-ideal point where different steps limit ideal vs scaling.

    G(O)=2.0, G(OH)=0.5, b=3.2:
        ideal:   max(0.5, 1.5, 1.46) - 1.23 = 0.27 V  (step 2 = ΔG_OH→O limits)
        scaling: G(OOH)=3.7, max(0.5, 1.5, 1.7, 1.22) - 1.23 = 0.47 V (step 3 limits)

    Reference energies:
        E_O  = -278.135 + (2.0  - 7.517) = -283.652
        E_OH = -278.135 + (0.5  - 11.193) = -288.828
    """
    surface_study, _, _ = TiO2_r_init_surface_study
    surface_study = copy.deepcopy(surface_study)

    df = surface_study.get_adsorption_sites_df()
    on_top_idx = int(df[df["site type"] == "on-top"]["Site index"].iloc[0])
    surface_study.initialize_adsorption_site_study(on_top_idx)
    ad_study = surface_study.get_adsorption_site_studies_dict()[on_top_idx]

    ad_study.set_relaxed_O_calc(
        ad_study.get_atoms_for_O_calc(), energy=-283.652, require_magmom=False
    )
    oh_init_atoms = ad_study.get_OH_adsorbed_atoms()
    ad_study.set_relaxed_OH_calc(oh_init_atoms, energies=[-288.828] * 3)

    surface_study.update_adsorption_energies()
    row = surface_study.get_adsorption_sites_df().loc[on_top_idx]

    assert row["G(O)"]  == pytest.approx(2.00, abs=1e-5)
    assert row["G(OH)"] == pytest.approx(0.50, abs=1e-5)
    assert row["ideal overpotential"] == pytest.approx(0.27, abs=1e-5)
    assert row["overpotential from OH-OOH scaling relation"] == pytest.approx(0.47, abs=1e-5)


def test_adsorbtion_sites_dict(TiO2_r_init_surface_study):
    surface_study, df, df_relaxed = TiO2_r_init_surface_study

    surface_study = copy.deepcopy(surface_study)

    i_inits = []
    for i in df_relaxed['Site index']:
        surface_study.initialize_adsorption_site_study(i)
        i_inits.append(i)

    ad_site_dict = surface_study.get_adsorption_site_studies_dict()
    assert set(i_inits) == set(ad_site_dict.keys())


def test_adsorbtion_sites_init(TiO2_r_init_surface_study):
    surface_study, df, df_relaxed = TiO2_r_init_surface_study

    surface_study = copy.deepcopy(surface_study)

    for i in df_relaxed['Site index']:
        surface_study.initialize_adsorption_site_study(i)

    ad_site_dict = surface_study.get_adsorption_site_studies_dict()

    for i in df_relaxed['Site index']:
        ad_site_study = ad_site_dict[i]
        assert isinstance(ad_site_study, OER_adsorption_site_study)

        init_atoms = ad_site_study.init_surface

        df_row = df_relaxed[df_relaxed['Site index'] == i].iloc[0]
        site_type = df_row['site type']
        if site_type == 'on-top':
            expected_element = df_row['ad site element']
        elif site_type == 'lattice O':
            expected_element = 'O'
        else:
            raise ValueError("Invalid site type in test dataframe:", site_type)

        found_site = False
        for atom in init_atoms:
            if atom.tag == 3:
                if atom.symbol == expected_element:
                    found_site = True
                    break

        assert found_site, (
            f'Failed to find expected site element {expected_element} '
            f'for site type {site_type} in adsorption site study with index {i}'
        )


def test_adsorbtion_update_O_adsorption_energies(TiO2_r_init_surface_study):
    surface_study, df, df_relaxed = TiO2_r_init_surface_study

    surface_study = copy.deepcopy(surface_study)

    for i, row in df_relaxed.iterrows():
        surface_study.initialize_adsorption_site_study(i)
        ad_site_study = surface_study.get_adsorption_site_studies_dict()[i]
        O_atoms = ad_site_study.get_atoms_for_O_calc()

        site_type = row['site type']
        if site_type == 'on-top':
            E = -283.19200000000006
        elif site_type == 'lattice O':
            E = -283.19200000000006 - 2 * (-5.057000000000016)
        else:
            raise ValueError("Invalid site type in test dataframe:", site_type)

        ad_site_study.set_relaxed_O_calc(O_atoms, energy=E, require_magmom=False)

    surface_study.update_adsorption_energies()
    df = surface_study.get_adsorption_sites_df()
    for i, row in df.iterrows():
        diff = row['G(O)'] - 2.46
        assert abs(diff) < 0.001, (
            f"Expected G(O) = 2.46, got {row['G(O)']} for site {i} of type {row['site type']}"
        )


# ---------------------------------------------------------------------------
# set_relaxed_base_surface error / locking tests
# ---------------------------------------------------------------------------

@pytest.mark.mini
def test_oer_surface_study_raises_for_non_slab_input(TiO2_r_bulk):
    """OER_surface_study.__init__ must reject non-Slab input (§3f guard)."""
    _, oer_study = TiO2_r_bulk
    for bad_input in (None, "not_a_slab", 42):
        with pytest.raises(ValueError, match="must be a Slab"):
            OER_surface_study(oer_study, bad_input)


def test_set_relaxed_base_surface_raises_structure_consistency_error(TiO2_r_bulk):
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(), H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )
    _finalize(study)
    study.predict_most_likely_surfaces(method='O-coordination')
    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]

    reordered_surface = surface_study.get_non_relaxed_surface()
    swap_index = next(
        i for i, atom in enumerate(reordered_surface)
        if atom.symbol != reordered_surface[0].symbol
    )
    reordered_indices = list(range(len(reordered_surface)))
    reordered_indices[0], reordered_indices[swap_index] = (
        reordered_indices[swap_index],
        reordered_indices[0],
    )
    reordered_surface = reordered_surface[reordered_indices]

    with pytest.raises(StructureConsistencyError, match="different symbols"):
        surface_study.set_relaxed_base_surface(
            reordered_surface, energy=-278.13500000000005, require_magmom=False
        )


def test_set_relaxed_base_surface_can_retry_after_missing_energy_error(TiO2_r_bulk):
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(), H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )
    _finalize(study)
    study.predict_most_likely_surfaces(method='O-coordination')

    termination_index = next(iter(study.get_terminations_dict()))
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]

    relaxed_surface = surface_study.get_non_relaxed_surface()

    # First attempt raises due to missing energy (no calculator):
    with pytest.raises(ValueError, match="Energy not provided and could not be obtained"):
        surface_study.set_relaxed_base_surface(relaxed_surface, require_magmom=False)

    success = surface_study.set_relaxed_base_surface(
        relaxed_surface, energy=-278.13500000000005, require_magmom=False
    )

    assert success is True
    assert surface_study.relaxed_surface_energy == -278.13500000000005
    assert surface_study.get_relaxed_surface() is not None


def test_set_relaxed_base_surface_locks_study_for_disconnected_surface(TiO2_r_bulk):
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(), H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )
    _finalize(study)
    study.predict_most_likely_surfaces(method='O-coordination')

    termination_index = next(iter(study.get_terminations_dict()))
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]

    # Make a disconnected surface:
    disconnected_surface = surface_study.get_non_relaxed_surface()
    disconnected_surface.set_pbc(False)
    disconnected_surface[-1].position += np.array([50.0, 50.0, 50.0])

    success = surface_study.set_relaxed_base_surface(
        disconnected_surface, energy=-278.13500000000005, require_magmom=False
    )

    assert success is False
    assert surface_study.study_locked is True
    assert surface_study.relaxed_surface_energy is None
    assert surface_study.get_relaxed_surface() is None

    # A later valid relaxed surface should unlock and be accepted:
    valid_surface = surface_study.get_non_relaxed_surface()
    retry_success = surface_study.set_relaxed_base_surface(
        valid_surface, energy=-278.13500000000005, require_magmom=False
    )

    assert retry_success is True
    assert surface_study.study_locked is False
    assert surface_study.relaxed_surface_energy == -278.13500000000005
    assert surface_study.get_relaxed_surface() == valid_surface


def test_determine_adsorption_sites_raises_specific_error_for_locked_study(TiO2_r_bulk):
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(), H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )
    _finalize(study)
    study.predict_most_likely_surfaces(method='O-coordination')

    termination_index = next(iter(study.get_terminations_dict()))
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]

    disconnected_surface = surface_study.get_non_relaxed_surface()
    disconnected_surface.set_pbc(False)
    disconnected_surface[-1].position += np.array([50.0, 50.0, 50.0])

    success = surface_study.set_relaxed_base_surface(
        disconnected_surface, energy=-278.13500000000005, require_magmom=False
    )
    assert success is False

    with pytest.raises(
        SurfaceStudyLockedError,
        match="surface study is locked due to a previously provided non-continuous relaxed surface",
    ):
        surface_study.determine_adsorption_sites()

    # A valid surface should unlock the study:
    success = surface_study.set_relaxed_base_surface(
        surface_study.get_non_relaxed_surface(), energy=10, require_magmom=False
    )
    assert success is True
    assert surface_study.study_locked is False

    # determine_adsorption_sites should now succeed:
    try:
        surface_study.determine_adsorption_sites()
    except SurfaceStudyLockedError:
        pytest.fail(
            "determine_adsorption_sites should not raise SurfaceStudyLockedError "
            "after providing a valid relaxed surface"
        )


# ---------------------------------------------------------------------------
# Surface magmom API tests
# ---------------------------------------------------------------------------

def test_get_slab_magmom_returns_list_of_correct_length(TiO2_r_init_surface_study):
    """get_slab_magmom() returns the bulk-propagated magmoms for every slab atom."""
    surface_study, _, _ = TiO2_r_init_surface_study
    magmom = surface_study.get_slab_magmom()
    assert magmom is not None
    assert len(magmom) == len(surface_study.original_slab)


def test_get_relaxed_surface_magmom_returns_none_without_calculator(
        TiO2_r_init_surface_study):
    """When the relaxed surface was set without a VASP calculator (require_magmom=False),
    get_relaxed_surface_magmom() returns None."""
    surface_study, _, _ = TiO2_r_init_surface_study
    assert surface_study.get_relaxed_surface_magmom() is None


def test_get_relaxed_surface_magmom_returns_values_when_set(TiO2_r_bulk):
    """Positive path: magmoms from a mock calculator are stored and returned correctly."""
    atoms, _ = TiO2_r_bulk
    study = OER_catalyst_study(
        atoms.copy(), H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )
    _finalize(study)
    study.predict_most_likely_surfaces(method='O-coordination')
    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]
    relaxed_surface = surface_study.get_non_relaxed_surface()
    n = len(relaxed_surface)
    fake_magmoms = [float(i % 3) + 0.5 for i in range(n)]  # non-trivial, non-zero
    relaxed_surface.calc = _MockCalc(fake_magmoms)
    ok = surface_study.set_relaxed_base_surface(relaxed_surface, energy=-278.0)
    assert ok is True
    result = surface_study.get_relaxed_surface_magmom()
    assert result is not None
    assert len(result) == n
    assert result == pytest.approx(fake_magmoms, abs=1e-6)


def test_set_relaxed_base_surface_raises_without_magmom_by_default(TiO2_r_bulk):
    """Default require_magmom=True raises ValueError when no calculator is attached."""
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(), H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )
    _finalize(study)
    study.predict_most_likely_surfaces(method='O-coordination')
    termination_index = next(iter(study.get_terminations_dict()))
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]
    init_surface = surface_study.get_non_relaxed_surface()  # no calculator

    with pytest.raises(ValueError, match="require_magmom"):
        surface_study.set_relaxed_base_surface(init_surface, energy=-278.0)


# ---------------------------------------------------------------------------
# Magmom propagation: bulk → surface
#
# These tests verify the full chain:
#   finalize_mag_bulk_batch (writes site_properties["magmom"] on bulk_structure)
#   → SlabGenerator (propagates site_properties to generated slabs)
#   → get_slab_magmom() (reads back from original_slab.site_properties["magmom"])
#
# Tests use element-specific sentinel values so that any mixing or loss of
# magmoms is unambiguously detectable. GNoME tests run the full
# enumerate_magnetic_structures → finalize_mag_bulk_batch → surface pipeline
# on real candidate structures from the GNoME database.
# ---------------------------------------------------------------------------

def test_magmom_propagates_from_bulk_to_surface_1(TiO2_r_bulk):
    """Element-specific magmoms set on the bulk structure must appear on every
    corresponding slab atom after SlabGenerator builds the surface.

    Mechanism: finalize_mag_bulk_batch writes magmoms into
    bulk_structure.site_properties["magmom"]; SlabGenerator propagates
    site_properties to each generated slab; get_slab_magmom() reads them back.
    """
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(), H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )

    # Assign distinct, non-trivial magmoms per element so propagation is unambiguous:
    Ti_MAG = 2.0
    O_MAG = 0.5
    init_atoms = study.get_init_atoms()
    magmoms = [Ti_MAG if a.symbol == 'Ti' else O_MAG for a in init_atoms]

    study.store_mag_bulk_candidate(0, init_atoms, -1.0, magmoms)
    study.finalize_mag_bulk_batch()

    study.predict_most_likely_surfaces(method='O-coordination')
    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]

    _assert_slab_magmoms_match_bulk(surface_study, study.get_bulk_structure())


def test_magmom_propagates_from_bulk_to_surface_2(TiO2_r_bulk):
    """Element-specific magmoms set on the bulk structure must appear on every
    corresponding slab atom after SlabGenerator builds the surface.

    Mechanism: finalize_mag_bulk_batch writes magmoms into
    bulk_structure.site_properties["magmom"]; SlabGenerator propagates
    site_properties to each generated slab; get_slab_magmom() reads them back.
    """
    atoms, _ = TiO2_r_bulk

    study = OER_catalyst_study(
        atoms.copy(), H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )

    # Assign distinct, non-trivial magmoms per element so propagation is unambiguous:
    Ti_MAG = 2.0
    O_MAG = 0.5
    init_atoms = study.get_init_atoms()
    magmoms = [Ti_MAG if a.symbol == 'Ti' else O_MAG for a in init_atoms]
    magmons = []
    for a in init_atoms:
        if a.symbol == 'Ti':
            magmons.append(Ti_MAG)
            Ti_MAG += 1 # Increment to make each Ti site distinct
        else:
            magmons.append(O_MAG)

    study.store_mag_bulk_candidate(0, init_atoms, -1.0, magmoms)
    study.finalize_mag_bulk_batch()

    study.predict_most_likely_surfaces(method='O-coordination')
    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]

    _assert_slab_magmoms_match_bulk(surface_study, study.get_bulk_structure())


@_needs_traj
def test_magmom_propagates_gnome_IrO4Pd_manually_assigned():
    """GNoME IrO4Pd (traj index 30, 6 atoms): element-specific sentinel magmoms
    set manually propagate correctly through the surface builder.
    Does NOT require enumlib — tests the propagation path with a real GNoME geometry.
    """
    atoms = _load_gnome_atoms(30)
    study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )
    init_atoms = study.get_init_atoms()
    # Assign a distinct sentinel per element
    elem_to_mag = {sym: (i + 1) * 0.7
                   for i, sym in enumerate(dict.fromkeys(a.symbol for a in init_atoms))}
    magmoms = [elem_to_mag[a.symbol] for a in init_atoms]
    _finalize_with_magmoms(study, magmoms)

    study.predict_most_likely_surfaces(method='O-coordination')

    term = sorted(study.terminations)[0]
    study.initialize_oer_surface_study(term)
    surface_study = study.get_surface_studies()[term]
    _assert_slab_magmoms_match_bulk(surface_study, study.get_bulk_structure())


@_needs_traj
@_needs_enumlib
def test_magmom_propagates_gnome_Fe2Ir2O12Rh2_enumerated():
    """GNoME Fe2Ir2O12Rh2 (traj index 8, 18 atoms): magmoms from the FM
    enumerated config propagate correctly to every slab atom.

    Fe is the magnetic element here. The FM config places the same positive
    value on all Fe sites. After propagation, every Fe atom in the slab must
    carry that value, and every non-magnetic element must carry its bulk value.
    """
    atoms = _load_gnome_atoms(8)
    study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )
    init_structure = study.get_init_structure()
    configs = enumerate_magnetic_structures(init_structure, n_max=4)

    # Use the first config (highest-symmetry — typically FM for Fe structures)
    best = configs[0]
    study.store_mag_bulk_candidate(0, study.get_init_atoms(), -1.0, best["magmom"])
    study.finalize_mag_bulk_batch()

    study.predict_most_likely_surfaces(max_miller=1)
    ranking_df = study.get_termination_rankings()
    valid_rows = ranking_df[ranking_df['Score'].notna()]
    assert len(valid_rows) >= 1, (
        "At least one termination must have a non-NaN score — a NaN score "
        "indicates no valid slab could be generated for any termination"
    )
    termination_index = int(valid_rows.iloc[0]['Termination index'])
    study.initialize_oer_surface_study(termination_index)
    surface_study = study.get_surface_studies()[termination_index]
    _assert_slab_magmoms_match_bulk(surface_study, study.get_bulk_structure())

    # Additionally verify that Fe atoms in the slab carry a non-trivial magmom
    slab_magmom = surface_study.get_slab_magmom()
    fe_magmoms = [
        m for site, m in zip(surface_study.original_slab.sites, slab_magmom)
        if site.species_string == 'Fe'
    ]
    assert fe_magmoms, "No Fe atoms found in slab — check structure"
    assert any(abs(m) > 0.1 for m in fe_magmoms), (
        f"Fe atoms in slab all have near-zero magmom: {fe_magmoms}"
    )


@_needs_traj
@_needs_enumlib
def test_magmom_propagates_gnome_Ag3BiCoIr3O14_co_only_magnetic():
    """GNoME Ag3BiCoIr3O14 (traj index 6, 22 atoms): Co is the only magnetic
    element. After enumeration the FM config places a non-zero magmom on Co
    and ~0 on all others. Propagation must preserve this.
    """
    atoms = _load_gnome_atoms(6)
    study = OER_catalyst_study(
        atoms, H2_gas_free_energy=-6.77, H2O_gas_free_energy=-14.217,
    )
    init_structure = study.get_init_structure()
    configs = enumerate_magnetic_structures(init_structure, n_max=4)

    best = configs[0]
    study.store_mag_bulk_candidate(0, study.get_init_atoms(), -1.0, best["magmom"])
    study.finalize_mag_bulk_batch()

    study.predict_most_likely_surfaces()

    study.initialize_oer_surface_study(0)
    surface_study = study.get_surface_studies()[0]
    _assert_slab_magmoms_match_bulk(surface_study, study.get_bulk_structure())

    # Co is the sole magnetic element — verify it carries a non-trivial magmom in the slab
    slab_magmom = surface_study.get_slab_magmom()
    co_magmoms = [
        m for site, m in zip(surface_study.original_slab.sites, slab_magmom)
        if site.species_string == 'Co'
    ]
    if co_magmoms:  # Co may not appear in every termination
        assert any(abs(m) > 0.1 for m in co_magmoms), (
            f"Co atoms in slab have near-zero magmom: {co_magmoms} — "
            "expected non-trivial value from FM enumeration"
        )


def test_pipeline_slab_kpoints_vacuum_direction(TiO2_r_init_surface_study):
    """Surface KPOINTS from a real pipeline slab must have k_c=1 and c must be the
    vacuum direction (verified by comparing slab atom thickness to cell c-length).

    Checks:
    1. The c-direction of the slab cell contains substantial vacuum (> 5 Å).
    2. RPBE_relax_surface_set applied to that slab produces KPOINTS with k_c = 1.
    3. In-plane k-points (a, b) match the ceil(30/lattice_param) formula.
    """
    surface_study, _, _ = TiO2_r_init_surface_study
    atoms = surface_study.get_non_relaxed_surface()  # ASE Atoms with vacuum along c

    # Confirm c is the vacuum direction
    cell = atoms.get_cell()
    c_vec = cell[2, :]
    c_length = np.linalg.norm(c_vec)
    c_normal = c_vec / c_length

    thickness = compute_slab_thickness(atoms, normal_vector=c_normal)
    vacuum = c_length - thickness
    assert vacuum > 5.0, (
        f"Expected > 5 Å vacuum in the c-direction, got {vacuum:.2f} Å. "
        "If this fails the slab is not oriented with vacuum along c."
    )

    # Check KPOINTS
    structure = AseAtomsAdaptor.get_structure(atoms)
    surface_set = RPBE_relax_surface_set(structure)
    kpts = surface_set.CONFIG["KPOINTS"]

    k_a, k_b, k_c = kpts.kpts[0]
    assert k_c == 1, (
        f"Surface KPOINTS must have 1 k-point in the vacuum (c) direction, got {k_c}"
    )
    a_len = np.linalg.norm(cell[0, :])
    b_len = np.linalg.norm(cell[1, :])
    assert k_a == math.ceil(30 / a_len), f"k_a mismatch: expected {math.ceil(30 / a_len)}, got {k_a}"
    assert k_b == math.ceil(30 / b_len), f"k_b mismatch: expected {math.ceil(30 / b_len)}, got {k_b}"
