from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest

from gnome_dreams_oer_screening import Explog
from gnome_dreams_oer_screening.vasp.job_validation import (
    ValidationResult,
    _extract_oh_init_index,
    _fail,
    _pass,
    validate_enter_process,
    validate_submit_process,
)

pytestmark = pytest.mark.mini
_DEFAULT_OH_CANDIDATES = object()


# *** Set of dummy classes to mimic the study object classes ***********
class DummyAdStudy:
    def __init__(
        self,
        relaxed_O_atoms=None,
        oh_adsorbed_atoms=_DEFAULT_OH_CANDIDATES,
    ):
        self.relaxed_O_atoms = relaxed_O_atoms
        if oh_adsorbed_atoms is _DEFAULT_OH_CANDIDATES:
            oh_adsorbed_atoms = [object()]
        self._oh_adsorbed_atoms = oh_adsorbed_atoms

    def get_OH_adsorbed_atoms(self):
        return self._oh_adsorbed_atoms

    def set_oh_batch_finalized(self, _value: bool):
        return None


class DummySurfaceStudy:
    def __init__(self, base_surface_relaxed=None, adsite_studies=None):
        self.base_surface_relaxed = base_surface_relaxed
        self._adsite_studies = {} if adsite_studies is None \
            else adsite_studies

    def get_adsorption_site_studies_dict(self):
        return self._adsite_studies

    def get_relaxed_surface(self):
        return self.base_surface_relaxed


class DummySurfaceStudyAdsitesNone:
    def __init__(self, base_surface_relaxed=None):
        self.base_surface_relaxed = base_surface_relaxed

    def get_adsorption_site_studies_dict(self):
        return None

    def get_relaxed_surface(self):
        return self.base_surface_relaxed


class DummyStudy:
    def __init__(self, bulk_atoms=None, surface_studies=None):
        self._bulk_atoms = bulk_atoms
        self._surface_studies = surface_studies or {}

    def get_bulk_atoms(self):
        return self._bulk_atoms

    def get_surface_studies(self):
        return self._surface_studies 


class DummyStudySurfaceStudiesNone:
    def __init__(self, bulk_atoms=None):
        self._bulk_atoms = bulk_atoms

    def get_bulk_atoms(self):
        return self._bulk_atoms

    def get_surface_studies(self):
        return None


def _make_adsorption_study(
    *,
    adsite_key: int = 3,
    relaxed_O=None,
    oh_adsorbed_atoms=_DEFAULT_OH_CANDIDATES,
    base_surface_relaxed=None,
) -> DummyStudy:
    ad_study = DummyAdStudy(
        relaxed_O_atoms=relaxed_O,
        oh_adsorbed_atoms=oh_adsorbed_atoms,
    )
    surface_study = DummySurfaceStudy(
        base_surface_relaxed=base_surface_relaxed,
        adsite_studies={adsite_key: ad_study},
    )
    return DummyStudy(surface_studies={0: surface_study})   
# **********************************************************************


@pytest.fixture()
def local_explog_dir() -> Path:
    base = Path(".pytest_tmp_validate_job")
    base.mkdir(exist_ok=True)
    path = base / f"test_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    rmtree(path, ignore_errors=True)


@pytest.fixture()
def explog(local_explog_dir: Path):
    log = Explog()
    log.init(
        vasp_calculation_dir=local_explog_dir,
        mode="test",
        prefix_vasp_dirs=False,
    )
    return log


def _add_process_row(
    explog: Explog,
    process_id: int,
    candidate_id: str,
    job_type: str,
    status: str,
    termination_index=None,
    site_index=None,
    process_note: str | None = None,
    vasp_dir: str | None = None,
) -> None:
    if vasp_dir is None:
        vasp_dir = str(Path(".") / f"calc_{process_id}")
    explog.relational_frame.processes.add_row(
        {
            "process_id": process_id,
            "candidate_id": candidate_id,
            "job_type": job_type,
            "status": status,
            "termination_index": termination_index,
            "site_index": site_index,
            "processNote": process_note,
            "VASP_dir": vasp_dir,
        }
    )


# Test that ValidationResult raises error when called on failed result:
def test_validation_result_raise_for_error_uses_explicit_exception():
    result = ValidationResult(
        ok=False,
        message="bad input",
        error_type=ValueError,
    )
    with pytest.raises(ValueError, match="bad input"):
        result.raise_for_error()


# Test that ValidationResult raises default RuntimeError:
def test_validation_result_raise_for_error_uses_default_runtime_error():
    result = ValidationResult(ok=False, code="fallback_code")
    with pytest.raises(RuntimeError, match="fallback_code"):
        result.raise_for_error()


# test correct data is stored in _pass and _fail results:
def test_pass_and_fail_helpers_return_expected_result_shapes():
    ok = _pass(a=1)
    assert ok.ok is True
    assert ok.context["a"] == 1

    bad = _fail("x", "msg", RuntimeError, k=2)
    assert bad.ok is False
    assert bad.code == "x"
    assert bad.message == "msg"
    assert bad.error_type is RuntimeError
    assert bad.context["k"] == 2


# Test _extract_oh_init_index correctly parses index from processNote:
def test_extract_oh_init_index_parses_expected_token_and_pipe():
    assert _extract_oh_init_index("note | OH_init_index=2 | x") == 2


# Test _extract_oh_init_index returns None when token is missing:
def test_extract_oh_init_index_returns_none_for_missing_or_wrong_type():
    assert _extract_oh_init_index("no token") is None
    assert _extract_oh_init_index(None) is None


# Test _extract_oh_init_index returns None for non-integer values:
def test_extract_oh_init_index_returns_none_for_non_integer_token():
    assert _extract_oh_init_index("OH_init_index=abc") is None


# *** VALIDATE ENTER PROCESS TESTS *************************************
# Test candidate_not_found is returned when candidate_id is not in log:
def test_validate_enter_process_returns_candidate_not_found_for_missing_id(
    explog: Explog,
):
    result = validate_enter_process(
        explog.relational_frame,
        candidate_id="missing",
        job_type="bulk_relaxation",
    )
    assert result.ok is False
    assert result.code == "candidate_not_found"
    assert result.error_type is KeyError


# Test that failed duplicates are allowed by default:
def test_validate_enter_process_allows_failed_duplicate_by_default(
    explog: Explog,
):
    candidate_id = "cand_failed_allowed"
    explog.add_candidate(candidate_id, DummyStudy())
    _add_process_row(
        explog=explog,
        process_id=0,
        candidate_id=candidate_id,
        job_type="bulk_relaxation",
        status="failed",
    )

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="bulk_relaxation",
    )
    assert result.ok is True


# Test that failed duplicates rejected are if reject_if_failed_exists:
def test_validate_enter_process_rejects_failed_duplicate_in_strict_mode(
    explog: Explog,
):
    candidate_id = "cand_failed_blocked"
    explog.add_candidate(candidate_id, DummyStudy())
    _add_process_row(
        explog=explog,
        process_id=0,
        candidate_id=candidate_id,
        job_type="bulk_relaxation",
        status="failed",
    )

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="bulk_relaxation",
        reject_if_failed_exists=True,
    )
    assert result.ok is False
    assert result.code == "duplicate_process_including_failed"


# Test that bulk relaxations cannot be entered with termination indices:
def test_validate_enter_process_rejects_bulk_with_indices(
        explog: Explog):
    candidate_id = "cand_bulk_idx"
    explog.add_candidate(candidate_id, DummyStudy())

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="bulk_relaxation",
        termination_index=0,
    )
    assert result.ok is False
    assert result.code == "bulk_indices_not_allowed"


# Test bulk relaxation cannot be entered if bulk atoms already set:
def test_validate_enter_process_rejects_bulk_if_already_relaxed(
    explog: Explog,
):
    candidate_id = "cand_bulk_set"
    explog.add_candidate(candidate_id, DummyStudy(bulk_atoms=object()))

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="bulk_relaxation",
    )
    assert result.ok is False
    assert result.code == "bulk_already_set"


# Test surface relaxations cannot be entered without termination index:
def test_validate_enter_process_rejects_surface_without_termination(
    explog: Explog,
):
    candidate_id = "cand_surf_no_term"
    explog.add_candidate(candidate_id, DummyStudy())

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="surface_relaxation",
    )
    assert result.ok is False
    assert result.code == "surface_missing_termination_index"


# Test that surface relaxations cannot be entered with site indices:
def test_validate_enter_process_rejects_surface_with_site_index(
    explog: Explog,
):
    candidate_id = "cand_surf_site"
    explog.add_candidate(candidate_id, 
                        DummyStudy(surface_studies={0: object()}))

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="surface_relaxation",
        termination_index=0,
        site_index=1,
    )
    assert result.ok is False
    assert result.code == "surface_site_index_not_allowed"


# Test surface relaxations cannot be entered if surface study not init:
def test_validate_enter_process_rejects_surface_if_study_not_initialized(
    explog: Explog,
):
    candidate_id = "cand_surf_missing"
    explog.add_candidate(candidate_id, DummyStudy(surface_studies={}))

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="surface_relaxation",
        termination_index=0,
    )
    assert result.ok is False
    assert result.code == "surface_study_not_initialized"


# Test surface relaxations cannot be entered if surface studies is None:
def test_validate_enter_process_rejects_surface_when_studies_none(
    explog: Explog,
):
    candidate_id = "cand_surf_studies_none"
    explog.add_candidate(candidate_id, DummyStudySurfaceStudiesNone())

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="surface_relaxation",
        termination_index=0,
    )
    assert result.ok is False
    assert result.code == "surface_studies_not_initialized"


# Test that no surface relaxation can be entered if surface already set:
def test_validate_enter_process_rejects_surface_if_relaxed_already_set(
    explog: Explog,
):
    candidate_id = "cand_surf_set"
    surface = DummySurfaceStudy(base_surface_relaxed=object())
    explog.add_candidate(candidate_id, 
                         DummyStudy(surface_studies={0: surface}))

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="surface_relaxation",
        termination_index=0,
    )
    assert result.ok is False
    assert result.code == "surface_already_relaxed"


# Test that O relaxation cannot be entered if adsite study not inited:
def test_validate_enter_process_rejects_adsorption_when_site_missing(
    explog: Explog,
):
    candidate_id = "cand_ad_missing_site"
    surface = DummySurfaceStudy(adsite_studies={})
    explog.add_candidate(candidate_id, 
                         DummyStudy(surface_studies={0: surface}))

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="O_adsorption",
        termination_index=0,
        site_index=3,
    )
    assert result.ok is False
    assert result.code == "adsite_study_not_initialized"


# Test that O relaxation cannot be entered if adsite studies is None:
def test_validate_enter_process_rejects_adsorption_when_adsites_none(
    explog: Explog,
):
    candidate_id = "cand_ad_adsites_none"
    surface = DummySurfaceStudyAdsitesNone()
    explog.add_candidate(
        candidate_id,
        DummyStudy(surface_studies={0: surface}),
    )

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="O_adsorption",
        termination_index=0,
        site_index=3,
    )
    assert result.ok is False
    assert result.code == "adsite_studies_not_initialized"


# Test that O relaxation cannot be entered if adsite studies is None:
def test_validate_enter_process_rejects_adsorption_when_studies_none(
    explog: Explog,
):
    candidate_id = "cand_ad_studies_none"
    explog.add_candidate(candidate_id, DummyStudySurfaceStudiesNone())

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="O_adsorption",
        termination_index=0,
        site_index=3,
    )
    assert result.ok is False
    assert result.code == "surface_studies_not_initialized"


# Test that O relaxation cannot be entered if O already relaxed/set:
def test_validate_enter_process_rejects_o_adsorption_if_o_already_set(
    explog: Explog,
):
    candidate_id = "cand_o_set"
    study = _make_adsorption_study(relaxed_O=object())
    explog.add_candidate(candidate_id, study)

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="O_adsorption",
        termination_index=0,
        site_index=3,
    )
    assert result.ok is False
    assert result.code == "o_already_relaxed"


# Test that OH cannot be entered if relaxed O is required but not set:
def test_validate_enter_process_rejects_oh_when_o_required_but_missing(
    explog: Explog,
):
    candidate_id = "cand_oh_requires_o"
    study = _make_adsorption_study(relaxed_O=None)
    explog.add_candidate(candidate_id, study)

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        termination_index=0,
        site_index=3,
        require_relaxed_o_for_oh=True,
    )
    assert result.ok is False
    assert result.code == "oh_requires_relaxed_o"


# Test that OH cannot be re-entered when a non-failed duplicate exists:
def test_validate_enter_process_rejects_second_oh_batch_after_submission(
    explog: Explog,
):
    candidate_id = "cand_oh_submitted"
    study = _make_adsorption_study(relaxed_O=object())
    explog.add_candidate(candidate_id, study)

    # Add process row, which is attempted to be validated again below:
    _add_process_row(
        explog=explog,
        process_id=0,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        status="completed",
        termination_index=0,
        site_index=3,
    )

    # Test if the same process can be entered again (it should fail)
    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        termination_index=0,
        site_index=3,
    )
    assert result.ok is False
    assert result.code == "duplicate_non_failed_process"


# Test that an valied OH job is accepted:
def test_validate_enter_process_returns_ok_for_valid_oh_case(
    explog: Explog,
):
    candidate_id = "cand_oh_ok"
    study = _make_adsorption_study(relaxed_O=object())
    explog.add_candidate(candidate_id, study)

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        termination_index=0,
        site_index=3,
        require_relaxed_o_for_oh=True,
    )
    assert result.ok is True
    assert "ad_study" in result.context


# Test O calc is rejected if termination index is missing:
def test_validate_enter_process_rejects_adsorption_without_termination_index(
    explog: Explog,
):
    candidate_id = "cand_ads_missing_term"
    study = _make_adsorption_study(relaxed_O=object())
    explog.add_candidate(candidate_id, study)

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="O_adsorption",
        termination_index=None,
        site_index=3,
    )
    assert result.ok is False
    assert result.code == "ads_missing_termination_index"


# Test OH calc is rejected if site index is missing:
def test_validate_enter_process_rejects_adsorption_without_site_index(
    explog: Explog,
):
    candidate_id = "cand_ads_missing_site"
    study = _make_adsorption_study(relaxed_O=object())
    explog.add_candidate(candidate_id, study)

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        termination_index=0,
        site_index=None,
    )
    assert result.ok is False
    assert result.code == "ads_missing_site_index"


# Test non-valid job type is rejected:
def test_validate_enter_process_rejects_unsupported_job_type(
    explog: Explog,
):
    candidate_id = "cand_bad_job_type"
    explog.add_candidate(candidate_id, DummyStudy())

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="not_a_real_job_type",
    )
    assert result.ok is False
    assert result.code == "unsupported_job_type"


# Test that valid bulk relaxation is accepted:
def test_validate_enter_process_returns_ok_for_valid_bulk_case(
    explog: Explog,
):
    candidate_id = "cand_bulk_ok"
    explog.add_candidate(candidate_id, DummyStudy())

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="bulk_relaxation",
    )
    assert result.ok is True
    assert "study_obj" in result.context


# Test that valid surface relaxation is accepted:
def test_validate_enter_process_returns_ok_for_valid_surface_case(
    explog: Explog,
):
    candidate_id = "cand_surface_ok"
    surface = DummySurfaceStudy(base_surface_relaxed=None)
    explog.add_candidate(
        candidate_id,
        DummyStudy(surface_studies={0: surface}),
    )

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="surface_relaxation",
        termination_index=0,
    )
    assert result.ok is True
    assert "surface_study" in result.context


# Test that valid O adsorption is accepted:
def test_validate_enter_process_returns_ok_for_valid_o_adsorption_case(
    explog: Explog,
):
    candidate_id = "cand_o_ok"
    study = _make_adsorption_study(relaxed_O=None)
    explog.add_candidate(candidate_id, study)

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="O_adsorption",
        termination_index=0,
        site_index=3,
    )
    assert result.ok is True
    assert "ad_study" in result.context


# Test that valid OH adsorption is accepted when relaxed O not required:
def test_validate_enter_process_allows_oh_without_relaxed_o_when_optional(
    explog: Explog,
):
    candidate_id = "cand_oh_optional_o_guard"
    study = _make_adsorption_study(relaxed_O=None)
    explog.add_candidate(candidate_id, study)

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        termination_index=0,
        site_index=3,
        require_relaxed_o_for_oh=False,
    )
    assert result.ok is True
    assert "ad_study" in result.context


# Test job is O rejected if termination missing from surface study dict:
def test_validate_enter_process_rejects_adsorption_when_termination_missing(
    explog: Explog,
):
    candidate_id = "cand_ads_term_missing_nonempty"
    surface = DummySurfaceStudy(adsite_studies={3: DummyAdStudy()})
    explog.add_candidate(
        candidate_id,
        DummyStudy(surface_studies={1: surface}),
    )

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="O_adsorption",
        termination_index=0,
        site_index=3,
    )
    assert result.ok is False
    assert result.code == "surface_study_not_initialized"


# Test duplicate processes is rejected when failed duplicates allowed:
def test_validate_enter_process_rejects_non_failed_duplicate_in_strict_mode(
    explog: Explog,
):
    candidate_id = "cand_nonfailed_strict_duplicate"
    study = _make_adsorption_study(relaxed_O=object())
    explog.add_candidate(candidate_id, study)
    _add_process_row(
        explog=explog,
        process_id=0,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        status="running",
        termination_index=0,
        site_index=3,
    )

    result = validate_enter_process(
        explog.relational_frame,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        termination_index=0,
        site_index=3,
        reject_if_failed_exists=True,
    )
    assert result.ok is False
    assert result.code == "duplicate_process_including_failed"
# **********************************************************************

# *** Test can_add_process wrapper in Explog ***************************
# Test non-mutating Explog wrapper around validate_enter_process:
def test_explog_can_add_process_returns_validation_result_without_mutation(
    explog: Explog,
):
    candidate_id = "cand_can_add_ok"
    explog.add_candidate(candidate_id, DummyStudy())
    n_before = len(explog.relational_frame.processes.df)

    result = explog.can_add_process(
        candidate_id=candidate_id,
        calculation_type="bulk_relaxation",
    )

    n_after = len(explog.relational_frame.processes.df)
    assert result.ok is True
    assert n_before == n_after


# Test Explog wrapper returns validator failure for missing candidate:
def test_explog_can_add_process_returns_candidate_not_found(
    explog: Explog,
):
    result = explog.can_add_process(
        candidate_id="missing_candidate",
        calculation_type="bulk_relaxation",
    )
    assert result.ok is False
    assert result.code == "candidate_not_found"


# Test enter_process fails cleanly when surface studies are missing:
def test_add_process_rejects_surface_when_surface_studies_none(
    explog: Explog,
):
    candidate_id = "cand_enter_surface_none"
    explog.add_candidate(candidate_id, DummyStudySurfaceStudiesNone())

    with pytest.raises(ValueError, match="No initialized surface studies"):
        explog.add_process(
            candidate_id=candidate_id,
            calculation_type="surface_relaxation",
            termination_index=0,
        )


# Test enter_process fails cleanly when adsite studies dict is None:
def test_add_process_rejects_adsorption_when_adsite_studies_none(
    explog: Explog,
):
    candidate_id = "cand_enter_adsite_none"
    surface = DummySurfaceStudyAdsitesNone()
    explog.add_candidate(
        candidate_id,
        DummyStudy(surface_studies={0: surface}),
    )

    with pytest.raises(
        ValueError,
        match="No initialized adsorption-site studies",
    ):
        explog.add_process(
            candidate_id=candidate_id,
            calculation_type="O_adsorption",
            termination_index=0,
            site_index=3,
        )


def test_add_process_allows_oh_reentry_after_failed_by_default(
    explog: Explog,
):
    candidate_id = "cand_oh_retry_default"
    study = _make_adsorption_study(relaxed_O=object())
    explog.add_candidate(candidate_id, study)
    _add_process_row(
        explog=explog,
        process_id=0,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        status="failed",
        termination_index=0,
        site_index=3,
    )

    process_ids = explog.add_process(
        candidate_id=candidate_id,
        calculation_type="OH_adsorption",
        termination_index=0,
        site_index=3,
    )
    assert isinstance(process_ids, list)
    assert len(process_ids) == 1


def test_add_process_rejects_oh_reentry_after_failed_in_strict_mode(
    local_explog_dir: Path,
):
    explog = Explog()
    explog.init(
        vasp_calculation_dir=local_explog_dir,
        mode="test",
        prefix_vasp_dirs=False,
        reject_if_failed_exists=True,
    )
    candidate_id = "cand_oh_retry_strict"
    study = _make_adsorption_study(relaxed_O=object())
    explog.add_candidate(candidate_id, study)
    _add_process_row(
        explog=explog,
        process_id=0,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        status="failed",
        termination_index=0,
        site_index=3,
    )

    with pytest.raises(RuntimeError, match="OH_adsorption"):
        explog.add_process(
            candidate_id=candidate_id,
            calculation_type="OH_adsorption",
            termination_index=0,
            site_index=3,
        )
# **********************************************************************


# *** VALIDATE SUBMIT PROCESS TESTS ************************************
# Test that submit validation fails cleanly if process candidate is missing:
def test_validate_submit_process_rejects_when_candidate_missing(
    explog: Explog,
):
    _add_process_row(
        explog=explog,
        process_id=0,
        candidate_id="missing_candidate",
        job_type="bulk_relaxation",
        status="un-submitted",
    )

    result = validate_submit_process(
        explog.relational_frame,
        process_id=0,
    )
    assert result.ok is False
    assert result.code == "candidate_not_found"
    assert result.error_type is KeyError


# Test that submit validation fails if adsite studies dict is None:
def test_validate_submit_process_rejects_when_adsite_studies_none(
    explog: Explog,
):
    candidate_id = "cand_submit_adsite_none"
    surface = DummySurfaceStudyAdsitesNone()
    explog.add_candidate(
        candidate_id,
        DummyStudy(surface_studies={0: surface}),
    )
    _add_process_row(
        explog=explog,
        process_id=0,
        candidate_id=candidate_id,
        job_type="O_adsorption",
        status="un-submitted",
        termination_index=0,
        site_index=3,
    )

    result = validate_submit_process(
        explog.relational_frame,
        process_id=0,
    )
    assert result.ok is False
    assert result.code == "adsite_studies_not_initialized"


# Test that submit validation fails if OH candidates are None:
def test_validate_submit_process_rejects_oh_when_candidates_none(
    explog: Explog,
):
    candidate_id = "cand_submit_oh_no_candidates"
    study = _make_adsorption_study(
        relaxed_O=object(),
        oh_adsorbed_atoms=None,
    )
    explog.add_candidate(candidate_id, study)
    _add_process_row(
        explog=explog,
        process_id=0,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        status="un-submitted",
        termination_index=0,
        site_index=3,
        process_note="OH_init_index=0",
    )

    result = validate_submit_process(
        explog.relational_frame,
        process_id=0,
    )
    assert result.ok is False
    assert result.code == "oh_no_candidates"


# Test malformed OH_init_index token is handled as validation failure:
def test_validate_submit_process_rejects_oh_with_non_integer_index_token(
    explog: Explog,
):
    candidate_id = "cand_submit_bad_oh_idx"
    study = _make_adsorption_study(
        relaxed_O=object(),
        oh_adsorbed_atoms=[object()],
    )
    explog.add_candidate(candidate_id, study)
    _add_process_row(
        explog=explog,
        process_id=0,
        candidate_id=candidate_id,
        job_type="OH_adsorption",
        status="un-submitted",
        termination_index=0,
        site_index=3,
        process_note="OH_init_index=abc",
    )

    result = validate_submit_process(
        explog.relational_frame,
        process_id=0,
    )
    assert result.ok is False
    assert result.code == "oh_init_index_missing"
# **********************************************************************
