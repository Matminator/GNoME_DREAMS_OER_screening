import pytest
import pandas as pd

from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from gnome_dreams_oer_screening import Explog


pytestmark = pytest.mark.mini


class DummyStudy:
    def get_bulk_atoms(self):
        return None

    def get_surface_studies(self):
        return {}


@pytest.fixture()
def local_explog_dir() -> Path:
    base = Path(".pytest_tmp_validate_job")
    base.mkdir(exist_ok=True)
    path = base / f"test_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    yield path
    rmtree(path, ignore_errors=True)


# Test that process ids are sorted numerically:
def test_explog_process_id_sort_is_numeric(local_explog_dir: Path):
    """
    Ensure process ids stay numerically ordered in the processes table.
    This guards against old string-based lexicographic ordering
    (e.g., 0, 1, 10, 11, 2, ...).
    """
    explog = Explog()
    explog.init(
        vasp_calculation_dir=local_explog_dir,
        mode="test",
        prefix_vasp_dirs=False,
    )

    n_processes = 112
    created_ids = []

    for idx in range(n_processes):
        candidate_id = f"cand_pid_sort_{idx}"
        explog.add_candidate(candidate_id, DummyStudy())
        process_id = explog.add_process(
            candidate_id=candidate_id,
            calculation_type="bulk_relaxation",
        )
        created_ids.append(process_id)

    # Correct number of processes created (and equivalent to range()):
    assert created_ids == list(range(n_processes))

    pdf = explog.relational_frame.processes.df.copy()
    process_id_series = pdf["process_id"]

    # Ensure process_id is stored as integer dtype in the DataFrame:
    assert pd.api.types.is_integer_dtype(process_id_series.dtype)

    # Ensure pandas numeric ascending sort yields 0..N-1:
    sorted_ids_asc = process_id_series.sort_values(
        ascending=True, ignore_index=True)
    
    # Test in pandas native integer dtype:
    expected_ids_asc = pd.Series(
        range(n_processes), dtype=sorted_ids_asc.dtype
    )
    assert sorted_ids_asc.equals(expected_ids_asc)
    # Test as plain Python integers:
    assert sorted_ids_asc.tolist() == list(range(n_processes))

    # Ensure pandas numeric descending sort yields N-1..0:
    sorted_ids_desc = process_id_series.sort_values(
        ascending=False, ignore_index=True
    )

    # Test in pandas native integer dtype:
    expected_ids_desc = pd.Series(
        range(n_processes - 1, -1, -1), dtype=sorted_ids_desc.dtype
    )
    assert sorted_ids_desc.equals(expected_ids_desc)
    # Test as plain Python integers:
    assert sorted_ids_desc.tolist() == \
        list(range(n_processes - 1, -1, -1))
