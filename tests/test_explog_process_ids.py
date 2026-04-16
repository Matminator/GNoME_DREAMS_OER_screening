# Tests for numeric process ID ordering in the Explog processes table.
#
# Guards against a regression where process IDs were stored as strings and
# sorted lexicographically ("0", "1", "10", "11", "2", ...) rather than
# numerically (0, 1, 2, ..., 111).  This matters when the processes table
# is sorted or filtered by process_id — string ordering gives wrong results
# once ID counts exceed 9.

import pytest
import pandas as pd

from pathlib import Path
from ase.io import read

from gnome_dreams_oer_screening import Explog, OER_catalyst_study


pytestmark = pytest.mark.mini


@pytest.fixture(scope="module")
def tio2_rutile_atoms():
    dir_path = Path(__file__).resolve().parent
    path = dir_path / "../notebooks/TiO2_rutile.cif"
    return read(path.resolve(), format="cif")


def test_explog_process_id_sort_is_numeric(tmp_path: Path, tio2_rutile_atoms):
    """Process IDs are integers and sort numerically, not lexicographically.

    Creates 112 candidates and submits one bulk_relaxation process each.
    add_process("bulk_relaxation") returns list[int] — one ID per magnetic
    config; for NM TiO2 this is always a single-element list.

    Verifies:
      - All IDs collected in creation order equal range(112).
      - The processes DataFrame stores process_id as an integer dtype.
      - Ascending and descending pandas sorts reproduce the expected ranges.
    """
    explog = Explog()
    explog.init(
        vasp_calculation_dir=tmp_path,
        mode="test",
        prefix_vasp_dirs=False,
    )

    n_processes = 112
    created_ids = []

    for idx in range(n_processes):
        candidate_id = f"cand_pid_sort_{idx}"
        explog.add_candidate(
            candidate_id=candidate_id,
            reason_or_hypothesis="process ID sort regression",
            notes="numeric ordering guard",
            study_obj=OER_catalyst_study(
                tio2_rutile_atoms,
                H2_gas_free_energy=-6.77,
                H2O_gas_free_energy=-14.217,
            ),
        )
        ids = explog.add_process(
            candidate_id=candidate_id,
            calculation_type="bulk_relaxation",
        )
        created_ids.extend(ids)

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
