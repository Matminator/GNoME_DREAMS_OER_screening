"""Tests for the 2026-09 EL8 -> EL9 submission cutover.

Covers the routing table, the atom-count boundaries, the idle-node promotion
added for partition='auto', and the rewritten slurm template. Nothing here
touches the scheduler: sinfo is monkeypatched, so these run anywhere.
"""
from pathlib import Path
from typing import get_args

import pytest

import gnome_dreams_oer_screening.vasp as vasp_pkg
from gnome_dreams_oer_screening.niflheim import slurm
from gnome_dreams_oer_screening.niflheim.slurm import (
    ACCESSIBLE_NODES,
    CORES_PER_PARTITION,
    PARTITION_ROUTING,
    PartitionName,
    TEST_PARTITION,
    choose_partition,
    idle_nodes,
    partition_for_atom_count,
)

pytestmark = pytest.mark.mini

TEMPLATE = Path(vasp_pkg.__file__).parent / "pysqa_setup" / "slurm_template.sh"
PKG_SRC = Path(vasp_pkg.__file__).parent.parent


# --- the routing table -------------------------------------------------------

def test_routing_table_is_ordered_by_capability():
    """choose_partition slices from the floor to get never-downgrade, so the
    table MUST be ascending in cores. If someone reorders it, promotion would
    silently start downgrading jobs."""
    cores = [c for _, _, c in PARTITION_ROUTING]
    assert cores == sorted(cores), f"PARTITION_ROUTING not ascending: {cores}"


def test_routing_table_boundaries_ascend_and_end_open():
    bounds = [b for b, _, _ in PARTITION_ROUTING]
    assert bounds[-1] is None, "last entry must be the open-ended catch-all"
    finite = [b for b in bounds if b is not None]
    assert finite == sorted(finite), f"boundaries not ascending: {finite}"
    assert len(finite) == len(bounds) - 1, "only the LAST entry may be None"


def test_every_partition_is_el9():
    for _, partition, _ in PARTITION_ROUTING:
        assert partition.endswith("el9"), partition
    assert TEST_PARTITION.endswith("el9_test"), TEST_PARTITION


def test_derived_constants_match_the_table():
    assert CORES_PER_PARTITION == {p: c for _, p, c in PARTITION_ROUTING}
    assert ACCESSIBLE_NODES == [p for _, p, _ in PARTITION_ROUTING]


def test_partition_name_literal_is_in_sync_with_the_table():
    """PartitionName is hand-maintained because typing.Literal needs static
    values. This is the test that catches it drifting from the table."""
    assert set(get_args(PartitionName)) == {p for _, p, _ in PARTITION_ROUTING} | {"auto"}


# --- atom-count routing ------------------------------------------------------

@pytest.mark.parametrize("n_atoms,expected", [
    (1, "xeon24el9"),
    (29, "xeon24el9"),    # boundaries are exclusive lower bounds
    (30, "xeon40el9"),
    (79, "xeon40el9"),
    (80, "xeon56el9"),
    (122, "xeon56el9"),   # largest cell seen in the 2026 campaign
    (10_000, "xeon56el9"),
])
def test_partition_for_atom_count(n_atoms, expected):
    assert partition_for_atom_count(n_atoms) == expected


def test_partition_for_atom_count_covers_every_boundary_in_the_table():
    """Derived from the table, so it keeps holding if the split is retuned."""
    for max_atoms, partition, _ in PARTITION_ROUTING:
        if max_atoms is None:
            continue
        assert partition_for_atom_count(max_atoms - 1) == partition
        assert partition_for_atom_count(max_atoms) != partition


# --- idle_nodes --------------------------------------------------------------

def _fake_sinfo(monkeypatch, output, exc=None):
    def fake(cmd, **kwargs):
        if exc is not None:
            raise exc
        assert cmd[0] == "sinfo"
        assert kwargs.get("timeout"), "sinfo must be called with a timeout"
        return output
    monkeypatch.setattr(slurm.subprocess, "check_output", fake)


def test_idle_nodes_parses_sinfo(monkeypatch):
    _fake_sinfo(monkeypatch,
                "xeon24el9* mix@ 4\n"
                "xeon24el9* alloc 108\n"
                "xeon40el9 drng@ 11\n"
                "xeon40el9 maint 11\n"
                "xeon40el9 alloc 184\n"
                "xeon56el9 alloc 30\n"
                "xeon56el9 idle 16\n")
    assert idle_nodes(["xeon24el9", "xeon40el9", "xeon56el9"]) == {
        "xeon24el9": 0, "xeon40el9": 0, "xeon56el9": 16}


def test_idle_nodes_strips_default_partition_marker(monkeypatch):
    """sinfo marks the cluster default partition with '*'."""
    _fake_sinfo(monkeypatch, "xeon24el9* idle 7\n")
    assert idle_nodes(["xeon24el9"]) == {"xeon24el9": 7}


def test_idle_nodes_sums_multiple_idle_rows(monkeypatch):
    _fake_sinfo(monkeypatch, "xeon56el9 idle 3\nxeon56el9 idle 4\n")
    assert idle_nodes(["xeon56el9"]) == {"xeon56el9": 7}


@pytest.mark.parametrize("state", ["mix", "mix@", "alloc", "drng@", "maint",
                                   "down", "drain*", "comp", "idle~", "idle*"])
def test_idle_nodes_counts_only_exact_idle(monkeypatch, state):
    """Every submitted partition is whole-node, so a partially free 'mix' node
    is useless. 'idle~' (powered down) and 'idle*' (unreachable) must not count
    either -- missing them is the safe direction."""
    _fake_sinfo(monkeypatch, f"xeon56el9 {state} 9\n")
    assert idle_nodes(["xeon56el9"]) == {"xeon56el9": 0}


def test_idle_nodes_ignores_malformed_lines(monkeypatch):
    _fake_sinfo(monkeypatch,
                "garbage\nxeon56el9 idle\nxeon56el9 idle 5\nxeon56el9 idle NaN\n")
    assert idle_nodes(["xeon56el9"]) == {"xeon56el9": 5}


def test_idle_nodes_ignores_unrequested_partitions(monkeypatch):
    _fake_sinfo(monkeypatch, "sm3090el9 idle 99\nxeon56el9 idle 2\n")
    assert idle_nodes(["xeon56el9"]) == {"xeon56el9": 2}


def test_idle_nodes_returns_empty_on_failure(monkeypatch, capsys):
    """{} means 'no information', which choose_partition must treat as
    'do not promote' -- never as 'nothing is idle'."""
    _fake_sinfo(monkeypatch, "", exc=OSError("sinfo not found"))
    assert idle_nodes(["xeon56el9"]) == {}
    assert "[ROUTE]" in capsys.readouterr().out


def test_idle_nodes_empty_output(monkeypatch):
    _fake_sinfo(monkeypatch, "")
    assert idle_nodes(["xeon24el9", "xeon56el9"]) == {
        "xeon24el9": 0, "xeon56el9": 0}


# --- choose_partition --------------------------------------------------------

def test_promotes_to_most_capable_idle_partition(monkeypatch):
    _fake_sinfo(monkeypatch, "xeon40el9 idle 2\nxeon56el9 idle 5\n")
    assert choose_partition("xeon24el9") == "xeon56el9"


def test_promotes_only_as_far_as_idle_allows(monkeypatch):
    _fake_sinfo(monkeypatch, "xeon40el9 idle 2\nxeon56el9 alloc 46\n")
    assert choose_partition("xeon24el9") == "xeon40el9"


def test_never_downgrades(monkeypatch):
    """The whole point: a 40-core job may go up to xeon56el9 but must never be
    dropped to xeon24el9, however idle xeon24el9 is."""
    _fake_sinfo(monkeypatch, "xeon24el9 idle 100\nxeon40el9 alloc 206\n"
                             "xeon56el9 alloc 46\n")
    assert choose_partition("xeon40el9") == "xeon40el9"


def test_stays_on_floor_when_nothing_idle(monkeypatch, capsys):
    _fake_sinfo(monkeypatch, "xeon24el9 alloc 112\nxeon40el9 alloc 206\n"
                             "xeon56el9 alloc 46\n")
    assert choose_partition("xeon24el9") == "xeon24el9"
    assert "nothing idle" in capsys.readouterr().out


def test_falls_back_to_floor_when_sinfo_fails(monkeypatch):
    _fake_sinfo(monkeypatch, "", exc=RuntimeError("scheduler down"))
    assert choose_partition("xeon24el9") == "xeon24el9"


def test_top_partition_short_circuits_without_calling_sinfo(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("sinfo must not be called for the top partition")
    monkeypatch.setattr(slurm.subprocess, "check_output", boom)
    assert choose_partition("xeon56el9") == "xeon56el9"


def test_unknown_floor_is_returned_unchanged(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("sinfo must not be called for an unknown floor")
    monkeypatch.setattr(slurm.subprocess, "check_output", boom)
    assert choose_partition("xeon40el8") == "xeon40el8"


def test_promotion_is_logged(monkeypatch, capsys):
    """grep '[ROUTE]' on the nohup log is how this gets audited in production."""
    _fake_sinfo(monkeypatch, "xeon56el9 idle 16\n")
    choose_partition("xeon24el9")
    out = capsys.readouterr().out
    assert "[ROUTE]" in out and "xeon24el9" in out and "xeon56el9" in out


# --- the slurm template ------------------------------------------------------

def test_template_uses_the_el9_stack():
    text = TEMPLATE.read_text()
    assert "VASP/6.5.1-intel-2025b" in text
    assert "impi/2021.16.1-intel-compilers-2025.2.0" in text
    assert "venv_el9/bin/activate" in text


def test_template_has_no_el8_remnants():
    text = TEMPLATE.read_text()
    for stale in ("2023a", "2021.9.0", "venv2/bin/activate"):
        assert stale not in text, f"template still references {stale}"


def test_template_does_not_load_a_python_module():
    """venv_el9 carries its own CPython 3.11.13; a module Python would shadow
    it and reintroduce the dependency on the per-arch module tree."""
    for line in TEMPLATE.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not (stripped.startswith("ml Python")
                    or stripped.startswith("module load Python")), line


def test_template_guards_the_environment():
    text = TEMPLATE.read_text()
    assert "|| exit 1" in text, "a failed venv activate must abort the job"
    assert "vasp_std" in text and "mpiexec" in text, "PATH guard missing"


def test_template_renders_with_every_pysqa_variable():
    from jinja2 import Template
    rendered = Template(TEMPLATE.read_text()).render(
        cores=40, nodes=1, partition="xeon40el9", walltime="50:00:00",
        slurm_out_name="/tmp/job_log", command="python3 run_vasp.py /tmp/calc")
    assert "{{" not in rendered, "unsubstituted placeholder left in template"
    for expected in ("xeon40el9", "50:00:00", "/tmp/job_log.err",
                     "python3 run_vasp.py /tmp/calc"):
        assert expected in rendered


# --- nothing EL8 left in the package ----------------------------------------

def test_no_el8_partitions_referenced_in_package_source():
    stale = ("xeon40el8", "xeon24el8", "'xeon56'", '"xeon56"')
    offenders = []
    for path in PKG_SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(errors="replace")
                                      .splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # explanatory comments about the old names are fine
            for token in stale:
                if token in line:
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "EL8 partitions still referenced:\n" + "\n".join(offenders)


# --- the composition job_handler relies on -----------------------------------

def _routing_scenarios():
    """(sinfo output, label) covering the idle states the router can meet."""
    return [
        ("", "nothing idle"),
        ("xeon24el9 idle 5\n", "only the smallest idle"),
        ("xeon40el9 idle 5\n", "only the middle idle"),
        ("xeon56el9 idle 5\n", "only the largest idle"),
        ("xeon24el9 idle 5\nxeon40el9 idle 5\nxeon56el9 idle 5\n", "all idle"),
    ]


@pytest.mark.parametrize("sinfo_out,label", _routing_scenarios())
@pytest.mark.parametrize("n_atoms", [1, 22, 29, 30, 55, 79, 80, 122])
def test_routed_partition_always_has_a_core_count(monkeypatch, sinfo_out, label,
                                                  n_atoms):
    """job_handler looks up CORES_PER_PARTITION[partition] straight after
    routing and raises if it misses. Whatever the idle state, the pair of
    functions must only ever produce a partition that is in that map."""
    _fake_sinfo(monkeypatch, sinfo_out)
    chosen = choose_partition(partition_for_atom_count(n_atoms))
    assert chosen in CORES_PER_PARTITION, f"{chosen} unroutable ({label})"


@pytest.mark.parametrize("sinfo_out,label", _routing_scenarios())
@pytest.mark.parametrize("n_atoms", [1, 22, 29, 30, 55, 79, 80, 122])
def test_promotion_never_reduces_cores(monkeypatch, sinfo_out, label, n_atoms):
    """The never-downgrade rule expressed in the unit that matters: a job must
    never end up on fewer cores than its atom count called for."""
    _fake_sinfo(monkeypatch, sinfo_out)
    floor = partition_for_atom_count(n_atoms)
    chosen = choose_partition(floor)
    assert CORES_PER_PARTITION[chosen] >= CORES_PER_PARTITION[floor], (
        f"{n_atoms} atoms: {floor} -> {chosen} lost cores ({label})")


def test_job_handler_auto_branch_routes_through_choose_partition():
    """Guards against the promotion call being reverted, leaving the static
    split silently in place. Source-level because reaching the branch for real
    needs a full Job_handler, EXPLOG and candidate."""
    import gnome_dreams_oer_screening.vasp.job_handler as jh
    src = Path(jh.__file__).read_text()
    assert "choose_partition" in src, "job_handler no longer imports the router"
    marker = "if partition == 'auto':"
    assert marker in src
    branch = src.split(marker, 1)[1][:400]
    assert "choose_partition(" in branch, (
        "the auto branch no longer promotes on idle nodes")
    assert "partition_for_atom_count(" in branch, (
        "the auto branch no longer derives a floor from the atom count")
