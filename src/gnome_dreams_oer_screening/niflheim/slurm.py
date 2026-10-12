import subprocess
from typing import Literal

# ---------------------------------------------------------------------------
# Partition routing -- the ONLY place the atom-count boundaries live.
# A structure goes to the first partition whose max_atoms it is BELOW; the
# final entry (None) is the catch-all. Retune by editing these numbers.
#
# 30/80 chosen 2026-09-28 from 3,975 historical jobs (May-Sep 2026).
# Predicted >50h timeout risk 0.1%. The same analysis supports moving the
# lower cut to 50, which would lift xeon24el9's share of core-hours from 2%
# to 14% -- deferred until the 24-core scaling benchmark has run.
# ---------------------------------------------------------------------------
PARTITION_ROUTING = (
    #  max_atoms, partition,     cores
    (30,          'xeon24el9',   24),
    (80,          'xeon40el9',   40),
    (None,        'xeon56el9',   56),
)

#: Short-walltime partition for slurm_test / MLIP_test modes. NB the old
#: xeon24el8_test no longer exists at all, so those modes are already broken;
#: this is a repair, not just a rename.
TEST_PARTITION = 'xeon24el9_test'

#: Keep in sync with PARTITION_ROUTING by hand -- typing.Literal needs static
#: values and cannot be built from the table at runtime.
PartitionName = Literal['xeon24el9', 'xeon40el9', 'xeon56el9', 'auto']

CORES_PER_PARTITION = {p: c for _, p, c in PARTITION_ROUTING}
ACCESSIBLE_NODES = [p for _, p, _ in PARTITION_ROUTING]


def partition_for_atom_count(n_atoms: int) -> str:
    """Route a structure by size. Boundaries are exclusive lower bounds:
    with 30/80, 29 atoms -> xeon24el9, 30 -> xeon40el9, 80 -> xeon56el9."""
    for max_atoms, partition, _ in PARTITION_ROUTING:
        if max_atoms is None or n_atoms < max_atoms:
            return partition
    raise ValueError(f"no partition for {n_atoms} atoms")  # unreachable


def idle_nodes(partitions: list[str]) -> dict[str, int]:
    """Whole idle nodes per partition, from one sinfo call (~30 ms).

    Only exact state "idle" counts. That is deliberate: every partition we
    submit to is whole-node (a site submit filter enforces it on xeon40el9 and
    xeon56el9, and a xeon24el9 node is 24 cores anyway), so a partially
    occupied "mix" node is no use to us.

    Returns {} on any failure. Callers must read that as "no information",
    never as "nothing is idle".
    """
    try:
        out = subprocess.check_output(
            ["sinfo", "-h", "-p", ",".join(partitions), "-o", "%P %t %D"],
            text=True, timeout=15,
        )
    except Exception as exc:  # sinfo missing, hung, or scheduler down
        print(f"[ROUTE] sinfo failed ({type(exc).__name__}: {exc})")
        return {}

    counts = {p: 0 for p in partitions}
    for line in out.strip().splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        part, state, num = fields
        part = part.rstrip("*")  # sinfo marks the default partition with '*'
        if state == "idle" and part in counts:
            try:
                counts[part] += int(num)
            except ValueError:
                continue
    return counts


def choose_partition(floor: str) -> str:
    """Pick the most capable partition at or above `floor` that has an idle node.

    `floor` is what the atom count requires (see partition_for_atom_count).
    PARTITION_ROUTING is ordered by capability, so slicing from `floor` makes
    the never-downgrade rule fall out: a job that needs 40 cores can be promoted
    to xeon56el9 but never dropped to xeon24el9.

    An idle whole node means the job starts immediately, which matters because
    64% of this campaign's job-time was queue wait, not compute. Falls back to
    `floor` when nothing is idle or the lookup fails -- i.e. to the behaviour
    this had before -- and says which in the log.

    Only ever called for partition='auto'. An explicitly requested partition is
    honoured as given.
    """
    order = [p for _, p, _ in PARTITION_ROUTING]
    if floor not in order:
        print(f"[ROUTE] {floor} is not in PARTITION_ROUTING; leaving as-is")
        return floor
    tier = order[order.index(floor):]
    if len(tier) == 1:  # already the most capable; nothing to promote to
        return floor

    idle = idle_nodes(tier)
    for partition in reversed(tier):  # most capable first
        if idle.get(partition, 0) > 0:
            print(f"[ROUTE] {floor} -> {partition} ({idle[partition]} idle)")
            return partition

    print(f"[ROUTE] {floor} (nothing idle in {'/'.join(tier)})")
    return floor


def get_my_jobs_status():
    out = subprocess.check_output(
        ["squeue", "--me", "-h", "-o", "%i %T"],
        text=True
    )
    return dict(line.split() for line in out.splitlines())


import subprocess
from collections import defaultdict

def _get_idle_alloc_counts(partition: str) -> tuple[int, int]:
    out = subprocess.check_output(
        ["sinfo", "-p", partition, "-h", "-o", "%t %D"],
        text=True
    )

    counts = defaultdict(int)

    for line in out.strip().splitlines():
        state, num = line.split()
        counts[state] += int(num)

    # sum idle and allocated (including mix as partially allocated)
    idle = counts.get("idle", 0)
    allocated = counts.get("alloc", 0) + counts.get("mix", 0)

    return idle, allocated


def get_node_status(nodes: list[str] | None = None):

    if nodes is None:
        nodes = ACCESSIBLE_NODES

    node_queues = {}
    partition_idle_alloc = {}
    for node_name in nodes:
        queue_info = subprocess.check_output(
                ["squeue", "-p", node_name, "-h", "-o", "%i %T"],
        text=True
        )

        num_pending = 0
        num_running = 0
        for line in queue_info.strip().splitlines():
            job_id, state = line.split()
            if state == "PENDING":
                num_pending += 1
            elif state == "RUNNING":
                num_running += 1

        node_queues[node_name] = (num_pending, num_running)
        partition_idle_alloc[node_name] = \
        _get_idle_alloc_counts(node_name)

    # --- TEMPORARY print ----------------------------------------------
    print('--- Node Status -------------------------------------------')
    for node in nodes:
        pending, running = node_queues[node]
        idle, allocated = partition_idle_alloc[node]
        print(f"Node: {node} | Pending: {pending} | Running: {running} | "
              f"Idle Nodes: {idle} | Allocated Nodes: {allocated}")
    print('-----------------------------------------------------------')
    # ------------------------------------------------------------------