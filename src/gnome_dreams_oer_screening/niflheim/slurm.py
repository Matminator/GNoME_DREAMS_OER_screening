import subprocess

ACCESSIBLE_NODES = [
    "xeon24el8", 'xeon40el8', 'xeon56'
]

def get_status_my_jobs():
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