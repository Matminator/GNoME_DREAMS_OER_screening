import subprocess

def get_status_all_jobs():
    out = subprocess.check_output(
        ["squeue", "-h", "-o", "%i %T"],
        text=True
    )
    return dict(line.split() for line in out.splitlines())