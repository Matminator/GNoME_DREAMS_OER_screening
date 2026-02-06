from pathlib import Path

def read_template(type: str):
    current_dir = Path(__file__).parent
    slurm_template = Path("slurm_templates/test_template.sh").read_text()
    return slurm_template