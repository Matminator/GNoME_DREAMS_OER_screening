import sys
from gnome_dreams_oer_screening.vasp.vasp_calculation import (
    run_vasp_via_custodian, BULK_RELAXATION_HANDLERS)

if __name__ == "__main__":
    run_vasp_via_custodian(
        directory=sys.argv[1],
        handlers=BULK_RELAXATION_HANDLERS,
        vasp_cmd=["mpiexec", "vasp_std"],
        auto_npar=True,
        max_errors=20,
    )
