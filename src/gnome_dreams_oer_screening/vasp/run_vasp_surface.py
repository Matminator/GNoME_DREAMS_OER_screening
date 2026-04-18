import sys
from gnome_dreams_oer_screening.vasp.vasp_calculation import run_surface_vasp_two_stage

if __name__ == "__main__":
    directory = sys.argv[1]
    use_dipole = sys.argv[2].lower() != "false" if len(sys.argv) > 2 else True
    run_surface_vasp_two_stage(
        directory=directory,
        use_dipole_correction=use_dipole,
        vasp_cmd=["mpiexec", "vasp_std"],
        auto_npar=True,
        max_errors=20,
    )
