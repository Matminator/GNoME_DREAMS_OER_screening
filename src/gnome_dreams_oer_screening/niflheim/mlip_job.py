import sys
from pathlib import Path

from ase.io import Trajectory
from ase.optimize import FIRE
from mace.calculators import mace_mp


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Usage: python mlip_job.py <calc_dir>")

    calc_dir = Path(sys.argv[1]).expanduser().resolve()
    if not calc_dir.exists() or not calc_dir.is_dir():
        raise NotADirectoryError(f"Invalid calc_dir: {calc_dir}")

    input_traj = calc_dir / "mlip_input.traj"
    if not input_traj.exists():
        raise FileNotFoundError(f"Missing input file: {input_traj}")

    traj = Trajectory(input_traj)
    if len(traj) == 0:
        raise RuntimeError(f"Input trajectory is empty: {input_traj}")
    atoms = traj[-1].copy()
    traj.close()

    # Use a MACE-MP-0 preset model.
    atoms.calc = mace_mp(model="medium", device="cpu")

    opt_traj = calc_dir / "mlip_opt.traj"
    dyn = FIRE(atoms, trajectory=str(opt_traj))
    dyn.run(fmax=0.05, steps=500) # terminate at 0.05 eV/Å or 500 steps

    result_traj = calc_dir / "mlip_result.traj"
    out = Trajectory(result_traj, "w")
    out.write(atoms)
    out.close()

    print(f"Saved MLIP result to: {result_traj}")


if __name__ == "__main__":
    main()
