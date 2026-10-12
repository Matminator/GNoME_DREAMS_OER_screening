# GNoME_DREAMS_OER_screening

The DFT engine behind an autonomous OER catalyst screening campaign: it turns a
bulk crystal structure into relaxed adsorbate geometries and the free energies
that rank it as a catalyst, and records every calculation and decision in a
relational log.

**[Browse the results](https://matminator.github.io/OER-GNoME/)** ·
**Data:** [doi.org/10.11583/DTU.33265158](https://doi.org/10.11583/DTU.33265158)

## What it does

```
oer/       surface and termination generation, adsorption-site determination,
           coordination-based surface ranking, free energies and overpotentials
vasp/      VASP input sets (RPBE), magnetic-configuration enumeration,
           job submission and validation, convergence recovery via custodian
explog/    the relational record: one row per candidate with its hypothesis and
           decision history, one per calculation with its site and energies
niflheim/  SLURM submission, partition routing by system size
```

The pipeline is a fixed ladder — bulk relaxation, surface relaxation, O\*, then
OH\* — with ordering enforced in validation, so an adsorbate calculation cannot
start from an unaccepted surface.

## Install

```bash
pip install .
```

Magnetic enumeration additionally needs the `enumlib` Fortran binaries
(`enum.x`, `makestr.x`) on PATH; `conda install -c conda-forge enumlib` on
Linux. Without them, enumeration falls back to a single non-magnetic job.
Running the DFT itself requires a licensed VASP installation.

## Notes

Termination indices are positional and follow pymatgen's miller-index
enumeration order, which is not a stable API across pymatgen releases. Indices
recorded in a log are only meaningful alongside the pymatgen version that
produced them.

## Licence

MIT — see [LICENSE](LICENSE). The accompanying dataset is separately licensed
CC BY-NC 4.0.
