# GNoME_DREAMS_OER_screening

## v1.1.2 Updates

**Production-mode tag restoration**: ASE atom tags (tag=3 for adsorption site, tag=4 for adsorbed O/H) are not preserved through VASP POSCAR/vasprun.xml round-trips. `job_handler.py` now explicitly restores tags from the in-memory study object before passing VASP-returned atoms to `set_relaxed_O_calc` / `set_relaxed_OH_calc`.

**Pickle checkpointing refactor**: `hpc_workflow_tryout.py` and the companion notebook no longer attempt to pickle the full `EXPLOG` object (which contains unpicklable `pysqa.QueueAdapter` functions). Mid-run and end-of-run checks now load and verify the `explog.pkl` payload dict written internally by `Explog._save_pickle()`.

## v1.0.0 Updates

IMPORTANT:
Now needs enumlib, used under pymatgen. This CAN NOT run on Windows. 
- see https://pymatgen.org/installation.html for installation of enumlib and configuring of pymatgen