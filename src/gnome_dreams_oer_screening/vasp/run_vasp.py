from vasp_calculation import run_vasp_via_custodian
import sys

from custodian.custodian import Custodian
from custodian.vasp.jobs import VaspJob
from custodian.vasp.handlers import ( 
    VaspErrorHandler,
    MeshSymmetryErrorHandler,
    UnconvergedErrorHandler,
    NonConvergingErrorHandler,
    PotimErrorHandler,
    PositiveEnergyErrorHandler,
    FrozenJobErrorHandler,
    StdErrHandler,
    LargeSigmaHandler,
    IncorrectSmearingHandler,
    KspacingMetalHandler,
    PotimErrorHandler, # only relaxation handler
    DriftErrorHandler, # only relaxation handler
    WalltimeHandler
)

HANDLERS = [VaspErrorHandler(),
        MeshSymmetryErrorHandler(), NonConvergingErrorHandler(),
        UnconvergedErrorHandler(), FrozenJobErrorHandler(),
        StdErrHandler(), PositiveEnergyErrorHandler()
        # LargeSigmaHandler(), IncorrectSmearingHandler(), 
        ]

if __name__ == "__main__":
    run_vasp_via_custodian(
    directory = sys.argv[1], 
    handlers = HANDLERS,
    vasp_cmd = ["mpiexec", "vasp_std"],
    auto_npar = True,
    max_errors = 20
    )