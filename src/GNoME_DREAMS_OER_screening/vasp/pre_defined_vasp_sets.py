from dataclasses import dataclass
from pymatgen.io.vasp.sets import VaspInputSet
from GNoME_DREAMS_OER_screening.vasp.test_relaxset import get_pre_defined_settings

@dataclass
class TestRelaxSet(VaspInputSet):
    """
    Custom VASP input set using Python dictionary instead of YAML.
    """
    CONFIG = get_pre_defined_settings()