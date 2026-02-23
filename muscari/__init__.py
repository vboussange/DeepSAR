import matplotlib as mpl
from .plotting import RCPARAMS_DICT
from .ensemble_model import MuScaRiEnsemble

mpl.rcParams.update(RCPARAMS_DICT)

__all__ = ["MuScaRiEnsemble"]