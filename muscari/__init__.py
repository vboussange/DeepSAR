import matplotlib as mpl
from .plotting import RCPARAMS_DICT
from .muscari_ensemble import MuScaRiEnsemble

mpl.rcParams.update(RCPARAMS_DICT)

__all__ = ["MuScaRi", "MuScaRiEnsemble"]