"""Stage 2 RCM-DCE research implementation.

This package is intentionally separate from the legacy ``ham_dce`` pilot.  It
contains the approximately one-million-parameter plain curve baseline and
the parameter-matched condition/residual controls used by the new protocol.
"""

from .model import RCMCurveNet, count_parameters, model_signature

__all__ = ["RCMCurveNet", "count_parameters", "model_signature"]
