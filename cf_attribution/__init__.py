"""Conditional counterfactual attribution on panel data.

Outputs are conditional-distribution contrasts, not structural causal effects.
The evaluation protocol is falsification: relative ranking, temporal
localization, observational calibration, proxy agreement, and sensitivity.
"""

from cf_attribution.semantics import SEMANTIC_DECLARATION

__all__ = ["SEMANTIC_DECLARATION"]
__version__ = "0.1.0"
