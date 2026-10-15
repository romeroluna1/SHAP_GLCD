"""Public text API for SHAP-GLCD.

This module preserves the validated implementations used for AG News and
20 Newsgroups while exposing them from a single stable import path.
"""

from ._text_explainer import SHAPTextExplainerLCD
from ._text_metrics import SHAPTextLCDMetrics
from ._text_global import SHAPGlobalTextGLCD
from ._text_20newsgroups import (
    SHAPGlobalTextGLCD20Newsgroups,
    SHAPGlobalTextGLCD20NG,
)
from ._text_token_visualizer import (
    SHAPGlobalTextTokenSizeArticleVisualizer,
)
from ._text_ranked_visualizer import (
    SHAPGlobalTextTwoColumnRankedArticleVisualizer,
)

__all__ = [
    "SHAPTextExplainerLCD",
    "SHAPTextLCDMetrics",
    "SHAPGlobalTextGLCD",
    "SHAPGlobalTextGLCD20Newsgroups",
    "SHAPGlobalTextGLCD20NG",
    "SHAPGlobalTextTokenSizeArticleVisualizer",
    "SHAPGlobalTextTwoColumnRankedArticleVisualizer",
]
