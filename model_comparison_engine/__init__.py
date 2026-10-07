"""Reusable paired regression-model diagnostics for arbitrary tabular datasets."""
# SETUP LOGIC: Public imports have no file reads, model inference, training or widget display.
from .data import attach_predictions, from_long_predictions, read_data
from .engine import Comparison, Model, compare_models, compare_predictions
from .model_sets import compare_model_set, compare_prediction_set
from .trade_view import TradeViewConfig
from .explainability import ExplanationResult, permutation_importance, tree_shap
from .slices import Slice, default_slices
from .inference import InferenceConfig
from .temporal import TemporalConfig
from .walk_forward import WalkForwardConfig, WalkForwardFold, walk_forward_splits
from .cross_validation import TrainableModel, WalkForwardResult, walk_forward_compare

# CONFIGURATION LOGIC: Public package identity is independent of any input data source.
__version__ = '0.5.0'

def show_comparison(data=None, **kwargs):
    # UI LOGIC: Load optional notebook controls only when explicitly requested.
    from .ui import show_comparison as show
    return show(data, **kwargs)


__all__ = ['Comparison','Model','Slice','InferenceConfig','TemporalConfig','compare_models','compare_predictions',
           'compare_model_set','compare_prediction_set',
           'TradeViewConfig','ExplanationResult','permutation_importance','tree_shap',
           'attach_predictions','from_long_predictions','read_data','default_slices','show_comparison',
           'WalkForwardConfig','WalkForwardFold','walk_forward_splits','TrainableModel',
           'WalkForwardResult','walk_forward_compare']
