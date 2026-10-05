"""Reusable paired regression-model diagnostics for arbitrary tabular datasets."""
# SETUP LOGIC: Public imports have no file reads, model inference, training or widget display.
from .data import attach_predictions, from_long_predictions, read_data
from .engine import Comparison, Model, compare_models, compare_predictions
from .slices import Slice, default_slices

# CONFIGURATION LOGIC: Public package identity is independent of any input data source.
__version__ = '0.1.0'

def show_comparison(data=None, **kwargs):
    # UI LOGIC: Load optional notebook controls only when explicitly requested.
    from .ui import show_comparison as show
    return show(data, **kwargs)


__all__ = ['Comparison','Model','Slice','compare_models','compare_predictions',
           'attach_predictions','from_long_predictions','read_data','default_slices','show_comparison']
