"""Boolean predicate heads, PyXAI explanations, and explicit crop dependencies."""
from .logic import BooleanHead, compile_boolean_function, fit_tree, threshold_scores

__all__ = ['BooleanHead', 'compile_boolean_function', 'fit_tree', 'threshold_scores']
