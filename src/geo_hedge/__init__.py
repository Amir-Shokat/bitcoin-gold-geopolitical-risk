"""Bitcoin and gold as hedges against geopolitical risk."""

from .data import build_dataset, simulate_dataset
from .models import fit, results_table

__all__ = ["build_dataset", "simulate_dataset", "fit", "results_table"]
