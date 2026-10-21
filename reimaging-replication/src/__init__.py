"""
Utility package for the (Re-)Imag(in)ing Price Trends replication project.

The modules inside src/ are shared by the CLI scripts under scripts/ so that
training, inference, and visualization can be configured via command-line
arguments without duplicating logic across files.
"""

from __future__ import annotations

__all__ = [
    "data_utils",
    "device",
    "i5_r5",
    "models",
    "training",
    "visualization",
    "yahoo_cleaning",
]
