"""Reproducible seeding across numpy, random, and torch."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed all common RNG sources for reproducibility.

    Torch is imported lazily so the utility works in environments that
    have not installed torch yet (e.g. lightweight smoke tests).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
