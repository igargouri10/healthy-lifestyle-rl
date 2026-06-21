"""Helper that lets scripts find the ``src/`` package without installation.

Each script can ``import _bootstrap`` at the top to ensure ``src/`` is on
``sys.path`` regardless of where ``python`` is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
