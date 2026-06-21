"""Test configuration: silence the matplotlib backend and pin a seed."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")
