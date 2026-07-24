"""Make the repo root importable under pytest.

The project uses top-level packages (tools/, data/, agent/, memory/) imported
as `from tools.x import y`. Placing this conftest at the repo root puts the
root on sys.path so `pytest` works from anywhere without an install step.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
