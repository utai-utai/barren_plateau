import os

from .barren_plateau import BP
from .barren_plateau_multi import BPs
from .plotting import PLOTTING
from setup_db import initialize_database

version = 1.0

__all__ = ['BP', 'BPs', 'PLOTTING', 'initialize_database']

DB_PATH = 'barren/data.db'
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Database path '{DB_PATH}' does not exist.")
