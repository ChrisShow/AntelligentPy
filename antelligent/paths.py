"""Percorsi noti del progetto, risolti rispetto alla posizione del package.

Cosi' l'app trova risorse, ``config.properties`` e la cartella ``results/`` a
prescindere dalla working directory da cui viene lanciata.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

RESOURCES_DIR = PACKAGE_DIR / "resources"
IMAGES_DIR = RESOURCES_DIR / "images"

CONFIG_PATH = PROJECT_ROOT / "config.properties"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "results.txt"
SCREENSHOTS_DIR = RESULTS_DIR / "screenshots"
