"""Entry point dell'applicazione.

Legge ``config.properties`` per precompilare il launcher, avvia la GUI di
confronto euristica vs RL e, alla fine di ogni prova, ripropone il launcher con i
valori salvati.
"""

from __future__ import annotations

import logging
import tkinter as tk

from . import paths
from .config import SimulationConfig
from .launcher import AntelligentLauncher
from .simulation.simulation import ComparisonSimulation

_RL_POLICY_FILE = "policy.pkl"

_LOGGER = logging.getLogger(__name__)


class AntelligentStarter:
    def __init__(self, root: tk.Tk) -> None:
        self._root = root

    def start(self) -> None:
        AntelligentLauncher(self._root, self._read_defaults(), self._launch)

    # ------------------------------------------------------------------ #

    def _read_defaults(self) -> SimulationConfig:
        properties = self._load_properties()
        return SimulationConfig(
            cols=_read_int(properties, "cols"),
            rows=_read_int(properties, "rows"),
            n_thread=_read_int(properties, "nThread"),
            n_ants=_read_int(properties, "nAnts"),
            n_seeds=_read_int(properties, "nSeeds"),
            seed_types=_read_int(properties, "seedTypes"),
            matrix_type=_read_int(properties, "type"),
            refresh_rate=_read_int(properties, "refreshRate"),
            stop_criterion=_read_int(properties, "stopCriterion"),
            max_iterations=_read_int(properties, "maxIterations"),
            entropy_threshold=_read_float(properties, "entropyThreshold"),
            master_seed=_read_int(properties, "masterSeed"),
            capture_screenshots=_read_bool(properties, "captureScreenshots"),
        )

    def _load_properties(self) -> dict[str, str]:
        """Preferisce il file su disco (riflette i salvataggi precedenti) al default del package."""
        for candidate in (paths.CONFIG_PATH, paths.RESOURCES_DIR / "config.properties"):
            if candidate.is_file():
                return _parse_properties(candidate.read_text(encoding="utf-8"))
        _LOGGER.warning("config.properties non trovato, uso valori a 0")
        return {}

    def _save_config(self, config: SimulationConfig) -> None:
        content = (
            f"cols={config.cols}\n"
            f"rows={config.rows}\n"
            f"nThread={config.n_thread}\n"
            f"nAnts={config.n_ants}\n"
            f"nSeeds={config.n_seeds}\n"
            f"seedTypes={config.seed_types}\n"
            f"type={config.matrix_type}\n"
            "# Numero di iterazioni della simulazione tra un repaint della board e il successivo.\n"
            f"refreshRate={config.refresh_rate}\n"
            "# Criterio di arresto della simulazione: 0 = numero massimo di iterazioni, 1 = soglia di entropia.\n"
            f"stopCriterion={config.stop_criterion}\n"
            "# Usato quando stopCriterion=0.\n"
            f"maxIterations={config.max_iterations}\n"
            "# Usato quando stopCriterion=1 (scala 0-100, valori piu' bassi = semi piu' raggruppati per tipo).\n"
            f"entropyThreshold={config.entropy_threshold}\n"
            "# Seed dello stato iniziale condiviso dalle due copie (euristica / RL). 0 = casuale a ogni run.\n"
            f"masterSeed={config.master_seed}\n"
            "# Cattura periodica di schermate della finestra in results/screenshots/ (0 = no, 1 = si').\n"
            f"captureScreenshots={1 if config.capture_screenshots else 0}\n"
        )
        try:
            paths.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            paths.CONFIG_PATH.write_text(content, encoding="utf-8")
        except OSError:
            _LOGGER.exception("Impossibile salvare %s", paths.CONFIG_PATH)

    def _launch(self, config: SimulationConfig) -> None:
        self._save_config(config)
        rl_policy_path = paths.RESULTS_DIR / _RL_POLICY_FILE
        # `start` come callback "Nuova iterazione": ricrea il launcher precompilato.
        ComparisonSimulation(
            self._root, config, self.start, rl_policy_path=rl_policy_path
        )


def _parse_properties(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


def _read_int(properties: dict[str, str], key: str) -> int:
    try:
        return int(properties.get(key, "0"))
    except ValueError:
        return 0


def _read_bool(properties: dict[str, str], key: str) -> bool:
    """Accetta ``1``/``true``/``yes``/``on`` (case-insensitive); assente o altro = ``False``."""
    return properties.get(key, "").strip().lower() in {"1", "true", "yes", "on"}


def _read_float(properties: dict[str, str], key: str) -> float:
    try:
        return float(properties.get(key, "0").replace(",", "."))
    except ValueError:
        return 0.0


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    root = tk.Tk()
    root.withdraw()
    AntelligentStarter(root).start()
    root.mainloop()


if __name__ == "__main__":
    main()
