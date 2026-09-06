"""Parametri di una simulazione (griglia, formiche, semi, criterio di arresto, RL)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Configurazione immutabile passata dal launcher alla simulazione.

    ``matrix_type`` e' mantenuto solo per compatibilita' col formato di
    ``config.properties`` / ``results.txt`` (chiave ``type``): esiste un'unica
    matrice (normale con lock), valore ``1``.

    ``stop_criterion``: ``0`` = numero massimo di iterazioni, ``1`` = soglia di entropia.

    ``master_seed``: genera lo **stato iniziale condiviso** dalle due copie gemelle
    (euristica / RL). Se ``0`` viene sorteggiato un seed casuale a ogni run e
    stampato/salvato.

    ``capture_screenshots``: se ``False`` (default) la run non cattura nulla.
    La cattura (``ImageGrab``) blocca il thread della GUI e su macOS richiede il
    permesso "Registrazione schermo", quindi e' opt-in dal launcher.
    """

    cols: int
    rows: int
    n_thread: int
    n_ants: int
    n_seeds: int
    seed_types: int
    matrix_type: int
    refresh_rate: int
    stop_criterion: int
    max_iterations: int
    entropy_threshold: float
    master_seed: int = 0
    capture_screenshots: bool = False
