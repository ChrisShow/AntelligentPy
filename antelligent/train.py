"""Harness headless di addestramento (``docs/rl-design.md`` §4.3, §5.1).

Addestra una :class:`TabularQPolicy` a parametri condivisi su molti episodi (campo
rigenerato a ogni episodio), salva la politica e il log delle curve di
apprendimento. La politica salvata viene poi caricata dalla GUI di confronto.

Esempio::

    python -m antelligent.train --episodes 400 --out results/policy.pkl
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from pathlib import Path

from . import paths
from .config import SimulationConfig
from .environment import Environment, RewardConfig
from .policies.tabular import QConfig, TabularQPolicy
from .world import build_initial_state

_LOGGER = logging.getLogger(__name__)


def _load_config(config_path: Path) -> SimulationConfig:
    properties: dict[str, str] = {}
    if config_path.is_file():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                properties[key.strip()] = value.strip()

    def _int(key: str, default: int = 0) -> int:
        try:
            return int(properties.get(key, str(default)))
        except ValueError:
            return default

    def _float(key: str, default: float = 0.0) -> float:
        try:
            return float(properties.get(key, str(default)).replace(",", "."))
        except ValueError:
            return default

    return SimulationConfig(
        cols=_int("cols", 40),
        rows=_int("rows", 25),
        n_thread=_int("nThread", 6),
        n_ants=_int("nAnts", 100),
        n_seeds=_int("nSeeds", 650),
        seed_types=_int("seedTypes", 5),
        matrix_type=_int("type", 1),
        refresh_rate=_int("refreshRate", 500),
        stop_criterion=_int("stopCriterion", 1),
        max_iterations=_int("maxIterations", 20_000),
        entropy_threshold=_float("entropyThreshold", 5.0),
        master_seed=_int("masterSeed", 0),
    )


def _run_episode(policy: TabularQPolicy, config: SimulationConfig, seed: int, max_ticks: int,
                 entropy_every: int, reward_cfg: RewardConfig) -> dict:
    init = build_initial_state(config, seed)
    env = Environment.from_initial_state(
        init, reward=reward_cfg, rng=random.Random(seed),
        entropy_every=entropy_every, window_radius=0,  # la Fase 1 tabellare non usa la finestra
    )
    if policy._needs_heuristic:  # noqa: SLF001 - ablazioni A1/A2
        policy.bind_environment(env)
    while env.iterations < max_ticks and not env.is_done(
        config.stop_criterion, max_ticks, config.entropy_threshold
    ):
        env.tick(policy)
    env.refresh_entropy()
    return {
        "ticks": env.iterations,
        "initial_entropy": env.initial_entropy,
        "final_entropy": env.entropy,
        "moves": env.total_moves,
        "seeds": env.seeds_collected,
    }


def train(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    rng = random.Random(args.seed)
    q_config = QConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay or max(1, args.episodes * 3 // 4),
        learn_move=not args.no_learn_move,
        learn_manip=not args.no_learn_manip,
    )
    policy = TabularQPolicy(config.seed_types, q_config)
    policy.rng.seed(args.seed)
    reward_cfg = RewardConfig()

    max_ticks = args.max_ticks or config.max_iterations
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_name("training_log.csv")

    rows: list[dict] = []
    start = time.monotonic()
    for episode in range(args.episodes):
        stats = _run_episode(policy, config, rng.randrange(1, 2**31), max_ticks,
                             args.entropy_every, reward_cfg)
        policy.end_episode()
        row = {
            "episode": episode,
            "ticks": stats["ticks"],
            "initial_entropy": round(stats["initial_entropy"], 3),
            "final_entropy": round(stats["final_entropy"], 3),
            "moves": stats["moves"],
            "seeds": stats["seeds"],
            "epsilon": round(policy.epsilon, 4),
            "elapsed_s": round(time.monotonic() - start, 1),
        }
        rows.append(row)
        if episode % args.log_every == 0 or episode == args.episodes - 1:
            print(
                f"ep {episode:4d}  ticks {stats['ticks']:5d}  "
                f"H0 {stats['initial_entropy']:5.1f} -> H {stats['final_entropy']:5.1f}  "
                f"moves {stats['moves']:7d}  seeds {stats['seeds']:5d}  eps {policy.epsilon:.3f}"
            )

    policy.save(out_path)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\npolitica salvata in {out_path}")
    print(f"log di addestramento in {log_path}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Addestra la politica RL tabellare (Fase 1).")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--config", default=str(paths.CONFIG_PATH))
    parser.add_argument("--out", default=str(paths.RESULTS_DIR / "policy.pkl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="tetto di tick per episodio (default: maxIterations della config)")
    parser.add_argument("--entropy-every", type=int, default=25,
                        help="ogni quanti tick ricalcolare l'entropia (per l'arresto/log)")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=int, default=0,
                        help="episodi su cui decade epsilon (default: 3/4 degli episodi)")
    parser.add_argument("--no-learn-move", action="store_true", help="ablazione A2: movimento euristico")
    parser.add_argument("--no-learn-manip", action="store_true", help="ablazione A1: pick/drop euristico")
    parser.add_argument("--log-every", type=int, default=10)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
