"""Addestramento con ricompensa "alla Lumer-Faieta" (variante sperimentale).

Stessa struttura di :mod:`antelligent.train` (episodi su campo rigenerato,
politica salvata + log delle curve), ma con
:class:`~antelligent.environment_lumer.LumerFaietaEnvironment` e
:class:`~antelligent.policies.lumer_tabular.LumerQPolicy`: la manipolazione e'
premiata con le **stesse funzioni di probabilita' che governano l'euristica**
invece che rispetto al livello del caso.

Esempi::

    # variante di default: centered, drop pesato il doppio, rifiuto + shaping
    python -m antelligent.train_lumer --episodes 400

    # ablazioni
    python -m antelligent.train_lumer --episodes 400 --mode advantage
    python -m antelligent.train_lumer --episodes 400 --shaping-scale 0 --decline-scale 0
    python -m antelligent.train_lumer --episodes 400 --decline-two-sided   # patologico

Il file prodotto (``results/policy_lumer.pkl`` di default) e' riconoscibile: la
GUI e :func:`~antelligent.policies.lumer_tabular.load_policy` scelgono da soli la
classe giusta. Per usarlo nel confronto affiancato basta salvarlo come
``results/policy.pkl`` (``--out``).
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
from .environment import RewardConfig
from .environment_lumer import LumerFaietaEnvironment
from .lumer_reward import MODES, LumerRewardConfig
from .policies.lumer_tabular import LumerQPolicy
from .policies.tabular import QConfig
from .train import _load_config
from .world import build_initial_state

_LOGGER = logging.getLogger(__name__)


def _run_episode(
    policy: LumerQPolicy,
    config: SimulationConfig,
    seed: int,
    max_ticks: int,
    entropy_every: int,
    reward_cfg: RewardConfig,
    lumer_cfg: LumerRewardConfig,
) -> dict:
    init = build_initial_state(config, seed)
    env = LumerFaietaEnvironment.from_initial_state(
        init, reward=reward_cfg, lumer=lumer_cfg, rng=random.Random(seed),
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
        "drops": env.drops_done,
        "carried_left": env.carried_count,
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
        f_bins=args.f_bins,
        allow_stay=args.allow_stay,
    )
    lumer_cfg = LumerRewardConfig(
        kp=args.kp,
        kd=args.kd,
        pick_scale=args.pick_scale,
        drop_scale=args.drop_scale,
        decline_scale=args.decline_scale,
        decline_two_sided=args.decline_two_sided,
        shaping_scale=args.shaping_scale,
        wall_penalty=args.wall_penalty,
        # lo shaping e' potenziale solo se usa lo stesso sconto della Q-learning
        gamma=args.gamma,
        isolated_guard=args.isolated_guard,
        mode=args.mode,
    )
    policy = LumerQPolicy(config.seed_types, q_config, lumer=lumer_cfg)
    policy.rng.seed(args.seed)
    reward_cfg = RewardConfig(
        step_penalty=args.step_penalty,
        contested_penalty=args.contested_penalty,
        invalid_penalty=args.invalid_penalty,
    )

    max_ticks = args.max_ticks or config.max_iterations
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_name(f"{out_path.stem}_log.csv")

    print(
        f"ricompensa Lumer-Faieta: mode={lumer_cfg.mode}  kp={lumer_cfg.kp}  kd={lumer_cfg.kd}  "
        f"f*={lumer_cfg.indifference_f:.4f}  pick={lumer_cfg.pick_scale}  drop={lumer_cfg.drop_scale}  "
        f"decline={lumer_cfg.decline_scale}{' (simmetrico)' if lumer_cfg.decline_two_sided else ''}  "
        f"shaping={lumer_cfg.shaping_scale}  wall={lumer_cfg.wall_penalty}"
    )

    rows: list[dict] = []
    start = time.monotonic()
    for episode in range(args.episodes):
        stats = _run_episode(policy, config, rng.randrange(1, 2**31), max_ticks,
                             args.entropy_every, reward_cfg, lumer_cfg)
        policy.end_episode()
        row = {
            "episode": episode,
            "ticks": stats["ticks"],
            "initial_entropy": round(stats["initial_entropy"], 3),
            "final_entropy": round(stats["final_entropy"], 3),
            "moves": stats["moves"],
            "picks": stats["seeds"],
            "drops": stats["drops"],
            "carried_left": stats["carried_left"],
            "epsilon": round(policy.epsilon, 4),
            "elapsed_s": round(time.monotonic() - start, 1),
        }
        rows.append(row)
        if episode % args.log_every == 0 or episode == args.episodes - 1:
            print(
                f"ep {episode:4d}  ticks {stats['ticks']:5d}  "
                f"H0 {stats['initial_entropy']:5.1f} -> H {stats['final_entropy']:5.1f}  "
                f"moves {stats['moves']:7d}  picks {stats['seeds']:5d}  drops {stats['drops']:5d}  "
                f"carried {stats['carried_left']:4d}  eps {policy.epsilon:.3f}"
            )

    policy.save(out_path)
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\npolitica salvata in {out_path}")
    print(f"log di addestramento in {log_path}")


#: I default della riga di comando sono presi *dalla dataclass*, non ricopiati:
#: cosi' non possono divergere in silenzio da :class:`LumerRewardConfig`.
_LF = LumerRewardConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Addestra la politica RL con ricompensa alla Lumer-Faieta (variante sperimentale).",
    )
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--config", default=str(paths.CONFIG_PATH))
    parser.add_argument("--out", default=str(paths.RESULTS_DIR / "policy_lumer.pkl"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="tetto di tick per episodio (default: maxIterations della config)")
    parser.add_argument("--entropy-every", type=int, default=25,
                        help="ogni quanti tick ricalcolare l'entropia (per l'arresto/log)")

    lf = parser.add_argument_group("ricompensa Lumer-Faieta")
    lf.add_argument("--mode", choices=MODES, default=_LF.mode,
                    help="centered = 2P-1, soglia selettiva (default, misurato migliore); "
                         "advantage = P(azione)-P(opposta), zero in f*=sqrt(kp*kd); "
                         "raw = P (ablazione: porta al churn)")
    lf.add_argument("--kp", type=float, default=_LF.kp,
                    help="costante di pick dell'euristica")
    lf.add_argument("--kd", type=float, default=_LF.kd,
                    help="costante di drop dell'euristica")
    lf.add_argument("--pick-scale", type=float, default=_LF.pick_scale,
                    help="peso del segnale di raccolta")
    lf.add_argument("--drop-scale", type=float, default=_LF.drop_scale,
                    help="peso del segnale di deposito, cioe' del PUNTO in cui si posa: in modalita' "
                         "'centered' il drop ha un quinto dell'escursione del pick, quindi il default lo riequilibra")
    lf.add_argument("--decline-scale", type=float, default=_LF.decline_scale,
                    help="peso della penalita' per aver rifiutato una manipolazione conveniente "
                         "(0 = rifiutare e' gratis)")
    lf.add_argument("--decline-two-sided", action="store_true",
                    help="ablazione: premia anche il rifiuto di una manipolazione sconveniente "
                         "(patologico, crea una rendita di posizione)")
    lf.add_argument("--wall-penalty", type=float, default=_LF.wall_penalty,
                    help="penalita' per una mossa scelta fuori dalla griglia (0 = gratis, sconsigliato)")
    lf.add_argument("--shaping-scale", type=float, default=_LF.shaping_scale,
                    help="peso dello shaping potenziale che guida verso i punti buoni (0 = disattivato)")
    lf.add_argument("--isolated-guard", action="store_true",
                    help="riproduce la guardia dell'euristica sul vicinato vuoto (P=0 per entrambe le regole)")

    rl = parser.add_argument_group("Q-learning")
    rl.add_argument("--alpha", type=float, default=0.1)
    rl.add_argument("--gamma", type=float, default=0.95,
                    help="sconto: usato anche dallo shaping potenziale, che altrimenti non sarebbe invariante")
    rl.add_argument("--f-bins", type=int, default=4, help="bin di discretizzazione di f")
    rl.add_argument("--epsilon-start", type=float, default=0.30)
    rl.add_argument("--epsilon-end", type=float, default=0.02)
    rl.add_argument("--epsilon-decay", type=int, default=0,
                    help="episodi su cui decade epsilon (default: 3/4 degli episodi)")
    rl.add_argument("--allow-stay", action="store_true",
                    help="lascia all'RL l'azione 'resta fermo' (sconsigliato: punto fisso assorbente)")
    rl.add_argument("--no-learn-move", action="store_true", help="ablazione A2: movimento euristico")
    rl.add_argument("--no-learn-manip", action="store_true", help="ablazione A1: pick/drop euristico")

    costs = parser.add_argument_group("costi di base (RewardConfig)")
    costs.add_argument("--step-penalty", type=float, default=0.01)
    costs.add_argument("--contested-penalty", type=float, default=0.05)
    costs.add_argument("--invalid-penalty", type=float, default=0.05)

    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
