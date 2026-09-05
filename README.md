# Antelligent — euristica vs Reinforcement Learning

Progetto per la **tesi magistrale**. La stessa colonia di formiche che fa
**clustering/sorting** ant-based di semi colorati su una griglia viene fatta
girare in **due copie gemelle e inizialmente identiche** dello stesso campo
(stesso `master_seed`):

- **euristica** — movimento e raccolta/deposito guidati dall'algoritmo
  probabilistico pre-impostato (regole di Lumer–Faieta, costanti fisse);
- **RL** — le formiche **imparano** movimento e pick/drop con reinforcement
  learning, usando l'**entropia del sistema** come costo da minimizzare.

Documento di progetto completo: [`docs/rl-design.md`](docs/rl-design.md).

## Come si usa

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e ".[test]"

python -m antelligent.train --episodes 400          # addestra la politica RL -> results/policy.pkl
python -m antelligent                               # GUI di confronto affiancato
pytest                                              # test
```

### GUI di confronto (`python -m antelligent`)

Launcher di configurazione → una finestra con le **due griglie affiancate**
(`EURISTICA` a sinistra, `RL` a destra), i pulsanti **Start** / **Visibility**
condivisi in basso e, **sotto ciascuna griglia**, il pannello statistiche della
copia: **tempo** (aggiornato una volta al secondo), **mosse totali**, **semi
raccolti**, **entropia totale**.

Il pannello RL carica `results/policy.pkl` se presente (inferenza greedy,
congelata); altrimenti la politica **apprende dal vivo** durante la run (utile per
vedere l'RL migliorare, ma i risultati veri vanno prodotti con `train.py`).

### Addestramento (`python -m antelligent.train`)

Harness headless: molti episodi su campo rigenerato ogni volta, salva la politica
(`results/policy.pkl`) e le curve di apprendimento (`results/training_log.csv`).
Opzioni principali: `--episodes`, `--config`, `--out`, `--seed`, `--max-ticks`,
`--alpha --gamma --epsilon-start --epsilon-end`, e per le ablazioni
`--no-learn-move` (A2: movimento euristico) / `--no-learn-manip` (A1: pick/drop
euristico).

## Architettura

```
antelligent/
├── __main__.py / starter.py     entry point, config.properties round-trip
├── launcher.py                  form di configurazione (tkinter)
├── config.py  paths.py  world.py   config + percorsi + stato iniziale condiviso (master_seed)
├── actions.py                   Action / Observation / Transition / Manipulation
├── environment.py               Environment: observe(), step(), tick(); fisica di occupazione
│                                celle (try_acquire + ripiego sulla contesa), contatori, entropia
├── policies/
│   ├── base.py                  AntPolicy (il "seam")
│   ├── heuristic.py             HeuristicPolicy — regole pre-impostate (baseline A0)
│   └── tabular.py               TabularQPolicy — Q-learning fattorizzato a parametri condivisi (Fase 1)
├── train.py                     harness headless di addestramento
├── simulation/
│   ├── seed_matrix.py           SeedMatrix (ABC): primitive + probabilita' Lumer–Faieta + entropia
│   ├── lock_seed_matrix.py      matrice normale + un lock di occupazione per cella
│   ├── ant.py                   la formica come "corpo" (posizione, heading, seme trasportato)
│   ├── simulation.py            ComparisonSimulation — GUI a due griglie gemelle
│   ├── results_dialog.py  simulation_result.py
├── seeds/  utilities/  resources/images/
docs/rl-design.md                obiettivi, metriche, protocollo, sviluppi futuri
tests/                           pytest
```

Il *seam* `AntPolicy` separa il **corpo** della formica (fisica, invariata) dal
suo **cervello** (la politica). Lo stesso `Environment` serve sia l'euristica sia
l'RL: cambia solo la politica passata a `tick()`.

> **GIL.** I due ambienti girano su due thread worker indipendenti (uno per
> griglia); per via del GIL non e' vero parallelismo di calcolo, ma la struttura e
> la semantica dei lock di cella (contesa, `try_acquire`, occupazione esclusiva)
> sono mantenute. L'addestramento e' single-thread.

## Configurazione

`config.properties` (formato `chiave=valore`, riscritto dal launcher):

```properties
cols=40
rows=25
nThread=6
nAnts=100
nSeeds=650
seedTypes=5
type=1                   # matrice normale con lock
refreshRate=500          # iterazioni tra un repaint della board e il successivo
stopCriterion=1          # 0 = numero massimo di iterazioni, 1 = soglia di entropia
maxIterations=20000
entropyThreshold=5.0
masterSeed=0             # 0 = stato iniziale casuale a ogni run
```

Vincoli del launcher: `nSeeds <= righe*colonne` e `nAnts <= righe*colonne`.

## Output

`results/` (git-ignored):

- `results/results.txt` — CSV, **due righe per run** (mode `heuristic` / `rl`) con
  timestamp, durata, iterazioni, mosse totali, semi raccolti, entropia
  iniziale/finale, `masterSeed`;
- `results/training_log.csv` — una riga per episodio di addestramento;
- `results/policy.pkl` — politica RL addestrata;
- `results/screenshots/<data-ora>/` — screenshot periodici della finestra.

## Stato

- Fase 1 (`docs/rl-design.md` §5.1): **Q-learning tabellare** implementato e
  funzionante. Reward locale allineato all'entropia (§3.3).
- Da fare: reward shaping potenziale/incrementale globale, Fase 2 deep RL
  (DQN/PPO), warm start per imitazione, campagna sperimentale completa (§6, §8).
