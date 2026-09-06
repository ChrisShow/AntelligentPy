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
python -m antelligent.train_lumer --episodes 400    # variante: ricompensa alla Lumer-Faieta
python -m antelligent                               # GUI di confronto affiancato
pytest                                              # test
```

### GUI di confronto (`python -m antelligent`)

Launcher di configurazione → una finestra con le **due griglie affiancate**
(`EURISTICA` a sinistra, `RL` a destra), i pulsanti **Start** / **Visibility**
condivisi in basso e, **sotto ciascuna griglia**, il pannello statistiche della
copia: **tempo** (aggiornato una volta al secondo), **mosse totali**, **semi
raccolti**, **entropia totale**.

Le due run sono **sequenziali**: parte prima l'euristica, e l'RL solo quando la
prima ha concluso. Ogni copia ha il proprio cronometro etichettato (`tempo
euristica` / `tempo RL`) che riporta lo stato — `in attesa`, `in corso`,
`conclusa` — e si **ferma** sul valore finale a fine run. Così il wall-clock di
ciascuna politica è misurato senza contesa del GIL con l'altra griglia.

Il pannello RL carica `results/policy.pkl` se presente (inferenza greedy,
congelata); altrimenti la politica **apprende dal vivo** durante la run (utile per
vedere l'RL migliorare, ma i risultati veri vanno prodotti con `train.py`).

### Addestramento (`python -m antelligent.train`)

Harness headless: molti episodi su campo rigenerato ogni volta, salva la politica
(`results/policy.pkl`) e le curve di apprendimento (`results/training_log.csv`).
Opzioni principali: `--episodes`, `--config`, `--out`, `--seed`, `--max-ticks`,
`--alpha --gamma --epsilon-start --epsilon-end`, i pesi della ricompensa
`--step-penalty --contested-penalty --invalid-penalty --manip-scale --pivot`, e
per le ablazioni `--no-learn-move` (A2: movimento euristico) / `--no-learn-manip`
(A1: pick/drop euristico) / `--allow-stay`.

> `--pivot` è tarato automaticamente sul **livello del caso** `1/seedTypes`, e la
> testa del movimento **non** può scegliere di restare ferma (`allow_stay=False`).
> Entrambi i default evitano che la politica congelata si blocchi: vedi
> [`docs/uso.md` §5](docs/uso.md).

### Variante sperimentale: ricompensa alla Lumer-Faieta (`python -m antelligent.train_lumer`)

Seconda formulazione della ricompensa, in **file separati** (l'impianto sopra
resta invariato e continua a funzionare). Invece di misurare la manipolazione
rispetto al livello del caso `1/k`, la misura con le **stesse funzioni di
probabilità che governano l'euristica**: `P_pick = (kp/(kp+f))²`,
`P_drop = (f/(kd+f))²`. L'euristica *campiona* da quelle probabilità; qui
diventano un **segnale denso** che l'agente impara a sfruttare.

Tre contributi, tutti pesabili da riga di comando:

| contributo | flag | a cosa serve |
|---|---|---|
| azione | `--pick-scale` `--drop-scale` | quanto vale raccogliere / posare **in quel punto** |
| rifiuto | `--decline-scale` | informa anche il braccio "non manipolare" (l'euristica *estrae* fra agire e non agire) |
| navigazione | `--shaping-scale` | shaping potenziale `F = γΦ(s′) − Φ(s)` con `Φ` = probabilità LF nella cella: premia **avvicinarsi** a un buon punto di deposito |

Lo zero del segnale non è arbitrario. Il default `--mode centered` lo mette a
`P = 0.5` (per il drop: `f > 0.72`, una soglia molto **selettiva**);
`--mode advantage` lo mette in `f* = √(kp·kd) ≈ 0.173`, il punto in cui
`P_pick = P_drop`, cioè dove l'euristica stessa è indifferente.

**È la prima configurazione che batte l'euristica**: su `config.properties`,
400 episodi, 10 `master_seed` di valutazione, entropia finale **13.6 ± 2.5**
contro **25.5 ± 1.9** dell'euristica — più bassa su **10 seed su 10**. La leva
decisiva è `--drop-scale 2` (default): in modalità `centered` il segnale di
deposito ha un quinto dell'escursione di quello di raccolta, quindi senza
riequilibrio la politica impara meglio *quando raccogliere* che *dove posare*.
Tabelle, ablazioni e caveat in [`docs/uso.md` §3bis](docs/uso.md).

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
│   ├── tabular.py               TabularQPolicy — Q-learning fattorizzato a parametri condivisi (Fase 1)
│   └── lumer_tabular.py         LumerQPolicy — variante LF (stato manip. riallineato) + load_policy
├── lumer_reward.py              formule pure P_pick/P_drop -> ricompensa, punto di indifferenza f*
├── environment_lumer.py         LumerFaietaEnvironment — Environment con la ricompensa LF
├── train.py                     harness headless di addestramento
├── train_lumer.py               idem, per la variante Lumer-Faieta
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

> **Thread.** Ogni ambiente gira su un thread worker, ma **uno alla volta**: la
> copia RL parte solo quando l'euristica ha concluso, così il tempo di ciascuna è
> confrontabile. La semantica dei lock di cella (contesa, `try_acquire`,
> occupazione esclusiva) resta invariata. I worker **non toccano mai Tkinter**
> (che non è thread-safe): accodano eventi su una `queue.Queue` che il thread
> principale drena a 50 ms. L'addestramento è single-thread.

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
captureScreenshots=0     # 1 = salva schermate periodiche (casella "Opzioni" nel launcher)
```

Vincoli del launcher: `nSeeds <= righe*colonne` e `nAnts <= righe*colonne`.

## Output

`results/` (git-ignored):

- `results/results.txt` — CSV, **due righe per run** (mode `heuristic` / `rl`) con
  timestamp, durata, iterazioni, mosse totali, semi raccolti, entropia
  iniziale/finale, `masterSeed`;
- `results/training_log.csv` — una riga per episodio di addestramento;
- `results/policy.pkl` — politica RL addestrata;
- `results/policy_lumer.pkl` + `results/policy_lumer_log.csv` — variante
  Lumer-Faieta (il pickle dichiara la propria famiglia: la GUI sceglie da sola la
  classe giusta, quindi puoi salvarla direttamente come `results/policy.pkl`);
- `results/screenshots/<data-ora>/` — screenshot periodici della finestra, **solo
  se** la casella "Cattura schermate della finestra" è flaggata nel launcher
  (`captureScreenshots=1`); di default la cattura è disattivata.

## Stato

- Fase 1 (`docs/rl-design.md` §5.1): **Q-learning tabellare** implementato e
  funzionante. Reward locale allineato all'entropia (§3.3). L'osservazione include
  un **gradiente locale** (`best_dir`) che indica dove raccogliere / dove posare, e
  la politica congelata mantiene un rumore residuo (`inference_epsilon`) senza il
  quale, su stati aliasati, cadrebbe in cicli limite — vedi `docs/uso.md` §5.
- Su `config.properties` (14×14, 130 semi, 5 tipi), 10 `master_seed`: euristica
  H ≈ 25.5, RL con ricompensa al livello del caso H ≈ 38.8, **RL con ricompensa
  alla Lumer-Faieta H ≈ 13.6** (più bassa dell'euristica su 10 seed su 10).
  La variante LF include anche il primo pezzo di **reward shaping potenziale**.
- Da fare: campagna con più repliche di addestramento (la varianza fra seed è
  dello stesso ordine dello sweep di `--drop-scale`), shaping incrementale
  globale, Fase 2 deep RL (DQN/PPO), warm start per imitazione (§6, §8).
