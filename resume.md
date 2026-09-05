# Resume — AntelligentPy

Diario di bordo del progetto (dal più recente in alto), a partire dal 2 settembre
2026. Contiene due tipi di voce, in ordine cronologico unico:

- **`[MODIFICA]`** — cambiamenti al progetto (codice, config, doc, test): *cosa*,
  *perché*, *quali file*.
- **`[RISULTATI]`** — esiti degli esperimenti (run di addestramento, run di
  confronto euristica vs RL, ablazioni): scenario, comandi usati, numeri, grafici
  e osservazioni. Serve a ricostruire l'evoluzione del lavoro e a riportare i
  dati nella tesi magistrale.

**Template `[MODIFICA]`:**

```
## AAAA-MM-GG — [MODIFICA] Titolo breve

**Obiettivo.** Perché questa modifica.
**Modifiche.**
- ...
**File.** `percorso/relativo`, ...
**Test.** cosa è stato verificato / stato di `pytest`.
**Note / follow-up.** eventuali rischi o cose da fare dopo.
```

**Template `[RISULTATI]`:**

```
## AAAA-MM-GG — [RISULTATI] Titolo breve

**Scenario.** griglia, nAnts, nSeeds, seedTypes, criterio di arresto, master_seed,
  numero di ripetizioni.
**Setup.** modalità/ablazione (A0–A3), iperparametri (QConfig), pesi ricompensa
  (RewardConfig), comandi esatti eseguiti, commit/data del codice.
**Metriche.** entropia finale, iterazioni/tempo alla soglia, AUC entropia–iter,
  mosse totali, semi raccolti, ecc. (tabella; media ± IC 95% se più ripetizioni).
**File.** `results/...` (log CSV, policy.pkl, screenshot, grafici) — dove sono.
**Osservazioni.** cosa se ne ricava, confronto con l'euristica, anomalie.
**Follow-up.** cosa provare dopo.
```

---

## 2026-09-05 — [MODIFICA] Pulizia dei riferimenti al progetto Java e alla tesi triennale

**Obiettivo.** Presentare `AntelligentPy` come lavoro autonomo per la tesi
magistrale: rimuovere da commenti/docstring del codice e da tutti i file `.md`
ogni riferimento al progetto Java di origine e alla tesi triennale.

**Modifiche.**

- **Docstring dei moduli** — tolto il "port di ``X.java``" da `config.py`,
  `launcher.py`, `starter.py`, `__init__.py`, `seeds/seed.py`,
  `seeds/seed_type.py`, `utilities/position.py`, `utilities/check_move.py`,
  `simulation/ant.py`, `simulation/seed_matrix.py`,
  `simulation/lock_seed_matrix.py`, `simulation/simulation_result.py` e dai test
  `test_check_move.py`, `test_seed_type.py`, `test_seed_matrix.py`. Sostituito con
  una descrizione autonoma del modulo.
- **Commenti interni** — `lock_seed_matrix.py` (niente più `ReentrantLock`,
  `ConcurrentLinkedQueue`, `Collections.synchronizedList`, `CustomMatrix`,
  `seeds[posY][posX]`, `new ReentrantLock(true)`); `check_move.py` (niente più
  "identiche al Java", `randomMove`, `do/while`); `seed_matrix.py` (niente più
  "Nel Java … NaN", "Fedele al Java"); `environment.py`, `policies/heuristic.py`
  (tolto "versione plain" / "vecchio ``Ant.__call__``"); `config.py`
  ("continuità" → "compatibilità" del formato).
- **File `.md`** — `README.md`, `docs/rl-design.md` (rimosso il punto
  "Continuità con la tesi triennale", gli "scenari ripresi dalla triennale", i
  rimandi alla triennale negli sviluppi futuri e nei riferimenti),
  `docs/uso.md` (riscritta la nota sul `results.txt` con intestazione diversa
  senza parlare di "versione vecchia"), questo `resume.md` (voce baseline).

**File.** `antelligent/{__init__,config,launcher,starter,environment}.py`,
`antelligent/seeds/{seed,seed_type}.py`,
`antelligent/utilities/{position,check_move}.py`,
`antelligent/simulation/{ant,seed_matrix,lock_seed_matrix,simulation_result}.py`,
`antelligent/policies/heuristic.py`,
`tests/{test_check_move,test_seed_type,test_seed_matrix}.py`, `README.md`,
`docs/rl-design.md`, `docs/uso.md`, `resume.md`.

**Test.** `pytest -q` → **36 passed** (solo commenti/docstring toccati).

**Note / follow-up.** Nessuna modifica di comportamento. Il progetto Java resta
in `../Antelligent/` come riferimento storico personale, ma non è più citato dal
codice o dalla documentazione del port.

---

## 2026-09-02 — [MODIFICA] Baseline: rifattorizzazione per il confronto euristica vs RL

**Obiettivo.** Impostare il progetto sul confronto tra l'**algoritmo euristico
pre-impostato** e una versione con **reinforcement learning**, secondo
`docs/rl-design.md`. Le due popolazioni girano su **due copie gemelle** della
stessa griglia (stesso `master_seed`); GUI a due griglie affiancate con
statistiche per copia.

**Modifiche.**

- **Seam `AntPolicy`** — nuovo package `antelligent/policies/`:
  - `base.py` — `AntPolicy` (Protocol) + `AntPolicyBase` (record/end_episode no-op);
  - `heuristic.py` — `HeuristicPolicy`: regole di Lumer–Faieta + random walk,
    algoritmo euristico pre-impostato (baseline A0);
  - `tabular.py` — `TabularQPolicy`: Q-learning fattorizzato (`q_move` 9 azioni +
    `q_manip` 2 azioni) a parametri condivisi, ε-greedy con decadimento,
    `save`/`load` (pickle), `freeze()`, `bind_environment()` per le ablazioni
    A1/A2; `QConfig` per gli iperparametri.
- **Ambiente** — nuovo `antelligent/environment.py`:
  - `Environment` con `observe()` (osservazione locale/parziale: carrying, `f`
    locale, `blocked`, `best_dir`, finestra categorica opzionale), `step()`
    (manipolazione + movimento; **fisica di occupazione celle `try_acquire` +
    ripiego sulla contesa**; contatori
    `total_moves` / `seeds_collected`), `tick()` (un giro di tutte le formiche in
    ordine casuale, delega alla politica), `is_done()`, `entropy`;
  - `RewardConfig` — ricompensa locale allineata all'entropia, centrata su
    `pivot` (raccogliere semi isolati / posare tra i simili premia, il "churn"
    no); `final_scale` predisposto ma non collegato.
- **Stato iniziale condiviso** — nuovo `antelligent/world.py`:
  `build_initial_state(config, master_seed)` → `InitialState` (posizioni semi,
  celle e heading formiche); `resolve_master_seed()` (0 → casuale).
  `Environment.from_initial_state()` materializza due ambienti identici.
- **Tipi condivisi** — nuovo `antelligent/actions.py`: `Manipulation`, `Action`,
  `Observation`, `Transition`, `StepResult`, sentinella `STAY`.
- **`SeedMatrix`** (`antelligent/simulation/seed_matrix.py`) rifattorizzata:
  aggiunte `pick_probability` / `drop_probability` (pure, no RNG),
  `commit_pick` / `commit_drop` (mutazioni incondizionate),
  `count_types_around` (era privata), `check_entropy_placed` (entropia sui soli
  semi posati, **mai −1**), helper `_local_entropy`. `pick_seed` / `drop_seed` /
  `check_entropy` mantenute per compatibilità e per i test.
- **`Ant`** (`antelligent/simulation/ant.py`) ridotta a "corpo": `code`,
  `position`, `direction`, `carried_seed`, `image_path`. Niente più logica di
  decisione né riferimento alla matrice.
- **GUI** — `antelligent/simulation/simulation.py`: `ComparisonSimulation`.
  Due griglie gemelle affiancate (EURISTICA / RL),
  pulsanti **Start** / **Visibility** condivisi in basso, pannello statistiche
  per copia (**tempo aggiornato a 1 Hz**, mosse totali, semi raccolti, entropia
  totale). Due thread worker indipendenti; repaint governato da `refreshRate`,
  separato dal timer statistiche.
  - `results_dialog.py` — riepilogo a colonne (una per copia).
  - `simulation_result.py` — esteso: `mode`, `total_moves`, `seeds_collected`,
    `master_seed`.
- **`check_move`** (`antelligent/utilities/check_move.py`): nuova
  `random_move_index()` (indice relativo, condiviso da euristica e ambiente) +
  sentinella `STAY` restituita quando nessuna adiacente è in griglia (niente più
  loop infinito su 1×1); `random_move()` riscritta su di essa, ora può
  restituire `(None, direction)`.
- **`SeedType.from_code()`** (`antelligent/seeds/seed_type.py`).
- **Config** — `antelligent/config.py`: nuovo campo `master_seed`.
  `config.properties` (+ copia in `antelligent/resources/`): nuova chiave
  `masterSeed`. `starter.py` legge/salva `masterSeed` e avvia
  `ComparisonSimulation` (carica `results/policy.pkl` se presente).
  `launcher.py`: campo "Seed iniziale (0 = casuale)".
- **Addestramento** — nuovo `antelligent/train.py`: harness headless
  (`python -m antelligent.train`), episodi su campo rigenerato, salva
  `results/policy.pkl` + `results/training_log.csv`. CLI: `--episodes --config
  --out --seed --max-ticks --entropy-every --alpha --gamma --epsilon-* 
  --no-learn-move --no-learn-manip --log-every`.
- **Output** — `results/results.txt`: nuova intestazione, **due righe per run**
  (`heuristic` / `rl`), colonne `mosseTotali` / `semiRaccolti` / `masterSeed`.
- **Packaging** — `pyproject.toml`: versione `3.0.0`, script
  `antelligent-train`, descrizione aggiornata.
- **Documentazione** — `docs/rl-design.md` (documento di progetto RL, già
  esistente), `docs/uso.md` (guida installazione/addestramento/confronto/
  troubleshooting), `README.md` riscritto, questo `resume.md`.

**File.** nuovi: `antelligent/actions.py`, `antelligent/environment.py`,
`antelligent/world.py`, `antelligent/train.py`, `antelligent/policies/*`,
`docs/uso.md`, `resume.md`. Modificati: `antelligent/config.py`,
`antelligent/starter.py`, `antelligent/launcher.py`,
`antelligent/simulation/{simulation,seed_matrix,ant,results_dialog,simulation_result}.py`,
`antelligent/seeds/seed_type.py`, `antelligent/utilities/check_move.py`,
`config.properties`, `antelligent/resources/config.properties`, `pyproject.toml`,
`README.md`. Rimosso: `tests/test_ant_movement.py`.

**Test.** `pytest` → **36 passed**. I test esistenti (`check_move`,
`seed_type`, `seed_matrix`) restano invariati + nuovi: `test_environment.py`,
`test_heuristic_policy.py`, `test_world.py`, `test_tabular_policy.py`,
`tests/conftest.py` (fixture `make_config`). Smoke verificati: training headless
(entropia decrescente, pick in calo), copie gemelle identiche all'avvio e
indipendenti dopo, invarianti (una formica per cella, semi conservati),
costruzione GUI + tick sincroni.

**Note / follow-up (Fase 2, da `docs/rl-design.md` §5-§8).**
- La Q-learning tabellare *impara* ma **non batte ancora l'euristica**: è atteso
  per la Fase 1. Tuning di `RewardConfig` / `QConfig` + più episodi = lavoro
  sperimentale della tesi.
- Da fare: reward shaping potenziale/incrementale globale; Fase 2 deep RL
  (DQN/PPO, dipendenza `torch` opzionale); warm start per imitazione; campagna
  sperimentale completa con test appaiati.
- `RewardConfig` non è ancora esposto come CLI in `train.py` (solo default).
- `final_scale` (bonus terminale) definito ma non collegato al runner.
