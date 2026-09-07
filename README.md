# Antelligent: euristica vs reinforcement learning

*Sviluppato nell'ambito di una tesi di laurea magistrale.*

## Progetto

Una colonia di formiche ordina semi colorati su una griglia con il classico
schema ant-based di clustering e sorting: una formica raccoglie un seme dove ne
trova pochi dello stesso tipo e lo posa dove ne trova tanti. Il campo viene fatto
girare in due copie che partono identiche (stesso `master_seed`), una per ciascuna
politica:

- **euristica**: movimento e raccolta/deposito decisi dall'algoritmo probabilistico
  di Lumer-Faieta, con costanti fisse `kp` e `kd`;
- **RL**: le formiche imparano quando muoversi e quando raccogliere o posare, con
  l'entropia del sistema come costo da minimizzare.

L'entropia misura quanto i semi di un tipo sono mescolati agli altri: più scende,
più il campo è diviso in gruppi omogenei. È la metrica su cui le due politiche
vengono confrontate.

Documento di progetto: [`docs/rl-design.md`](docs/rl-design.md). Guida d'uso
dettagliata: [`docs/uso.md`](docs/uso.md). Diario delle modifiche e degli
esperimenti: [`resume.md`](resume.md).

## Obiettivi

Le domande a cui il confronto deve rispondere:

| | Domanda | Come si misura |
|---|---|---|
| RQ1 | A parità di budget di iterazioni, l'RL arriva a un'entropia finale minore dell'euristica? | Entropia finale su una griglia di scenari, con test appaiati sullo stesso `master_seed` |
| RQ2 | L'RL raggiunge la soglia di entropia in meno iterazioni o meno tempo? | Iterazioni alla soglia, area sotto la curva entropia-iterazioni |
| RQ3 | Il guadagno viene dal movimento appreso, dalla manipolazione appresa o da entrambi? | Ablazioni A1 (solo movimento), A2 (solo manipolazione), A3 (entrambi) |
| RQ4 | La politica generalizza a scenari mai visti in addestramento? | Addestramento su una configurazione, test su densità e dimensioni diverse |
| RQ5 | L'RL apprende qualcosa di qualitativamente diverso dall'euristica? | Curve pick/drop apprese, tasso di contesa sulle celle, heatmap |
| RQ6 | Quanto costa l'RL rispetto al beneficio? | Curve di apprendimento, wall-clock di addestramento, dimensione della tabella |

## Architettura

```
antelligent/
├── __main__.py / starter.py     entry point, round-trip di config.properties
├── launcher.py                  form di configurazione (tkinter)
├── config.py  paths.py  world.py   config, percorsi, stato iniziale condiviso
├── actions.py                   Action, Observation, Transition, Manipulation
├── environment.py               observe(), step(), tick(), occupazione delle celle,
│                                contatori, entropia
├── policies/
│   ├── base.py                  AntPolicy, il "seam" fra corpo e cervello
│   ├── heuristic.py             HeuristicPolicy: regole pre-impostate (baseline A0)
│   ├── tabular.py               TabularQPolicy: Q-learning fattorizzato a parametri condivisi
│   └── lumer_tabular.py         LumerQPolicy: variante LF, più il dispatch load_policy
├── lumer_reward.py              formule pure P_pick/P_drop, punto di indifferenza f*
├── environment_lumer.py         LumerFaietaEnvironment: Environment con la ricompensa LF
├── train.py                     harness headless di addestramento
├── train_lumer.py               idem, per la variante Lumer-Faieta
├── simulation/
│   ├── seed_matrix.py           SeedMatrix (ABC): primitive, probabilità LF, entropia
│   ├── lock_seed_matrix.py      matrice con un lock di occupazione per cella
│   ├── ant.py                   la formica come corpo: posizione, heading, seme in mano
│   ├── simulation.py            ComparisonSimulation: GUI a due griglie gemelle
│   └── results_dialog.py  simulation_result.py
└── seeds/  utilities/  resources/images/
docs/rl-design.md                obiettivi, metriche, protocollo sperimentale
tests/                           pytest
```

Il seam `AntPolicy` separa il corpo della formica, che è fisica e resta invariato,
dal suo cervello, che è la politica. Lo stesso `Environment` serve entrambe le
politiche: cambia solo l'oggetto passato a `tick()`.

**Thread.** Ogni ambiente gira su un thread worker, ma uno alla volta: la copia RL
parte quando l'euristica ha finito, così il tempo di ciascuna è confrontabile. I
worker non toccano mai Tkinter, che non è thread-safe; accodano eventi su una
`queue.Queue` che il thread principale drena ogni 50 ms. L'addestramento è
single-thread.

## Installazione

Serve Python 3.11 o superiore.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  su macOS/Linux: source .venv/bin/activate
pip install -e ".[test]"
pytest
```

Senza `pip install`, i comandi `python -m antelligent ...` funzionano solo dalla
cartella `AntelligentPy/`.

La configurazione sta in `config.properties`, formato `chiave=valore`, riscritto
dal launcher a ogni avvio:

```properties
cols=40
rows=25
nThread=6
nAnts=100
nSeeds=650
seedTypes=5
type=1                   # matrice normale con lock
refreshRate=500          # iterazioni fra un repaint della board e il successivo
stopCriterion=1          # 0 = numero massimo di iterazioni, 1 = soglia di entropia
maxIterations=20000
entropyThreshold=5.0
masterSeed=0             # 0 = stato iniziale casuale a ogni run
captureScreenshots=0     # 1 = salva schermate periodiche
```

Il launcher richiede `nSeeds <= righe*colonne` e `nAnts <= righe*colonne`.

## Funzionamento

```bash
python -m antelligent.train --episodes 400          # addestra la politica RL
python -m antelligent.train_lumer --episodes 400    # variante con ricompensa Lumer-Faieta
python -m antelligent                               # GUI di confronto affiancato
```

### GUI di confronto

Il launcher di configurazione apre una finestra con le due griglie affiancate,
`EURISTICA` a sinistra e `RL` a destra, i pulsanti Start e Visibility condivisi in
basso e, sotto ciascuna griglia, il pannello statistiche della copia: tempo,
mosse totali, semi raccolti, entropia totale.

Le due run sono sequenziali. Ogni copia ha il proprio cronometro etichettato
(`tempo euristica`, `tempo RL`) che segnala lo stato (`in attesa`, `in corso`,
`conclusa`) e si ferma sul valore finale. Così il wall-clock di ciascuna politica
è misurato senza contesa del GIL con l'altra griglia.

Il pannello RL carica `results/policy.pkl` se esiste, in inferenza greedy
congelata. Se il file manca, la politica apprende dal vivo durante la run: serve
a vedere l'RL migliorare, ma i numeri da riportare si producono con `train.py`.

### Addestramento

`python -m antelligent.train` esegue molti episodi su un campo rigenerato ogni
volta, salva la politica in `results/policy.pkl` e le curve di apprendimento in
`results/training_log.csv`. Opzioni principali: `--episodes`, `--config`, `--out`,
`--seed`, `--max-ticks`, `--alpha --gamma --epsilon-start --epsilon-end`, i pesi
della ricompensa `--step-penalty --contested-penalty --invalid-penalty
--manip-scale --pivot`, e per le ablazioni `--no-learn-move` (A2, movimento
euristico), `--no-learn-manip` (A1, pick/drop euristico), `--allow-stay`.

`--pivot` è tarato sul livello del caso `1/seedTypes`, e la testa del movimento
non può scegliere di restare ferma (`allow_stay=False`). Entrambi i default
servono a impedire che la politica congelata si blocchi: il motivo è spiegato in
[`docs/uso.md` §5](docs/uso.md).

### Variante con ricompensa Lumer-Faieta

`python -m antelligent.train_lumer` usa una seconda formulazione della ricompensa,
in file separati, che lascia intatto l'impianto precedente. Invece di misurare la
manipolazione rispetto al livello del caso `1/k`, la misura con le stesse funzioni
di probabilità che governano l'euristica: `P_pick = (kp/(kp+f))²` e
`P_drop = (f/(kd+f))²`. L'euristica campiona da quelle probabilità; qui diventano
un segnale denso che l'agente impara a sfruttare.

Tre contributi, tutti pesabili da riga di comando:

| contributo | flag | a cosa serve |
|---|---|---|
| azione | `--pick-scale` `--drop-scale` | quanto vale raccogliere o posare in quel punto |
| rifiuto | `--decline-scale` | informa anche il braccio "non manipolare", visto che l'euristica estrae fra agire e non agire |
| navigazione | `--shaping-scale` | shaping potenziale `F = γΦ(s′) − Φ(s)` con `Φ` = probabilità LF nella cella, per premiare l'avvicinarsi a un buon punto di deposito |

Lo zero del segnale si può spostare. Il default `--mode centered` lo mette a
`P = 0.5`, che per il deposito vuol dire `f > 0.72`, una soglia molto selettiva.
`--mode advantage` lo mette in `f* = √(kp·kd) ≈ 0.173`, il punto in cui
`P_pick = P_drop` e l'euristica stessa è indifferente.

### Output

Tutto finisce in `results/`, che è git-ignored:

- `results.txt`, CSV con due righe per run (`heuristic` e `rl`): timestamp,
  durata, iterazioni, mosse totali, semi raccolti, entropia iniziale e finale,
  `masterSeed`;
- `training_log.csv`, una riga per episodio di addestramento;
- `policy.pkl`, la politica addestrata caricata dalla GUI;
- `policy_lumer.pkl` e `policy_lumer_log.csv` per la variante LF. Il pickle
  dichiara la propria famiglia, quindi la GUI sceglie da sola la classe giusta e
  puoi salvare il file direttamente come `results/policy.pkl`;
- `screenshots/<data-ora>/`, schermate periodiche della finestra, solo se la
  casella "Cattura schermate della finestra" è flaggata nel launcher.

## Risultati

Su `config.properties` (14×14, 130 semi, 5 tipi), 400 episodi di addestramento e
10 `master_seed` di valutazione, entropia finale media:

| politica | entropia finale | seed vinti su 10 |
|---|---|---|
| euristica | 25.5 ± 1.9 | riferimento |
| RL, ricompensa al livello del caso `1/k` | 38.8 ± 1.9 | 0 |
| RL, ricompensa Lumer-Faieta | **13.6 ± 2.5** | 10 |

La variante Lumer-Faieta è la prima configurazione che scende sotto l'euristica, e
lo fa su tutti e dieci i seed di valutazione. La leva decisiva è `--drop-scale 2`,
che è il default: in modalità `centered` il segnale di deposito ha un quinto
dell'escursione di quello di raccolta, quindi senza riequilibrio la politica impara
meglio quando raccogliere che dove posare.

Tre cautele sui numeri, tutte misurate:

- lo sweep di `--drop-scale` su `1.0 / 1.5 / 2 / 3 / 4` dà
  `19.9 / 22.1 / 13.6 / 18.1 / 18.0`, che non è monotono;
- la varianza fra seed di addestramento (13.6, 13.1, 19.5 con lo stesso
  `--drop-scale 2`) è dello stesso ordine dello sweep, quindi `2` è il punto
  migliore misurato e non un ottimo dimostrato;
- 800 episodi danno un risultato peggiore di 400 (21.7 contro 13.6): con ε già
  decaduto la politica smette di esplorare e deriva.

Tabelle complete, ablazioni e note metodologiche in
[`docs/uso.md` §3bis](docs/uso.md).

### Stato del lavoro

La Fase 1 (`docs/rl-design.md` §5.1), Q-learning tabellare, è implementata e
funziona. L'osservazione include un gradiente locale (`best_dir`) che indica dove
raccogliere e dove posare, e la politica congelata mantiene un rumore residuo
(`inference_epsilon`) senza il quale, su stati aliasati, cadrebbe in cicli limite.
La variante LF include anche il primo pezzo di reward shaping potenziale.

Restano da fare: una campagna con più repliche di addestramento, lo shaping
incrementale globale, la Fase 2 con deep RL (DQN o PPO) e un warm start per
imitazione.
