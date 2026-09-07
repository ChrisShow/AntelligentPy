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

## Obiettivi

Le domande a cui il confronto deve rispondere:

| | Domanda | Come si misura |
|---|---|---|
| RQ1 | A parità di budget di iterazioni, l'RL arriva a un'entropia finale minore dell'euristica? | Entropia finale su una griglia di scenari, con test appaiati sullo stesso `master_seed` |
| RQ2 | L'RL raggiunge la soglia di entropia in meno iterazioni o meno tempo? | Iterazioni alla soglia, area sotto la curva entropia-iterazioni |
| RQ3 | Il guadagno viene dal movimento appreso, dalla manipolazione appresa o da entrambi? | Ablazioni A1, A2, A3 |
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
cartella `AntelligentPy/`. Con `pip install -e .` sono disponibili anche gli
eseguibili `antelligent`, `antelligent-train` e `antelligent-train-lumer`.

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

Il launcher richiede `nSeeds <= righe*colonne` e `nAnts <= righe*colonne`. Tieni
`nAnts` ben sotto il numero di celle: se ci si avvicina, il piazzamento iniziale
delle formiche fatica a trovare celle libere e la run sembra bloccata all'avvio.

`captureScreenshots` corrisponde alla casella "Cattura schermate della finestra"
nel launcher, ed è disattivata di default: la cattura blocca il thread della GUI a
ogni scatto e su macOS richiede il permesso in Impostazioni di Sistema, Privacy e
sicurezza, Registrazione schermo.

## Funzionamento

```bash
python -m antelligent.train --episodes 400          # addestra la politica RL
python -m antelligent.train_lumer --episodes 400    # variante con ricompensa Lumer-Faieta
python -m antelligent                               # GUI di confronto affiancato
```

### GUI di confronto

Il launcher di configurazione apre una finestra con le due griglie affiancate,
`EURISTICA` a sinistra e `RL` a destra, i pulsanti Start e Visibility condivisi in
basso e, sotto ciascuna griglia, il pannello statistiche della copia: tempo, mosse
totali, semi raccolti, entropia totale.

Le due run sono sequenziali. Ogni copia ha il proprio cronometro etichettato
(`tempo euristica`, `tempo RL`) che segnala lo stato (`in attesa`, `in corso`,
`conclusa`) e si ferma sul valore finale. Così il wall-clock di ciascuna politica
è misurato senza contesa del GIL con l'altra griglia.

Il pannello RL carica `results/policy.pkl` se esiste, in inferenza greedy
congelata. Se il file manca, la politica apprende dal vivo durante la run: serve a
vedere l'RL migliorare, ma i numeri da riportare si producono con `train.py`.

### Addestramento

`python -m antelligent.train` esegue molti episodi su un campo rigenerato ogni
volta, salva la politica in `results/policy.pkl` e le curve di apprendimento in
`results/training_log.csv`.

Opzioni: `--episodes`, `--config`, `--out`, `--seed`, `--max-ticks`,
`--entropy-every`, `--log-every`, gli iperparametri `--alpha --gamma
--epsilon-start --epsilon-end --epsilon-decay`, e per le ablazioni
`--no-learn-move`, `--no-learn-manip`, `--allow-stay`.

I pesi della ricompensa (`RewardConfig` in `antelligent/environment.py`) sono
esposti come flag:

| Flag | Default | Effetto |
|---|---|---|
| `--step-penalty` | 0.01 | penalità per tick, spinge a essere veloce |
| `--contested-penalty` | 0.05 | penalità se la cella scelta è occupata |
| `--invalid-penalty` | 0.05 | penalità per un pick o drop non valido |
| `--manip-scale` | 1.0 | peso del segnale di manipolazione |
| `--pivot` | automatico, `1/seedTypes` | soglia di `f_own`, la frazione di vicini dello stesso tipo del seme manipolato: il pick è premiato sotto, il drop sopra |

`RewardConfig.final_scale` (bonus terminale `α·(H0 − H_final)`) è definito ma non
ancora collegato al runner.

### Variante con ricompensa Lumer-Faieta

`python -m antelligent.train_lumer` usa una seconda formulazione della ricompensa,
in file separati, che lascia intatto l'impianto precedente. Le due varianti
convivono e si confrontano.

L'euristica decide campionando dalle probabilità di Lumer-Faieta:

```
P_pick(f) = (kp/(kp+f))²     kp = 0.1     raccogli tanto più volentieri quanto il seme è isolato
P_drop(f) = (f/(kd+f))²      kd = 0.3     posa tanto più volentieri quanto è circondato dai suoi simili
```

dove `f` è la frazione, fra gli 8 vicini, dei semi dello stesso tipo di quello
manipolato. Qui le stesse formule diventano valore invece che probabilità:
l'agente non tira più il dado, ma riceve un rinforzo tanto più alto quanto più
l'azione scelta è quella che l'euristica giudicherebbe conveniente in quel punto.
È la stessa conoscenza di dominio dell'euristica, ma appresa invece che cablata, e
quindi migliorabile: l'RL può imparare a cercare i punti buoni, cosa che
l'euristica con il suo random walk non fa.

Il segnale deve avere un segno, altrimenti manipolare batte sempre non manipolare e
la colonia entra in churn, cioè raccoglie e riposa a caso. Lo zero non è scelto a
mano: `--mode advantage` lo mette nel valore di `f` in cui le due regole si
equivalgono,

```
P_pick(f) = P_drop(f)   ⟺   kp/(kp+f) = f/(kd+f)   ⟺   f* = √(kp·kd) ≈ 0.1732
```

Sotto `f*` l'euristica preferisce raccogliere, sopra preferisce posare. Il default
`--mode centered` mette invece lo zero a `P = 0.5`, che per il deposito vuol dire
`f > 0.72`, una soglia molto più selettiva, ed è quello che ha dato i risultati
migliori.

Quattro contributi, tutti pesabili da riga di comando:

| contributo | flag | cosa insegna |
|---|---|---|
| azione | `--pick-scale` `--drop-scale` | quanto vale raccogliere o posare in quel punto |
| rifiuto | `--decline-scale` | se poteva manipolare con vantaggio e non l'ha fatto, paga quel vantaggio: anche il braccio "non agire" viene informato |
| navigazione | `--shaping-scale` | shaping potenziale `F = γ·Φ(s′) − Φ(s)` con `Φ` = probabilità LF della manipolazione desiderata nella cella corrente: avvicinarsi a un buon punto di deposito paga prima di posare |
| muro | `--wall-penalty` | costo di una mossa scelta fuori dalla griglia |

Lo shaping è potenziale nel senso di Ng, Harada e Russell (1999): la somma su un
cammino dipende solo dagli estremi, quindi girare in tondo non frutta niente e la
politica ottima non cambia, cambia solo la velocità con cui la si trova. Per
questo `--gamma` viene passato anche allo shaping: con uno sconto diverso
l'invarianza si perde.

Oltre a tutte le opzioni di `train.py`:

| Flag | Default | Significato |
|---|---|---|
| `--mode {advantage,centered,raw}` | `centered` | `centered` = `2P − 1`, zero a probabilità 50 %, il migliore misurato; `advantage` = `P(azione) − P(opposta)`, zero in `f*`; `raw` = `P`, mai negativo, ablazione che mostra il churn |
| `--pick-scale` | 1.0 | peso del segnale di raccolta |
| `--drop-scale` | 2.0 | peso del segnale di deposito, cioè del punto in cui si posa |
| `--decline-scale` | 1.0 | peso della penalità per aver rifiutato una manipolazione conveniente (`0` la rende gratis) |
| `--decline-two-sided` | | ablazione: premia anche il rifiuto di una manipolazione sconveniente (patologico, vedi sotto) |
| `--shaping-scale` | 1.0 | peso dello shaping potenziale (`0` lo disattiva) |
| `--wall-penalty` | 0.05 | costo di una mossa scelta fuori dalla griglia (`0` la rende gratis, sconsigliato) |
| `--kp` `--kd` | 0.1 / 0.3 | costanti di Lumer-Faieta, le stesse dell'euristica; spostarle sposta `f*` |
| `--isolated-guard` | | riproduce alla lettera la guardia dell'euristica sul vicinato vuoto (`P = 0` per entrambe le regole) |
| `--f-bins` | 4 | granularità della discretizzazione di `f` |
| `--out` | `results/policy_lumer.pkl` | il log finisce accanto, come `<nome>_log.csv` |

Su `--isolated-guard`: con un vicinato completamente vuoto `f = 0/0` non è definita
e l'euristica restituisce `0` per entrambe le regole. Di default qui le formule si
estendono per continuità (`f = 0`), quindi raccogliere un seme completamente
isolato vale `+1`, perché è il seme più fuori posto che esista, e posarne uno nel
deserto vale `−1`. Riprodurre la guardia renderebbe il deserto un posto neutro
dove scaricare i semi.

`LumerQPolicy` eredita da `TabularQPolicy`: stessa Q-learning fattorizzata, stesso
`_move_state` compatto col gradiente locale, stesso `allow_stay = False`, stesso
`inference_epsilon`. L'unica differenza è la discretizzazione della testa
pick/drop, che qui ignora il tipo del seme e guarda solo `f` e la sua posizione
rispetto a `f*`.

Per usarla nella GUI basta salvarla dove la GUI la cerca:

```bash
python -m antelligent.train_lumer --episodes 400 --out results\policy.pkl
python -m antelligent
```

Il pickle dichiara la propria famiglia, quindi la GUI sceglie da sola la classe
giusta. Serve perché le due varianti codificano la testa di manipolazione in modo
diverso, e caricare l'una come l'altra fallirebbe in silenzio: letture sistematiche
su righe mai viste, cioè azioni a caso senza nessun errore. Se punti al file
sbagliato, il log lo dice.

### Quattro default che conviene non toccare

Sono tutti rimedi a patologie osservate durante l'addestramento, non scelte di
gusto.

**`--pivot` automatico.** Su una cella a caso la frazione attesa di vicini dello
stesso tipo è `1/k` con `k = seedTypes`. Un `pivot` fisso a `0.5` con `k = 5` rende
il drop penalizzato in media (`0.2 − 0.5 = −0.3`): la politica greedy impara a non
posare mai e le formiche restano bloccate col seme in mano. Il default `1/k` mette
lo zero al livello del caso, così il segno della ricompensa distingue "meglio del
caso" da "peggio del caso".

**`QConfig.allow_stay = False`.** La testa del movimento sceglie fra le 8 direzioni
relative, come l'euristica. Con `STAY` disponibile la Q-learning ci cade dentro:
restare fermi non rischia mai `contested_penalty`, quindi è l'azione meno costosa
ovunque non ci sia una ricompensa positiva raggiungibile, e il valore
`Q(s, STAY) = −step_penalty/(1−γ)` si auto-sostiene come self-loop assorbente. La
fisica gestisce comunque il caso "non riesco a muovermi" con un ripiego su
un'adiacente libera. `--allow-stay` la riabilita per le ablazioni.

**`QConfig.inference_epsilon = 0.05`.** È la probabilità di mossa casuale che resta
anche a politica congelata. L'osservazione è parziale e fortemente aliasata, cioè
stati diversi del mondo appaiono identici alla formica, e una politica
deterministica su stati aliasati cade in cicli limite, avanti e indietro fra le
stesse due celle, da cui non può uscire perché la scelta dipende solo
dall'osservazione, che non cambia. In un POMDP la politica ottima è in generale
stocastica, e l'euristica di confronto lo è già, quindi il paragone resta equo.
Misurato su `config.properties`, 5 `master_seed`:

| `inference_epsilon` | H finale | ritorni sulla cella `t-2` | picks |
|---|---|---|---|
| 0.00 (greedy pura) | 44.7 | 30.2 % | 69 |
| 0.05 (default) | 36.6 | 20.6 % | 206 |
| 0.10 | 36.8 | 18.5 % | 303 |
| 0.20 | 36.7 | 15.2 % | 465 |

`freeze()` sospende l'apprendimento, non il rumore.

**Rifiuto unilaterale e `--wall-penalty`.** Nella prima versione della ricompensa
LF il rifiuto era simmetrico, cioè rifiutare una manipolazione sconveniente veniva
premiato. Questo paga una formica a ogni tick perché "sta correttamente ferma su un
buon grappolo", cioè crea una rendita di posizione. La politica l'ha sfruttata nel
modo più diretto: scegliere una direzione fuori griglia, che `_apply_move` tratta
come "resta ferma" senza addebitare `contested_penalty`. Misurato: dal 39 al 44 %
dei passi contro un bordo, contro lo 0 % dell'euristica e il 2 % dell'RL base, e
metà delle mosse perse. È la stessa patologia di `allow_stay`, rientrata dalla
porta di servizio. I rimedi sono il rifiuto solo punitivo e la penalità sul muro;
`--decline-two-sided` riproduce l'ablazione.

### Ablazioni

| Sigla | Come |
|---|---|
| A0 | pannello EURISTICA della GUI, sempre presente |
| A1 | `python -m antelligent.train --no-learn-manip ...` (pick/drop euristico) |
| A2 | `python -m antelligent.train --no-learn-move ...` (movimento euristico) |
| A3 | `python -m antelligent.train ...` (default, entrambe apprese) |
| LF | `python -m antelligent.train_lumer ...`, con le stesse tre varianti |

Per una mini-campagna riproducibile, un ciclo sui seed con un `--out` diverso per
run, dato che `training_log.csv` viene sovrascritto:

```powershell
foreach ($s in 1..20) {
  python -m antelligent.train --config docs\train-small.properties --episodes 150 --seed $s --out results\pol_s$s.pkl
}
```

### Output

Tutto finisce in `results/`, che è git-ignored:

- `results.txt`, CSV con due righe per run (`heuristic` e `rl`), campi
  `timestamp;mode;durataMs;iterazioni;righe;colonne;thread;formiche;semi;mosseTotali;semiRaccolti;entropiaIniziale;entropiaFinale;masterSeed;tipo`;
- `training_log.csv`, una riga per episodio di addestramento;
- `policy.pkl`, la politica addestrata caricata dalla GUI;
- `policy_lumer.pkl` e `policy_lumer_log.csv` per la variante LF;
- `screenshots/<data-ora>/`, schermate periodiche della finestra, solo se la
  cattura è attiva nel launcher.

### Problemi frequenti

| Sintomo | Causa e rimedio |
|---|---|
| La GUI avverte "è una politica 'tabular', non 'lumer'" o viceversa | Hai puntato la GUI a un pickle dell'altra variante. Riaddestra con il comando indicato nel messaggio. |
| La GUI avverte "addestrata con la codifica di stato v1, questa versione usa la v2" | `results/policy.pkl` viene da un addestramento precedente a un cambio di `_move_state` o `_manip_state`: la tabella non è riutilizzabile, riaddestra. |
| `nessuna politica RL addestrata (...)` all'avvio della GUI | Normale se non hai ancora lanciato `train.py`. Il pannello RL apprende dal vivo. |
| La GUI non parte, `TclError` o `no display` | Serve un ambiente grafico, non SSH o headless. |
| `ModuleNotFoundError: No module named 'antelligent'` | Lancia dalla cartella `AntelligentPy/` oppure fai `pip install -e .`. |
| `ModuleNotFoundError: No module named 'PIL'` | `pip install pillow`. |
| Addestramento troppo lento | Abbassa `--max-ticks` e `--episodes`, alza `--entropy-every`, usa una griglia più piccola con `--config`. |
| Intestazioni miste in `results.txt` | Il file esisteva con un'intestazione diversa: cancellalo. |
| La run non si ferma mai con `stopCriterion=1` | La soglia scatta solo quando nessuna formica sta trasportando un seme e l'entropia è sotto soglia. Alza `entropyThreshold` o passa a `stopCriterion=0`. |

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
che è il default: in modalità `centered` il segnale di deposito arriva a `+0.18`
mentre quello di raccolta arriva a `+1.0`, un quinto dell'escursione, quindi senza
riequilibrio la politica impara meglio quando raccogliere che dove posare.

Le due ablazioni patologiche si vedono a occhio: `--mode raw`, con il segnale mai
negativo, produce 4330 pick contro 310, che è il churn previsto;
`--decline-two-sided` fa perdere su tutti e dieci i seed.

Tre cautele sui numeri, tutte misurate:

- lo sweep di `--drop-scale` su `1.0 / 1.5 / 2 / 3 / 4` dà
  `19.9 / 22.1 / 13.6 / 18.1 / 18.0`, che non è monotono;
- la varianza fra seed di addestramento (13.6, 13.1, 19.5 con lo stesso
  `--drop-scale 2`) è dello stesso ordine dello sweep, quindi `2` è il punto
  migliore misurato e non un ottimo dimostrato;
- 800 episodi danno un risultato peggiore di 400 (21.7 contro 13.6): con ε già
  decaduto la politica smette di esplorare e deriva.

Va tenuto presente anche che la variante LF è più selettiva sul deposito e a fine
run tiene in mano più semi (3.4 su 10 contro 1.7 su 10). Con `stopCriterion=1` la
run termina solo quando nessuna formica trasporta più niente, quindi la selettività
si paga in iterazioni.

### Stato del lavoro

La Fase 1, Q-learning tabellare, è implementata e funziona. L'osservazione include
un gradiente locale (`Observation.best_dir`) che indica dove raccogliere e dove
posare: per una formica a mani libere è la cella adiacente col seme più fuori
posto, per una che trasporta è la cella adiacente vuota i cui vicini sono più dello
stesso tipo del seme in mano. La variante LF include anche il primo pezzo di reward
shaping potenziale.

Restano da fare: una campagna con più repliche di addestramento, lo shaping
incrementale globale, la Fase 2 con deep RL (DQN o PPO) e un warm start per
imitazione.
