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

## 2026-09-06 — [MODIFICA] Variante sperimentale: ricompensa dalle probabilità di Lumer-Faieta

**Richiesta.** "Forse la feature su cui dobbiamo puntare è far capire alle formiche
quanto sia importante il punto in cui lasciare un seme. Utilizziamo le funzioni di
pick e drop probabilistico per assegnare rinforzi e penitenze agli agenti, in modo
da avvicinarci all'algoritmo euristico. Implementa tutto questo in nuovi file
lasciando intatto quello fatto finora e predisponi nuovi comandi da terminale."

**Impostazione.** L'euristica **campiona** da `P_pick = (kp/(kp+f))²` e
`P_drop = (f/(kd+f))²`. La variante usa le *stesse* formule come **valore**: niente
più tiro di dado, ma un rinforzo proporzionale a quanto l'euristica giudicherebbe
conveniente quell'azione *in quel punto*. Stessa conoscenza di dominio, ma appresa
invece che cablata — e quindi migliorabile, perché l'RL può imparare a **cercare**
i punti buoni, cosa che l'euristica (random walk puro) non fa.

**Il punto di indifferenza non è arbitrario.** Il segnale dev'essere con segno,
altrimenti "manipolare" batte sempre "non manipolare" e la colonia entra in churn.
Lo zero cade dove le due regole si equivalgono:

```
P_pick(f) = P_drop(f)  ⟺  kp/(kp+f) = f/(kd+f)  ⟺  f* = √(kp·kd) ≈ 0.1732
```

Sotto `f*` l'euristica preferisce raccogliere, sopra preferisce posare. È lo stesso
ruolo che aveva il livello del caso `1/k`, ma qui **deriva dalle costanti
dell'euristica**. Il vantaggio risulta antisimmetrico: `adv(pick, f) = −adv(drop, f)`.

**Tre contributi, tutti pesabili da CLI.**

| contributo | flag | cosa insegna |
|---|---|---|
| azione | `--pick-scale` `--drop-scale` | quanto vale raccogliere / posare *in quel punto* |
| rifiuto | `--decline-scale` | se poteva manipolare **con vantaggio** e non l'ha fatto, paga quel vantaggio (unilaterale) |
| navigazione | `--shaping-scale` | shaping potenziale `F = γΦ(s′) − Φ(s)`, `Φ` = probabilità LF nella cella: **avvicinarsi** a un buon punto di deposito paga prima di posare |
| muro | `--wall-penalty` | costo di una mossa scelta fuori dalla griglia |

Lo shaping è potenziale nel senso di Ng-Harada-Russell (1999): la somma su un
cammino dipende solo dagli estremi (test dedicato), quindi girare in tondo non
frutta niente e la politica ottima non cambia. Per questo `--gamma` viene passato
anche allo shaping. È il primo pezzo del "reward shaping potenziale" che era in
lista da `docs/rl-design.md` §3.3.

**Le correzioni anti-stallo sono ereditate, non ricopiate.**
`LumerFaietaEnvironment` estende `Environment` e `LumerQPolicy` estende
`TabularQPolicy`: gradiente locale `best_dir` in entrambe le fasi, `_move_state`
compatto, `allow_stay = False`, `inference_epsilon = 0.05`, tie-break casuale.
Se cambiano là, cambiano anche qui — con test di regressione dedicati.

**L'unica cosa che cambia nella politica** è la testa pick/drop:

| | `TabularQPolicy` | `LumerQPolicy` |
|---|---|---|
| stato | `(trasporta, tipo, bin di f, f ≥ 1/k)` | `(trasporta, bin di f, f ≥ f*)` |
| stati (k=5, 4 bin) | 80 | **16** |

Il codice del tipo sparisce perché con questa ricompensa il valore è `g(azione, f)`:
una funzione della sola frazione di simili. Quale tipo sia entra già tutto in `f`;
tenerlo dividerebbe per `k` i dati per stato.

**Guardia contro il degrado silenzioso.** Le due varianti codificano la testa di
manipolazione in modo diverso: caricare l'una come l'altra darebbe letture
sistematiche su righe mai viste → azioni a caso, senza alcun errore. Il pickle ora
dichiara la propria famiglia (`policy_kind`) e `load_policy()` sceglie la classe;
la GUI lo usa, e chi sbaglia file riceve un warning esplicito.

**File.** Nuovi: `antelligent/lumer_reward.py` (formule pure),
`antelligent/environment_lumer.py`, `antelligent/policies/lumer_tabular.py`,
`antelligent/train_lumer.py`, `tests/test_lumer_reward.py`,
`tests/test_lumer_policy.py`. Modificati (solo aggiunte):
`antelligent/simulation/simulation.py` (2 righe: dispatch di `load_policy`),
`pyproject.toml` (script `antelligent-train-lumer`), `README.md`, `docs/uso.md`
(nuovo §3bis), questo `resume.md`. **Nulla del percorso precedente è stato
toccato**: `environment.py`, `policies/tabular.py` e `train.py` sono invariati.

`docs/rl-design.md` **non** è stato aggiornato (è materiale di tesi): §3.3 va
esteso con questa seconda formulazione della ricompensa e §6.2 con le nuove
ablazioni (`--mode`, `--decline-two-sided`, `--wall-penalty`, sweep di
`--drop-scale`). Da fare alla prossima revisione del documento.

**[RISULTATI] Due patologie trovate misurando (non a priori).**

1. **Rendita di posizione.** Nella prima versione il rifiuto era simmetrico:
   rifiutare una manipolazione *sconveniente* veniva premiato. Questo paga una
   formica ogni tick perché "sta correttamente ferma sul posto giusto". La
   politica l'ha sfruttata nel modo più diretto: scegliere una direzione **fuori
   griglia**, che `_apply_move` tratta come "resta ferma" **senza** addebitare
   `contested_penalty`. Misurato: **39–44 %** dei passi contro un bordo (euristica
   0.0 %, RL base 2.1 %), metà delle mosse perse. È la stessa patologia di
   `allow_stay`, rientrata dalla porta di servizio.
   Rimedi: rifiuto **unilaterale** (`decline_two_sided` conserva l'ablazione) e
   nuovo `wall_penalty`. Verificato: `--decline-two-sided` perde 0/10 seed,
   `--wall-penalty 0` scende a 15 409 mosse su 20 000.
2. **`advantage` è troppo permissiva.** Mette lo zero in `f* ≈ 0.173`: posare
   conviene appena si è sopra il livello del caso. `centered` lo mette a `P = 0.5`
   → per il drop `f > 0.72`: soglia molto più selettiva. 33.0 contro 13.6.
   `centered` è diventata il default.

**[RISULTATI] Campagna** (400 ep × 2000 tick; valutazione 10 `master_seed` × 2000
tick, politica congelata; "vince" = entropia più bassa dell'euristica sullo
**stesso** seed):

| variante | H finale | vince su | picks | in mano |
|---|---|---|---|---|
| euristica (A0) | 25.5 ± 1.9 | — | 210 | 1.7/10 |
| RL base (pivot `1/k`) | 38.8 ± 1.9 | 0/10 | 385 | 1.4/10 |
| **LF default** (`centered`, drop×2) | **13.6 ± 2.5** | **10/10** | 310 | 3.4/10 |
| LF `--drop-scale 1` | 19.9 ± 3.6 | 9/10 | 319 | 4.7/10 |
| LF `--mode advantage` | 33.0 ± 3.5 | 1/10 | 446 | 1.3/10 |
| LF `--shaping-scale 0` | 19.3 ± 4.1 | 9/10 | 217 | 6.1/10 |
| LF `--decline-scale 0` | 24.2 ± 2.0 | 7/10 | 391 | 0.5/10 |
| LF `--wall-penalty 0` | 23.0 ± 2.7 | 6/10 | 360 | 3.3/10 |
| LF `--decline-two-sided` | 35.7 ± 1.6 | 0/10 | 424 | 2.0/10 |
| LF `--mode raw` | 44.9 ± 2.5 | 0/10 | **4330** | 5.2/10 |

**È la prima configurazione che batte l'euristica**, e la batte su tutti i seed di
valutazione. `--mode raw` produce 4330 pick contro 310: il *churn* previsto da un
segnale mai negativo, visibile a occhio nudo.

**Perché il deposito va pesato di più.** In `centered` i due segnali hanno
escursione molto diversa: il pick arriva a `+1.0` (seme isolato), il drop solo a
`2·(1/1.3)² − 1 = +0.18` — un quinto. Senza riequilibrio la politica impara molto
meglio *quando raccogliere* che *dove posare*. Da qui il default `drop_scale = 2`.

**Cautele da riportare in tesi.**
- Lo sweep `--drop-scale 1.0/1.5/2/3/4` dà `19.9 / 22.1 / 13.6 / 18.1 / 18.0`:
  **non monotono**. La varianza fra seed di addestramento (13.6 / 13.1 / 19.5 con
  lo stesso `--drop-scale 2`) è dello stesso ordine dello sweep: `2` è il punto
  migliore *misurato*, non un ottimo dimostrato. Servono più repliche.
- **800 episodi peggiorano** (21.7 contro 13.6): con ε già decaduto la politica
  smette di esplorare e deriva.
- La variante LF è più selettiva sul deposito e tiene in mano più semi a fine run
  (3.4/10 contro 1.7/10). Con `stopCriterion=1` la run termina solo quando
  *nessuna* formica trasporta: per i confronti a budget fisso usare
  `stopCriterion=0`.

**Comandi.**

```powershell
python -m antelligent.train_lumer --episodes 400                  # variante di default
python -m antelligent.train_lumer --episodes 400 --drop-scale 2   # pesa di più DOVE si posa
python -m antelligent.train_lumer --episodes 400 --shaping-scale 0 --decline-scale 0   # ablazione
python -m antelligent.train_lumer --episodes 400 --mode raw       # ablazione: P non centrata -> churn
python -m antelligent.train_lumer --episodes 400 --out results\policy.pkl   # per la GUI
```

---

## 2026-09-06 — [MODIFICA] Le formiche RL vagavano: gradiente locale, stato compatto, rumore in inferenza

**Osservazione di partenza.** Con la politica addestrata (400 ep) le formiche
"non capivano niente": pochi pick, andirivieni fra le stesse celle, a tratti ferme.
Ipotesi dell'utente: problema di **località dell'entropia**, meglio calcolarla
vicino a ogni formica.

**Verifica dell'ipotesi.** La ricompensa è *già* interamente locale: dipende solo da
`f_own` (frazione di simili nelle 8 celle adiacenti). L'entropia globale
`check_entropy_placed` **non compare mai** in `_reward` — serve solo al criterio di
arresto e alla statistica GUI. Quindi il costo era già "vicino alla formica".
L'intuizione era però giusta nella sostanza: mancava la località
dell'**informazione**, non del costo.

**Diagnosi misurata** (config 14×14, 130 semi, 10 formiche, 5 tipi):

- `best_dir` era calcolato **solo `if carrying`**: nel **100 %** degli stati "formica
  libera" valeva `8` = nessun bersaglio. Una formica scarica non sapeva *mai* dove
  fossero i semi; la sua mossa dipendeva solo da heading + celle bloccate → riflesso
  fisso.
- Solo il **2 %** delle situazioni replicate sui 4 heading sceglieva la stessa mossa,
  benché le azioni siano già relative all'heading: la tabella era frammentata in 4
  copie incoerenti (~92 000 stati).
- Movimento: ritorni sulla cella `t-2` **39.8 %** (euristica 6.7 %), copertura della
  griglia **121/196**, celle distinte/passi 7.6 % (euristica 23.1 %).

**Modifiche.**

- **`environment.py`** — `best_dir` diventa un vero **gradiente locale in entrambe le
  fasi** (`_target_dir`): trasportando → la cella adiacente *vuota* i cui vicini sono
  più dello stesso tipo del seme in mano (dove posare bene; prima puntava a una cella
  che *conteneva* già un seme simile, dove posare è impossibile); a mani libere → la
  cella col seme più fuori posto. Nuovi `_seed_block` (una sola scansione del blocco
  5×5, riusata da `f_here` e dal bersaglio) e `_counts_from_block`.
- **`policies/tabular.py`** —
  - `_move_state` ridotto a `(carrying, best_dir, blocked)`: via `direction` (le mosse
    sono già relative all'heading) e `carried_type` (già riassunto da `best_dir`).
    Da ~92 000 a ~4 600 stati.
  - `_manip_state` guadagna il bit `above_chance`: il segno della ricompensa si ribalta
    al livello del caso `1/k` (0.2 con k=5), che **non** cade su un confine dei bin —
    dentro il bin `[0, 0.25)` il pick è premiato a `f=0.10` e punito a `f=0.24`. Senza
    quel bit lo stato non può rappresentare il confine decisionale.
  - nuovo `QConfig.inference_epsilon` (default `0.05`): rumore residuo **a politica
    congelata**. Su osservazioni aliasate una politica deterministica cade in cicli
    limite da cui non può uscire; in un POMDP l'ottimo è in generale stocastico, e
    l'euristica di confronto è già stocastica. `freeze()` ora sospende
    l'*apprendimento*, non il rumore.
  - `_STATE_VERSION = 3` salvato nel pickle: `load()` **avverte** se la tabella è di
    una codifica precedente (prima degradava in silenzio, con tutte le letture a vuoto).

**File.** `antelligent/{actions,environment}.py`, `antelligent/policies/tabular.py`,
`tests/{test_environment,test_tabular_policy}.py`, `docs/uso.md`, `README.md`.

**Test.** `pytest -q` → **77 passed** (66 + 11 nuovi: bersaglio della formica libera
sul seme più fuori posto, bersaglio del trasporto su cella *vuota*, `best_dir = STAY`
senza bersagli, `_counts_from_block` coincidente con `count_types_around`, stato di
movimento invariante a heading/tipo e sensibile al bersaglio, confine decisionale
rappresentabile, avviso sulla codifica obsoleta, congelata rumorosa ma che non impara).

**[RISULTATI] Effetto di `inference_epsilon`** — policy 400 ep, 5 `master_seed`:

| `inference_epsilon` | H finale | ritorni `t-2` | picks |
|---|---|---|---|
| 0.00 (greedy pura) | 44.7 | 30.2 % | 69 |
| **0.05 (default)** | **36.6** | 20.6 % | 206 |
| 0.10 | 36.8 | 18.5 % | 303 |
| 0.20 | 36.7 | 15.2 % | 465 |

**[RISULTATI] A/B finale** — 2000 tick, media ± sd su 5 `master_seed`:

| variante | H finale | picks | drops | in mano | mosse |
|---|---|---|---|---|---|
| euristica (A0) | **23.5 ± 3.8** | 203 | 201 | 1.8/10 | 20 000 |
| casuale (riferimento) | 69.2 ± 2.3 | 2313 | 2307 | 6.6/10 | 17 877 |
| RL addestrata (A3) | **33.4 ± 3.3** | 423 | 420 | 2.2/10 | 19 451 |

Movimento (800 tick, 5 seed): ritorni `t-2` **39.8 % → 20.5 %** (euristica 6.2 %),
celle distinte **7.6 % → 20.8 %** (euristica 23.0 %), copertura **62 % → 100 %**,
ferma 14 % → 2.5 %.

**Note / follow-up.** La patologia del movimento è risolta e l'RL è ora nettamente
sopra il riferimento casuale (33.4 contro 69.2), ma **resta dietro all'euristica**
(23.5). Il tasso di manipolazione non è più il collo di bottiglia (423 pick contro
203). Il divario residuo è la domanda di ricerca di Fase 1: probabile che serva il
reward shaping potenziale globale (`docs/rl-design.md` §3.3) e/o il bonus terminale
`final_scale`, tuttora non collegato. Da valutare anche `inference_epsilon` come
iperparametro da riportare nella tesi (non è un trucco: è la controparte stocastica
dell'euristica).

---

## 2026-09-06 — [MODIFICA] Run sequenziali nella GUI, cronometri per copia, handoff thread-safe

**Obiettivo.** Far girare le due copie **una alla volta** (prima euristica, poi
RL) invece che in parallelo, con etichette del tempo distinte per copia che si
**fermano** quando la rispettiva run è conclusa.

**Modifiche.**

- **Esecuzione sequenziale** (`simulation/simulation.py`) — ogni `_Pane` ha uno
  `status` (`waiting` / `running` / `done`; `done` è ora una property derivata).
  `Start` chiama `_start_next_pane()`, che avvia **una sola** copia; alla fine di
  una run `_pane_finished()` la marca conclusa e avvia la successiva; quando non
  ne restano chiama `_maybe_finish()`. `_run_pane` non imposta più `start_time`
  (lo fa il thread principale all'avvio effettivo).
- **Cronometri per copia** — riga etichettata `tempo euristica` / `tempo RL`
  (`_TIME_ROW_LABEL`), valore `mm:ss` seguito dallo stato (`in attesa` / `in
  corso` / `conclusa`, `_STATUS_TEXT`) e colorato di conseguenza
  (`_STATUS_COLOR`). `_elapsed_seconds()` restituisce `duration_ms` fisso quando
  la copia è conclusa, così **l'orologio si ferma** invece di continuare a
  scorrere; `_refresh_stats` smette di riarmarsi quando tutte hanno concluso.
  Anche il titolo sopra la griglia riporta lo stato.
- **Handoff worker → GUI thread-safe** — *bug latente preesistente*: i worker
  chiamavano `root.after` (via `_schedule`), che **solleva `RuntimeError: main
  thread is not in main loop`**; l'eccezione veniva ingoiata, quindi repaint,
  screenshot e fine-run potevano essere persi silenziosamente. Ora i worker
  chiamano solo `_post(evento, pane)` su una `queue.Queue` e il thread principale
  la drena in `_pump_events()` ogni 50 ms (`_EVENT_POLL_MS`). Il riarmo del timer
  è in `finally`: una callback che fallisce non può più uccidere il pump e
  piantare la run. `_schedule` / `_schedule_render` rimossi.
- **`results_dialog.py`** — nuovo parametro `with_screenshots`: la nota finale
  cita `results/screenshots/` solo se la cattura era attiva.

**File.** `antelligent/simulation/{simulation,results_dialog}.py`,
`tests/test_comparison_gui.py` (nuovo), `README.md`, `docs/uso.md`.

**Test.** `pytest -q` → **66 passed** (57 + 9 nuovi in `test_comparison_gui.py`:
stato iniziale "in attesa", testo diverso per i tre stati, orologio fermo da
conclusa e avanzante da in corso, `Start` che avvia solo la prima copia, assenza
di sovrapposizione con verifica sui tempi registrati (`rl.start_time >=` fine
euristica), seconda copia intatta durante la prima, dialog solo a run entrambe
concluse, etichette congelate oltre un giro del timer). Verificato anche
end-to-end con un `mainloop` reale: 0 sovrapposizioni, `heuristic` 114 ms →
`rl` 140 ms, etichette finali `00:00 (conclusa)`, dialog mostrato.

**Note / follow-up.** Il tempo per copia ora è una metrica di confronto
utilizzabile (prima le due copie si contendevano il GIL e il tempo di una
dipendeva dal carico dell'altra). `docs/rl-design.md` §4.5 descrive ancora
l'esecuzione parallela: da riallineare alla prossima revisione del documento.

---

## 2026-09-06 — [MODIFICA] Casella "cattura schermate" + sblocco delle formiche ferme

**Obiettivo.** (1) Rendere opzionale la cattura delle schermate, che prima era
sempre attiva. (2) Dopo un training da 400 episodi le formiche restavano ferme
nella loro cella, soprattutto **mentre trasportavano un seme**.

**Diagnosi (2).** Valutazione strumentata della politica **congelata** sulla
config reale (14×14, 150 semi = 77 % di densità, 10 formiche, 5 tipi), 2000 tick:
`carried_left = 10/10`, 19 980 decisioni su 20 000 prese *mentre si trasporta*,
`STAY` scelto nel **66 %** dei casi, **3 drop su 7941 occasioni valide**. Due
cause indipendenti che si rinforzano a vicenda:

1. **`pivot` fisso a 0.5 con `seedTypes = 5`.** La ricompensa di drop è
   `f_own − pivot`; su una cella a caso `f_own ≈ 1/k = 0.2`, quindi il drop vale
   in media `−0.30`: **posare è sempre punito** e la greedy impara a non posare
   mai. L'euristica non ha il problema perché `P_drop = (f/(kd+f))²` non è mai
   zero (a `f = 0.2` posa comunque il 16 % delle volte).
2. **`STAY` fra le azioni di movimento.** Restare fermi non rischia mai
   `contested_penalty`, quindi è l'azione meno costosa ovunque non ci sia una
   ricompensa positiva raggiungibile; il valore `Q(s,STAY) = −step_penalty/(1−γ)`
   si auto-sostiene (self-loop assorbente). L'euristica non ha questa azione:
   `check_move.random_move_index` restituisce sempre una direzione reale.

Combinate: la formica raccoglie subito un seme, non lo posa mai (causa 1) e
smette di muoversi (causa 2).

**Modifiche.**

- **`config.py`** — nuovo campo `SimulationConfig.capture_screenshots` (default
  `False`).
- **`launcher.py`** — nuova sezione *Opzioni* con la casella "Cattura schermate
  della finestra (results/screenshots/)" e una nota sul costo/permesso macOS.
- **`starter.py`** — round-trip della chiave `captureScreenshots` in
  `config.properties`; nuovo helper `_read_bool` (accetta `1/true/yes/on`, chiave
  assente = `False`).
- **`simulation/simulation.py`** — `_take_screenshot` / `_save_screenshots` escono
  subito se il flag è spento, e il worker non pianifica più il callback Tk ogni
  200 iterazioni.
- **`environment.py`** — `RewardConfig.pivot` diventa `Optional[float] = None`;
  `Environment.pivot` lo risolve al **livello del caso `1/seed_types`** quando non
  è dato esplicitamente. `_reward` usa `self.pivot`.
- **`policies/tabular.py`** — nuovo `QConfig.allow_stay` (default `False`) e
  proprietà `_n_move_actions`: la testa del movimento sceglie fra le 8 direzioni
  reali. `_argmax` accetta un `limit`; il bootstrap in `record()` usa
  `max(next_row[:n])` per non contaminare il target con la colonna di `STAY`.
  Le righe restano lunghe 9 → i pickle esistenti si caricano ancora.
- **`train.py`** — `--pivot` ora ha default `None` (auto), nuovo `--allow-stay`.

**File.** `antelligent/{config,launcher,starter,environment,train}.py`,
`antelligent/policies/tabular.py`, `antelligent/simulation/simulation.py`,
`config.properties`, `antelligent/resources/config.properties`,
`tests/{conftest,test_environment,test_tabular_policy,test_launcher_config}.py`,
`README.md`, `docs/uso.md`.

**Test.** `pytest -q` → **57 passed** (41 + 16 nuovi: round-trip della casella
launcher↔`config.properties`, `pivot` automatico/esplicito, drop al livello del
caso a ricompensa nulla, `STAY` mai scelto di default e riabilitabile, bootstrap
che ignora la colonna `STAY`).

**[RISULTATI] A/B della politica congelata** — config 14×14, 150 semi, 10
formiche, 5 tipi; 2000 tick; media su 5 `master_seed` (11/22/33/44/55); policy
addestrate 400 episodi, `--max-ticks 2500 --seed 7`:

| variante | H finale | picks | drops | in mano | STAY % |
|---|---|---|---|---|---|
| euristica (A0) | **30.0** | 228 | 225 | 2.8/10 | 0 % |
| RL prima (pivot 0.5 + STAY) | 71.5 | 13 | 4 | **9.6/10** | **43.1 %** |
| RL solo fix STAY | 67.9 | 20 | 10 | 10.0/10 | 0 % |
| RL dopo (pivot 1/k + no STAY) | **57.9** | 48 | 45 | **3.4/10** | 0 % |

I due fix sono **complementari**: togliere `STAY` da solo sblocca il movimento
(mosse 5805 → 9665) ma le formiche restano tutte con il seme in mano perché
continuano a non posare; serve anche il `pivot` corretto.

**Note / follow-up.** Il blocco è risolto, ma **l'RL resta dietro all'euristica**
(H 57.9 vs 30.0): manipola ancora troppo poco (48 pick contro 228). La causa
probabile è che il **movimento non ha gradiente**: `Observation.best_dir` è
calcolato solo quando la formica trasporta, e punta alla prima cella adiacente che
contiene un seme dello stesso tipo — cioè proprio una cella dove **non** si può
posare. Quando non trasporta vale sempre `8`, quindi la formica libera non ha
alcuna informazione su dove siano i semi. Prossimo passo suggerito: rendere
`best_dir` un vero segnale di navigazione (portando → cella adiacente *vuota* con
più vicini dello stesso tipo; libera → cella adiacente con un seme *fuori posto*),
misurando il costo di `observe()`. Restano aperti il reward shaping potenziale
globale e `final_scale`.

---

## 2026-09-06 — [MODIFICA] Fix: la politica RL congelata non raccoglieva più semi

**Obiettivo.** Dopo un addestramento headless + `freeze()`, la GUI in modalità
"RL (addestrata)" non eseguiva quasi nessun `pick`: le formiche lasciavano i semi
dov'erano. In modalità "apprendimento live" il problema non si vedeva solo perché
la GUI non chiama mai `end_episode()`, quindi ε resta a 0.30 e il ~15% dei pick
sono forzati casuali.

**Causa.** Lo stato di manipolazione della `TabularQPolicy` era degenere per il
`pick`: in `_manip_state`, `relevant = obs.carried_type if obs.carrying else 0`,
cioè quando la formica **non** trasporta (decisione di pick) l'indice era cablato
a `0` e lo stato guardava `f_here[0]` — la densità dei semi di **tipo 0** — non il
tipo del seme sotto la formica. Tutte le decisioni di pick collassavano su 4
celle `(0, 0, f_bin)`, mentre la ricompensa dipende dal tipo *reale* del seme.
La Q convergeva alla media della ricompensa di pick, che diventa negativa quando
il campo si ordina (un pick casuale finisce su un seme già ben posizionato →
`pivot − f_own < 0`). Appena `Q[pick] ≤ Q[no-op] = 0` il greedy sceglie sempre
`no-op`, e `_argmax` rompeva anche le parità esatte a favore dell'indice 0.
`freeze()` (ε = 0) cementava il tutto.

**Modifiche.**

- **`policies/tabular.py`** — `_manip_state`: `relevant = obs.carried_type if
  obs.carrying else obs.seed_here_type` (per il pick usa il tipo del seme sotto
  la formica). `_argmax`: in caso di parità sceglie a caso tra i massimi invece
  di preferire sempre l'indice 0 (no-op / STAY).
- **`actions.py`** — `Observation` ha un nuovo campo `seed_here_type` (codice del
  seme sulla cella della formica, 0 se assente).
- **`environment.py`** — `observe()` popola `seed_here_type`. `step()`
  rifattorizzato: `will_pick` / `will_drop` calcolati una volta, `f_own` (frazione
  di vicini dello stesso tipo del seme manipolato) messo in `info` insieme a
  `manipulation`; nuovo contatore `drops_done`. `_reward()` non prende più
  `action` (legge `info["manipulation"]`). Nessun cambio alla formula della
  ricompensa (`pivot`-relativa, invariata).
- **`train.py`** — `RewardConfig` ora configurabile da CLI: `--step-penalty
  --contested-penalty --invalid-penalty --manip-scale --pivot`. Il log per
  episodio sostituisce `seeds` con `picks, drops, carried_left` (e la riga a
  console li stampa) per rendere visibile un eventuale collasso.

**File.** `antelligent/actions.py`, `antelligent/environment.py`,
`antelligent/policies/tabular.py`, `antelligent/train.py`,
`tests/test_environment.py`, `tests/test_tabular_policy.py`, `README.md`,
`docs/uso.md`.

**Test.** `pytest -q` → **41 passed** (36 + 5 nuovi: pick di seme fuori
posto/ben raggruppato, drop accanto a tipo estraneo/tra simili, stato di pick
dipendente dal tipo del seme sotto la formica). Smoke: addestramento headless su
griglia piccola (20×14, 25 formiche, 4 tipi, 100 episodi) + valutazione della
politica **congelata** su 5 `master_seed` non visti → picks 48–80, drops
allineati, `carried_left` 0–6 (niente più collasso), entropia ~65 → ~30
(l'euristica arriva a ~13–20 sullo stesso campo: divario di Fase 1 atteso).

**Note / follow-up.** Il fix strutturale è `_manip_state`; il tie-break di
`_argmax` e i default invariati sono contorno. Resta il divario RL vs euristica
(Fase 1): ora si può lavorarci con `--pivot` / `--manip-scale` / `--step-penalty`
e la nuova telemetria pick/drop/carried. `final_scale` ancora non collegato.

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
