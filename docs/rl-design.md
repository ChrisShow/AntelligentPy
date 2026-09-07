# Antelligent RL: documento di progetto

Il focus è il confronto tra:

- **`heuristic`** — la versione attuale: matrice normale con lock, movimento e
  raccolta/deposito guidati da un **algoritmo probabilistico pre-impostato**
  (regole di tipo Lumer–Faieta con costanti fisse `kp`, `kd` e distribuzione di
  movimento fissa);
- **`rl`** — una versione in cui le formiche **imparano** come muoversi e come
  raccogliere/depositare i semi tramite **reinforcement learning**, usando
  l'**entropia del sistema** come segnale di ricompensa (funzione di costo da
  minimizzare).

Le due popolazioni non condividono la griglia: girano su **due copie distinte e
inizialmente identiche** dello stesso campo. A partire da un unico `master_seed`
si genera un solo stato iniziale (posizioni dei semi, celle di partenza delle
formiche, parametri), poi si istanziano **due ambienti gemelli** — uno con
`HeuristicPolicy`, l'altro con la politica RL — con la stessa fisica di
occupazione delle celle (`try_acquire`) e lo stesso criterio di arresto. Da lì in
poi ogni copia evolve in modo indipendente sotto la propria politica. È un
**confronto appaiato** (paired): a parità assoluta di condizioni iniziali,
l'unica variabile è la politica.

---

## 1. Motivazioni

1. **Da regole disegnate a mano a politiche apprese.** L'algoritmo attuale
   discende dai lavori storici sul clustering ant-based (Deneubourg et al. 1991;
   Lumer & Faieta 1994): funziona, ma dipende da costanti calibrate a mano
   (`kp = 0.1`, `kd = 0.3`, probabilità di movimento `25/5/15/15/15/15/5/5`). La
   domanda di ricerca è se un agente RL, senza alcuna conoscenza di quelle
   formule, riesca a **eguagliare o superare** l'euristica sul suo stesso
   terreno, e *cosa* impara.
2. **Adattività.** L'euristica usa le stesse costanti a prescindere da densità dei
   semi, dimensione della griglia e numero di categorie. Una politica appresa può
   **adattarsi** alle condizioni del campo (o generalizzare a condizioni non
   viste).
3. **Funzione di costo pulita.** L'entropia del sistema è una metrica densa, già
   implementata, normalizzata (`0` = ordine perfetto, `100` = massimo disordine):
   un banco di prova naturale per l'RL.
4. **Multi-agent RL con vincoli di concorrenza.** Il setting è quello di *agenti
   situati* multi-thread che condividono un ambiente con contesa fisica (i lock di
   cella): un caso di **multi-agent RL** con vincoli di concorrenza, poco
   esplorato in letteratura.
5. **Explainability.** Confrontare le curve di probabilità di pick/drop *apprese*
   con quelle *analitiche* `(kp/(kp+f))²` e `(f/(kd+f))²` permette di capire se
   l'RL riscopre la stessa struttura o ne trova una diversa.

---

## 2. Obiettivi e domande di ricerca

| # | Domanda | Come si risponde |
|---|---------|------------------|
| RQ1 | A parità di budget di iterazioni, l'RL raggiunge un'entropia finale **minore** dell'euristica? | Confronto entropia finale (§6) su griglia di scenari, con test statistici |
| RQ2 | L'RL raggiunge la **soglia di entropia** in **meno iterazioni** (e/o meno tempo)? | Iterazioni/tempo al raggiungimento della soglia; area sotto la curva entropia–iterazioni |
| RQ3 | Quale componente porta il guadagno: **movimento**, **manipolazione** o entrambi? | Ablazioni A1/A2/A3 (§6.2) |
| RQ4 | La politica appresa **generalizza** a scenari non visti in addestramento (densità, dimensione, numero di categorie)? | Train su una configurazione, test su configurazioni diverse |
| RQ5 | L'RL apprende comportamenti **qualitativamente diversi** (specializzazione, gestione della contesa sulle celle, pattern di movimento)? | Analisi comportamentale: heatmap, tasso di contesa, curve pick/drop apprese |
| RQ6 | Qual è il **costo** dell'RL (campioni, tempo di addestramento, complessità) rispetto al beneficio? | Curve di apprendimento, wall-clock di training, numero di parametri |

**Non-obiettivi (per questa fase):** ottimizzare il throughput multi-thread
(il GIL lo limita comunque), comunicazione esplicita tra formiche, RL su hardware
distribuito. Vedi §8.

---

## 3. Formulazione come problema di Reinforcement Learning

Il sistema è un **Dec-POMDP** (decentralized partially observable MDP):
`N` agenti omogenei (le formiche), ambiente parzialmente osservabile, ricompensa
cooperativa. Si adotta lo schema **CTDE con parametri condivisi** (centralized
training, decentralized execution): **una sola politica** `π_θ` condivisa da tutte
le formiche, addestrata sulle transizioni raccolte da tutte, eseguita in modo
indipendente da ciascuna su osservazioni locali. È la scelta naturale per uno
sciame di agenti identici e riduce la non-stazionarietà tipica del multi-agent RL.

### 3.1 Osservazione (locale, parziale) — `o_t`

Ogni formica osserva solo un intorno della propria cella (coerente con la
caratteristica MAS "punto di vista limitato"):

- **stato di trasporto:** `carrying ∈ {0,1}` + one-hot del tipo di seme
  trasportato (dimensione `seedTypes`);
- **finestra locale** (Moore, es. `5×5` centrata sulla formica): per ogni cella un
  canale categorico ∈ {vuota, seme di tipo `k`, altra formica, fuori griglia};
- **istogramma di prossimità** `f = (f_0, …, f_{seedTypes−1})`: frazione di semi
  di ciascun tipo nelle 8 celle adiacenti (la stessa informazione usata
  dall'euristica — così il confronto è equo);
- **direzione corrente** `direction ∈ {0,1,2,3}` (momentum di movimento);
- (opzionale) **entropia locale** nell'intorno.

L'osservazione è volutamente **compatta**: consente sia un approccio tabellare
(dopo discretizzazione) sia una piccola rete.

### 3.2 Azione — `a_t`

Ad ogni tick la formica compie una **manipolazione** e un **movimento**. Azione
fattorizzata:

- **manipolazione** `∈ {no-op, pick, drop}` — `pick` valida solo se non trasporta
  ed è presente un seme nella sua cella; `drop` valida solo se trasporta e la
  cella è libera da semi. Azioni non valide → `no-op` + penalità (§3.3);
- **movimento** `∈ {N, NE, E, SE, S, SW, W, NW, stay}` (9 valori) oppure
  `∈ {avanti, dietro, dx, sx, avanti-dx, …}` relativi alla direzione (8 valori),
  per parallelismo diretto con `check_move`.

Spazio d'azione discreto piatto ≈ `3 × 9 = 27` (o fattorizzato con due teste).
La **fisica di occupazione resta identica all'euristica**: la politica *propone*
una cella di destinazione, poi il tentativo di `try_acquire` non bloccante decide
se il passo va a buon fine; in caso di collisione la formica non perde il tick e
ripiega su un'adiacente libera (o resta ferma). L'agente può *imparare* a evitare
le celle contese.

### 3.3 Ricompensa — `r_t` (dall'entropia)

Osservazione chiave: **il movimento non cambia l'entropia del sistema** (l'entropia
dipende solo dalla posizione dei semi, non delle formiche). L'entropia varia
**solo dopo un `pick` o un `drop` andati a buon fine**, e solo *localmente*, nelle
celle entro raggio 2 dalla cella toccata. Questo permette un calcolo
**incrementale** e a basso costo del reward.

Ricompensa (cooperativa, per-agente, con *reward shaping* potenziale):

- **shaping basato su potenziale** con `Φ(s) = −H_placed(s)`, dove `H_placed` è
  l'entropia calcolata sui **soli semi già posati** (sempre ben definita, a
  differenza di `check_entropy` che restituisce `-1` mentre un seme è trasportato):
  `r_shape = γ·Φ(s_{t+1}) − Φ(s_t)`. Il termine è ≈ 0 per i passi di solo
  movimento e diverso da 0 solo per pick/drop; per costruzione **non altera la
  politica ottima** (Ng, Harada & Russell 1999);
- **penalità di passo** `−ε` per incentivare a raggiungere la soglia in meno tick;
- **penalità** `−δ` per manipolazione non valida e per destinazione occupata
  (contesa);
- **bonus terminale** all'arresto: `+α·(H_0 − H_final)` (guadagno di ordine) oppure
  `−β·T` se si è raggiunta la soglia in `T` tick (velocità).

Le componenti locali danno un segnale **denso** (apprendimento rapido, buon credit
assignment); il termine terminale allinea la politica all'**obiettivo vero** del
task. I pesi `ε, δ, α, β, γ` sono iperparametri (§5.4).

### 3.4 Transizione ed episodio

- Un **tick** = tutte le formiche agiscono una volta, in ordine casuale (o in
  parallelo su thread, con politica congelata — §4.4).
- **Episodio** = una corsa completa su una griglia rigenerata casualmente
  (posizioni iniziali di semi e formiche, e — per la generalizzazione — anche
  numero di semi/tipi entro range).
- **Terminazione:** identica alle due modalità dell'euristica — numero massimo di
  iterazioni **oppure** entropia (stretta: tutti i semi posati) `≤ soglia`.

---

## 4. Architettura software

### 4.1 Il "seam": `AntPolicy`

Si introduce un'interfaccia che separa il **corpo** della formica (posizione, seme
trasportato, applicazione di mosse/pick/drop sulla matrice, lock di occupazione —
fisica **invariata**) dal suo **cervello** (la politica):

```
class AntPolicy(Protocol):
    def select_action(self, observation) -> Action: ...
    def record(self, transition) -> None: ...      # no-op per l'euristica
    def end_episode(self) -> None: ...             # no-op per l'euristica
```

Implementazioni:

| Politica | Descrizione |
|---|---|
| `HeuristicPolicy` | incapsula le regole attuali: `random_move` (distribuzione fissa) + probabilità di pick/drop `(kp/(kp+f))²`, `(f/(kd+f))²`. Nessun apprendimento. |
| `TabularQPolicy` | Q-learning / SARSA a tabella condivisa, su stato discretizzato. Fase 1 (baseline RL, senza dipendenze aggiuntive). |
| `DQNPolicy` / `PPOPolicy` | rete piccola (MLP o CNN sulla finestra locale), parametri condivisi. Fase 2 (richiede PyTorch). |

### 4.2 Rifattorizzazione di `Ant` e della matrice

- Le formule di pick/drop **escono** da `SeedMatrix` e finiscono in
  `HeuristicPolicy`; `SeedMatrix` mantiene solo le primitive *meccaniche*
  (`try_place`, `try_pick`, `has_seed`, `count_types_around`, lock di cella) e il
  calcolo di entropia (globale e **locale/incrementale**).
- `Ant.__call__` diventa:
  `obs = env.observe(self)` → `action = policy.select_action(obs)` →
  `reward, done = env.step(self, action)` → `policy.record(Transition(obs, action, reward, next_obs, done))`.
- Nuovo modulo `environment.py`: `observe(ant)`, `step(ant, action)`,
  `local_entropy_delta(cell)`, gestione dell'ordine dei tick e del reset episodio.

### 4.3 Modalità di esecuzione e persistenza

- Nuovi campi in `SimulationConfig`: `master_seed` (genera lo stato iniziale
  condiviso dalle due copie) e un blocco `rl` con gli iperparametri / percorso
  della politica addestrata da caricare. Il campo `type=1` resta per
  compatibilità del formato di `config.properties`.
- **`train.py`** — harness **headless** (senza Tk): esegue molti episodi, aggiorna
  la politica, logga le curve di apprendimento, salva la politica addestrata
  (`policy.pkl` per la tabella / `policy.pt` per la rete) insieme a config e
  metadati. È qui che si producono i risultati statistici del confronto.
- **GUI** — dopo l'addestramento, mostra le due copie **affiancate** (§4.5),
  ciascuna con la propria politica **congelata** in **sola inferenza** (greedy,
  nessun apprendimento). La politica congelata è read-only ⇒ thread-safe.
- Output: `results/results.txt` guadagna una colonna `mode` e scrive **due righe
  per run** (una per copia), con in più `mosse_totali` e `semi_raccolti`; nuovi
  file `results/training_log.csv` (episodio, ritorno, entropia finale, lunghezza,
  epsilon/entropia della policy, tempo) e `results/eval_log.csv`.
- **Riproducibilità:** `master_seed` per lo stato iniziale condiviso; RNG con seed
  per episodio in training; set di seed di valutazione fissi e separati.

### 4.4 Concorrenza e RL

L'addestramento con formiche su thread reali **e** politica condivisa che muta
(soprattutto una rete) è una fonte di race condition e, causa GIL, non porta
comunque speedup. Scelta di progetto:

- **Training:** stepping dell'ambiente **sequenziale** (un tick = ciclo sulle
  formiche in ordine casuale). Semplice, deterministico, veloce da debuggare. La
  *fisica* dei lock di cella resta in funzione (una formica per cella, contesa,
  `try_acquire`), solo che le collisioni sono rare perché si procede una formica
  alla volta — il che va bene: la contesa vera si misura in esecuzione.
- **Esecuzione / benchmark finale:** opzionalmente **multi-thread**
  (`ThreadPoolExecutor`, come oggi) con politica **congelata**. Così il confronto
  `heuristic` vs `rl` esercita anche il percorso concorrente, a parità di
  modello di esecuzione.
- Alternativa documentata (non prioritaria): schema A3C-like con un thread
  *learner* che consuma da una coda di transizioni prodotte dai thread *worker*.

### 4.5 GUI — confronto affiancato (dual-pane)

Flusso:

1. **Launcher** (invariato nella sostanza): si scelgono i parametri di
   configurazione e si preme *Avvia*.
2. Si apre **una sola finestra** con le **due griglie affiancate** —
   `EURISTICA` a sinistra, `RL` a destra — costruite dallo stesso stato iniziale
   (`master_seed`).
3. In basso, **centrati e condivisi**, i pulsanti **Start** e **Visibility**.
4. **Sotto ciascuna griglia**, il pannello statistiche della rispettiva copia.

```
┌─────────────────────────────┬─────────────────────────────┐
│          EURISTICA          │             RL              │
│                             │                             │
│        [ griglia A ]        │        [ griglia B ]        │
│                             │                             │
├─────────────────────────────┼─────────────────────────────┤
│ tempo     00:07             │ tempo     00:07             │
│ mosse     12 480            │ mosse     11 905            │
│ semi      317               │ semi      342               │
│ entropia  41.3              │ entropia  33.8              │
└─────────────────────────────┴─────────────────────────────┘
                  [ Start ]   [ Visibility ]
```

**Pulsanti (agiscono su entrambe le copie):**

- **Start** — avvia i due loop di simulazione. I due ambienti girano su **due
  thread worker indipendenti**, ciascuno alla propria velocità: il *tempo* per
  copia è così una metrica di confronto reale (l'inferenza RL può essere più
  lenta dell'euristica). Alla seconda pressione: salva screenshot + risultati e
  chiude, come nella versione attuale.
- **Visibility** — mostra/nasconde il rendering di **entrambe** le griglie; i
  pannelli statistiche continuano ad aggiornarsi (utile per far girare le run
  senza il costo del repaint).

**Pannello statistiche (uno per copia).** Quattro valori:

| Voce | Significato | Aggiornamento |
|---|---|---|
| **tempo** | wall-clock dall'avvio della run, formato `mm:ss` | **1 volta al secondo** (`root.after(1000, …)`), non a decimi/millesimi, per non caricare la CPU |
| **mosse totali** | contatore cumulativo degli spostamenti a buon fine di tutte le formiche di quella copia | letto a 1 Hz |
| **semi raccolti** | contatore cumulativo dei `pick` andati a buon fine | letto a 1 Hz |
| **entropia totale** | entropia del sistema `H_placed` (sui semi posati, sempre definita) | letto a 1 Hz dall'ultimo valore già calcolato dal loop |

**Note implementative:**

- L'aggiornamento delle 4 label è un **unico** callback `root.after(1000, …)` sul
  thread principale che legge contatori a buon mercato (interi/float) da entrambi
  gli ambienti — nessun ricalcolo pesante nel timer. `mosse_totali` e
  `semi_raccolti` sono `int` incrementati dai worker e letti dal main (sicuro
  sotto GIL). `entropia` è l'ultimo valore già prodotto dal loop per il criterio
  di arresto: il timer non ricalcola nulla.
- Il **repaint delle griglie** resta governato da `refreshRate` (iterazioni tra
  un disegno e il successivo), **separato** dal timer statistiche a 1 Hz.
- Ogni worker termina alla propria condizione di arresto; la schermata dei
  risultati finali (o il salvataggio su `results.txt`) avviene quando **entrambe**
  le copie hanno concluso, riportando le due righe (`heuristic`, `rl`) con
  tempo, iterazioni, mosse totali, semi raccolti, entropia iniziale/finale.
- Layout Tk: un `Frame` contenitore con due colonne (`grid`), ogni colonna =
  `Label` titolo + `Canvas` griglia + `Frame` statistiche; sotto, una riga con i
  due pulsanti centrati. Le dimensioni di cella si calcolano su **metà** larghezza
  schermo disponibile per lasciare spazio a due griglie.
- La GUI è una `ComparisonSimulation` che possiede **due** `Environment` + **due**
  pannelli; la modalità a griglia singola resta un caso particolare (una sola
  copia).

---

## 5. Algoritmi

### 5.1 Fase 1 — Q-learning tabellare (baseline RL)

- Stato discreto compatto: `carrying` + tipo trasportato + `f` quantizzato in
  pochi bin + hint di direzione migliore. Tabella `Q[stato, azione]` **condivisa**.
- Aggiornamento: `Q(s,a) ← Q(s,a) + η [r + γ max_a' Q(s',a') − Q(s,a)]`
  (o SARSA on-policy). Esplorazione ε-greedy con decadimento.
- Pro: nessuna dipendenza nuova, veloce, **completamente interpretabile**
  (si può stampare la tabella e confrontarla con le formule analitiche),
  riproducibile. Contro: osservazione necessariamente povera.

### 5.2 Fase 2 — Deep RL (risultati principali)

- **DQN** (Mnih et al. 2015) con replay buffer + target network, oppure **PPO**
  (Schulman et al. 2017) — PPO è più naturale per parametri condivisi in
  multi-agent ed evita l'instabilità del replay in ambiente non stazionario.
- Rete: encoder della finestra locale (CNN leggera o MLP) + testa azione (+ testa
  valore per PPO). Poche decine di migliaia di parametri.
- **Parameter sharing** (Gupta et al. 2017): tutte le formiche → stessa rete; le
  transizioni di tutte confluiscono nello stesso aggiornamento.
- Dipendenza: `torch` (CPU sufficiente per queste dimensioni). Da aggiungere a
  `pyproject.toml` come extra opzionale `rl`.

### 5.3 Warm start (opzionale ma consigliato)

Pre-addestrare per **imitazione** sulle traiettorie generate dall'euristica
(behavior cloning), poi fine-tuning con RL. Riduce drasticamente i campioni
necessari e dà un confronto "RL parte dall'euristica e la migliora?".

### 5.4 Iperparametri (da esplorare)

`γ` (discount), `η`/learning rate, `ε` e suo decadimento, dimensione finestra
locale, pesi del reward `ε_step, δ_penalty, α_final, β_time`, dimensione rete,
frequenza di update, dimensione batch/replay. Ricerca via griglia o random search
su un sottoinsieme di scenari, con criterio di selezione = entropia finale media
sul set di validazione.

---

## 6. Metriche e protocollo sperimentale

### 6.1 Metriche

**Qualità della soluzione**
- entropia finale del sistema a budget di iterazioni fissato (metrica primaria);
- numero di cluster distinti; **purezza** dei cluster; dimensione del cluster più
  grande per categoria; frazione di semi in cluster "puri".

**Efficienza**
- iterazioni (e wall-clock) per raggiungere una soglia di entropia obiettivo;
- **curva entropia vs iterazioni** e sua area (AUC): misura la velocità di
  ordinamento lungo tutta la corsa, non solo al termine;
- **mosse totali** e **mosse per unità di ordine guadagnato** (`Δmosse / ΔH`):
  quanto "lavoro" impiega ogni politica per lo stesso risultato;
- **semi raccolti** (pick cumulativi) e rapporto pick/drop utili su pick/drop
  totali (efficienza delle manipolazioni).

**Statistiche live della GUI** (§4.5) — mostrate sotto ciascuna griglia e loggate
a fine run: `tempo` (wall-clock, aggiornato a 1 Hz), `mosse totali`,
`semi raccolti`, `entropia totale`. Servono al confronto qualitativo/dimostrativo
affiancato; i numeri statisticamente validi vengono dalla campagna headless.

**Scalabilità**
- andamento delle metriche al variare di: numero di formiche, dimensione della
  griglia, numero di tipi di seme, densità dei semi.

**Generalizzazione**
- train su una configurazione, test (sola inferenza) su configurazioni non viste;
- degrado rispetto a una politica addestrata direttamente sullo scenario di test.

**Costo dell'RL**
- episodi / passi ambiente fino a convergenza; wall-clock di training; numero di
  parametri; sensibilità agli iperparametri.

**Comportamento (analisi qualitativa)**
- heatmap di attività / di pick / di drop sulla griglia;
- **tasso di contesa** (frazione di `try_acquire` falliti) heuristic vs rl, al
  variare del numero di thread;
- confronto tra **probabilità di pick/drop apprese** (stimate dalla politica in
  funzione di `f`) e le curve analitiche `(kp/(kp+f))²`, `(f/(kd+f))²`;
- statistiche di movimento (lunghezza dei tratti rettilinei, frequenza di
  inversione) vs la distribuzione fissa dell'euristica.

### 6.2 Confronti e ablazioni

| Sigla | Movimento | Pick/Drop | Scopo |
|---|---|---|---|
| **A0** | euristico | euristico | baseline (versione attuale) |
| **A1** | RL | euristico | il guadagno viene dal movimento? |
| **A2** | euristico | RL | il guadagno viene dalla manipolazione? |
| **A3** | RL | RL | politica completamente appresa |
| **A3-BC** | RL (warm start) | RL (warm start) | l'RL migliora l'euristica da cui parte? |

### 6.3 Protocollo

- **Scenari:** griglie `30×60` e `37×74`, semi `1225`/`1800`, formiche
  `50`/`100`, `seedTypes ∈ {3,5}`; più uno scenario piccolo per iterazioni rapide.
- **Ripetizioni:** ≥ 20 `master_seed` indipendenti; per ciascuno si eseguono
  **tutte** le modalità (A0–A3…) sulle **copie gemelle** dello stesso stato
  iniziale. Riportare **media ± intervallo di confidenza al 95 %**.
- **Confronto appaiato:** poiché le modalità partono da campi identici (stesso
  `master_seed`), si usa un test **appaiato** (Wilcoxon signed-rank) tra A0 e
  ciascuna variante RL, oltre al confronto tra medie; *effect size* riportato.
  Il confronto appaiato è possibile sia per-run (metriche finali) sia per-tick
  (curve entropia–iterazioni) grazie all'identità delle condizioni iniziali.
- **Correzione** per confronti multipli (Holm) sui vari scenari/ablazioni.
- **Stesso budget:** a parità di iterazioni massime e stesso criterio di arresto.
- **Fairness:** copie gemelle dello stesso ambiente, stessa fisica dei lock,
  stesso modello di esecuzione; l'unica differenza è la politica.
- **Separazione train/test:** iperparametri scelti su un set di validazione;
  numeri finali su un set di test mai usato per la selezione.

---

## 7. Ipotesi e analisi attese

- **H1 (RQ1):** l'RL raggiunge un'entropia finale ≤ euristica su almeno gli
  scenari di training; incerto sugli scenari di generalizzazione.
- **H2 (RQ2):** vantaggio maggiore sulla **velocità** (iterazioni-alla-soglia,
  AUC) che sull'entropia asintotica — l'euristica converge bene ma lentamente.
- **H3 (RQ3):** il contributo dominante viene dalla **manipolazione appresa**
  (A2 ≈ A3), perché è lì che si crea/distrugge ordine; il movimento appreso aiuta
  soprattutto riducendo il tempo perso e la contesa.
- **H4 (RQ5):** l'RL riscopre una relazione pick/drop qualitativamente simile a
  Lumer–Faieta (monotòna in `f`) ma con soglie diverse e dipendenti dal contesto;
  possibile emergere di **specializzazione** implicita pur con politica condivisa
  (perché lo stato include `carrying`/tipo).
- **Rischio principale:** non-stazionarietà multi-agente e reward ritardato per il
  movimento ⇒ training instabile. Mitigazioni: shaping potenziale, parametri
  condivisi, PPO invece di DQN, warm start, learning rate basso, valutazioni
  frequenti con early stopping.

Altri rischi: costo di calcolo dell'entropia (mitigato dal calcolo
locale/incrementale e dal fatto che il movimento non la cambia); overfitting allo
scenario di training (mitigato dalla randomizzazione degli episodi e dal set di
test separato); sensibilità agli iperparametri (documentata come risultato).

---

## 8. Sviluppi futuri

- **Azioni parametriche / meta-policy:** apprendere `kp`, `kd` (o direttamente le
  probabilità) come funzione del contesto, invece di costanti — ibrido euristica/RL.
- **Stigmergia appresa:** canale di comunicazione tipo feromone appreso
  (l'agente scrive/legge una traccia sulla cella), per confrontarsi con ACO.
- **Curriculum / transfer:** addestrare su griglie piccole e poche categorie,
  poi trasferire a griglie grandi; politica basata su GNN/attenzione per
  vicinati di dimensione variabile.
- **Multi-obiettivo:** funzioni di costo alternative (numero di cluster,
  silhouette, purezza) e ottimizzazione multi-obiettivo.
- **Offline RL:** apprendere solo da un dataset di traiettorie euristiche, senza
  interazione, e confrontare con l'online.
- **Parallelo / distribuito:** training vettorizzato su molti ambienti in
  parallelo e, in esecuzione, valutazione dello scaling multi-thread
  heuristic vs rl.
- **Robotica di sciame / Smart City:** riformulare il task su agenti fisici.

---

## 9. Riferimenti essenziali

- Deneubourg J.-L. et al. (1991). *The dynamics of collective sorting: robot-like
  ants and ant-like robots.*
- Lumer E. D., Faieta B. (1994). *Diversity and adaptation in populations of
  clustering ants.* — origine delle probabilità di pick/drop `(k/(k+f))²`.
- Sutton R., Barto A. (2018). *Reinforcement Learning: An Introduction.*
- Ng A., Harada D., Russell S. (1999). *Policy invariance under reward
  transformations: theory and application to reward shaping.*
- Mnih V. et al. (2015). *Human-level control through deep reinforcement
  learning (DQN).*
- Schulman J. et al. (2017). *Proximal Policy Optimization Algorithms (PPO).*
- Tan M. (1993). *Multi-agent reinforcement learning: independent vs. cooperative
  agents.*
- Gupta J. K., Egorov M., Kochenderfer M. (2017). *Cooperative multi-agent
  control using deep reinforcement learning (parameter sharing).*
- Cicirelli F. et al. (2016). *Transparent and efficient parallelization of swarm
  algorithms* — per l'angolo parallelo/distribuito.

---

## 10. Piano di lavoro (alto livello)

1. Rifattorizzazione: `AntPolicy` seam, `environment.py`, `master_seed` per lo
   stato iniziale condiviso, spostamento delle regole euristiche in
   `HeuristicPolicy` — **a parità di comportamento osservabile** (i test esistenti
   restano verdi).
2. `train.py` headless + logging + persistenza politica; `mode`/blocco `rl` in
   config; contatori `mosse_totali` / `semi_raccolti` nell'ambiente.
3. `TabularQPolicy` (Fase 1) + prime curve di apprendimento su scenario piccolo.
4. Ambiente vettorizzato / episodi randomizzati; reward shaping incrementale.
5. `DQNPolicy` / `PPOPolicy` (Fase 2) con `torch` opzionale.
6. Warm start per imitazione (opzionale).
7. **GUI `ComparisonSimulation`** (§4.5): due griglie gemelle affiancate, pulsanti
   Start/Visibility condivisi, pannello statistiche per copia con timer a 1 Hz.
8. Campagna sperimentale completa (§6) + analisi (§6.1, §7).
