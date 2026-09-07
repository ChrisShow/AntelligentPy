# Guida d'uso — installazione, addestramento, confronto

Passi dettagliati per far girare il progetto. I comandi assumono **Windows +
PowerShell** e di trovarsi nella cartella `AntelligentPy/`.

```powershell
cd C:\percorso\alla\cartella\AntelligentPy
```

---

## 1. Ambiente Python

Serve **Python 3.11+** (tu hai il 3.13). Verifica:

```powershell
python --version
```

### 1a. Ambiente virtuale (consigliato)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se PowerShell blocca lo script di attivazione:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

e riprova. Da Git Bash l'attivazione è invece `source .venv/Scripts/activate`.

### 1b. Installazione

```powershell
pip install -e ".[test]"
```

Installa il pacchetto in *editable mode* (le modifiche al codice hanno effetto
subito) con le dipendenze: **Pillow** (sprite + screenshot) e **pytest**.

Dopo l'installazione questi comandi funzionano da qualsiasi cartella:

| Comando | Equivalente |
|---|---|
| `antelligent` | `python -m antelligent` (GUI di confronto) |
| `antelligent-train` | `python -m antelligent.train` (addestramento) |

Senza `pip install`, usa `python -m antelligent ...` **solo dalla cartella
`AntelligentPy/`** (altrimenti `ModuleNotFoundError: antelligent`).

### 1c. Verifica

```powershell
pytest -q
```

Atteso: `36 passed`.

---

## 2. Configurazione (`config.properties`)

Il file `config.properties` nella cartella del progetto contiene i parametri di
default del launcher. **Viene riscritto ogni volta che avvii una run dal
launcher**, quindi riflette l'ultima scelta.

```properties
cols=40                  # colonne della griglia
rows=25                  # righe
nThread=6                # (non più critico: i due pannelli usano 1 thread ciascuno)
nAnts=100                # formiche per griglia   (vincolo: nAnts  <= cols*rows)
nSeeds=650               # semi per griglia       (vincolo: nSeeds <= cols*rows)
seedTypes=5              # 1-5 colori di seme
type=1                   # matrice normale con lock
refreshRate=500          # iterazioni tra un repaint della board e il successivo
stopCriterion=1          # 0 = numero massimo di iterazioni, 1 = soglia di entropia
maxIterations=20000      # usato con stopCriterion=0
entropyThreshold=5.0     # usato con stopCriterion=1 (scala 0-100; più basso = più ordinato)
masterSeed=0             # 0 = stato iniziale casuale a ogni run; >0 = riproducibile
captureScreenshots=0     # 1 = salva schermate periodiche in results/screenshots/
```

> **`captureScreenshots`** corrisponde alla casella **"Cattura schermate della
> finestra"** nel launcher (sezione *Opzioni*). È **disattivata di default**: la
> cattura (`ImageGrab`) blocca il thread della GUI a ogni scatto e su macOS
> richiede il permesso *Impostazioni di Sistema → Privacy e sicurezza →
> Registrazione schermo*. Attivala solo quando ti servono davvero le immagini.

Puoi modificarlo a mano oppure dal campo corrispondente nel launcher.

> **`masterSeed`**: le due griglie (euristica / RL) partono **sempre dalla stessa
> identica configurazione iniziale**. Con `masterSeed=0` è casuale a ogni run
> (comunque condivisa dalle due copie); con un valore fisso la run è
> riproducibile — utile per il confronto appaiato.

### Pulisci un `results.txt` con intestazione diversa

Se `results/results.txt` esiste già con un'intestazione diversa da quella
descritta in §4b (il file viene scritto con **due righe per run**), cancellalo
per evitare intestazioni miste:

```powershell
Remove-Item results\results.txt -ErrorAction SilentlyContinue
Remove-Item -Recurse results\screenshots -ErrorAction SilentlyContinue
```

(La cartella `results/` è ignorata da Git: nessun dato versionato viene perso.)

---

## 3. Addestrare la politica RL (`python -m antelligent.train`)

Il pannello "RL" della GUI diventa interessante **solo dopo** aver addestrato una
politica: senza `results/policy.pkl` la GUI fa apprendere l'RL *dal vivo* durante
la run (lo vedi migliorare, ma i risultati statistici vanno prodotti qui).

### 3a. Prova rapida (~1-2 minuti)

Usa una griglia piccola per iterare in fretta. Crea `docs/train-small.properties`:

```properties
cols=20
rows=14
nThread=4
nAnts=25
nSeeds=120
seedTypes=4
type=1
refreshRate=200
stopCriterion=0
maxIterations=1500
entropyThreshold=5.0
masterSeed=0
```

poi:

```powershell
python -m antelligent.train --config docs\train-small.properties --episodes 20 --seed 1
```

Output atteso (ogni 10 episodi + l'ultimo):

```
ep    0  ticks  1500  H0  72.6 -> H  64.8  moves   ...  seeds  ...  eps 0.253
...
politica salvata in ...\results\policy.pkl
log di addestramento in ...\results\training_log.csv
```

Dovresti vedere il divario `H0 -> H` allargarsi e il numero di `seeds` (pick
cumulativi) **calare** episodio dopo episodio: la politica impara a smettere di
rimescolare a caso.

### 3b. Run completa

Con la config "vera" (`config.properties`), riducendo i tick per episodio:

```powershell
python -m antelligent.train --episodes 300 --max-ticks 2500 --entropy-every 50 --seed 1
```

Indicazioni di durata: circa `episodi × max-ticks × nAnts` passi-agente. Su
`config.properties` di default (100 formiche) conta **decine di secondi per
episodio** → una run da 300 episodi può richiedere **qualche ora**. Parti con
`--episodes 100` e aumenta.

### 3c. Opzioni di `train.py`

| Flag | Default | Significato |
|---|---|---|
| `--config PATH` | `config.properties` | scenario di addestramento |
| `--out PATH` | `results/policy.pkl` | dove salvare la politica |
| `--episodes N` | 400 | numero di episodi |
| `--seed N` | 0 | seed globale (riproducibilità) |
| `--max-ticks N` | `maxIterations` della config | tetto di tick per episodio |
| `--entropy-every N` | 25 | ogni quanti tick ricalcolare l'entropia (costo/arresto) |
| `--alpha` `--gamma` | 0.1 / 0.95 | learning rate / discount |
| `--epsilon-start` `--epsilon-end` | 0.30 / 0.02 | esplorazione ε-greedy iniziale/finale |
| `--epsilon-decay N` | 3/4 degli episodi | su quanti episodi decade ε |
| `--step-penalty` | 0.01 | penalità per tick (`RewardConfig`) |
| `--contested-penalty` | 0.05 | penalità per destinazione occupata |
| `--invalid-penalty` | 0.05 | penalità per pick/drop non valido |
| `--manip-scale` | 1.0 | peso del segnale di pick/drop |
| `--pivot` | `1/seedTypes` | soglia di `f_own`: sopra = "tra i simili", sotto = "isolato" |
| `--allow-stay` | — | ridà all'RL l'azione "resta fermo" (**sconsigliato**, vedi §6) |
| `--no-learn-move` | — | **ablazione A2**: movimento euristico, pick/drop appreso |
| `--no-learn-manip` | — | **ablazione A1**: pick/drop euristico, movimento appreso |
| `--log-every N` | 10 | frequenza di stampa a console |

### 3d. Cosa guardare

- **`results/training_log.csv`** — una riga per episodio: `episode, ticks,
  initial_entropy, final_entropy, moves, picks, drops, carried_left, epsilon,
  elapsed_s`. Traccia `final_entropy` in funzione di `episode` (in Excel o con
  pandas/matplotlib): deve scendere. `picks`/`drops` devono restare **vicini fra
  loro** e non crollare a zero; `carried_left` (formiche ancora con un seme in
  mano a fine episodio) deve restare basso — se cresce, la politica raccoglie
  senza più posare (alza `--pivot` o `--manip-scale`, o abbassa `--step-penalty`).
- **`results/policy.pkl`** — la politica addestrata; la GUI la carica in
  automatico se si trova in `results/`.

Per addestrare più varianti tienile separate:

```powershell
python -m antelligent.train --episodes 200 --out results\policy_A3.pkl
python -m antelligent.train --episodes 200 --no-learn-manip --out results\policy_A1.pkl
```

e copia in `results\policy.pkl` quella che vuoi vedere nella GUI.

---

## 3bis. Variante: ricompensa alla Lumer-Faieta (`python -m antelligent.train_lumer`)

Seconda formulazione della ricompensa, **in file separati**: `antelligent/lumer_reward.py`,
`antelligent/environment_lumer.py`, `antelligent/policies/lumer_tabular.py`,
`antelligent/train_lumer.py`. Tutto l'impianto del §3 resta com'era e continua a
funzionare; le due varianti convivono e si confrontano.

### 3bis-a. L'idea

L'euristica decide **campionando** dalle probabilità di Lumer-Faieta:

```
P_pick(f) = (kp/(kp+f))²     kp = 0.1     ← raccogli tanto più volentieri quanto il seme è isolato
P_drop(f) = (f/(kd+f))²      kd = 0.3     ← posa tanto più volentieri quanto è circondato dai suoi simili
```

dove `f` è la frazione, fra gli **8 vicini**, dei semi dello stesso tipo di quello
manipolato. Qui le stesse formule diventano **valore**, non probabilità: l'agente
non tira più il dado, ma riceve un rinforzo tanto più alto quanto più l'azione
scelta è quella che l'euristica giudicherebbe conveniente *in quel punto*.

Stessa conoscenza di dominio dell'euristica, ma **appresa** invece che cablata — e
quindi migliorabile: l'RL può imparare a *cercare* i punti buoni, cosa che
l'euristica (random walk puro) non fa.

### 3bis-b. Il punto di indifferenza `f*`

Il segnale dev'essere **con segno**, altrimenti "manipolare" batte sempre "non
manipolare" e la colonia entra in *churn* (raccogli-e-riposa a caso). Lo zero non
è scelto a mano: è il valore di `f` in cui le due regole si equivalgono,

```
P_pick(f) = P_drop(f)   ⟺   kp/(kp+f) = f/(kd+f)   ⟺   f* = √(kp·kd) ≈ 0.1732
```

Sotto `f*` l'euristica preferisce raccogliere, sopra preferisce posare. È lo stesso
ruolo che nel §3 aveva il livello del caso `1/k`, ma qui **deriva dalle costanti
dell'euristica** invece che dal numero di tipi — ed è esattamente lì che il segnale
di ricompensa cambia segno.

### 3bis-c. I tre contributi

| contributo | flag | cosa insegna |
|---|---|---|
| **azione** | `--pick-scale` `--drop-scale` | quanto vale raccogliere / posare *in quel punto* |
| **rifiuto** | `--decline-scale` | se poteva manipolare **con vantaggio** e non l'ha fatto, paga quel vantaggio: anche il braccio "non agire" viene informato |
| **navigazione** | `--shaping-scale` | shaping potenziale `F = γ·Φ(s′) − Φ(s)` con `Φ` = probabilità LF della manipolazione desiderata nella cella corrente: avvicinarsi a un buon punto di deposito paga **prima** di posare |
| **muro** | `--wall-penalty` | costo di una mossa scelta **fuori dalla griglia** — vedi il riquadro sotto |

> **Il rifiuto è unilaterale, e il muro si paga.** Sono due correzioni fatte
> *dopo* aver misurato, non scelte a priori. Nella prima versione il rifiuto era
> simmetrico (rifiutare una manipolazione sconveniente veniva *premiato*): questo
> paga una formica ogni tick perché "sta correttamente ferma su un buon
> grappolo", cioè crea una **rendita di posizione**. La politica l'ha sfruttata
> nel modo più diretto: scegliere una direzione **fuori griglia**, che
> `_apply_move` tratta come "resta ferma" senza addebitare `contested_penalty`.
> Misurato: **39–44 %** dei passi contro un bordo (euristica 0 %, RL base 2 %), e
> metà delle mosse perse. È la stessa patologia di `QConfig.allow_stay`, rientrata
> dalla porta di servizio. Rimedi: rifiuto solo punitivo (`decline_two_sided`
> riproduce l'ablazione) e `wall_penalty` che chiude la porta.

Lo shaping è **potenziale** nel senso di Ng, Harada & Russell (1999): la somma su
un cammino dipende solo dagli estremi, quindi girare in tondo non frutta niente e
la politica ottima non cambia — cambia solo la velocità con cui la si trova. Per
questo `--gamma` viene passato *anche* allo shaping: con uno sconto diverso
l'invarianza si perde.

`--drop-scale` è la manopola che risponde direttamente alla domanda "quanto conta
il **punto** in cui lascio il seme": alzarlo pesa di più il deposito rispetto alla
raccolta.

### 3bis-d. Opzioni di `train_lumer.py`

Oltre a tutte quelle del §3c (`--episodes --config --out --seed --max-ticks
--entropy-every --alpha --gamma --epsilon-* --allow-stay --no-learn-move
--no-learn-manip --step-penalty --contested-penalty --invalid-penalty
--log-every`):

| Flag | Default | Significato |
|---|---|---|
| `--mode {advantage,centered,raw}` | `centered` | `centered` = `2P − 1`, zero a probabilità 50 % (**il migliore misurato**); `advantage` = `P(azione) − P(opposta)`, zero in `f*`; `raw` = `P`, mai negativo (**ablazione**: mostra il churn) |
| `--pick-scale` | 1.0 | peso del segnale di raccolta |
| `--drop-scale` | **2.0** | peso del segnale di deposito, cioè del **punto** in cui si posa |
| `--decline-scale` | 1.0 | peso della penalità per aver rifiutato una manipolazione conveniente (`0` = gratis) |
| `--decline-two-sided` | — | **ablazione**: premia anche il rifiuto di una manipolazione sconveniente (patologico) |
| `--shaping-scale` | 1.0 | peso dello shaping potenziale (`0` = disattivato) |
| `--wall-penalty` | 0.05 | costo di una mossa scelta fuori dalla griglia (`0` = gratis, **sconsigliato**) |
| `--kp` `--kd` | 0.1 / 0.3 | costanti di Lumer-Faieta (le stesse dell'euristica; spostarle sposta `f*`) |
| `--isolated-guard` | — | riproduce alla lettera la guardia dell'euristica sul vicinato vuoto (`P = 0` per entrambe le regole) |
| `--f-bins` | 4 | granularità della discretizzazione di `f` |
| `--out` | `results/policy_lumer.pkl` | il log finisce accanto, come `<nome>_log.csv` |

> **`--isolated-guard`.** Con un vicinato completamente vuoto `f = 0/0` non è
> definita e l'euristica restituisce `0` per entrambe le regole. Di default qui le
> formule si estendono per continuità (`f = 0`): raccogliere un seme completamente
> isolato vale `+1` (è il seme più fuori posto che esista) e posarne uno nel
> deserto vale `−1`. Riprodurre la guardia renderebbe il deserto un posto
> *neutro* dove scaricare i semi.

### 3bis-e. Usarla nella GUI

Il pickle **dichiara la propria famiglia** (`policy_kind`), e la GUI sceglie da
sola la classe giusta: la codifica della testa di manipolazione è diversa fra le
due varianti, e caricare l'una come l'altra fallirebbe *in silenzio* (letture
sistematiche su righe mai viste → azioni a caso). Quindi basta:

```powershell
python -m antelligent.train_lumer --episodes 400 --out results\policy.pkl
python -m antelligent
```

Se carichi il file sbagliato, il log lo dice: `... è una politica 'tabular', non
'lumer' ... Riaddestrala con: python -m antelligent.train_lumer`.

### 3bis-f. Cosa cambia nella politica

`LumerQPolicy` **eredita** da `TabularQPolicy`: stessa Q-learning fattorizzata,
stesso `_move_state` compatto col gradiente locale `best_dir`, stesso
`allow_stay = False`, stesso `inference_epsilon`. Tutte le correzioni che hanno
sbloccato le formiche (§5) valgono quindi anche qui *per costruzione*: se cambiano
là, cambiano anche qua.

L'unica differenza è la discretizzazione della testa pick/drop:

| | `TabularQPolicy` | `LumerQPolicy` |
|---|---|---|
| stato | `(trasporta, tipo, bin di f, f ≥ 1/k)` | `(trasporta, bin di f, f ≥ f*)` |
| stati (k=5, 4 bin) | 80 | **16** |

Il **codice del tipo sparisce** perché con questa ricompensa il valore di una
manipolazione è `g(azione, f)`: una funzione della sola frazione di simili. Quale
tipo sia non cambia nulla — entra già tutto in `f`. Tenerlo replicherebbe la stessa
decisione su `k` righe distinte, dividendo per `k` i dati per stato.

### 3bis-g. Cosa è stato misurato

Protocollo: `config.properties` (14×14, 130 semi ≈ 66 % di densità, 10 formiche,
5 tipi), 400 episodi di addestramento da 2000 tick, valutazione su **10
`master_seed`** da 2000 tick con politica congelata. `H` = entropia finale
(0 = ordinato). "vince" = entropia più bassa dell'euristica **sullo stesso seed**.

| variante | H finale | vince su | picks | in mano |
|---|---|---|---|---|
| euristica (A0) | 25.5 ± 1.9 | — | 210 | 1.7/10 |
| RL base (`train.py`, pivot `1/k`) | 38.8 ± 1.9 | 0/10 | 385 | 1.4/10 |
| **LF default** (`centered`, drop×2) | **13.6 ± 2.5** | **10/10** | 310 | 3.4/10 |
| LF `--drop-scale 1` | 19.9 ± 3.6 | 9/10 | 319 | 4.7/10 |
| LF `--mode advantage` | 33.0 ± 3.5 | 1/10 | 446 | 1.3/10 |
| LF `--shaping-scale 0` | 19.3 ± 4.1 | 9/10 | 217 | 6.1/10 |
| LF `--decline-scale 0` | 24.2 ± 2.0 | 7/10 | 391 | 0.5/10 |
| LF `--wall-penalty 0` | 23.0 ± 2.7 | 6/10 | 360 | 3.3/10 |
| LF `--decline-two-sided` | 35.7 ± 1.6 | 0/10 | 424 | 2.0/10 |
| LF `--mode raw` | 44.9 ± 2.5 | 0/10 | **4330** | 5.2/10 |

**È la prima configurazione che batte l'euristica**, e la batte su tutti i seed di
valutazione. Quattro letture:

1. **`centered` batte `advantage`** (13.6 contro 33.0). `advantage` mette lo zero
   in `f* ≈ 0.173`, cioè posare conviene appena si è sopra il livello del caso;
   `centered` lo mette a `P = 0.5`, cioè il drop conviene solo da `f > 0.72`. È
   una soglia molto più **selettiva**: la formica non posa "abbastanza bene", posa
   solo dove è *davvero* giusto. Il prezzo è che a fine run più formiche hanno
   ancora un seme in mano (3.4/10 contro 1.3/10) — vedi il riquadro sotto.
2. **Pesare il deposito paga.** Con `centered` i due segnali hanno escursione molto
   diversa: il pick arriva a `+1.0` (seme isolato), il drop solo a `+0.18` (seme fra
   soli simili) — un quinto. Senza riequilibrio la politica impara molto meglio
   *quando raccogliere* che *dove posare*. `--drop-scale 2` porta H da ~20 a ~15
   (media su 3 seed di addestramento). Lo sweep `1.0 / 1.5 / 2 / 3 / 4` dà
   `19.9 / 22.1 / 13.6 / 18.1 / 18.0`: **non è monotono**, e la varianza fra seed di
   addestramento (13.6 / 13.1 / 19.5 con lo stesso `--drop-scale 2`) è dello stesso
   ordine dello sweep. `2` è il punto migliore *misurato*, non un ottimo dimostrato:
   il confronto va rifatto con più repliche.
3. **Le due ablazioni patologiche si vedono.** `--mode raw` (segnale mai negativo)
   produce **4330 pick** contro 310: è il *churn* previsto, raccogli-e-riposa senza
   criterio. `--decline-two-sided` (rendita di posizione) fa perdere 0/10.
4. **Più episodi non aiutano.** 800 episodi invece di 400 danno H 21.7 contro 13.6:
   con ε già decaduto la politica smette di esplorare e deriva. Il budget giusto
   qui è ~400 episodi.

> **Attenzione al criterio di arresto.** La variante LF è più selettiva sul
> deposito, quindi a fine run tiene in mano più semi (3.4/10 contro 1.7/10
> dell'euristica). Con `stopCriterion=1` la run termina solo quando **nessuna**
> formica trasporta un seme *e* l'entropia è sotto soglia: con questa variante può
> volerci di più. Per i confronti a budget fisso usa `stopCriterion=0` con
> `maxIterations`.

Riproduzione:

```powershell
python -m antelligent.train_lumer --episodes 400 --max-ticks 2000 --entropy-every 100 --seed 0
python -m antelligent.train_lumer --episodes 400 --max-ticks 2000 --mode advantage --seed 0
python -m antelligent.train_lumer --episodes 400 --max-ticks 2000 --mode raw --seed 0
python -m antelligent.train_lumer --episodes 400 --max-ticks 2000 --decline-two-sided --seed 0
```

---

## 4. GUI di confronto (`python -m antelligent`)

```powershell
python -m antelligent
```

Serve un **display grafico** (non funziona via SSH/headless).

### 4a. Flusso

1. Si apre il **launcher**: rivedi i parametri (compreso "Seed iniziale") e premi
   **Avvia simulazione**. I valori scelti vengono salvati in `config.properties`.
2. Si apre **una finestra** con **due griglie affiancate**:
   - **EURISTICA** (sinistra) — algoritmo pre-impostato;
   - **RL** — titolo `RL (addestrata)` se ha caricato `results/policy.pkl`,
     `RL (apprendimento live)` se non c'è una politica salvata.
3. **Start** (in basso, centrato) — avvia le simulazioni **una alla volta**:
   prima l'euristica, poi l'RL quando la prima ha concluso. Il pulsante diventa
   **Close**.
4. **Visibility** — mostra/nasconde il disegno di **entrambe** le griglie; le
   statistiche continuano ad aggiornarsi (fai girare le run senza il costo del
   repaint).
5. Sotto ciascuna griglia, il pannello statistiche di quella copia:
   - **tempo euristica** / **tempo RL** — cronometro della copia, formato `mm:ss`,
     aggiornato **1 volta al secondo**, con lo stato accanto: `in attesa` (non è
     ancora il suo turno), `in corso`, `conclusa`. A fine run **si ferma** sul
     valore finale invece di continuare a scorrere;
   - **mosse totali** — spostamenti a buon fine di tutte le formiche;
   - **semi raccolti** — `pick` cumulativi;
   - **entropia totale** — entropia media del campo (0 = ordinato).
6. Quando **entrambe** le copie hanno raggiunto il criterio di arresto compare la
   schermata di riepilogo, con i valori affiancati e i pulsanti **Nuova
   iterazione** (torna al launcher) / **Esci**.

> **Perché sequenziali.** Girando insieme, le due copie si contendevano il GIL e
> il *tempo* di ciascuna dipendeva da quanto era carica l'altra — inutilizzabile
> come metrica di confronto. Una alla volta, il wall-clock di ogni politica è
> misurato su una macchina scarica.
7. Seconda pressione di **Close**: salva screenshot + risultati e chiude.

### 4b. Output di una run

`results/` (ignorata da Git):

- **`results/results.txt`** — CSV `;`-separato, **due righe per run**
  (`mode = heuristic` / `rl`):
  `timestamp;mode;durataMs;iterazioni;righe;colonne;thread;formiche;semi;mosseTotali;semiRaccolti;entropiaIniziale;entropiaFinale;masterSeed;tipo`
- **`results/screenshots/<data-ora>/`** — screenshot periodici della finestra.

---

## 5. Il lavoro sperimentale

Le "manopole" per gli esperimenti di `docs/rl-design.md` §5-§8:

### Ricompensa — `antelligent/environment.py`, classe `RewardConfig`

| Campo | Default | Effetto |
|---|---|---|
| `step_penalty` | 0.01 | penalità per tick (spinge a essere veloce) |
| `contested_penalty` | 0.05 | penalità se la cella scelta è occupata |
| `invalid_penalty` | 0.05 | penalità per pick/drop non valido |
| `manip_scale` | 1.0 | peso del segnale di manipolazione |
| `pivot` | `None` → `1/seedTypes` | soglia di `f_own` (frazione di vicini dello stesso tipo del seme manipolato): pick premiato sotto, drop premiato sopra |
| `final_scale` | 0.0 | bonus terminale `α·(H0 − H_final)` (da collegare nel runner) |

> **`pivot` va lasciato automatico.** Su una cella a caso la frazione attesa di
> vicini dello stesso tipo è `1/k` con `k = seedTypes`. Un `pivot` fisso a `0.5`
> con `k = 5` rende il drop **penalizzato in media** (`0.2 − 0.5 = −0.3`): la
> politica greedy impara a non posare mai e le formiche restano bloccate con il
> seme in mano. Il default `1/k` mette lo zero al livello del caso, così il segno
> della ricompensa distingue "meglio del caso" da "peggio del caso".

`step_penalty`, `contested_penalty`, `invalid_penalty`, `manip_scale` e `pivot`
sono esposti da `train.py` come `--step-penalty --contested-penalty
--invalid-penalty --manip-scale --pivot`. `final_scale` non è ancora collegato al
runner.

### Il gradiente locale (`Observation.best_dir`)

La ricompensa è **già interamente locale**: dipende solo da `f_own`, la frazione di
semi dello stesso tipo nelle 8 celle adiacenti alla formica. L'entropia globale
(`check_entropy_placed`) **non entra mai** nella ricompensa — serve solo al criterio
di arresto e alla statistica mostrata nella GUI.

Quello che mancava non era la località del *costo*, ma la località
dell'*informazione*: `best_dir` era calcolato solo mentre la formica trasportava
un seme, quindi una formica a mani libere non sapeva **mai** dove fossero i semi
(nel 100 % di quegli stati valeva `8` = "nessun bersaglio"). Il suo movimento
diventava un riflesso fisso su heading + celle bloccate, e con la politica congelata
(deterministica) degenerava in cicli: avanti e indietro fra le stesse due celle.

Ora `best_dir` è un vero gradiente locale in **entrambe** le fasi:

| fase | bersaglio |
|---|---|
| trasporta | la cella adiacente **vuota** i cui vicini sono più dello stesso tipo del seme in mano (dove posare *bene*) |
| a mani libere | la cella adiacente col seme più **fuori posto** (frazione di simili più bassa) — quello che conviene raccogliere |

Puntare, come prima, a una cella che *contiene* già un seme simile era controproducente:
lì non si può posare.

Il calcolo riusa una sola scansione del blocco 5×5 attorno alla formica
(`Environment._seed_block`), condivisa da `f_here` e dal bersaglio.

### Perché la politica congelata mantiene un po' di rumore

`QConfig.inference_epsilon` (default `0.05`) è la probabilità di mossa casuale che
resta **anche a politica congelata**. Non è una svista: l'osservazione è parziale e
fortemente aliasata (stati diversi del mondo appaiono identici alla formica), e una
politica *deterministica* su stati aliasati cade in **cicli limite** — avanti e
indietro fra le stesse due celle — da cui non può uscire, perché la scelta dipende
solo dall'osservazione, che non cambia. In un POMDP la politica ottima è in generale
stocastica; e l'euristica di confronto lo è già (mossa estratta da una distribuzione,
pick/drop probabilistici), quindi il paragone resta equo.

Misurato su `config.properties` (14×14, 130 semi, 10 formiche, 5 tipi), 5 `master_seed`:

| `inference_epsilon` | H finale | ritorni sulla cella `t-2` | picks |
|---|---|---|---|
| 0.00 (greedy pura) | 44.7 | 30.2 % | 69 |
| **0.05 (default)** | **36.6** | **20.6 %** | 206 |
| 0.10 | 36.8 | 18.5 % | 303 |
| 0.20 | 36.7 | 15.2 % | 465 |

`freeze()` sospende l'**apprendimento**, non il rumore. Con `inference_epsilon = 0`
si ottiene la greedy pura di prima.

### Perché l'RL non ha l'azione "resta fermo"

`QConfig.allow_stay` è `False` di default: la testa del movimento sceglie fra le
**8 direzioni relative**, come l'euristica (`check_move.random_move_index` non
restituisce mai `STAY` su una griglia normale). Con `STAY` disponibile la
Q-learning ci cade dentro: restare fermi non rischia mai `contested_penalty`,
quindi è l'azione meno costosa ovunque non ci sia una ricompensa positiva
raggiungibile, e il valore `Q(s, STAY) = −step_penalty/(1−γ)` si auto-sostiene
(self-loop assorbente). La fisica gestisce comunque il caso "non riesco a
muovermi" (contesa → ripiego su un'adiacente libera → resta ferma), quindi
l'azione esplicita non serve. `--allow-stay` la riabilita per le ablazioni.

### Iperparametri RL — `antelligent/policies/tabular.py`, classe `QConfig`

Già esposti in `train.py`: `--alpha --gamma --epsilon-start --epsilon-end
--epsilon-decay`. `f_bins` (granularità della discretizzazione di `f`) si cambia
nel codice.

### Ablazioni (`docs/rl-design.md` §6.2)

| Sigla | Come |
|---|---|
| **A0** | pannello EURISTICA della GUI (sempre presente) |
| **A1** | `python -m antelligent.train --no-learn-manip ...` |
| **A2** | `python -m antelligent.train --no-learn-move ...` |
| **A3** | `python -m antelligent.train ...` (default) |
| **LF** | `python -m antelligent.train_lumer ...` (§3bis) — stessa ablazione A1/A2/A3 con `--no-learn-manip` / `--no-learn-move` |

### Mini-campagna riproducibile

```powershell
foreach ($s in 1..20) {
  python -m antelligent.train --config docs\train-small.properties --episodes 150 --seed $s --out results\pol_s$s.pkl
}
```

Aggrega i `training_log.csv` (uno per run, sovrascritto: rinominali o cambia
`--out` in una sottocartella), calcola media ± IC 95 %, test appaiato di Wilcoxon
tra A0 e le varianti (design §6.3).

---

## 6. Problemi frequenti

| Sintomo | Causa / rimedio |
|---|---|
| La GUI avverte "è una politica 'tabular', non 'lumer'" (o viceversa) | Hai puntato la GUI a un pickle dell'altra variante. Le due codificano la testa di manipolazione in modo diverso: riaddestra con il comando indicato nel messaggio (`train.py` o `train_lumer.py`). |
| La GUI avverte "addestrata con la codifica di stato v1, questa versione usa la v2" | `results/policy.pkl` è di un addestramento precedente a un cambio di `_move_state`/`_manip_state`: la tabella non è riutilizzabile. Riaddestra con `python -m antelligent.train`. |
| `nessuna politica RL addestrata (...)` all'avvio GUI | Normale se non hai ancora fatto `train.py`. Il pannello RL apprende dal vivo. Lancia l'addestramento per avere `results/policy.pkl`. |
| La GUI non parte, errore `TclError` / `no display` | Serve un ambiente grafico (non SSH/headless). |
| `ModuleNotFoundError: No module named 'antelligent'` | Lancia dalla cartella `AntelligentPy/` **oppure** fai `pip install -e .`. |
| `ModuleNotFoundError: No module named 'PIL'` | `pip install pillow`. |
| Addestramento troppo lento | Abbassa `--max-ticks` e `--episodes`, alza `--entropy-every`, usa una griglia più piccola (`--config`). |
| La simulazione sembra bloccata a inizio run | Se `nAnts` è vicino a `righe*colonne` il piazzamento delle formiche fatica a trovare celle libere. Tieni `nAnts` ben sotto il numero di celle (il launcher blocca solo il caso `nAnts > righe*colonne`). |
| Intestazioni miste in `results.txt` | Il file esisteva con un'intestazione diversa: cancellalo (vedi §2). |
| La run non si ferma mai con `stopCriterion=1` | La soglia scatta solo quando **nessuna** formica sta trasportando un seme **e** l'entropia è ≤ soglia. Alza `entropyThreshold` o passa a `stopCriterion=0` con `maxIterations`. |
