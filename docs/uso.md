# Guida d'uso — installazione, addestramento, confronto

Passi dettagliati per far girare il progetto. I comandi assumono **Windows +
PowerShell** e di trovarsi nella cartella `AntelligentPy/`.

```powershell
cd C:\Users\crist\OneDrive\Documenti\Tesi\AntelligentPy
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
```

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
| `--no-learn-move` | — | **ablazione A2**: movimento euristico, pick/drop appreso |
| `--no-learn-manip` | — | **ablazione A1**: pick/drop euristico, movimento appreso |
| `--log-every N` | 10 | frequenza di stampa a console |

### 3d. Cosa guardare

- **`results/training_log.csv`** — una riga per episodio: `episode, ticks,
  initial_entropy, final_entropy, moves, seeds, epsilon, elapsed_s`. Traccia
  `final_entropy` in funzione di `episode` (in Excel o con pandas/matplotlib):
  deve scendere.
- **`results/policy.pkl`** — la politica addestrata; la GUI la carica in
  automatico se si trova in `results/`.

Per addestrare più varianti tienile separate:

```powershell
python -m antelligent.train --episodes 200 --out results\policy_A3.pkl
python -m antelligent.train --episodes 200 --no-learn-manip --out results\policy_A1.pkl
```

e copia in `results\policy.pkl` quella che vuoi vedere nella GUI.

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
3. **Start** (in basso, centrato) — avvia entrambe le simulazioni su due thread
   indipendenti. Il pulsante diventa **Close**.
4. **Visibility** — mostra/nasconde il disegno di **entrambe** le griglie; le
   statistiche continuano ad aggiornarsi (fai girare le run senza il costo del
   repaint).
5. Sotto ciascuna griglia, il pannello statistiche di quella copia:
   - **tempo** — cronometro dall'avvio, formato `mm:ss`, aggiornato **1 volta al
     secondo**;
   - **mosse totali** — spostamenti a buon fine di tutte le formiche;
   - **semi raccolti** — `pick` cumulativi;
   - **entropia totale** — entropia media del campo (0 = ordinato).
6. Quando **entrambe** le copie hanno raggiunto il criterio di arresto compare la
   schermata di riepilogo, con i valori affiancati e i pulsanti **Nuova
   iterazione** (torna al launcher) / **Esci**.
7. Seconda pressione di **Close**: salva screenshot + risultati e chiude.

### 4b. Output di una run

`results/` (ignorata da Git):

- **`results/results.txt`** — CSV `;`-separato, **due righe per run**
  (`mode = heuristic` / `rl`):
  `timestamp;mode;durataMs;iterazioni;righe;colonne;thread;formiche;semi;mosseTotali;semiRaccolti;entropiaIniziale;entropiaFinale;masterSeed;tipo`
- **`results/screenshots/<data-ora>/`** — screenshot periodici della finestra.

---

## 5. Il lavoro sperimentale (tesi)

Le "manopole" per gli esperimenti di `docs/rl-design.md` §5-§8:

### Ricompensa — `antelligent/environment.py`, classe `RewardConfig`

| Campo | Default | Effetto |
|---|---|---|
| `step_penalty` | 0.01 | penalità per tick (spinge a essere veloce) |
| `contested_penalty` | 0.05 | penalità se la cella scelta è occupata |
| `invalid_penalty` | 0.05 | penalità per pick/drop non valido |
| `manip_scale` | 1.0 | peso del segnale di manipolazione |
| `pivot` | 0.5 | soglia di `f`: sopra = "tra i simili", sotto = "isolato" |
| `final_scale` | 0.0 | bonus terminale `α·(H0 − H_final)` (da collegare nel runner) |

Oggi `train.py` usa i default. Per sperimentare: modifica i default, **oppure**
estendi `train.py` per esporli come argomenti CLI (è ~5 righe in `main()` +
`RewardConfig(...)` in `train()`).

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
| `nessuna politica RL addestrata (...)` all'avvio GUI | Normale se non hai ancora fatto `train.py`. Il pannello RL apprende dal vivo. Lancia l'addestramento per avere `results/policy.pkl`. |
| La GUI non parte, errore `TclError` / `no display` | Serve un ambiente grafico (non SSH/headless). |
| `ModuleNotFoundError: No module named 'antelligent'` | Lancia dalla cartella `AntelligentPy/` **oppure** fai `pip install -e .`. |
| `ModuleNotFoundError: No module named 'PIL'` | `pip install pillow`. |
| Addestramento troppo lento | Abbassa `--max-ticks` e `--episodes`, alza `--entropy-every`, usa una griglia più piccola (`--config`). |
| La simulazione sembra bloccata a inizio run | Se `nAnts` è vicino a `righe*colonne` il piazzamento delle formiche fatica a trovare celle libere. Tieni `nAnts` ben sotto il numero di celle (il launcher blocca solo il caso `nAnts > righe*colonne`). |
| Intestazioni miste in `results.txt` | Il file esisteva con un'intestazione diversa: cancellalo (vedi §2). |
| La run non si ferma mai con `stopCriterion=1` | La soglia scatta solo quando **nessuna** formica sta trasportando un seme **e** l'entropia è ≤ soglia. Alza `entropyThreshold` o passa a `stopCriterion=0` con `maxIterations`. |
