# llama.cpp su Singularity — guida all'utilizzo

Questa guida descrive come **utilizzare** `llama.cpp` una volta disponibile
l'immagine (in locale) o la sandbox (su IBiSCo). La procedura di build non è
qui trattata: per la costruzione dell'immagine `.sif` e della sandbox CUDA si
rimanda a [installazione_container.md](installazione_container.md).

L'utilizzo si articola in due momenti distinti:

- **l'avvio del servizio** (`llama-server`), che carica un modello GGUF in VRAM
  ed espone un'interfaccia web e un endpoint HTTP compatibile con l'API OpenAI;
- **l'interrogazione del servizio**, dalla stessa macchina o da remoto, tramite
  interfaccia web, `curl` o un client (ad esempio la libreria `openai` di
  Python).

La guida è articolata in due parti:

- **Parte 1 — Utilizzo in locale**, con l'immagine `llamacpp-cuda.sif` costruita
  al termine della Parte 1 della guida di installazione.
- **Parte 2 — Utilizzo su IBiSCo**, con la sandbox `cu122-sm70/` e i binari già
  compilati su Lustre (Parte 2 della guida di installazione).

Il presupposto comune è che il modello GGUF da servire sia già stato scaricato
secondo quanto descritto in [download_modelli.md](download_modelli.md).

---

## Parte 1 — Utilizzo in locale

### 1.1 Avvio del servizio

L'avvio del servizio ricalca la sezione § 1.4 della guida di installazione. La
`%runscript` dell'immagine inoltra gli argomenti direttamente a `llama-server`,
per cui è sufficiente:

```bash
singularity run --nv \
    -B $HOME/.cache/huggingface:/hf \
    llamacpp-cuda.sif \
    -m /hf/hub/models--<org>--<repo>/snapshots/<hash>/mio-modello.gguf \
    --host 127.0.0.1 --port 8080 \
    -ngl 99 -c 4096 --jinja
```

- `--nv` — espone driver e librerie GPU dell'host nel container (indispensabile).
- `-B $HOME/.cache/huggingface:/hf` — monta la cache di Hugging Face, in cui
  risiedono i modelli scaricati con `hf download`.
- `-m` — percorso del file GGUF all'interno della cache montata.
- `--host 127.0.0.1` — in locale il servizio ascolta sul solo loopback.
- `-ngl 99` — offload di tutti i layer in VRAM (da ridurre se il modello non vi
  entra interamente).
- `-c 4096` — dimensione della finestra di contesto.
- `--jinja` — abilita il template di chat nativo del modello (necessario per il
  corretto formato dei ruoli nell'endpoint `/v1/chat/completions`, in particolare
  per i modelli con reasoning o tool-calling).

Il processo resta in primo piano: la shell rimane occupata e il log del server
scorre a terminale. Per interrompere il servizio è sufficiente `Ctrl-C`.

> **Nota.** Per lasciare il servizio attivo dopo la chiusura del terminale lo si
> può avviare in una sessione persistente (`tmux` o `screen`), oppure
> reindirizzando l'output e ponendolo in background:
> `... llamacpp-cuda.sif ... > server.log 2>&1 &`.

### 1.2 Verifica dello stato del servizio

All'avvio, `llama-server` stampa a log il caricamento dei layer sulla GPU e la
riga di ascolto (`server is listening on http://127.0.0.1:8080`). Lo stato può
essere verificato interrogando l'endpoint di salute:

```bash
curl http://127.0.0.1:8080/health
```

La risposta `{"status":"ok"}` indica che il modello è caricato e pronto a
ricevere richieste. Durante il caricamento la risposta può invece riportare
`{"status":"loading model"}`.

### 1.3 Interfaccia web

A servizio avviato, l'interfaccia web è raggiungibile dal browser all'indirizzo:

```
http://127.0.0.1:8080
```

Consente di dialogare con il modello in modalità chat e di regolare i parametri
di campionamento (temperatura, `top_p`, ecc.) senza scrivere codice. È il modo
più rapido per una verifica funzionale del modello appena avviato.

### 1.4 Interrogazione via API (endpoint OpenAI-compatibile)

`llama-server` espone un endpoint compatibile con l'API OpenAI su
`http://127.0.0.1:8080/v1`. Ciò consente di riutilizzare qualunque client già
predisposto per OpenAI, cambiando soltanto `base_url` e ignorando la chiave API
(che `llama-server` non verifica, salvo diversa configurazione con `--api-key`).

**Esempio con `curl`:**

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "user", "content": "Spiega in breve cos'\''è un GGUF."}
        ],
        "temperature": 0.7
    }'
```

Si noti che il campo `model` è facoltativo: il server serve l'unico modello
caricato all'avvio, per cui il valore eventualmente indicato viene ignorato.

**Esempio con la libreria `openai` di Python:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed",  # llama-server non richiede autenticazione
)

resp = client.chat.completions.create(
    model="local",  # ignorato: il server serve l'unico modello caricato
    messages=[
        {"role": "user", "content": "Spiega in breve cos'è un GGUF."}
    ],
    temperature=0.7,
)

print(resp.choices[0].message.content)
```

L'endpoint supporta lo streaming (`"stream": true`) e i principali parametri di
campionamento (`temperature`, `top_p`, `max_tokens`, `stop`).

### 1.5 Esecuzione one-shot con `llama-cli`

Per una singola generazione da riga di comando, senza avviare il server, si può
invocare `llama-cli` sovrascrivendo la `%runscript` con `singularity exec`:

```bash
singularity exec --nv \
    -B $HOME/.cache/huggingface:/hf \
    llamacpp-cuda.sif \
    llama-cli \
    -m /hf/hub/models--<org>--<repo>/snapshots/<hash>/mio-modello.gguf \
    -ngl 99 -c 4096 \
    -p "Scrivi un haiku sul mare."
```

`singularity exec ... llama-cli` scavalca la `%runscript` (che punta a
`llama-server`) ed esegue direttamente il binario indicato. È utile per test
rapidi, benchmark o generazioni batch da script; per l'uso interattivo o
multi-client è invece preferibile il server (§ 1.1).

---

## Parte 2 — Utilizzo su IBiSCo

Su IBiSCo non esiste un `.sif` con `%runscript`: i binari sono compilati su
Lustre e vengono eseguiti tramite `singularity exec` sulla sandbox `cu122-sm70/`,
impostando `LD_LIBRARY_PATH` sulla directory dei binari (che contiene anche le
librerie condivise `.so`). Valgono inoltre i vincoli di rete del cluster: i nodi
di calcolo sono isolati e il servizio va esposto su `0.0.0.0` per essere
raggiungibile.

### 2.1 Allocazione di un nodo di calcolo

L'inferenza va eseguita su un nodo dotato di GPU, non sul login node. Si alloca
un nodo interattivo tramite SLURM (adattare partizione, numero di GPU e durata
alle proprie esigenze):

```bash
salloc --partition=<gpu_partition> --gres=gpu:1 --time=02:00:00
```

Ottenuta l'allocazione, si annoti il nome del nodo assegnato (`hostname`): sarà
necessario per raggiungere il servizio da altri nodi o via tunnel SSH (§ 2.4).

### 2.2 Avvio del servizio

L'avvio ricalca la sezione § 2.4 della guida di installazione:

```bash
LUSTRE=/lustre/home/$USER
BIN=$LUSTRE/llama.cpp-master/build/bin

singularity exec --nv --bind /lustre:/lustre \
    --env LD_LIBRARY_PATH=$BIN \
    cu122-sm70/ \
    $BIN/llama-server \
    -m $LUSTRE/models/mio-modello.gguf \
    --device CUDA0 \
    --host 0.0.0.0 --port 8081 \
    -ngl 99 -c 8192 --jinja
```

- `--env LD_LIBRARY_PATH=$BIN` — necessario per localizzare a runtime le librerie
  condivise di `llama.cpp`, residenti su Lustre.
- `--device CUDA0` — selezione della GPU tramite l'opzione nativa di `llama.cpp`.
  Non si utilizzi `CUDA_VISIBLE_DEVICES` (ha determinato fallimenti in fase di
  compilazione JIT del PTX in questa configurazione).
- `--host 0.0.0.0` — rende il servizio raggiungibile da altri nodi (rete
  InfiniBand), diversamente dal loopback impiegato in locale.
- Il modello è indicato con percorso esplicito su Lustre (`-m`), poiché la build
  di IBiSCo è priva del backend CURL (`LLAMA_CURL=OFF`).

Lo stato del servizio si verifica come in locale, sostituendo host e porta con
quelli del nodo:

```bash
curl http://<nodo>:8081/health
```

### 2.3 Serving multi-modello

Come descritto nella § 2.5 della guida di installazione, il medesimo schema
avviato più volte su porte distinte sostiene uno stack a più modelli. Ciascuna
istanza va lanciata in una propria sessione persistente (`tmux`/`screen`) o in
background:

| Modello                              | Porta | GPU              |
|--------------------------------------|-------|------------------|
| `gpt-oss-20b-mxfp4.gguf`             | 8081  | `CUDA0`          |
| `Ministral-3-8B-Instruct-2512-Q8_0`  | 8082  | `CUDA0`          |
| `Qwen3.6-27B-Q8_0.gguf`              | 8083  | `CUDA1,CUDA3`    |

Per i modelli che non risiedono in una singola V100 (32 GB) si adotta la
suddivisione multi-GPU con `--device CUDA1,CUDA3` e una finestra di contesto
adeguata (ad esempio `-c 32768`).

Ogni istanza espone un proprio endpoint OpenAI-compatibile sulla rispettiva
porta (`http://<nodo>:8081/v1`, `:8082/v1`, `:8083/v1`), interrogabile con gli
stessi client della § 1.4.

### 2.4 Accesso al servizio da remoto (tunnel SSH)

Poiché il servizio ascolta su un nodo di calcolo interno al cluster, per
raggiungerlo dalla propria macchina si stabilisce un **tunnel SSH** attraverso
il login node, che inoltra una porta locale verso `<nodo>:<porta>` del cluster:

```bash
ssh -L 8081:<nodo>:8081 <utente>@<login-node-ibisco>
```

- `-L 8081:<nodo>:8081` — mappa la porta locale `8081` sulla porta `8081` del
  nodo di calcolo `<nodo>`, instradando il traffico attraverso il login node.
- `<nodo>` — nome del nodo di calcolo assegnato da `salloc` (§ 2.1).

A tunnel attivo, il servizio remoto è utilizzabile dalla propria macchina come
se fosse locale: l'interfaccia web è su `http://127.0.0.1:8081` e l'endpoint API
su `http://127.0.0.1:8081/v1`, interrogabili con i medesimi client della § 1.4.

> **Nota.** Nel caso multi-modello (§ 2.3) si aprono più inoltri nella stessa
> connessione, concatenando le opzioni `-L`:
> `ssh -L 8081:<nodo>:8081 -L 8082:<nodo>:8082 -L 8083:<nodo>:8083 <utente>@<login-node>`.

### 2.5 Esecuzione one-shot con `llama-cli`

Analogamente alla § 1.5, per una singola generazione senza avviare il server si
invoca `llama-cli` dalla directory dei binari:

```bash
LUSTRE=/lustre/home/$USER
BIN=$LUSTRE/llama.cpp-master/build/bin

singularity exec --nv --bind /lustre:/lustre \
    --env LD_LIBRARY_PATH=$BIN \
    cu122-sm70/ \
    $BIN/llama-cli \
    -m $LUSTRE/models/mio-modello.gguf \
    --device CUDA0 \
    -ngl 99 -c 8192 \
    -p "Scrivi un haiku sul mare."
```

---

## Parametri di runtime ricorrenti

I parametri seguenti si applicano sia a `llama-server` sia a `llama-cli`, in
entrambi gli ambienti.

| Opzione            | Significato                                                        |
|--------------------|-------------------------------------------------------------------|
| `-m <file.gguf>`   | Percorso del modello GGUF da caricare.                            |
| `-ngl <n>`         | Numero di layer in offload su GPU (`99` = tutti; ridurre se la VRAM è insufficiente). |
| `-c <n>`           | Dimensione della finestra di contesto (in token).                 |
| `--jinja`          | Abilita il template di chat nativo del modello (ruoli, reasoning, tool-calling). |
| `--host <addr>`    | Indirizzo di ascolto (`127.0.0.1` in locale, `0.0.0.0` su IBiSCo).|
| `--port <n>`       | Porta di ascolto del server.                                      |
| `--device CUDAn`   | Selezione esplicita della/e GPU (su IBiSCo; mai `CUDA_VISIBLE_DEVICES`). |
| `-np <n>`          | Numero di richieste servite in parallelo (slot concorrenti).      |
| `--api-key <k>`    | Richiede l'autenticazione delle richieste con la chiave indicata. |

---

## Prospetto riepilogativo delle differenze

| Aspetto              | Locale (Parte 1)                    | IBiSCo (Parte 2)                                  |
|----------------------|-------------------------------------|---------------------------------------------------|
| Comando di avvio     | `singularity run` (via `%runscript`)| `singularity exec` sul binario in Lustre          |
| Librerie a runtime   | dentro l'immagine (`PATH`)          | su Lustre (`LD_LIBRARY_PATH=$BIN`)                |
| Sorgente del modello | cache HF montata (`-B ...:/hf`)     | percorso esplicito su Lustre (`-m`)               |
| Selezione GPU        | non necessaria                      | `--device CUDAn`                                   |
| Host binding         | `127.0.0.1` (loopback)              | `0.0.0.0` (multi-nodo)                            |
| Accesso da remoto    | non necessario                      | tunnel SSH attraverso il login node               |
| Nodo di esecuzione   | macchina locale                     | nodo di calcolo allocato con `salloc`             |

In entrambi gli ambienti l'interfaccia web e l'endpoint OpenAI-compatibile
(`/v1`) restano identici: cambia soltanto la modalità di avvio del servizio e di
accesso alla porta.
