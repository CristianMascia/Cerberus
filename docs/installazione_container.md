# llama.cpp su Singularity — guida al build dell'immagine

Questa guida descrive la procedura per ottenere un'immagine Singularity con
`llama.cpp` compilato con backend CUDA, finalizzata al **serving di un modello
GGUF** indicato come parametro a runtime.

Il modello non è mai incorporato nell'immagine: viene montato dall'esterno
tramite un bind, cosicché la medesima immagine possa servire qualunque file
GGUF.

La guida è articolata in due parti:

- **Parte 1 — Build in locale**, consigliata per la fase di sviluppo. In presenza
  dei privilegi di root (o dell'opzione `--fakeroot`) l'immagine `.sif` viene
  costruita in modo pulito a partire da un file di definizione `.def`.
- **Parte 2 — Build su IBiSCo**, dove l'ambiente è soggetto a vincoli specifici
  (assenza di privilegi di root e di `--fakeroot`, versione del driver che limita
  CUDA, toolchain di sistema non aggiornata). In tale contesto il file `.def` non
  è utilizzabile e si ricorre a una **sandbox** compilando da sorgente con la
  sandbox in sola lettura.

---

## Parte 1 — Build in locale (consigliata per lo sviluppo)

### 1.1 Determinazione dell'architettura CUDA della GPU

`llama.cpp` deve essere compilato per la *compute capability* della scheda in
uso. In assenza di tale valore, il codice ricade a runtime sulla compilazione
JIT del PTX, con conseguente penalizzazione delle prestazioni e, su determinate
architetture, con l'impossibilità di utilizzare alcuni kernel nativi. Il valore
richiesto è privo del punto decimale: a `8.9` corrisponde `89`.

Lo si ricava direttamente da riga di comando:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d .
```

Scomposizione del comando:

- `nvidia-smi` — interroga il driver NVIDIA.
- `--query-gpu=compute_cap` — richiede il solo campo *compute capability*.
- `--format=csv,noheader` — restituisce un output privo di intestazione
  (ad esempio `8.9`).
- `| tr -d .` — rimuove il punto decimale, producendo il valore pronto per CMake
  (`89`).

In presenza di più GPU, il comando restituisce una riga per ciascuna scheda:
si utilizzi il valore relativo alla GPU destinata all'inferenza.

> **Nota.** Il campo `compute_cap` è disponibile con driver di versione ≥ 495.
> Qualora il comando restituisca un errore, si ricavi il modello della scheda con
> `nvidia-smi --query-gpu=name --format=csv,noheader` e si effettui la conversione
> manualmente (Pascal `61`, Turing `75`, Ampere `86`, Ada `89`, Blackwell `120`).

### 1.2 File di definizione `llamacpp-local.def`

Il template di questo file e' presente nella cartella `/containers/`.

```singularity
Bootstrap: docker
From: nvidia/cuda:12.4.1-devel-ubuntu22.04

%post
    # <<< impostare con il valore determinato al passo 1.1 >>>
    ARCH=89

    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ccache \
        libcurl4-openssl-dev ca-certificates \
        && rm -rf /var/lib/apt/lists/*

    cd /opt
    git clone --depth 1 --single-branch https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp

    cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${ARCH} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"

    cmake --build build --config Release -j "$(nproc)"
    cmake --install build --prefix /usr/local
    cp -n build/bin/* /usr/local/bin/ 2>/dev/null || true

%environment
    export PATH=/usr/local/bin:$PATH
    export LC_ALL=C

%runscript
    exec llama-server "$@"
```

### 1.3 Struttura del comando di build

```bash
singularity build --fakeroot llamacpp-cuda.sif llamacpp-local.def
```

- `singularity build` — comando che assembla l'immagine.
- `--fakeroot` — consente l'esecuzione della sezione `%post` (installazione dei
  pacchetti, compilazione) senza privilegi di root effettivi, tramite gli user
  namespace. In alternativa, disponendo dei privilegi:
  `sudo singularity build llamacpp-cuda.sif llamacpp-local.def`.
- `llamacpp-cuda.sif` — **output**: l'immagine immutabile risultante.
- `llamacpp-local.def` — **input**: la ricetta elaborata da Singularity.

Operazioni svolte nella sezione `%post` (fase centrale del build):

- `From: nvidia/cuda:12.4.1-devel-ubuntu22.04` — l'immagine di partenza include
  il toolkit CUDA completo (`nvcc`) e una versione recente di gcc. Il tag deve
  essere allineato alla versione supportata dal driver: si verifichi il valore
  "CUDA Version" riportato in alto a destra nell'output di `nvidia-smi` e, qualora
  inferiore a 12.4, si adotti un tag più basso (ad esempio `12.2.2-devel`).
- `git clone ... llama.cpp` — acquisizione dei sorgenti aggiornati, requisito
  necessario poiché i file GGUF recenti richiedono build correnti.
- `cmake -B build ...` — configurazione della build. Opzioni principali:
  - `-DGGML_CUDA=ON` — abilita il backend CUDA (l'opzione `LLAMA_CUBLAS` è
    deprecata).
  - `-DCMAKE_CUDA_ARCHITECTURES=${ARCH}` — genera codice nativo per la GPU
    indicata, eliminando la compilazione JIT.
  - `-DLLAMA_CURL=ON` — abilita il caricamento di modelli anche tramite URL/HF.
  - `-DCMAKE_BUILD_TYPE=Release` — build ottimizzata.
- `cmake --build ... -j "$(nproc)"` — compilazione su tutti i core disponibili.
- `cmake --install` e la successiva copia — collocano gli eseguibili
  (`llama-server`, `llama-cli`, ecc.) nel `PATH`.

La sezione `%runscript` fa sì che il comando
`singularity run llamacpp-cuda.sif <argomenti>` inoltri gli argomenti
direttamente a `llama-server`.

### 1.4 Serving di un modello

```bash
singularity run --nv \
    -B $HOME/.cache/huggingface:/hf \
    llamacpp-cuda.sif \
    -m /hf/hub/models--<org>--<repo>/snapshots/<hash>/mio-modello.gguf \
    --host 127.0.0.1 --port 8080 \
    -ngl 99 -c 4096
```

- `--nv` — espone il driver e le librerie GPU dell'host all'interno del container
  (opzione indispensabile).
- `-B $HOME/.cache/huggingface:/hf` — monta la cache di Hugging Face (bind), in cui
  risiedono i modelli scaricati con `hf download`. All'interno del container la
  cache è accessibile sotto `/hf`.
- `-m` — specifica il file GGUF da servire, indicandone il percorso all'interno
  della cache. I modelli scaricati seguono la struttura
  `/hf/hub/models--<org>--<repo>/snapshots/<hash>/<file>.gguf`; i segnaposto
  `<org>`, `<repo>` e `<hash>` vanno sostituiti con i valori effettivi.
- `--host 127.0.0.1` — in ambiente locale il servizio è in ascolto sul solo
  loopback, senza esposizione in rete.
- `-ngl 99` — effettua l'offload di tutti i layer in VRAM; il valore va ridotto
  qualora il modello non risieda interamente in memoria.
- `-c 4096` — dimensione della finestra di contesto.

Al termine, l'interfaccia web è raggiungibile all'indirizzo
`http://127.0.0.1:8080`; l'endpoint compatibile con l'API OpenAI è disponibile su
`http://127.0.0.1:8080/v1`.

Per ricavare rapidamente il percorso esatto dello snapshot (evitando di comporre a mano `<org>`, `<repo>` e `<hash>`), è utile:

```bash
find $HOME/.cache/huggingface/hub -name '*.gguf'
```

> **Nota.** In alternativa al percorso esplicito dello snapshot, `llama-server`
> può risolvere il modello direttamente dalla cache tramite l'opzione
> `-hf <org>/<repo>:<quant>` (ad esempio `-hf ggml-org/gpt-oss-20b-GGUF`), a
> condizione che il backend CURL sia abilitato (`-DLLAMA_CURL=ON`, come nella
> build locale). In ambiente offline è comunque preferibile il percorso esplicito.
---

## Parte 2 — Build su IBiSCo

> ## 📌 Nota — Build già effettuata
>
> La procedura di build descritta nella presente Parte 2 è già stata completata
> su IBiSCo: la sandbox CUDA `tools/cu122-sm70/` è disponibile e operativa, e i binari
> di `llama.cpp` risultano già compilati per l'architettura sm_70.
>
> Non è pertanto necessario rieseguire i passi di compilazione (§ 2.3). Per
> l'utilizzo immediato si può procedere direttamente alla sezione
> [§ 2.4 — Serving di un modello](#24-serving-di-un-modello) o, per lo stack
> multi-modello, alla sezione
> [§ 2.5 — Serving multi-modello](#25-serving-multi-modello).
>
> I passaggi di build sono mantenuti nella documentazione a titolo di riferimento,
> per l'eventualità in cui la sandbox debba essere ricreata da zero.

### 2.1 Motivazioni dell'impossibilità di utilizzare il file `.def`

L'approccio della Parte 1 non è applicabile sul cluster a causa di tre vincoli
concomitanti:

1. **Assenza di privilegi di root e dell'opzione `--fakeroot`.** Il comando
   `singularity build immagine.sif file.def` richiede privilegi per eseguire la
   sezione `%post` e assemblare l'immagine squashfs immutabile. Tali privilegi
   non sono disponibili su IBiSCo, il che preclude la costruzione di un `.sif` a
   partire da un `.def`.
2. **Toolchain di sistema non aggiornata.** I nodi sono basati su CentOS 7
   (gcc 4.8.5, glibc 2.17), configurazione non sufficiente a compilare
   `llama.cpp`, che richiede un compilatore conforme a C++17. La compilazione
   direttamente sul login node fallisce per questo motivo; è necessario il
   compilatore aggiornato incluso in un'immagine CUDA basata su Ubuntu 22.04.
3. **Incompatibilità delle immagini precompilate con le GPU V100.** L'immagine
   ufficiale `llama.cpp:server-cuda` fallisce su architettura sm_70 con errori
   in fase di compilazione JIT del PTX (kernel `rms_norm_fused` / `can_use_pdl`,
   supportati nativamente soltanto su architettura Hopper, sm_90). Inoltre il
   driver del cluster (`535.104.05`) supporta CUDA fino alla versione 12.2,
   rendendo incompatibili a runtime le immagini basate su versioni superiori.

Ne consegue la necessità di **compilare `llama.cpp` da sorgente per l'architettura
sm_70**, in un ambiente dotato di CUDA 12.2 e di un compilatore aggiornato, in
assenza di privilegi di root.

### 2.2 Ruolo della sandbox e vincolo dei permessi

La sandbox costituisce, di fatto, l'unica soluzione compatibile con i vincoli
sopra esposti. Le ragioni sono le seguenti:

- Un `.sif` è **immutabile**: non consente l'installazione di pacchetti al proprio
  interno né la compilazione, e non può essere costruito da un `.def` in assenza
  di privilegi di root o dell'opzione `--fakeroot`.
- Una **sandbox** è invece una *directory scrivibile*, creabile a partire da
  un'immagine Docker **senza privilegi di root** (`singularity build --sandbox`,
  che sfrutta gli user namespace).

È tuttavia essenziale una precisazione sui permessi, poiché determina l'intera
procedura di build. L'opzione `--writable`, che consentirebbe di modificare la
sandbox (ad esempio installando pacchetti con `apt-get`), **richiede
`--fakeroot`**: `apt-get` opera come root, e senza fakeroot l'utente all'interno
del container è mappato al proprio UID non privilegiato, per cui l'installazione
fallisce. Poiché su IBiSCo `--fakeroot` non è disponibile, **la sandbox può essere
creata e utilizzata, ma non modificata tramite gestore di pacchetti.**

Ciò non costituisce un ostacolo, in virtù di una circostanza favorevole:
l'immagine `nvidia/cuda:12.2.2-devel` include **già** l'intero toolchain di
compilazione (`gcc`, `g++`, `make`) e CUDA (`nvcc`). Mancano soltanto `cmake` e
`git`, che si procurano come binario statico e tarball **su Lustre**, senza root.
La compilazione avviene pertanto con la **sandbox in sola lettura**, tramite
`singularity exec --bind /lustre:/lustre` (senza `--writable`), scrivendo tutti
gli artefatti di build su Lustre. È questa la modalità adottata su IBiSCo.

Essendo un ambiente CUDA generico, la sandbox `cu122-sm70/` è inoltre
riutilizzabile in altri progetti che richiedano la medesima configurazione.

### 2.3 Procedura di build

Tutte le operazioni che richiedono connettività di rete (download di `cmake` e
dei sorgenti) devono essere eseguite sul **login node**, poiché i nodi di calcolo
isolati non dispongono di accesso alla rete. Qualora il login node presenti
limitazioni di memoria o di thread durante la compilazione, si consiglia di
allocare un nodo tramite `salloc` ed eseguire la compilazione al suo interno.

**1. Creazione della sandbox base con CUDA 12.2** (compatibile con il driver 535;
non richiede privilegi):

```bash
cd /lustre/home/$USER
singularity build --sandbox cu122-sm70/ docker://nvidia/cuda:12.2.2-devel-ubuntu22.04
```

**2. Verifica del toolchain già presente nella sandbox** (sola lettura):

```bash
singularity exec cu122-sm70/ bash -c 'which gcc g++ make nvcc'
```

Output atteso: `/usr/bin/gcc`, `/usr/bin/g++`, `/usr/bin/make`,
`/usr/local/cuda/bin/nvcc`. Confermato ciò, mancano soltanto `cmake` e `git`.

**3. Acquisizione di `cmake` e dei sorgenti su Lustre** (login node, senza root):

```bash
cd /lustre/home/$USER

# cmake: binario statico precompilato, nessuna installazione
wget https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.tar.gz
tar -xzf cmake-3.30.5-linux-x86_64.tar.gz

# sorgenti di llama.cpp: tarball (git non necessario)
wget https://github.com/ggml-org/llama.cpp/archive/refs/heads/master.tar.gz -O llamacpp.tar.gz
tar -xzf llamacpp.tar.gz
```

**4. Compilazione per l'architettura sm_70** (sandbox in sola lettura, output su
Lustre):

```bash
LUSTRE=/lustre/home/$USER
CMAKE=$LUSTRE/cmake-3.30.5-linux-x86_64/bin/cmake
SRC=$LUSTRE/llama.cpp-master

singularity exec --bind /lustre:/lustre cu122-sm70/ bash -c "
  $CMAKE -S $SRC -B $SRC/build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs' \
    -DCMAKE_SHARED_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs' &&
  $CMAKE --build $SRC/build --config Release -j 2
"
```

Considerazioni specifiche per IBiSCo:

- **Nessun `--writable`.** La compilazione avviene con la sandbox in sola lettura,
  impiegando il `nvcc`/`g++` già presenti nell'immagine e scrivendo l'output su
  Lustre. È questo l'elemento che aggira il vincolo dei permessi (assenza di
  fakeroot).
- `CMAKE_CUDA_ARCHITECTURES=70` — genera codice nativo per le GPU V100. È
  l'opzione che elimina i fallimenti in fase di compilazione JIT del PTX.
- `LLAMA_CURL=OFF` — in assenza di `apt-get` non sono disponibili gli header di
  `libcurl`; poiché i modelli vengono caricati da percorsi locali su Lustre
  (`-m`), la funzione di download tramite URL non è necessaria e viene disattivata.
- Gli **stub del linker** (`lib64/stubs`) sono necessari poiché in fase di build
  il driver non è presente; in loro assenza il collegamento fallisce.
- `-j 2` — livello di parallelismo volutamente contenuto per non saturare il
  login node (la compilazione richiede circa 15–20 minuti). Su un nodo allocato
  il valore può essere incrementato.

Al termine, gli eseguibili risiedono in `$SRC/build/bin/` su Lustre.

### 2.4 Serving di un modello

Gli eseguibili compilati risiedono **su Lustre**, in
`.../llama.cpp-master/build/bin/`. Poiché nella medesima directory si trovano
anche le librerie condivise (`.so`), è necessario impostare `LD_LIBRARY_PATH`. Il
servizio si avvia mediante `exec`, esponendo la GPU e montando Lustre:

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

Dettagli specifici per IBiSCo:

- `--env LD_LIBRARY_PATH=$BIN` — necessario affinché le librerie condivise di
  `llama.cpp`, residenti su Lustre, siano correttamente localizzate a runtime.
- `--device CUDA0` — selezione della GPU tramite l'opzione nativa di `llama.cpp`.
  Non si utilizzi `CUDA_VISIBLE_DEVICES`, che in questa configurazione ha
  determinato fallimenti in fase di compilazione JIT del PTX.
- `--host 0.0.0.0` — necessario per rendere il servizio raggiungibile da altri
  nodi (rete InfiniBand), diversamente dal loopback impiegato in locale.

### 2.5 Serving multi-modello

Il medesimo schema, avviato tre volte su porte distinte, sostiene lo stack a tre
modelli (i percorsi e l'assegnazione delle GPU vanno adattati alla disponibilità
del nodo):

| Modello                              | Porta | GPU              |
|--------------------------------------|-------|------------------|
| `gpt-oss-20b-mxfp4.gguf`             | 8081  | `CUDA0`          |
| `Ministral-3-8B-Instruct-2512-Q8_0`  | 8082  | `CUDA0`          |
| `Qwen3.6-27B-Q8_0.gguf`              | 8083  | `CUDA1,CUDA3`    |

Per i modelli di dimensioni tali da non risiedere in una singola V100 (32 GB) si
adotta la suddivisione multi-GPU con `--device CUDA1,CUDA3` e una finestra di
contesto adeguata (ad esempio `-c 32768`).

---

## Prospetto riepilogativo delle differenze

| Aspetto            | Locale (Parte 1)          | IBiSCo (Parte 2)                              |
|--------------------|---------------------------|-----------------------------------------------|
| Metodo             | `.sif` da `.def`          | sandbox in sola lettura, build su Lustre      |
| Privilegi          | root / `--fakeroot`       | nessun privilegio (né root né fakeroot)       |
| Modifica container | consentita (`%post`)      | non consentita (no `apt-get`, no `--writable`)|
| cmake / sorgenti   | installati in `%post`     | binario statico e tarball su Lustre           |
| Versione CUDA base | allineata al driver       | `12.2.2-devel` (driver 535)                   |
| Architettura       | `compute_cap` della GPU   | `70` (V100 / sm_70)                           |
| Backend CURL       | `ON`                      | `OFF` (nessun header libcurl; modelli locali) |
| Sede dei binari    | dentro l'immagine (`PATH`)| su Lustre (`LD_LIBRARY_PATH` a runtime)       |
| Selezione GPU      | non necessaria            | `--device CUDAn` (mai `CUDA_VISIBLE_DEVICES`) |
| Host binding       | `127.0.0.1`               | `0.0.0.0` (multi-nodo)                        |
| Stub del linker    | non necessari             | necessari (driver assente in fase di build)   |