# Download modelli da Hugging Face su IBiSCo

Questa guida descrive la procedura per scaricare modelli da Hugging Face sul
cluster **IBiSCo**, propedeutica al serving tramite `llama.cpp` (si veda
[llama.cpp_local_img.md](llama.cpp_local_img.md)).

È già stato predisposto sul cluster un ambiente conda dedicato, `hf_download`,
configurato con una API key di Hugging Face che consente il download.

> **Titolare della API key.** La API key associata all'ambiente `hf_download`
> è di proprietà di **Cristian Mascia** (`cristian.mascia@unina.it`). Alcuni
> repository su Hugging Face sono *gated*, ovvero richiedono l'approvazione
> esplicita del titolare del repository prima di poter essere scaricati (tipicamente
> tramite l'accettazione di una licenza sulla pagina del modello). Qualora il
> download fallisca per questo motivo, si contatti Cristian Mascia per verificare
> lo stato della richiesta di accesso.

---

## 1. Accesso al login node

Il download va effettuato dal **login node**, l'unico dotato di accesso alla
rete esterna (i nodi di calcolo sono isolati):

## 2. Attivazione di conda

Su IBiSCo conda non è disponibile di default nella shell: va caricato ad ogni
nuova sessione (login node o nodo di calcolo) tramite lo script di setup di
sistema:

```bash
source /nfsexports/SOFTWARE/anaconda3.OK/setupconda.sh
```

## 3. Attivazione dell'ambiente `hf_download`

```bash
conda activate hf_download
```

L'ambiente include il pacchetto `huggingface_hub` (comando `hf`) già
autenticato tramite la API key di cui sopra. Per verificare che l'autenticazione
sia attiva:

```bash
hf auth whoami
```

## 4. Cache Hugging Face

I comandi di download di questa guida **non** specificano `--local-dir`: in
sua assenza `hf download` salva i file nella cache standard di Hugging Face
(`$HF_HOME/hub`, o `~/.cache/huggingface/hub` se `HF_HOME` non è impostata),
nella struttura `models--<org>--<repo>/snapshots/<hash>/...`. È il percorso
che [llama.cpp_local_img.md](llama.cpp_local_img.md#14-serving-di-un-modello)
si aspetta per il bind mount in fase di serving (`-B $HOME/.cache/huggingface:/hf`).

Poiché la home su IBiSCo ha quota limitata, si consiglia di reindirizzare la
cache su `/ibiscostorage`, impostando `HF_HOME` **prima** del download (ad
esempio nel proprio `~/.bashrc`, o in sessione):

```bash
export HF_HOME=/ibiscostorage/$USER/hf_cache
```

Con questa variabile impostata, i comandi `hf download` della sezione 5
scaricano automaticamente sotto `/ibiscostorage/$USER/hf_cache/hub`, senza
bisogno di indicare un percorso esplicito.

## 5. Formato del modello: GGUF per gli LLM, safetensors per gli embedder

Poiché il serving avviene tramite `llama.cpp`, gli **LLM** vanno scaricati in
formato **GGUF**. Come fonte si consiglia l'organizzazione **[ggml-org](https://huggingface.co/ggml-org)**
su Hugging Face, che pubblica conversioni GGUF mantenute e aggiornate dei
principali modelli.

Gli **embedder**, invece, non passano da `llama.cpp`: vengono impiegati
direttamente con `sentence-transformers` o con la libreria `transformers` di
Hugging Face, per cui vanno scaricati nel loro formato nativo **safetensors**
(nessuna conversione GGUF necessaria).

## 6. Download di un modello con `hf download`

### 6.1 LLM in formato GGUF (quantizzato)

Una repo GGUF su `ggml-org` contiene tipicamente **più quantizzazioni** dello
stesso modello (`Q4_K_M`, `Q5_K_M`, `Q8_0`, ecc.). Si sconsiglia di scaricare
l'intera repository: tramite il parametro `--include` con un pattern glob si
seleziona il solo file relativo alla quantizzazione desiderata.

```bash
hf download ggml-org/Qwen3-4B-GGUF --include "*Q4_K_M*"
```

- `hf download` — comando di `huggingface_hub` per il download da un repository.
- `ggml-org/Qwen3-4B-GGUF` — identificativo `<org>/<repo>` del modello.
- `--include "*Q4_K_M*"` — filtro glob che limita il download ai soli file il cui
  nome contiene la quantizzazione indicata, evitando di scaricare le altre
  versioni presenti nella repo.

Non specificando `--local-dir`, il file scaricato finisce nella cache HF
(sezione 4), pronto per essere referenziato da `llama.cpp` tramite il percorso
`.../snapshots/<hash>/<file>.gguf` oppure tramite `-hf ggml-org/Qwen3-4B-GGUF:Q4_K_M`.

Per un modello non quantizzato (precisione originale, ad esempio `F16`/`BF16`),
si applica lo stesso schema adattando il pattern:

```bash
hf download ggml-org/Qwen3-4B-GGUF --include "*F16*"
```

> **Suggerimento.** Prima del download, i file disponibili in una repo GGUF
> possono essere consultati direttamente sulla pagina Hugging Face del modello
> (scheda "Files"), per individuare il pattern esatto della quantizzazione
> desiderata.

### 6.2 Embedder in formato safetensors

Per un embedder, non essendo necessaria alcuna quantizzazione GGUF, si scarica
l'intera repository nel formato nativo:

```bash
hf download <org>/<repo-embedder>
```

Il modello così scaricato risiede nella cache HF e può essere caricato con
`sentence-transformers` o `transformers` passando direttamente l'identificativo
`<org>/<repo-embedder>` (le librerie risolvono automaticamente la cache locale
tramite `HF_HOME`), senza necessità di indicarne il percorso esplicito.

---

## Riepilogo

| Tipo di modello | Formato    | Fonte consigliata | Selezione file            |
|------------------|-----------|--------------------|-----------------------------|
| LLM (serving `llama.cpp`) | GGUF | `ggml-org` | `--include "*<QUANT>*"` (una sola quantizzazione) |
| Embedder (`sentence-transformers`/`transformers`) | safetensors | repo ufficiale del modello | repo completa |

In caso di errore di accesso su repository gated, contattare **Cristian Mascia**
(`cristian.mascia@unina.it`), titolare della API key configurata nell'ambiente
`hf_download`.
