# Cerberus — batch API

`cerberus batch` esegue un file JSONL di richieste di chat sui modelli di un
`models.conf`, con **ripetizioni** e **split in più job paralleli**. Ogni job
(shard) è un task di un job array SLURM e avvia **i propri server llama.cpp**:
nessun server da tenere acceso a mano, nessun secondo file di configurazione.

```bash
cerberus download -c models.conf
cerberus batch submit -c models.conf -i domande.jsonl --shards 4 --repeats 3
cerberus batch status --batch-dir batch_20261027_1530
cerberus batch merge  --batch-dir batch_20261027_1530
```

---

## Formato dell'input

Una richiesta per riga. `custom_id` e `messages` sono obbligatori.

```json
{"custom_id": "q-0001", "model": "qwen-big", "messages": [{"role": "user", "content": "Ciao"}], "params": {"temperature": 0.2, "max_tokens": 512, "seed": 7}}
```

- `custom_id`: stringa unica nel file.
- `model`: label di `models.conf`. Se manca si usa `--model`.
- `params`: parametri della singola richiesta (vedi sotto).

È accettato anche il **formato OpenAI Batch**: un file preparato per OpenAI gira
senza modifiche (`max_completion_tokens` diventa `max_tokens`).

```json
{"custom_id": "q-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "qwen-big", "messages": [...], "temperature": 0.2}}
```

### Parametri ammessi

| Parametro | Dove finisce | Note |
|-----------|--------------|------|
| `temperature`, `top_p`, `max_tokens`, `seed`, `stop`, `presence_penalty`, `frequency_penalty` | argomenti OpenAI | `max_tokens` non può superare `max_output_tokens` del modello; se manca vale `max_output_tokens` |
| `top_k`, `min_p`, `repeat_penalty` | `extra_body` | campi specifici di llama.cpp |
| `reasoning` (bool) | `chat_template_kwargs.enable_thinking` | come `CerberusClient.chat` |
| `reasoning_effort` | `chat_template_kwargs.reasoning_effort` | per gpt-oss |

Un parametro non in elenco è un errore di validazione. Se una richiesta ha `seed`,
la ripetizione `k` usa `seed + k`. Senza `seed` le ripetizioni sono campioni
indipendenti.

---

## La sezione `[batch]` di models.conf

Opzionale. Dà i default dei comandi; i flag della CLI vincono.

```toml
[batch]
shards          = 4            # job paralleli, ognuno con i propri server
repeats         = 1            # ripetizioni per richiesta
partition       = "gpus"
# account       = "mio_progetto"
time            = "04:00:00"   # time limit di ogni task
cpus_per_task   = 8
# max_parallel  = 2            # task in esecuzione contemporanea (--array=0-3%2)
max_retries     = 3            # retry per richiesta su errori transitori
request_timeout = 600          # secondi per richiesta
```

Dei `[[model]]` il batch avvia **solo quelli citati nel JSONL**. Il campo `parallel`
di un modello è anche la concorrenza massima verso il suo server.
`max_input_tokens` viene controllato con una stima per eccesso (caratteri / 3):
le righe a rischio danno un avviso, e se il contesto viene davvero superato
l'errore finisce nella riga del risultato.

---

## Comandi

| Comando | Dove | Cosa fa |
|---------|------|---------|
| `cerberus batch validate -i in.jsonl [--model L] [--schema-only]` | login node | Controlla JSONL, label, `custom_id`, parametri e limiti; senza `--schema-only` stima anche GPU e nodi per shard (serve rete) |
| `cerberus batch submit -i in.jsonl [--model L] [--shards N] [--repeats R] [--batch-dir D] [--dry-run]` | login node | Valida, calcola il placement **una volta**, scrive la cartella del batch e lancia `sbatch --array=0-(N-1)`; con `--dry-run` si ferma prima di `sbatch` |
| `cerberus batch run --batch-dir D [--shard I] [--endpoints E] [--retry-failed]` | dentro l'allocazione | Il lavoro di un task: avvia i server, elabora lo shard, scrive i risultati, spegne i server. `--shard` di default è `SLURM_ARRAY_TASK_ID` |
| `cerberus batch status --batch-dir D` | ovunque | Per shard: unità fatte / totali / errori e stato SLURM (`sacct`/`squeue`) |
| `cerberus batch merge --batch-dir D [-o out.jsonl]` | ovunque | Unisce gli shard in `results.jsonl` nell'ordine dell'input; esce con 1 se mancano unità |

`run --endpoints endpoints.json` salta l'avvio dei server e usa quelli già attivi
(per esempio quelli di un `cerberus up` in corso): utile per i test e senza SLURM.

### Cartella del batch

```
batch_<timestamp>/
  manifest.json            impostazioni, placement, id del job SLURM
  requests.jsonl           copia normalizzata dell'input
  shards/shard_000.jsonl   le unità (custom_id, repeat) di ogni shard
  results/shard_000.jsonl  una riga per unità completata
  logs/                    output SLURM dei task
  job.sbatch               il job array
  results.jsonl            scritto da merge
  .cerberus/<job>/         log e plan dei server di ogni task
```

---

## Ripetizioni e split

L'unità di lavoro è la coppia `(custom_id, repeat)`. Le ripetizioni si espandono
**prima** dello split, così il carico è bilanciato anche con poche richieste e
molte ripetizioni. Le unità sono assegnate agli shard in round-robin: lo split è
deterministico e le ripetizioni della stessa richiesta finiscono in shard diversi.
Il numero di shard è limitato al numero di unità.

Ogni shard paga l'avvio dei server (caricamento del modello). Con poche richieste
conviene `--shards 1`; `submit` avvisa se le unità per shard sono poche.

---

## Output

Una riga per (richiesta, ripetizione):

```json
{"custom_id": "q-0001", "repeat": 0, "model": "qwen-big", "response": {"content": "...", "reasoning": null, "finish_reason": "stop", "usage": {"prompt_tokens": 12, "completion_tokens": 40}}, "error": null, "shard": 0, "latency_s": 1.8}
```

Una richiesta fallita (contesto superato, timeout, errore del server dopo i retry)
ha `response: null` ed `error: {"type", "message"}`, e non ferma lo shard.

### Ripresa

I risultati sono scritti riga per riga con flush. Se un task scade per il time
limit o viene cancellato, rilanciarlo (`sbatch` dello stesso `job.sbatch`, o
`cerberus batch run --shard I`) salta le unità già presenti. Con `--retry-failed`
rielabora anche quelle finite con errore. Se un'unità compare più volte, vale
l'ultima riga.

---

## Provarlo

- In locale, senza SLURM: [`test/demo/batch_local/`](../test/demo/batch_local/README.md).
- Controllo automatico dei risultati: `python test/demo/batch_check.py --batch-dir D`.
