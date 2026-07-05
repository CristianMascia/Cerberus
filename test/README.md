# Test di Cerberus

Due livelli di verifica:

## `unit/` — test unitari
Verificano i componenti di Cerberus **in isolamento**, senza rete, GPU, SLURM,
Singularity né dipendenze pesanti (solo `unittest` della standard library).

```bash
python -m unittest discover -s test/unit -v
```

Coprono: parsing/validazione di `models.conf` (`cerberus.config`), bin-packing del
posizionamento (`cerberus.placement`), separazione del reasoning e risoluzione
della mappa endpoint (`cerberus.client`), espansione dei GGUF spezzati
(`cerberus.download`).

## `demo/` — test d'integrazione (su IBiSCo)
Demo eseguibili end-to-end che usano il tool completo (allocazione, avvio dei
server, interrogazione via client):

- `ibisco/` — singolo nodo (un modello grande auto-split su 2 GPU + un piccolo);
- `ibisco_multinode/` — due nodi (due modelli grandi auto-distribuiti);
- `local/` — variante locale con immagine `.sif` (indipendente dal tool).

Istruzioni passo-passo e diagnosi degli errori: [../docs/demo.md](../docs/demo.md).
Il client condiviso è `demo/demo_client.py` (usato da entrambe le demo IBiSCo).
