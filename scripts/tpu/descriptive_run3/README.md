# Descriptive TPU run #3 (v3.3.1 paper, Section "Descriptive TPU run")

Scripts that produced the numbers in `results/gwenlaya_v4/tpu_run3/`. **Superseded for any comparison
by `scripts/tpu/gen_batch.py`**, which uses the registered raw-prompt protocol. This older run used the
vLLM chat template with its own short system prompts and no code-fence prefill, so it is only a loose
consistency check against the llama.cpp CPU arm (hardware, quantisation, engine and code prompt all differ).

- `drive3.sh` - creates one on-demand v5litepod-1, uploads, runs, downloads, deletes (no cap check; see
  `deploy/tpu/drive.sh` for the capped driver).
- `run3.sh` - on the VM: bubblewrap, rustup, vllm-tpu, then `gen_tasks.py` per model.
- `gen_tasks.py` - generation **and in-process scoring**. The in-VM Python scoring was invalid (every item
  failed), so all outputs were re-scored locally; use the `*.rescored.jsonl` files.
