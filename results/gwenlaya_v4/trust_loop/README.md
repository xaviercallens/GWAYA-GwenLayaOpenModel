# A26 trust loop: committed dev artefacts

Copied from the data disk (trust_loop/) on 2026-10-14 so every dev number in the paper has a committed source.
Produced by scripts/loop/{build_table,baseline_dev,lever_D_*,lever_U_*}.py (dev only, grouped CV by cluster, seed 0).

Not committed: test_table.jsonl (sealed E rows; the sealed scorer output is results/gwenlaya_v4/laya/sealed_E_D3_lr_core.json),
lever_D_exec_cache.json (cache of executed stub audits, 2 MB) and lever_U_prompt_emb_qwen25_05b.npz (prompt embeddings, 2 MB); both are regenerable.
