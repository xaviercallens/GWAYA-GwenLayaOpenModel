# GwenLaya v4 Data Manifest

**Generated:** 2026-10-07T19:23:50.521997Z

**Plan:** GwenLaya v4 (written 2026-10-07)

## Evaluation Datasets

| ID | HF ID | Domain | License | Rows |
|----|----|--------|---------|------|
| EVAL000 | evalplus/mbppplus | python | apache-2.0 | 378 |
| EVAL001 | evalplus/humanevalplus | python | apache-2.0 | 164 |
| EVAL002 | google-research-datasets/mbpp | python | cc-by-4.0 | 427 |

## Laya RL Model Weights

- **model.safetensors**: 803.57 MB (SHA256: 891102d372688fc2)
- **rl_energy_model_unified.pt**: 0.09 MB (SHA256: 5387fe9c0699b7e6)
- **rl_100_formal_critic.pt**: 0.09 MB (SHA256: 27335eace9cb1310)
- **surrogate_energy_predictor_v2.pt**: 0.26 MB (SHA256: ea24a4153b027dc3)
- **rl_agent_config.json**: 0.00 MB (SHA256: ae287b56bbcf5f8c)
- **rl_energy_model_unified_200.pt**: 0.09 MB (SHA256: a5f32b894a801629)
- **rl_multidisciplinary_critic.pt**: 0.09 MB (SHA256: ce1c7c31a689062b)
- **rl_common.py**: 0.02 MB (SHA256: c35dc070546aed7b)
- **rl_agent_api.py**: 0.00 MB (SHA256: be3b46819c9999c3)
- **rl_nightly_epochs_1_to_100.tar.gz**: 127.18 MB (SHA256: d38f713755c8f647)

## Lean Oracle Archive

- **Path:** gs://socrateai-datalake-gen-lang-client-0625573011/formal_verification/lean_oracle_v5.tar.gz
- **Size:** 32.38 MB
- **MD5:** bjY/gywarXKeA6IE7++66A==
- **Note:** Contains no Mathlib source or lean-toolchain (inspected 2026-10-07)

## Statistics

- **Total eval rows:** 969
- **Est. eval bytes:** 0.5 MB
- **Laya weights:** 0.91 GB
