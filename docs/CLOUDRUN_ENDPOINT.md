# GwenLaya Cloud Run L4 endpoint (plan.json stage S7)

Tooling only. **Nothing here has been deployed or run**; no latency, cost or cold-start number exists yet
(all `TBD` until `deploy/cloudrun/smoke_test.py` writes a results file). Targets H14 are in
`docs/GWENLAYA_PREREGISTRATION.md`.

| File | Purpose |
|---|---|
| `deploy/cloudrun/Dockerfile` | `ghcr.io/ggml-org/llama.cpp:server-cuda` + FastAPI front (engine: llama.cpp, because tiers are q4_K_M GGUF) |
| `deploy/cloudrun/entrypoint.sh` | starts `llama-server` (loopback, `--parallel 8`) then the front on `$PORT` |
| `deploy/cloudrun/front.py` | `/healthz` (503 until the model is loaded), `/generate`, `/v1/chat/completions`, `/route` (forwards to the existing Laya service with a metadata-server ID token; 503 if `LAYA_SERVICE_URL` unset) |
| `deploy/cloudrun/deploy.sh` | read-only quota preflight, Cloud Build, `gcloud run deploy`. Dry-run default; real run needs `--yes-spend` and `GWAYA_CONFIRM_SPEND=1` |
| `deploy/cloudrun/smoke_test.py` | unauth check, cold call, warm calls, wait for instance count 0. Same two-key spend gate |
| `deploy/cloudrun/teardown.sh` | deletes `gwenlaya-*` services only. Dry-run default; `--yes` and `GWAYA_CONFIRM_SPEND=1` |

Deploy flags: `--gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --cpu 4 --memory 16Gi --no-cpu-throttling
--min-instances 0 --max-instances 1 --no-allow-unauthenticated --labels purpose=gwenlaya`, region `us-central1`.
`--no-gpu-zonal-redundancy` is used because the quota read on 2026-10-07 (`gcloud beta quotas info list`,
read-only) showed a value only for `NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion` in us-central1 (3); the
zonal-redundancy quota showed no value there. Other services (`deepseek-prover-v2`) may already use some of it.

## Known deviations and open points (not resolved by this tooling)

1. **Weights source.** The work item asks for a Cloud Storage volume mount of
   `gs://socrateai-datalake-gen-lang-client-0625573011/gwenlaya_v4/quantized/`. The pre-registration (section 11)
   says the 9B q4 GGUF is baked into the image to avoid cross-region reads (bucket US-EAST4, service us-central1).
   This tooling implements the mount (the entrypoint also accepts a GGUF baked into `/models`). Cold start from a
   FUSE mount of a multi-GB file is slower and is exactly what H14 measures; the frozen prereg is not edited here.
2. **Topology.** Prereg lists a second CPU service `gwenlaya-gate`. It is not built here; the front calls the
   existing Laya service (read-only use) through `/route`. Its request schema and `LAYA_ROUTE_PATH` are unverified.
3. **Unverified without a deploy:** the `mount-options=only-dir=...` value syntax in `--add-volume`, the
   `server-cuda` image's `llama-server` path, startup time against the probe budget (120 x 5 s), and that the
   service account has `roles/storage.objectViewer` on the bucket (pass `--service-account`).
4. **Artifact Registry repo `gwenlaya` in us-central1** must exist; no script creates it.
5. **Scale-to-zero check** reads Cloud Monitoring `container/instance_count`, which lags by minutes; GPU instances
   may idle up to 10 minutes (prereg section 0), so the default wait is 1200 s. An empty window counts as zero
   only after a positive count was seen in the same run.
