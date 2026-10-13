# GWAYA / GwenLaya serverless endpoint on TPU (OpenAI format)

Status: **tooling written and unit-tested against a fake TPU API; not deployed, no cold-start number measured yet.**
Every latency/cost figure below is marked *estimate* until `deploy/tpu_endpoint/smoke.py` has been run.

## What it is

```
client (OpenAI SDK / curl)
   |  HTTPS, Authorization: Bearer <API key>
   v
Cloud Run "gwenlaya-endpoint"   CPU gateway, min instances 0, HTTP autoscale (max 5), public URL, key enforced by the app
   |  creates on demand / proxies over the VPC (tcp:8000, bearer key, firewall tag)
   v
Cloud TPU VM "gwenlaya-endpoint" (v5litepod-1, on-demand)   `vllm serve Qwen/Qwen3.5-4B`, OpenAI API
   deletes ITSELF after IDLE_MIN (10) minutes without requests, or after MAX_UP_MIN (120) minutes regardless
```

Why not "Cloud Run on TPU": Cloud Run offers GPUs, not TPUs. A TPU endpoint that scales to zero therefore has to
create and delete a TPU VM, so **a cold request costs minutes, not seconds**. That is the price of using TPU quota
(16 v5e chips per region here) instead of the 3-L4 Cloud Run GPU quota. If seconds matter more than quota, the
existing `deploy/cloudrun` L4 design (llama.cpp, baked GGUF) is the alternative.

What the endpoint serves: the **GwenLaya generator tier** (Qwen3.5-4B, bf16) behind an OpenAI-compatible API. The GWAYA
gate and the Laya router/calibrator are *not* inside the serverless image: they run client-side against this endpoint
(see "Using GWAYA with it"). Responses are plain model answers; they carry no verdict or calibrated P(correct).

## Cold start: what is optimised, what is not (estimates)

| Phase | Typical | Optimisation |
|---|---|---|
| Gateway Cloud Run cold start | 1-3 s | tiny CPU image, startup CPU boost |
| TPU VM create + boot | 1.5-3 min (seen 15:26 -> 15:29 on 2026-10-11) | on-demand, zone fallback us-west4-a / us-central1-a / us-east5-b |
| Python env (`vllm-tpu==0.31.0`) | ~30-40 s (measured in earlier runs) | pinned wheel, uv |
| Weights | ~30-90 s from the endpoint's own bucket (first boot ever: HF download, then cached) | `gs://gwenlaya-endpoint-<project>/models` |
| vLLM start + XLA warm-up | 2-4 min (Qwen3.5 uses the PyTorch fallback path on TPU, `SKIP_JAX_PRECOMPILE=1`) | no precompile, `--max-num-seqs 16` |
| **Total cold request** | **about 6-9 min (estimate)** | |

Not possible: keeping "weights already loaded" with zero instances. After a scale-to-zero the TPU VM no longer
exists; weights are re-read from the cache bucket, not from the network of Hugging Face.

Timeout handling:
- A cold request is **held** by the gateway for up to 540 s (`HOLD_S`), then answered `503` with `Retry-After: 30`
  while the TPU keeps starting. The OpenAI SDK retries 503 by itself; set `timeout=600` and `max_retries=3`.
- `X-Gwaya-Wait: 0` returns the 503 immediately instead of holding.
- Best practice for a test session: **`POST /v1/warmup` first, poll `GET /v1/status` until `ready`, then test.** Warm
  requests then take generation time only.
- Cloud Run request timeout is 3600 s; the client timeout is the one to raise.

## Cost (estimates, list prices, not the bill)

| Item | Cost |
|---|---|
| Gateway, idle | $0 (min instances 0; free tier covers test traffic) |
| TPU VM | $1.20 per chip-hour, billed per second while the VM exists (cold start + serving + idle window) |
| One test session (cold start ~7 min + 10 min idle window) | about $0.34 |
| Hard ceiling per VM lifetime | `MAX_UP_MIN` = 120 min = $2.40 |
| Weights bucket | about $0.25 per month |

The VM deletes itself (`watchdog.py`); if the watchdog failed you can also `POST /v1/stop`, or
`gcloud compute tpus tpu-vm delete gwenlaya-endpoint --zone <zone>`. Check with
`gcloud compute tpus tpu-vm list --zone us-west4-a`.

## Deploy (once)

```bash
deploy/tpu_endpoint/deploy.sh                                   # dry run: prints every command
GWAYA_CONFIRM_SPEND=1 deploy/tpu_endpoint/deploy.sh --yes-spend # creates the resources below
```

Creates, all named `gwenlaya-*`: service account + custom role (create/get/list/delete TPU nodes only), bucket
`gwenlaya-endpoint-<project>`, firewall rule `gwenlaya-endpoint-8000` (10.128.0.0/9 -> tag `gwenlaya-endpoint`,
tcp:8000), Artifact Registry repo `gwenlaya`, Cloud Run service `gwenlaya-endpoint` (public URL, `--min-instances 0`,
`--max-instances 5`). It does not touch the data-lake bucket, its IAM, or any other service. The API key is generated
once into `$GWAYA_DATA_ROOT/endpoint/keys.json` (mode 600).

Tear down: `gcloud run services delete gwenlaya-endpoint --region us-central1`, delete the TPU VM if present, then the
bucket/firewall/SA/role if you no longer want them.

## User guide: calling it as a public endpoint

Base URL: the Cloud Run URL printed by `deploy.sh` (e.g. `https://gwenlaya-endpoint-xxxxx-uc.a.run.app`).
The URL is public; **every route except `/health` (`/healthz` is intercepted by Cloud Run's front end with a 404) requires `Authorization: Bearer <API key>`**. With no key
configured the gateway rejects everything (fail closed). Share the key only with people you want to spend your TPU
time. Rotate with `gcloud run services update gwenlaya-endpoint --update-env-vars GWAYA_API_KEYS=<new>,...` (several
keys may be comma-separated).

```bash
export GWAYA_ENDPOINT=https://gwenlaya-endpoint-xxxxx-uc.a.run.app
export GWAYA_API_KEY=gwk_...

# 1. wake it (returns 202 at once) and wait until ready
curl -s -X POST  $GWAYA_ENDPOINT/v1/warmup -H "Authorization: Bearer $GWAYA_API_KEY"
until curl -s $GWAYA_ENDPOINT/v1/status -H "Authorization: Bearer $GWAYA_API_KEY" | grep -q '"ready"'; do sleep 10; done

# 2. chat (OpenAI format)
curl -s $GWAYA_ENDPOINT/v1/chat/completions -H "Authorization: Bearer $GWAYA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gwenlaya-4b","messages":[{"role":"user","content":"Write a Python function that reverses a string."}],"max_tokens":256,"temperature":0}'

# 3. stream
curl -sN $GWAYA_ENDPOINT/v1/chat/completions -H "Authorization: Bearer $GWAYA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gwenlaya-4b","stream":true,"messages":[{"role":"user","content":"Say hi"}],"max_tokens":32}'

# 4. stop early (otherwise it deletes itself after 10 idle minutes)
curl -s -X POST $GWAYA_ENDPOINT/v1/stop -H "Authorization: Bearer $GWAYA_API_KEY"
```

OpenAI Python SDK:

```python
from openai import OpenAI
c = OpenAI(base_url=f"{ENDPOINT}/v1", api_key=KEY, timeout=600, max_retries=3)
r = c.chat.completions.create(model="gwenlaya-4b", messages=[{"role": "user", "content": "2+2?"}], max_tokens=32)
print(r.choices[0].message.content)
```

Routes: `GET /health` (open), `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`,
`POST /v1/warmup`, `GET /v1/status`, `POST /v1/stop`.
Limits enforced by the gateway: body at most 512 KiB, `max_tokens` clamped to 2,048 (default 512), `n` forced to 1,
`model` forced to `gwenlaya-4b`, context 8,192 tokens, at most 16 concurrent sequences on the TPU.
Thinking mode is turned off by default (`enable_thinking=false` where vLLM supports the flag); pass
`"chat_template_kwargs": {"enable_thinking": true}` to enable it.

Measure it: `GWAYA_ENDPOINT=... GWAYA_API_KEY=... python3 deploy/tpu_endpoint/smoke.py --stop` prints the cold-start
seconds and the latency of three warm requests as JSON.

## Using GWAYA (gate + Laya) with it

The generator is remote; verification stays in your process so that the sandbox and hidden tests never leave your
machine. The registered tooling talks to llama.cpp-style servers (`gwaya/llama_tier.py`); for this endpoint use any
OpenAI client to obtain the candidate, then run the gate locally, e.g. for Python:
`gwaya` gate on `(candidate, visible_tests)`, and the frozen Laya (`gwaya/laya_runtime.py`) for P(correct). Note that
the Laya was calibrated on **bf16 vLLM-TPU log-probs of Qwen3.5**; this endpoint does not return log-probs unless you
send `"logprobs": true, "top_logprobs": 1`, and a different serving stack or quantisation shifts the calibration
(A19 measures that for Q4_K_M). Treat P(correct) from other stacks as uncalibrated.

## Known limits and unverified points

1. Never deployed: the exact vLLM-TPU flags for Qwen3.5 (`--limit-mm-per-prompt`, `--max-num-batched-tokens 1024`)
   are those that worked in batch runs (2026-10-08); the HTTP server mode itself is untested on this model.
2. Direct VPC egress to a TPU VM in another region and the firewall source range are assumptions to confirm on first
   deploy (`deploy.sh` prints the commands; `smoke.py` fails with 502 if the gateway cannot reach the VM).
3. The custom IAM role lists the permissions needed to create TPU nodes; a missing permission shows up as a 403 from
   the TPU API on the first warm-up and is fixed by adding it to the role.
4. Capacity: on-demand v5e in us-west4-a worked on 2026-10-10/11; capacity is not guaranteed, hence the zone fallback.
5. Qwen3.5 is not JAX-native on tpu-inference (PyTorch fallback): expect tens of tokens per second per stream, not
   thousands. Single-stream latency was not measured.
6. One TPU VM serves all callers (16 sequences); `max-instances 5` scales the gateway, not the TPU.
