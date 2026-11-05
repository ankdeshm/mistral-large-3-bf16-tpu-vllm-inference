# `Mistral-Large-3-675B-Instruct-2512-BF16` Serving on TPU `v7x` with `vllm-torchtpu`

Multi-host deployment manifest, OpenAI SSE streaming benchmark harness (`c=1..256` concurrency, `32K`–`128K` context, and greedy `natural-chat`), and latency/throughput results for unquantized **[`mistralai/Mistral-Large-3-675B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-BF16)** (`675B` total / `41B` active parameters, `1,352.0 GB` / `1.23 TiB` BF16 across `272` consolidated safetensors shards) on a **single TPU `v7x-16` slice (`2x2x4`, 16 chips / 32 TensorCores, `3,072 GiB` HBM3e)** with **`vllm 0.32.0` + `vllm-torchtpu`**.

- **Benchmark Outputs:** [`REPORT.md`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`summary.json`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) · Benchmark Job: [`v7x16-ml3-suite.yaml`](./k8s/v7x16-ml3-suite.yaml) · Reference LWS manifest (not used for the measured runs): [`v7x-16-ml3-bf16-tp2-dp16-ep32.yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml)

---

## 1. Model & Hardware Overview

| Property | Value |
|---|---|
| **Hugging Face Checkpoint** | [`https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-BF16) (`272` consolidated safetensors shards, `1,352.0 GB` BF16 staged via [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py)) |
| **Architecture** | `MistralLarge3ForCausalLM` (`--language-model-only`, `61` Transformer layers: `3` dense + `58` MoE layers, `128` routed experts + `1` shared expert, top-4 active routing, Multi-head Latent Attention `MLA` with `kv_lora_rank=512`, `qk_rope_head_dim=64`, `294,912` max position embeddings) |
| **Parallelism Layout** | `TP=2, DP=16, EP=32` across `32` TensorCores (`4` routed BF16 experts per TensorCore via EP-filtered Run:AI GCS streamer, `--enable-ep-weight-filter` in [`scripts/plan_mistral_large3_v7x16.json`](./scripts/plan_mistral_large3_v7x16.json); Tensor Parallelism stays intra-chip across the 2 TensorCores of each `v7x` chip) |
| **KV-Cache Capacity (`--kv-cache-dtype=fp8`)** | **`12,255,232` (`12.26M`) FP8 MLA KV-cache tokens** across the 16-chip slice (`765,952` tokens per DP engine × `16` DP engines = `5.80×` concurrency at `132,096` tokens per engine, or **`92.8×` concurrent `128K` requests** alongside `1.35 TB` of unquantized BF16 weights) |
| **Runtime Stack** | Measured runs: image `us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/vllm/vllm-torchtpu-bench:v0.32.0` — `vllm 0.32.0` + `vllm-torchtpu`, `torch 2.13.0+cpu` wheel, `jax 0.11.2`, `ray 2.59.0`; Ray multi-host executor driven by [`scripts/run_vllm_tpu_multihost_plan.py`](./scripts/run_vllm_tpu_multihost_plan.py) from [`k8s/v7x16-ml3-suite.yaml`](./k8s/v7x16-ml3-suite.yaml). [`docker/Dockerfile`](./docker/Dockerfile) is a reference build, not used for the measured runs |
| **Boot & Memory** | Server ready in `880 s` (`vllm serve` launch to `/v1/models` responding); engine init `653.2 s`, of which compilation `523.4 s` (maximum over DP engines); each TensorCore streams only its expert-parallel shard from GCS (`85.0 GB`); BF16 weights use `57.38 GiB` per TensorCore of the `94.74 GiB` runtime-reported HBM (`--gpu-memory-utilization=0.90` cap: `85.26 GiB`) |

---

## 2. Benchmark Results (`v7x-16` BF16 vs FP8 Weights at `TP=2, DP=16, EP=32`)

TPU `v7x-16` provides `3,072 GiB` HBM3e (`118 TB/s` aggregate HBM bandwidth), enabling the entire `1.35 TB` unquantized BF16 checkpoint to run on a single 16-chip slice at `132,096` max context with `12.26M` FP8 MLA KV-cache tokens and zero failed requests across all `11` benchmark rows (`0` errors out of `1,024` requests):

- **High-Throughput Batching (`4,096` in / `1,024` out, `c=1..256`):** Unquantized BF16 scales cleanly to **`3,156.49` output tok/s (`15,785.51` total tok/s)** at `c=256`, achieving `82.8%` of native FP8 throughput (`3,812.96` output tok/s) despite reading `2×` larger BF16 expert weights per decode step (`39.75` stream tok/s and `25.17 ms` median TPOT on single-stream greedy decode).
- **`128K` Long Context (`131,072` in / `512` out, `c=4`, `8` requests):** BF16 reaches **`2,940.83` total tok/s (`11.44` output tok/s, `31.31` stream tok/s)**, `+10.9%` above the FP8 row `ml3-fp8-tp2-dp16-ep32-fp8kv` (`2,650.90` total tok/s / `10.31` output tok/s). The FP8 row has the lower median TTFT (`154.47 s` vs `163.48 s`) and includes two requests with TTFT ≈ `217.5 s`. At `32K` / `64K` context, BF16 total tok/s is `9.5%` / `10.3%` below FP8.

| Workload Suite (`Script Anchor`) | Input : Output | Temp | Conc | BF16 Out TPS | BF16 Total TPS | vs FP8 (`TP=2, DP=16, EP=32`) | Mean Stream TPS | TTFT p50/p90 (ms) | TPOT p50/p90 (ms) | OK/Fail | Manifest & Result Files |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `1` | `35.70` | `178.55` | `-22.3%` (`45.92` out) | `39.27` | `2,631.4 / 2,644.7` | `25.48 / 25.48` | `4/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `8` | `251.23` | `1,256.41` | `-22.7%` (`324.95` out) | `35.48` | `3,643.9 / 4,361.7` | `28.47 / 28.48` | `16/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `32` | `854.67` | `4,274.19` | `-20.9%` (`1,080.69` out) | `29.39` | `3,966.8 / 4,774.7` | `35.36 / 35.87` | `64/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `64` | `1,388.41` | `6,943.43` | `-20.3%` (`1,741.51` out) | `23.86` | `3,865.6 / 5,884.9` | `42.44 / 50.43` | `128/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `128` | `2,185.18` | `10,928.02` | `-17.0%` (`2,634.02` out) | `18.88` | `5,690.7 / 9,821.3` | `53.46 / 57.67` | `256/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |
| [`concurrency_4096in` (`#L380`)](./scripts/run_tpu_benchmark_sweep.py#L380-L386) | `4,096 : 1,024` | `0.6` | `256` | **`3,156.49`** | **`15,785.51`** | `-17.2%` (`3,812.96` out) | `14.16` | `10,140.3 / 16,600.0` | `64.83 / 89.08` | `512/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | `32,768 : 512` | `0.6` | `4` | `50.66` | `3,293.27` | `-9.5%` (`3,638.24` tot) | `34.22` | `26,015.0 / 26,250.0` | `29.22 / 31.22` | `8/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | `65,536 : 512` | `0.6` | `4` | `24.72` | `3,188.31` | `-10.3%` (`3,552.50` tot) | `33.90` | `68,344.6 / 68,457.5` | `28.36 / 33.72` | `8/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |
| [`context_sweep` (`#L395`)](./scripts/run_tpu_benchmark_sweep.py#L395-L401) | **`131,072 : 512`** | `0.6` | `4` | **`11.44`** | **`2,940.83`** | **`+10.9%`** (`2,650.90` tot) | **`31.31`** | `163,476.3 / 163,670.5` | `30.19 / 38.86` | `8/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |
| [`greedy_natural` (`#L402`)](./scripts/run_tpu_benchmark_sweep.py#L402-L407) | `natural-chat (39.5) : 1,024` | `0.0` | `1` | **`39.62`** | `41.15` | `-20.6%` (`49.89` out) | **`39.75`** | `77.6 / 202.5` | **`25.17 / 25.18`** | `4/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |
| [`greedy_natural` (`#L402`)](./scripts/run_tpu_benchmark_sweep.py#L402-L407) | `natural-chat (38.3) : 1,024` | `0.0` | `8` | **`277.72`** | `288.11` | `-24.3%` (`367.04` out) | `34.88` | `142.8 / 216.5` | `28.67 / 28.68` | `16/0` | [`yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) · [`REPORT`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) · [`JSON`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/summary.json) |

Conventions: `vs FP8` = (BF16 − FP8) / FP8 at the same workload and concurrency, using output TPS for concurrency and greedy rows and total TPS (`tot`) for context rows; FP8 = `ml3-fp8-tp2-dp16-ep32-fp8kv` from [`mistral-large-3-fp8-tpu-vllm-inference`](https://github.com/ankdeshm/mistral-large-3-fp8-tpu-vllm-inference). Output TPS = generated tokens / wall time of the row; Total TPS = (prompt + generated) tokens / wall time of the row; Mean Stream TPS = mean per-request `(output tokens − 1) / decode time`. TTFT is measured client-side from request send to first streamed token and includes queueing.

### Methodology Notes

- **Patch:** the BF16 checkpoint has no quantization config and needs no model patch. The benchmark Job runs [`scripts/apply_ml3_patches.py`](./scripts/apply_ml3_patches.py) on every host for all configurations in the plan; its default path changes only the compressed-tensors FP8 MoE code path, which the BF16 checkpoint does not use. The Eagle draft-model patches were not applied.
- **Weights:** BF16 MoE expert weights run unquantized (BF16 grouped-matmul path).
- **KV cache:** `--kv-cache-dtype=fp8` (e4m3) without calibrated KV scales, as in the FP8 run.
- **Memory utilization:** the BF16 run used `--gpu-memory-utilization=0.90`; the FP8 comparison run used `0.88`. All other server flags are the same.
- **Accuracy:** no accuracy evaluation was run; outputs were checked for coherent text only.
- **`--language-model-only`:** skips the Pixtral vision encoder (`48` layers, ~`2.5B` parameters) and its image-input memory profiling; image inputs were not benchmarked.
- **Workloads:** sampled rows use `/v1/completions` with `T=0.6`, `ignore_eos` and a fixed output length on random-word prompts truncated to the stated token length with the model tokenizer; greedy `natural-chat` rows use `/v1/chat/completions` with `T=0` on natural-language prompts (mean prompt tokens in parentheses). Concurrency rows send `max(2×c, 4)` requests; context rows send `8` requests at `c=4`. Percentiles are nearest-rank over the requests in a row.

### Supported Configurations & Known Limitations

- **Measured on this stack:** BF16 weights at `TP=2, DP=16, EP=32` on one `v7x-16` slice; `--max-model-len=132096`; FP8 KV cache; async scheduling; prefix caching enabled (this sweep has no shared-prefix rows); text-only serving (`--language-model-only`).
- **Speculative decoding** was not measured with BF16 weights. Eagle speculative decoding (`mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle`) does not run on this stack yet: the vLLM `0.32.0` Mistral-Large-3 Eagle draft model does not yet initialize and load under `vllm-torchtpu`.
- **Eagle patches** in [`scripts/apply_ml3_patches.py`](./scripts/apply_ml3_patches.py) are opt-in (`ML3_EAGLE_PATCHES=1`, default off) and were not used for any result in this repository.
- **`131,072`-token comparison:** each row is an 8-request sample; the FP8 row includes two slower requests (TTFT ≈ `217.5 s` vs `154.47 s` median).

---

## 3. Repository Structure & Reproduction

- [`docker/Dockerfile`](./docker/Dockerfile) — Reference image build (`vllm/vllm-tpu` base + benchmark tooling); not used for the measured runs
- [`scripts/run_vllm_tpu_multihost_plan.py`](./scripts/run_vllm_tpu_multihost_plan.py) — Multi-host harness used for the measured runs: for each experiment, all hosts join a fresh Ray cluster, host 0 runs `vllm serve` (Ray executor) and, once `/v1/models` responds, the benchmark sweep; writes `boot_*.json` and `suite_status.json` next to the results
- [`scripts/plan_mistral_large3_v7x16.json`](./scripts/plan_mistral_large3_v7x16.json) — Executed plan for the full Mistral-Large-3 suite; the BF16 configuration is experiment `ml3_bf16_tp2_dp16_ep32`
- [`scripts/apply_ml3_patches.py`](./scripts/apply_ml3_patches.py) — Runtime patch script run by the benchmark Job on every host (default path targets the FP8 checkpoint only; Eagle draft-model patches opt-in via `ML3_EAGLE_PATCHES=1`)
- [`scripts/run_tpu_benchmark_sweep.py`](./scripts/run_tpu_benchmark_sweep.py) — Async OpenAI SSE streaming benchmark (`concurrency_<in>in`, `cached_<ratio>pct_prefix`, `context_sweep`, `greedy_natural`)
- [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) — Parallel `272`-shard Hugging Face -> GCS stager
- [`k8s/v7x16-ml3-suite.yaml`](./k8s/v7x16-ml3-suite.yaml) — Benchmark Job used for the measured runs: 4-pod Indexed Job (one pod per `v7x` host) + headless Service
- [`k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) — Reference GKE LeaderWorkerSet (`4` hosts × `4` chips = `32` TensorCores, `TP=2, DP=16, EP=32`; same vLLM server flags as the plan) + Benchmark Job; not used for the measured runs
- [`results/`](./results/v7x-16-2x2x4-32cores__mistralai__Mistral-Large-3-675B-Instruct-2512-BF16__ml3-bf16-tp2-dp16-ep32-fp8kv__20261103T070908Z/REPORT.md) — Raw Markdown & JSON benchmark artifacts

```bash
# 1. Stage mistralai/Mistral-Large-3-675B-Instruct-2512-BF16 (1.35 TB, 272 shards) to GCS
python3 scripts/stage_hf_to_gcs.py --repos mistralai/Mistral-Large-3-675B-Instruct-2512-BF16 --bucket YOUR_GCS_BUCKET --prefix models

# 2. Scripts ConfigMap and Hugging Face token Secret for the benchmark Job
kubectl create configmap v7x16-ml3-scripts \
  --from-file=run_vllm_tpu_multihost_plan.py=scripts/run_vllm_tpu_multihost_plan.py \
  --from-file=plan.json=scripts/plan_mistral_large3_v7x16.json \
  --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py \
  --from-file=apply_ml3_patches.py=scripts/apply_ml3_patches.py
kubectl create secret generic hf-token --from-literal=HF_TOKEN=YOUR_HF_TOKEN

# 3. Add env SUITE_ONLY=ml3_bf16_tp2_dp16_ep32 to the Job to run only the BF16 configuration, then apply it
kubectl apply -f k8s/v7x16-ml3-suite.yaml
```

The plan's BF16 sweep builds exact-length prompts with `tokenizer.json` from `/models/models/mistralai/Mistral-Large-3-675B-Instruct-2512` (the FP8 checkpoint path; same tokenizer). Stage that file to `gs://YOUR_GCS_BUCKET/models/mistralai/Mistral-Large-3-675B-Instruct-2512/`, or point `bench_tokenizer` in the plan at a directory that contains it.

Reference deployment (not used for the measured runs):

```bash
kubectl create configmap tpu-bench-script --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py
kubectl apply -f k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml
```
