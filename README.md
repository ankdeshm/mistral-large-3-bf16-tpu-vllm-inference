# Mistral Large 3 (675B, BF16) on Google Cloud TPU v7x

Multi-host serving benchmarks for unquantized [`mistralai/Mistral-Large-3-675B-Instruct-2512-BF16`](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-BF16) on a single Google Cloud TPU `v7x-16` slice (`2x2x4`, 16 chips / 32 TensorCores) with `vllm-torchtpu`.

**Summary:** The full `1.35 TB` unquantized `bfloat16` checkpoint fits on one `v7x-16` slice (`TP=2, DP=16, EP=32`) alongside `12.26M` FP8 MLA KV-cache tokens (`132,096` max context) with 0 failed requests across 1,024 measured requests, reaching **3,156.49 output tok/s (`15,785.51` total tok/s) at `4k/1k` concurrency 256 (17.2% lower than the 3,812.96 output tok/s native FP8 checkpoint on the same `TP=2, DP=16, EP=32` layout while reading 2× larger expert weights)** and **39.75 stream tok/s (`25.17 ms` median TPOT) on single-stream greedy decode (20.6% lower output tok/s than FP8's `49.89` tok/s / `19.97 ms` TPOT)**. At `128k` context (`c=4`, 8 requests), BF16 reaches **2,940.83 total tok/s (10.9% higher wall-clock total tok/s than the 8-request FP8 sample due to two tail-delayed FP8 requests, though BF16 median TTFT is 5.8% slower at `163.48 s` vs. `154.47 s`)**.

---

## 1. Model, Framework, and Hardware

- **Model:** `mistralai/Mistral-Large-3-675B-Instruct-2512-BF16` (`MistralLarge3ForCausalLM`, text-only via `--language-model-only`), a **675B total / 41B active** Mixture-of-Experts model (`1,352.0 GB` across 272 consolidated safetensors shards, `294,912` max position embeddings).
  - **Precision:** Unquantized **`bfloat16` weights** (`57.38 GiB` per TensorCore, BF16 grouped-matmul MoE path) paired with **FP8 (`e4m3`) KV cache** (`--kv-cache-dtype=fp8`, `765,952` KV tokens per DP engine × 16 engines = `12,255,232` slice KV tokens at `--gpu-memory-utilization=0.90`).
  - **Architecture:** 61 Transformer layers (3 dense layers + 58 MoE layers with **128 routed experts + 1 shared expert**, top-4 active routing per token) and Multi-head Latent Attention (**MLA**, `kv_lora_rank = 512`, `qk_rope_head_dim = 64`).
- **Framework:** `vllm 0.32.0` + `vllm-torchtpu`, `torch 2.13.0`, `jax 0.11.2`, and `ray 2.59.0` (Ray multi-host executor with `--enable-ep-weight-filter` streaming `85.0 GB` per TensorCore via `runai_streamer`, `--enable-prefix-caching`, and `--async-scheduling`).
- **Hardware:** One Google Cloud TPU **v7x-16** slice on GKE (`2x2x4` 3D ICI torus: 4 hosts × 4 chips/host = **16 chips / 32 TensorCores**, `192 GB` HBM3e per chip / `94.74 GiB` usable per TensorCore, `3,072 GB` aggregate HBM3e). Tensor Parallelism (`TP=2`) stays intra-chip across the two TensorCores of each `v7x` chip while Expert Parallelism (`EP=32`) places 4 routed experts on every TensorCore across 16 data-parallel engines (`DP=16`).

---

## 2. Main Results

Measured on one `v7x-16` slice (`TP=2, DP=16, EP=32`, `--max-model-len=132096`, `--max-num-batched-tokens=1024`). Comparisons in the right-hand column are `(BF16 − FP8) / FP8` at the identical workload, concurrency, and `TP=2, DP=16, EP=32` topology from [`mistral-large-3-fp8-tpu-vllm-inference`](https://github.com/ankdeshm/mistral-large-3-fp8-tpu-vllm-inference) (using output TPS for concurrency/greedy suites and total TPS for long-context prefill rows).

| Suite | Input : Output Tokens | Temp | Concurrency | Output TPS | Total TPS | Mean Stream TPS | TTFT p50 / p90 (ms) | TPOT p50 / p90 (ms) | Comparison vs. Native FP8 (`TP=2, DP=16, EP=32`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `concurrency_4096in` | `4,096 : 1,024` | 0.6 | 1 | 35.70 | 178.55 | 39.27 | 2,631.4 / 2,644.7 | 25.48 / 25.48 | `-22.3%` out TPS (vs. `45.92` FP8) |
| `concurrency_4096in` | `4,096 : 1,024` | 0.6 | 8 | 251.23 | 1,256.41 | 35.48 | 3,643.9 / 4,361.7 | 28.47 / 28.48 | `-22.7%` out TPS (vs. `324.95` FP8) |
| `concurrency_4096in` | `4,096 : 1,024` | 0.6 | 32 | 854.67 | 4,274.19 | 29.39 | 3,966.8 / 4,774.7 | 35.36 / 35.87 | `-20.9%` out TPS (vs. `1,080.69` FP8) |
| `concurrency_4096in` | `4,096 : 1,024` | 0.6 | 64 | 1,388.41 | 6,943.43 | 23.86 | 3,865.6 / 5,884.9 | 42.44 / 50.43 | `-20.3%` out TPS (vs. `1,741.51` FP8) |
| `concurrency_4096in` | `4,096 : 1,024` | 0.6 | 128 | 2,185.18 | 10,928.02 | 18.88 | 5,690.7 / 9,821.3 | 53.46 / 57.67 | `-17.0%` out TPS (vs. `2,634.02` FP8) |
| `concurrency_4096in` | `4,096 : 1,024` | 0.6 | 256 | **3,156.49** | **15,785.51** | 14.16 | 10,140.3 / 16,600.0 | 64.83 / 89.08 | **`-17.2%` out TPS** (`82.8%` of `3,812.96` FP8) |
| `context_sweep` | `32,768 : 512` | 0.6 | 4 | 50.66 | 3,293.27 | 34.22 | 26,015.0 / 26,250.0 | 29.22 / 31.22 | `-9.5%` total TPS (vs. `3,638.24` FP8) |
| `context_sweep` | `65,536 : 512` | 0.6 | 4 | 24.72 | 3,188.31 | 33.90 | 68,344.6 / 68,457.5 | 28.36 / 33.72 | `-10.3%` total TPS (vs. `3,552.50` FP8) |
| `context_sweep` | `131,072 : 512` | 0.6 | 4 | **11.44** | **2,940.83** | 31.31 | 163,476.3 / 163,670.5 | 30.19 / 38.86 | **`+10.9%` total TPS** (vs. `2,650.90` FP8; p50 TTFT `+5.8%` slower) |
| `greedy_natural` | `natural-chat (39.5) : 1,024` | 0.0 | 1 | **39.62** | 41.15 | **39.75** | 77.6 / 202.5 | **25.17 / 25.18** | `-20.6%` out TPS (vs. `49.89` FP8, `19.97 ms` TPOT) |
| `greedy_natural` | `natural-chat (38.3) : 1,024` | 0.0 | 8 | 277.72 | 288.11 | 34.88 | 142.8 / 216.5 | 28.67 / 28.68 | `-24.3%` out TPS (vs. `367.04` FP8) |

*Notes:* All 11 rows completed with `0` failed requests (`4..512` requests per concurrency row; `8` requests per long-context row). The BF16 run used `--gpu-memory-utilization=0.90` vs. `0.88` for the FP8 run; all other server flags were identical. No formal benchmark accuracy evaluation was run (outputs were verified for coherent generation).

---

## 3. Reproducing the Main Results

Stage the 272-shard BF16 checkpoint (`1.35 TB`) to GCS with [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py), create the scripts `ConfigMap` and `hf-token` `Secret`, and launch the 4-pod `v7x-16` Indexed Job [`k8s/v7x16-ml3-suite.yaml`](./k8s/v7x16-ml3-suite.yaml) (which runs [`scripts/run_vllm_tpu_multihost_plan.py`](./scripts/run_vllm_tpu_multihost_plan.py) and [`scripts/run_tpu_benchmark_sweep.py`](./scripts/run_tpu_benchmark_sweep.py)):

```bash
# 1. Stream mistralai/Mistral-Large-3-675B-Instruct-2512-BF16 from Hugging Face to GCS
HF_TOKEN=YOUR_HF_TOKEN python3 scripts/stage_hf_to_gcs.py \
  --repos mistralai/Mistral-Large-3-675B-Instruct-2512-BF16 \
  --bucket YOUR_GCS_BUCKET --prefix models

# 2. Create scripts ConfigMap and Hugging Face Secret
kubectl create configmap v7x16-ml3-scripts \
  --from-file=run_vllm_tpu_multihost_plan.py=scripts/run_vllm_tpu_multihost_plan.py \
  --from-file=plan.json=scripts/plan_mistral_large3_v7x16.json \
  --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py \
  --from-file=apply_ml3_patches.py=scripts/apply_ml3_patches.py
kubectl create secret generic hf-token --from-literal=HF_TOKEN=YOUR_HF_TOKEN

# 3. Launch the BF16 experiment from scripts/plan_mistral_large3_v7x16.json (set SUITE_ONLY=ml3_bf16_tp2_dp16_ep32)
kubectl apply -f k8s/v7x16-ml3-suite.yaml
```

*(Ensure `tokenizer.json` is present at `bench_tokenizer` in [`scripts/plan_mistral_large3_v7x16.json`](./scripts/plan_mistral_large3_v7x16.json) so prompts are tokenized to exact lengths; a standalone LeaderWorkerSet manifest is also provided in [`k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml).)*

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Boot, compilation, and per-TensorCore HBM accounting, full `p10/p50/p90/p99` latency metrics, Eagle speculative decoding status and opt-in patch notes, reference `LeaderWorkerSet` deployment instructions, and BF16-vs-FP8 comparison caveats.
- **[`results/tp2-dp16-ep32/summary.json`](./results/tp2-dp16-ep32/summary.json):** Raw benchmark output JSON for all 11 measured rows and sanity-check metadata.
