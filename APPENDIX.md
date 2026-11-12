# Appendix: Boot Profiling, Full Latency Percentiles, and Engineering Notes (Mistral Large 3 BF16 on TPU v7x)

This document records server boot and XLA compilation timings, per-TensorCore HBM and KV-cache capacity math, complete percentile tables (`p10/p50/p90/p99`), unsupported/experimental features (Eagle speculative decoding and multimodal vision tower), and methodology caveats for `mistralai/Mistral-Large-3-675B-Instruct-2512-BF16` on Google Cloud TPU `v7x-16`. For headline tables and reproduction commands, see [`README.md`](./README.md).

---

## 1. Boot Timing, EP-Filtered Weight Streaming, and HBM Accounting

Measured on `v7x-16` (`2x2x4`, 4 hosts × 4 chips = 32 TensorCores) under `TP=2, DP=16, EP=32` (`--gpu-memory-utilization=0.90`, `--max-model-len=132096`, `--max-num-seqs=64`, `--max-num-batched-tokens=1024`, `--compilation-config='{"compile_sizes":[4,16,64,256,512,1024]}'`):

| Metric | Measured BF16 Value (`TP=2, DP=16, EP=32`) | Native FP8 Reference (`TP=2, DP=16, EP=32`) | Notes |
| :--- | ---: | ---: | :--- |
| Checkpoint Size on GCS | `1,352.0 GB` (`272` shards) | `681.5 GB` | Staged via [`scripts/stage_hf_to_gcs.py`](./scripts/stage_hf_to_gcs.py) |
| Per-TensorCore GCS Stream (`--enable-ep-weight-filter`) | `85.0 GB` | `47.9 GB` | Each core loads only its `4` of `128` routed experts + `TP=2` shared/MLA shards via `runai_streamer` |
| Usable HBM3e per TensorCore (`vllm` reported) | `94.74 GiB` (`85.26 GiB` @ `0.90` cap) | `94.74 GiB` (`83.37 GiB` @ `0.88` cap) | `3,031.68 GiB` usable across `32` TensorCores (`3,072 GB` physical HBM3e) |
| Weight HBM per TensorCore | `57.38 GiB` | `31.11 GiB` | Unquantized BF16 grouped-matmul MoE weights vs. per-channel FP8 MoE weights |
| FP8 MLA KV-Cache Tokens per DP Engine | `765,952` tokens | `1,436,672` tokens | `5.80×` max concurrency per DP engine at `132,096` tokens (`BF16`) |
| Total Slice FP8 MLA KV-Cache Capacity (`× 16` DP) | **`12,255,232` (`12.26M`) tokens** | `22,986,752` (`22.99M`) tokens | Supports `~92.8` concurrent `128K` requests across the 16-chip slice |
| End-to-End Server Ready Time (`vllm serve` → `/v1/models`) | `880 s` | `980 s` | Includes Ray cluster init, GCS streaming, XLA compilation, and KV allocation |
| Engine Init / XLA Compilation Time (max across DP ranks) | `653.2 s` / `523.4 s` | `782.6 s` / `597.7 s` | Reported by vLLM `init engine (profile, create kv cache, warmup model)` |
| Post-Boot Sanity Check (`20` prompt + `64` completion tokens) | `2.071 s` | — | Verified coherent output on `/v1/chat/completions` (`T=0.0`) before sweep |

---

## 2. Complete Benchmark Table (`p10 / p50 / p90 / p99` Percentiles)

Raw JSON: [`results/tp2-dp16-ep32/summary.json`](./results/tp2-dp16-ep32/summary.json)

| Suite | Input (Mean Actual) | Output | Temp | Conc | Prompts (`OK/Fail`) | Wall (s) | Output TPS | Total TPS | Stream TPS Mean / p50 / p10 | TTFT Mean / p50 / p90 / p99 (ms) | TPOT Mean / p50 / p90 / p99 (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `concurrency_4096in` | `4,096` (`4,097.0`) | 1,024 | 0.6 | 1 | `4 / 0` | 114.73 | 35.70 | 178.55 | 39.27 / 39.27 / 39.24 | 2,633.2 / 2,631.4 / 2,644.7 / 2,644.7 | 25.46 / 25.48 / 25.48 / 25.48 |
| `concurrency_4096in` | `4,096` (`4,097.0`) | 1,024 | 0.6 | 8 | `16 / 0` | 65.21 | 251.23 | 1,256.41 | 35.48 / 35.57 / 35.12 | 3,746.7 / 3,643.9 / 4,361.7 / 4,362.0 | 28.19 / 28.47 / 28.48 / 29.64 |
| `concurrency_4096in` | `4,096` (`4,097.0`) | 1,024 | 0.6 | 32 | `64 / 0` | 76.68 | 854.67 | 4,274.19 | 29.39 / 28.28 / 27.88 | 3,330.8 / 3,966.8 / 4,774.7 / 5,488.8 | 34.13 / 35.36 / 35.87 / 36.00 |
| `concurrency_4096in` | `4,096` (`4,097.0`) | 1,024 | 0.6 | 64 | `128 / 0` | 94.40 | 1,388.41 | 6,943.43 | 23.86 / 26.46 / 19.83 | 3,450.9 / 3,865.6 / 5,884.9 / 7,714.8 | 42.59 / 42.44 / 50.43 / 51.41 |
| `concurrency_4096in` | `4,096` (`4,097.0`) | 1,024 | 0.6 | 128 | `256 / 0` | 119.96 | 2,185.18 | 10,928.02 | 18.88 / 18.70 / 17.34 | 5,197.0 / 5,690.7 / 9,821.3 / 11,765.2 | 53.19 / 53.46 / 57.67 / 60.54 |
| `concurrency_4096in` | `4,096` (`4,097.0`) | 1,024 | 0.6 | 256 | `512 / 0` | 166.10 | **3,156.49** | **15,785.51** | 14.16 / 15.43 / 11.23 | 8,762.4 / 10,140.3 / 16,600.0 / 19,508.9 | 71.90 / 64.83 / 89.08 / 94.05 |
| `context_sweep` | `32,768` (`32,769.0`) | 512 | 0.6 | 4 | `8 / 0` | 80.85 | 50.66 | 3,293.27 | 34.22 / 34.91 / 32.03 | 25,410.6 / 26,015.0 / 26,250.0 / 26,250.6 | 29.33 / 29.22 / 31.22 / 33.05 |
| `context_sweep` | `65,536` (`65,537.0`) | 512 | 0.6 | 4 | `8 / 0` | 165.73 | 24.72 | 3,188.31 | 33.90 / 35.26 / 29.66 | 67,687.9 / 68,344.6 / 68,457.5 / 68,458.5 | 29.67 / 28.36 / 33.72 / 33.85 |
| `context_sweep` | `131,072` (`131,073.0`) | 512 | 0.6 | 4 | `8 / 0` | 357.95 | **11.44** | **2,940.83** | 31.31 / 33.13 / 25.73 | 162,438.3 / 163,476.3 / 163,670.5 / 163,673.1 | 32.33 / 30.19 / 38.86 / 38.97 |
| `greedy_natural` | `natural-chat` (`39.5`) | 1,024 | 0.0 | 1 | `4 / 0` | 103.39 | **39.62** | 41.15 | **39.75** / 39.77 / 39.71 | 108.4 / 77.6 / 202.5 / 202.5 | 25.16 / **25.17** / 25.18 / 25.18 |
| `greedy_natural` | `natural-chat` (`38.3`) | 1,024 | 0.0 | 8 | `16 / 0` | 59.00 | 277.72 | 288.11 | 34.88 / 34.87 / 34.87 | 154.4 / 142.8 / 216.5 / 217.0 | 28.67 / 28.67 / 28.68 / 28.70 |

---

## 3. Runtime Patching, Unsupported Features, and What Did Not Run

1. **BF16 Checkpoint Needs No Weight-Loader Patch:** Unlike the `compressed-tensors` FP8 checkpoint (`mistralai/Mistral-Large-3-675B-Instruct-2512`), the BF16 checkpoint has no quantization config and runs out of the box on `vllm-torchtpu v0.32.0` via the standard BF16 grouped-matmul MoE path. Although [`k8s/v7x16-ml3-suite.yaml`](./k8s/v7x16-ml3-suite.yaml) runs [`scripts/apply_ml3_patches.py`](./scripts/apply_ml3_patches.py) on every host across the shared suite plan, its default `PATCH 1` only touches `compressed_tensors_moe.py`, which is not exercised by BF16 weights.
2. **Eagle Speculative Decoding (`Mistral-Large-3-675B-Instruct-2512-Eagle`) — Not Yet Functional on `vllm-torchtpu v0.32.0`:**
   - Upstream vLLM `0.32.0`'s `EagleMistralLarge3Model` (`mistral_large_3_eagle.py`, 1 MoE layer with per-tensor FP8 weights and static activation scales) fails during initialization and weight loading on `vllm-torchtpu` due to missing `is_fused_shared_expert_enabled` / `use_aux_hidden_state` / `fc_input_size` attributes, upstream CUDA `Fp8Config` dispatch (`torch.ops._C.static_scaled_fp8_quant`), missing `combine_hidden_states()`, and double transposition of per-tensor FP8 `kv_b_proj` inside `mla_attention_op.py`.
   - Opt-in patches 2–5 (`ML3_EAGLE_PATCHES=1`, default off) in [`scripts/apply_ml3_patches.py`](./scripts/apply_ml3_patches.py) document work-in-progress fixes for those four issues, but Eagle speculative decoding does not yet run end-to-end on this stack and was not enabled for any benchmark in this repository.
3. **Text-Only Serving (`--language-model-only`):** Skips loading and profiling the 48-layer (`~2.5B`-parameter) Pixtral vision encoder to maximize HBM available for the 675B BF16 text backbone and `132,096`-token KV cache; image inputs were not benchmarked.
4. **Higher Tensor Parallelism (`TP=4` / `TP=8`) Not Required for BF16 Fit:** Because `TP=2, DP=16, EP=32` already fits the `57.38 GiB`/core BF16 shard comfortably inside `94.74 GiB`/core HBM while keeping Tensor Parallelism strictly intra-chip (zero inter-chip TP all-reduces), higher TP degrees were only swept on the FP8 checkpoint.

---

## 4. Reference Standalone Deployment (`LeaderWorkerSet`)

The measured runs used the 4-pod Indexed Job [`k8s/v7x16-ml3-suite.yaml`](./k8s/v7x16-ml3-suite.yaml) driven by [`scripts/run_vllm_tpu_multihost_plan.py`](./scripts/run_vllm_tpu_multihost_plan.py). For long-lived HTTP serving via Kubernetes `LeaderWorkerSet` (matching the exact `vllm serve` flags from the plan), use [`k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml`](./k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml) (and [`docker/Dockerfile`](./docker/Dockerfile) as a reference container build):

```bash
kubectl create configmap tpu-bench-script \
  --from-file=run_tpu_benchmark_sweep.py=scripts/run_tpu_benchmark_sweep.py
kubectl apply -f k8s/v7x-16-ml3-bf16-tp2-dp16-ep32.yaml
```

---

## 5. Methodology & Comparison Caveats

1. **`131,072`-Token Sample Variance (`BF16 +10.9%` vs. `FP8`):** Long-context rows send `8` requests at `c=4` (`--max-num-batched-tokens=1024`). On `131,072 : 512`, the FP8 run had a faster median TTFT (`154,470.0 ms` vs. BF16's `163,476.3 ms`), but two of the 8 FP8 requests experienced tail queueing (`p90 TTFT = 217,482.7 ms`, `p90 TPOT = 143.43 ms`), extending total wall time and lowering FP8 row-level `total_tps_incl_prompt` (`2,650.90` vs. BF16's `2,940.83`, where `p90 TTFT = 163,670.5 ms`). At `32,768` and `65,536` input tokens without tail outliers, BF16 `total_tps_incl_prompt` is `9.5%` and `10.3%` lower than FP8, as expected.
2. **Memory Utilization Flag (`0.90` vs. `0.88`):** BF16 used `--gpu-memory-utilization=0.90` (to allocate sufficient KV cache alongside `57.38 GiB`/core BF16 weights), whereas the FP8 comparison run used `0.88`. Neither run hit KV-cache preemption at the tested concurrencies.
3. **Uncalibrated FP8 KV Cache:** Both BF16 and FP8 runs used `--kv-cache-dtype=fp8` (`e4m3`) without calibrated KV scaling factors.
4. **Prompt Construction & Metrics:** Sampled rows (`concurrency_4096in`, `context_sweep`) call `/v1/completions` (`temperature=0.6`, `ignore_eos=True`) with exact-length tokenized synthetic prompts (`max(2×c, 4)` requests for concurrency rows; `8` requests for context rows). Greedy rows (`greedy_natural`) call `/v1/chat/completions` (`temperature=0.0`) on natural-language essay prompts. Client-side TTFT is measured from HTTP request dispatch to first SSE token chunk and includes request queueing.
