# vLLM-TorchTPU benchmark - `v7x-16-2x2x4-32cores` / `ml3-bf16-tp2-dp16-ep32-fp8kv`

- Timestamp (UTC): `20261103T070908Z`
- Served model: `mistralai/Mistral-Large-3-675B-Instruct-2512-BF16`
- Sanity check (2.071s): `Mixture-of-Experts (MoE) routing on TPUs dynamically selects a subset of specialized expert models (via a gating network) for each input, optimizing computational efficiency. TPUs accelerate this proc`

| Suite | In (mean actual) | Out | Prefix | Conc | Total out tok/s | Mean stream tok/s/user | TTFT p50/p90/p99 (ms) | TPOT p50/p90 (ms) | Accept rate / len | OK/Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 1 | **35.7** | **39.27** | 2631.4/2644.7/2644.7 | 25.48/25.48 | - | 4/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 8 | **251.23** | **35.48** | 3643.9/4361.7/4362.0 | 28.47/28.48 | - | 16/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 32 | **854.67** | **29.39** | 3966.8/4774.7/5488.8 | 35.36/35.87 | - | 64/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 64 | **1388.41** | **23.86** | 3865.6/5884.9/7714.8 | 42.44/50.43 | - | 128/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 128 | **2185.18** | **18.88** | 5690.7/9821.3/11765.2 | 53.46/57.67 | - | 256/0 |
| concurrency_4096in | 4096 (4097.0) | 1024.0 | 0% | 256 | **3156.49** | **14.16** | 10140.3/16600.0/19508.9 | 64.83/89.08 | - | 512/0 |
| context_sweep | 32768 (32769.0) | 512.0 | 0% | 4 | **50.66** | **34.22** | 26015.0/26250.0/26250.6 | 29.22/31.22 | - | 8/0 |
| context_sweep | 65536 (65537.0) | 512.0 | 0% | 4 | **24.72** | **33.9** | 68344.6/68457.5/68458.5 | 28.36/33.72 | - | 8/0 |
| context_sweep | 131072 (131073.0) | 512.0 | 0% | 4 | **11.44** | **31.31** | 163476.3/163670.5/163673.1 | 30.19/38.86 | - | 8/0 |
| greedy_natural | natural-chat (39.5) | 1024.0 | 0% | 1 | **39.62** | **39.75** | 77.6/202.5/202.5 | 25.17/25.18 | - | 4/0 |
| greedy_natural | natural-chat (38.3) | 1024.0 | 0% | 8 | **277.72** | **34.88** | 142.8/216.5/217.0 | 28.67/28.68 | - | 16/0 |
