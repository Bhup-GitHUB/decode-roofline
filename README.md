# gpu-memory-wall

When an LLM generates one token, the GPU has to stream every weight out of HBM.
At batch 1 that read takes far longer than the math, so the tensor cores sit
idle and decode speed is set by memory bandwidth, not by FLOPS. Batching
reuses each weight read across many sequences, so throughput climbs until the
compute ceiling finally becomes the limit.

`wall` puts numbers on that: the memory ceiling, the compute ceiling, where
they cross, and how a real vLLM server compares against them.

> **The numbers `wall` prints are theoretical upper bounds, not forecasts.**
> A real server will always land below them. That gap is what `wall report`
> measures.

## Formulas

| Quantity | Formula |
|---|---|
| weights_gb | `params * bytes_per_weight / 1e9` (fp16 = 2, fp8 = 1, int8 = 1, int4 = 0.5) |
| fits_in_vram | `weights_gb < vram_gb` |
| memory_ceiling | `bandwidth_gbs / weights_gb` tokens/sec per sequence |
| flops_per_token | `2 * params` |
| compute_ceiling | `tflops * 1e12 / flops_per_token` tokens/sec total |
| tokens_per_sec(B) | `min(memory_ceiling * B, compute_ceiling)` |
| idle_compute_ratio | `compute_ceiling / memory_ceiling` at batch 1 |
| crossover_batch | `ceil(idle_compute_ratio)`, the first batch that is compute bound |
| kv_cache_gb | `2 * layers * hidden * context * batch * 2 bytes / 1e9` |

Assumptions behind every number:

- all weights are read from HBM exactly once per decode step
- KV cache traffic and all software overheads (kernels, scheduling, sampling, networking) are ignored
- the GPU reaches its rated peak bandwidth and dense FP16 TFLOPS
- the compute ceiling uses dense FP16 TFLOPS for every precision
- the KV cache estimate assumes full multi-head attention in fp16, so it overestimates models with grouped-query attention (Llama 3.1 8B uses 8 KV heads out of 32, so its real KV cache is 4x smaller)

## Built-in GPUs

| Name | VRAM GB | Bandwidth GB/s | Dense FP16 TFLOPS |
|---|---|---|---|
| h100-sxm | 80 | 3350 | 989 |
| h100-pcie | 80 | 2000 | 989 |
| h200-sxm | 141 | 4800 | 989 |
| a100-80 | 80 | 1935 | 312 |

Anything else works with `--gpu custom --vram 48 --bandwidth 864 --tflops 181`.

## Running

Requires Go 1.22 or newer. There are no dependencies outside the standard library.

```sh
make run                                # h100-sxm, 8e9 params, fp16, batch 1
make run PARAMS=70e9 PRECISION=fp8 BATCH=16
make sweep                              # batch 1..256, writes sweep.csv
make compare                            # one model across GPUs and precisions
make test
```

Or call the binary directly:

```sh
go run ./cmd/wall --gpu h100-sxm --params 8e9 --precision fp16 --batch 8 \
    --context 8192 --layers 32 --hidden 4096
go run ./cmd/wall --gpu h200-sxm --params 70e9 --precision fp8 --sweep
go run ./cmd/wall compare --params 8e9 --gpu h100-sxm
```

### Sample sweep

`make sweep` for an 8B model in fp16 on an H100 SXM:

```
BATCH  TOK/S   PER SEQ  LIMIT
1      209.4   209.4    memory
2      418.8   209.4    memory
4      837.5   209.4    memory
8      1,675   209.4    memory
16     3,350   209.4    memory
32     6,700   209.4    memory
64     13,400  209.4    memory
128    26,800  209.4    memory
256    53,600  209.4    memory

crossover at batch 296, wrote sweep.csv
```

At batch 1 the GPU could do about 295x more math than the memory system lets
it. Shrink the weights to int4 on an H200 and the wall moves: the crossover
drops to batch 52, and everything past that is compute bound.

## Measuring a real vLLM server

`bench/bench_vllm.py` needs only the Python standard library and a running
vLLM OpenAI-compatible server.

```sh
vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype float16

python3 bench/bench_vllm.py \
    --base-url http://localhost:8000/v1 \
    --model meta-llama/Llama-3.1-8B-Instruct
```

What it does:

- sends a fixed prompt with fixed `max_tokens` (default 256), `temperature 0` and `ignore_eos`, so every request generates the same number of tokens
- streams every response so each token is timed as it arrives
- runs one warmup request and throws the result away
- single stream: `(tokens - 1) / (last_token_time - first_token_time)`, which leaves out time to first token
- concurrency 1, 8 and 32: total output tokens divided by wall time across all requests in flight
- repeats every setting 3 times, keeps the median, and writes `measured.json`

Then compare it with the prediction:

```sh
make report PARAMS=8e9 PRECISION=fp16 GPU=h100-sxm
```

```
CONCURRENCY      PREDICTED CEIL  LIMIT   MEASURED  % OF CEIL
1 (decode only)  209.4           memory  ...       ...
1                209.4           memory  ...       ...
8                1,675           memory  ...       ...
32               6,700           memory  ...       ...
```

The percentage column shows how close the server gets to the bandwidth limit.
Expect it to drop as concurrency grows, because KV cache reads, which this
model ignores, grow with every extra sequence.

## Layout

```
cmd/wall/            CLI: summary, --sweep, compare, report
internal/gpu/        built-in GPU table and custom GPUs
internal/model/      weights size, FLOPs per token, KV cache estimate
internal/ceiling/    memory and compute ceilings, crossover, sweep
internal/report/     tables, CSV, predicted vs measured
bench/bench_vllm.py  vLLM throughput benchmark
```

## Sources

- NVIDIA H100 Tensor Core GPU datasheet: https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA H200 Tensor Core GPU: https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA A100 Tensor Core GPU: https://www.nvidia.com/en-us/data-center/a100/
- Meta Llama 3.1 model card: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

NVIDIA lists FP16 tensor throughput with sparsity, so the dense numbers used
here are half of the headline figures (1,979 becomes 989 on H100).
