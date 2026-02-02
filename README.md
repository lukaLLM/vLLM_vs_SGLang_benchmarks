# vLLM vs SGLang Benchmark Comparison

A comprehensive benchmark comparison between vLLM and SGLang inference engines using the Qwen3-Coder-30B model.

## 📋 Overview

This repository contains benchmarking results and configurations for comparing vLLM and SGLang performance on the same hardware and model setup. The benchmarks use the ShareGPT dataset with 1000 prompts to evaluate throughput, latency, and resource utilization.

## 🖥️ System Requirements

### Hardware & OS
- **OS**: Linux (tested on Ubuntu 24.04)
  - Kernel: `6.14.0-29-generic #29~24.04.1-Ubuntu`
- **GPU**: NVIDIA GPU with CUDA support
- **Python**: 3.12+
- **Drivers** I use Driver Version: 570.133.20 that supports up to CUDA Version: 12.8 NVIDIA L40S-48C 
### Prerequisites
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker](https://docs.docker.com/engine/install/) with Docker Compose
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- HuggingFace account with access token

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Initialize Python environment with uv
uv init
uv python install 3.12
uv venv --python 3.12

# Activate virtual environment
source .venv/bin/activate

# Install dependencies from pyproject.toml
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```bash
HUGGING_FACE_HUB_TOKEN=your_token_here
```

## 🐳 Running Inference Servers

### SGLang Server

```bash
docker compose -f "docker compose SGLang.yaml" up -d
```

**Server Configuration:**
- Host: `0.0.0.0`
- Port: `30000`
- Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`
- Context Length: `10000`
- GPU Memory Utilization: `0.8`

### vLLM Server

```bash
docker compose -f "docker-compose vLLM.yaml" up -d
```

**Server Configuration:**
- Host: `0.0.0.0`
- Port: `8000`
- Model: `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`
- Max Model Length: `10000`
- GPU Memory Utilization: `0.8`

## 📊 Running Benchmarks

### SGLang Benchmark

Install SGLang for benchmarking tools:

```bash
uv pip install sglang
```

Run the benchmark:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host 0.0.0.0 \
  --port 30000 \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --request-rate inf \
  --warmup-requests 10
```

### vLLM Benchmark

We use here same tool from sglang to benchmark vLLM:

Run the benchmark (using SGLang's bench_serving with vLLM backend):

```bash
python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://0.0.0.0:8000 \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --request-rate inf \
  --warmup-requests 10
```

## 📁 Repository Contents

- `docker-compose vLLM.yaml` - Docker Compose configuration for vLLM
- `docker compose SGLang.yaml` - Docker Compose configuration for SGLang
- `vllm_0128_1000_sharegpt.jsonl` - vLLM benchmark results
- `sglang_0128_1000_sharegpt.jsonl` - SGLang benchmark results
- `Benchmarked vLLM vs SGLang So You Don't Have To - benchmark_comparison_neatly_categorized.csv.csv` - Comparison results
- `notes.md` - Additional notes and observations

## 📝 Benchmark Parameters

- **Model**: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
- **Dataset**: ShareGPT
- **Number of Prompts**: 1000
- **Request Rate**: Infinite (maximum throughput)
- **Warmup Requests**: 10
- **Context Length**: 10000 tokens
- **GPU Memory Utilization**: 0.8 (80%)

## 🔧 Troubleshooting

### GPU Not Detected
Ensure NVIDIA Container Toolkit is properly installed and configured:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Port Already in Use
Change the port mappings in the respective docker-compose files if ports 8000 or 30000 are already occupied.

### Model Download Issues
Ensure your HuggingFace token has access to the model and is properly set in your environment.

## � Resources

- **GitHub Repo**: [vLLM vs SGLang Benchmarks](https://github.com/lukaLLM/vLLM_vs_SGLang_benchmarks)
- **vLLM Documentation**: 
  - [vLLM GitHub](https://github.com/vllm-project/vllm)
  - [vLLM Docs](https://docs.vllm.ai)
- **SGLang Documentation**:
  - [SGLang GitHub](https://github.com/sgl-project/sglang)
  - [SGLang Docs](https://docs.sglang.io/references/learn_more.html)
- **Benchmark Tool**: [SGLang bench_serving](https://docs.sglang.io/developer_guide/bench_serving.html)
- **Paper**: [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

## �📄 License

This benchmark project is for educational and comparison purposes.