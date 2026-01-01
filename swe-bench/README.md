# SWE-bench Evaluation for IQuest-Coder

This directory contains configuration and scripts for evaluating IQuest-Coder models on the SWE-bench benchmark using the OpenAI-compatible vLLM endpoint.

## What is SWE-bench?

[SWE-bench](https://www.swebench.com) is a benchmark for evaluating large language models on real-world software engineering tasks. It consists of GitHub issues from popular Python repositories that require code changes to resolve.

## Prerequisites

### 1. Install SWE-agent

SWE-agent is the framework that runs the evaluation:

```bash
# Using pip
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

### 2. Start the vLLM Server

Make sure your IQuest-Coder model is running via vLLM:

```bash
# For Instruct model
iquest-serve start

# For Thinking model
iquest-serve start --model IQuestLab/IQuest-Coder-V1-40B-Thinking --thinking

# Check status and get endpoint
iquest-serve status
```

### 3. Docker (Required)

SWE-agent uses Docker to create isolated environments for each evaluation. Make sure Docker is installed and running:

```bash
docker --version
```

## Quick Start

### Evaluate on a Single Issue

Test with a single GitHub issue first:

```bash
python run_swe_bench.py --issue "scikit-learn/scikit-learn#12345"
```

### Evaluate on SWE-bench Datasets

#### SWE-bench Lite (300 instances)

The Lite version is a curated subset of 300 challenging instances:

```bash
python run_swe_bench.py --dataset lite
```

#### SWE-bench Verified (500 instances)

Human-validated subset with high-quality test cases:

```bash
python run_swe_bench.py --dataset verified
```

#### Full SWE-bench (2,294 instances)

The complete benchmark:

```bash
python run_swe_bench.py --dataset full
```

## Configuration

### Model Variants

#### Instruct Model (Default)

Optimized for general coding assistance:

```bash
python run_swe_bench.py --dataset lite --model instruct
```

Configuration file: `configs/iquest_coder_instruct.yaml`

#### Thinking Model

Uses reasoning-driven approach for complex problems:

```bash
python run_swe_bench.py --dataset lite --model thinking
```

Configuration file: `configs/iquest_coder_thinking.yaml`

### Custom Endpoint

If your vLLM server is running on a different host:

```bash
# Remote server
python run_swe_bench.py --dataset lite --api-base http://10.0.0.1:8000/v1

# SSH tunnel
ssh -L 8000:<REMOTE_IP>:8000 -N user@remote-host
python run_swe_bench.py --dataset lite --api-base http://localhost:8000/v1
```

### Parallel Evaluation

Run multiple evaluations in parallel (requires more resources):

```bash
python run_swe_bench.py --dataset lite --max-workers 4
```

## Output and Results

Results are saved to `./swe_bench_results` by default. You can specify a custom directory:

```bash
python run_swe_bench.py --dataset lite --output-dir ./my_results
```

The output directory contains:
- **Trajectory files**: Step-by-step agent actions
- **Patches**: Generated code changes
- **Logs**: Detailed execution logs
- **Results summary**: Pass/fail for each instance

## Configuration Files

### Model Configuration

The YAML configuration files in `configs/` specify:
- Model name and API endpoint
- Sampling parameters (temperature, top_p)
- Token limits and context window
- Agent parsing strategy

You can customize these files or create new ones for different setups.

### Key Parameters

```yaml
model:
  model_name: "openai/IQuestLab/IQuest-Coder-V1-40B-Instruct"
  model_kwargs:
    custom_llm_provider: "openai"
    api_base: "http://localhost:8000/v1"
    temperature: 0.6  # Recommended for IQuest-Coder
    top_p: 0.85       # Recommended for IQuest-Coder
```

## Advanced Usage

### Direct SWE-agent CLI

You can also use SWE-agent directly with our config files:

```bash
sweagent run \
  --config configs/iquest_coder_instruct.yaml \
  --dataset princeton-nlp/SWE-bench_Lite \
  --output-dir ./results
```

### Custom Datasets

Evaluate on a custom dataset:

```bash
python run_swe_bench.py --dataset "your-org/your-dataset"
```

### Modify Agent Behavior

Edit the config files to customize:
- Number of API calls per instance (`per_instance_call_limit`)
- Maximum tokens per completion (`max_tokens`)
- Parsing strategy (`agent.tools.parse_function.type`)

## Troubleshooting

### Connection Errors

```
❌ Connection refused. Is the server running?
```

**Solution**: Make sure vLLM is running (`iquest-serve status`)

### Docker Errors

```
❌ Docker daemon not running
```

**Solution**: Start Docker Desktop or the Docker daemon

### Memory Issues

If you encounter OOM errors, try:
1. Reduce `max_workers` to 1
2. Reduce `max_tokens` in the config
3. Use a smaller dataset (Lite instead of Full)

### Rate Limiting

For self-hosted models, rate limiting should not be an issue. If you see rate limit errors, check your vLLM server logs:

```bash
iquest-serve logs -f
```

## Performance Expectations

Based on the IQuest-Coder technical report:

- **SWE-bench Verified**: ~81.4% (state-of-the-art)
- **SWE-bench Lite**: High performance expected
- **Full SWE-bench**: Comprehensive evaluation

Note: Actual results may vary based on:
- Model variant (Instruct vs Thinking)
- Sampling parameters
- Agent configuration
- Infrastructure (GPU, network latency)

## Resources

- [SWE-bench Website](https://www.swebench.com)
- [SWE-agent Documentation](https://swe-agent.com)
- [SWE-agent GitHub](https://github.com/SWE-agent/SWE-agent)
- [IQuest-Coder Technical Report](../knowledge/IQuest_Coder_Technical_Report.pdf)

## Citation

If you use this setup for research, please cite both SWE-bench and IQuest-Coder:

```bibtex
@inproceedings{jimenez2024swebench,
  title={SWE-bench: Can Language Models Resolve Real-world Github Issues?},
  author={Jimenez, Carlos E and Yang, John and Wettig, Alexander and Yao, Shunyu and Pei, Kexin and Press, Ofir and Narasimhan, Karthik},
  booktitle={ICLR},
  year={2024}
}

@article{iquest-coder-v1-2025,
  title={IQuest-Coder-V1 Technical Report},
  author={IQuest Coder Team},
  year={2025}
}
```
