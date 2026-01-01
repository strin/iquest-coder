# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

This repository is a scaffold for testing and evaluating **IQuest-Coder-V1**, a family of code large language models (LLMs) designed for autonomous software engineering and code intelligence. The goal is to test the model on real coding tasks.

## Key Model Information

- **Model Family**: IQuest-Coder-V1 (7B to 40B parameters)
- **Variants**: Base, Instruct, Thinking, Loop-Instruct
- **Context Length**: 128K tokens native support
- **Hugging Face**: `IQuestLab/IQuest-Coder-V1-40B-Instruct`

## Recommended Sampling Parameters

When using IQuest-Coder-V1-Instruct:
- Temperature: 0.6
- TopP: 0.85
- TopK: 20

## Repository Structure

```
iquest-coder/
├── README.md          # Project documentation and quickstart guide
├── CLAUDE.md          # This file - guidance for Claude
├── knowledge/         # Reference materials and documentation
│   └── IQuest_Coder_Technical_Report.pdf
├── scripts/           # Utility scripts
│   ├── test_api.py    # API endpoint testing
│   └── iquest_serve.py # SLURM cluster serving CLI
├── swe-bench/         # SWE-bench evaluation framework
│   ├── configs/       # Model configurations for SWE-agent
│   ├── run_swe_bench.py # Evaluation script
│   └── README.md      # SWE-bench documentation
└── ...                # Additional project files
```

## Knowledge Directory

The `knowledge/` directory contains reference materials and documentation for the project. This includes technical reports, papers, and other resources that provide context for working with IQuest-Coder.

## Development Guidelines

1. **Testing Focus**: This repo is meant for testing IQuest-Coder on real coding tasks
2. **Code Validation**: Always validate generated code in sandboxed environments
3. **Model Limitations**: Be aware that the model may generate plausible but incorrect code

## Useful Commands

```bash
# Install transformers (recommended version)
pip install transformers>=4.52.4

# For vLLM deployment
vllm serve IQuestLab/IQuest-Coder-V1-40B-Instruct --tensor-parallel-size 8
```

## Testing

### API Endpoint Testing

Use the test script to verify the vLLM API endpoint is working:

```bash
# Test with default settings (localhost:8000)
python scripts/test_api.py

# Test a specific endpoint
python scripts/test_api.py --base-url http://10.0.0.1:8000/v1

# Quick health check only
python scripts/test_api.py --quick
```

The test script validates:
- Server health and connectivity
- Model listing (`/models` endpoint)
- Chat completions (streaming and non-streaming)
- Legacy completions endpoint
- Code generation capability

### Streaming Server Logs

To monitor vLLM logs and see incoming requests:

```bash
# Stream all logs in real-time
iquest-serve logs -f

# Stream only API request logs (POST, completions, etc.)
iquest-serve logs -f -r

# View stderr logs
iquest-serve logs -e

# Show recent logs (last 100 lines)
iquest-serve logs -n 100
```

### SWE-bench Evaluation

Evaluate IQuest-Coder on the SWE-bench benchmark for real-world software engineering tasks:

```bash
# Install SWE-agent
cd swe-bench
pip install -r requirements.txt

# Ensure vLLM server is running
iquest-serve start

# Run evaluation on SWE-bench Lite (300 instances)
python run_swe_bench.py --dataset lite

# Run on SWE-bench Verified (500 instances)
python run_swe_bench.py --dataset verified

# Use Thinking model for complex reasoning
python run_swe_bench.py --dataset verified --model thinking

# Test on a single GitHub issue
python run_swe_bench.py --issue "scikit-learn/scikit-learn#12345"
```

For detailed documentation, see `swe-bench/README.md`.

## Resources

- [GitHub Repository](https://github.com/IQuestLab/IQuest-Coder-V1)
- [Hugging Face Models](https://huggingface.co/IQuestLab)
- [Technical Report](https://github.com/IQuestLab/IQuest-Coder-V1/blob/main/papers/IQuest_Coder_Technical_Report.pdf)
