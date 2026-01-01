# IQuest-Coder-V1 Model Family

> The goal of this repository is to create a scaffold for iQuest Coder and test out the model on real coding tasks.

[![🤗 Hugging Face - Base Stage1](https://img.shields.io/badge/🤗%20Hugging%20Face-Base%20Stage1-yellow)](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Base-Stage1)
[![🤗 Hugging Face - Base](https://img.shields.io/badge/🤗%20Hugging%20Face-Base-yellow)](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Base)
[![🤗 Hugging Face - Instruct](https://img.shields.io/badge/🤗%20Hugging%20Face-Instruct-yellow)](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Instruct)
[![🤗 Hugging Face - Loop Instruct](https://img.shields.io/badge/🤗%20Hugging%20Face-Loop%20Instruct-yellow)](https://huggingface.co/IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct)

---

## Sampling Parameters

For the IQuest-Coder-V1-Instruct: We suggest using **Temperature=0.6**, **TopP=0.85**, **TopK=20**.

---

## IQuest-Coder-V1 Highlights

IQuest-Coder-V1 is a new family of code large language models (LLMs) designed to advance autonomous software engineering and code intelligence. Built on the innovative **code-flow multi-stage training paradigm**, IQuest-Coder-V1 captures the dynamic evolution of software logic, delivering state-of-the-art performance across critical dimensions:

- **State-of-the-Art Performance**: Achieves leading results on SWE-Bench Verified (81.4%), BigCodeBench (49.9%), LiveCodeBench v6 (81.1%), and other major coding benchmarks, surpassing competitive models across agentic software engineering, competitive programming, and complex tool use.

- **Code-Flow Training Paradigm**: Moving beyond static code representations, our models learn from repository evolution patterns, commit transitions, and dynamic code transformations to understand real-world software development processes.

- **Dual Specialization Paths**: Bifurcated post-training delivers two specialized variants—**Thinking models** (utilizing reasoning-driven RL for complex problem-solving) and **Instruct models** (optimized for general coding assistance and instruction-following).

- **Efficient Architecture**: The IQuest-Coder-V1-Loop variant introduces a recurrent mechanism that optimizes the trade-off between model capacity and deployment footprint.

- **Native Long Context**: All models natively support up to **128K tokens** without requiring additional scaling techniques.

---

## Model Overview

The IQuest-Coder-V1 series includes models ranging from **7B to 40B parameters**, with both standard and Loop variants:

### Architecture Features

| Feature | Specification |
|---------|--------------|
| Attention Mechanism | Grouped Query Attention (GQA) for efficient inference |
| Context Length | Native 128K context length support |
| Vocabulary Size | 76,800 tokens |
| Loop Variants | Recurrent transformer design with shared parameters across two iterations |

For more details, please refer to the [Technical Report](https://github.com/IQuestLab/IQuest-Coder-V1/blob/main/papers/IQuest_Coder_Technical_Report.pdf) and [GitHub](https://github.com/IQuestLab/IQuest-Coder-V1).

---

## Quickstart

IQuest-Coder-V1 uses custom modeling code via Hugging Face's `auto_map` feature. We recommend using `transformers>=4.52.4`.

### Basic Usage with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "IQuestLab/IQuest-Coder-V1-40B-Instruct"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Prepare the input
prompt = "Write a Python function to calculate the Fibonacci sequence using dynamic programming."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate response
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=8192
)
generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)
print(response)
```

### Using Thinking Models

For complex reasoning tasks, use the Thinking variant:

```python
model_name = "IQuestLab/IQuest-Coder-V1-40B-Thinking"

# The Thinking model includes explicit reasoning traces
# Use similar code as above, but expect longer, more detailed responses
# with step-by-step problem decomposition
```

### Deployment with vLLM

For production deployment, you can use vLLM to create an OpenAI-compatible API endpoint. Please refer to the [vLLM PR](https://github.com/vllm-project/vllm/pull/31575/files) for implementation details.

```bash
vllm serve IQuestLab/IQuest-Coder-V1-40B-Instruct --tensor-parallel-size 8
```

For Thinking models with reasoning support:

```bash
vllm serve IQuestLab/IQuest-Coder-V1-40B-Thinking --reasoning-parser qwen3 --tensor-parallel-size 8
```

### SLURM Cluster Deployment (IQuest Lab)

For deployment on SLURM clusters, we provide a CLI tool that handles job submission, monitoring, and exposes an OpenAI-compatible API.

#### Installation

```bash
# Using pip
pip install -e .

# Or using uv
uv pip install -e .
```

#### First-Time Setup

The first time you run `iquest-serve start`, it will automatically set up vLLM on the SLURM cluster if it's not already installed. You can also run setup manually:

```bash
# Install vLLM on the SLURM cluster (optional - runs automatically on first start)
iquest-serve setup

# If you need to reinstall vLLM
iquest-serve setup --reinstall
```

The setup process will:
- Create a virtual environment on the SLURM cluster
- Install vLLM and its dependencies
- Verify the installation

#### Usage

```bash
# Start serving with default settings (IQuest-Coder-V1-40B-Instruct, 8 GPUs)
# This will automatically run setup if vLLM is not installed
iquest-serve start

# Start with specific model and settings
iquest-serve start --model IQuestLab/IQuest-Coder-V1-40B-Thinking --thinking

# Start with custom configuration
iquest-serve start --model IQuestLab/IQuest-Coder-V1-40B-Instruct \
    --tensor-parallel 8 \
    --port 8000 \
    --gpus 8

# Check status and get OpenAI-compatible API endpoint
iquest-serve status

# View server logs
iquest-serve logs
iquest-serve logs -f  # Follow logs in real-time

# Stop the server
iquest-serve stop
```

#### API Access

After starting the server, use `iquest-serve status` to get the API endpoint. Example output:

```
🌐 OpenAI-Compatible API Endpoint
============================================

Base URL: http://<NODE_IP>:8000/v1

# List available models
curl http://<NODE_IP>:8000/v1/models

# Chat completion
curl http://<NODE_IP>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "IQuestLab/IQuest-Coder-V1-40B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.6,
    "top_p": 0.85
  }'
```

**Python with OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://<NODE_IP>:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="IQuestLab/IQuest-Coder-V1-40B-Instruct",
    messages=[{"role": "user", "content": "Write a hello world in Python"}],
    temperature=0.6,
    top_p=0.85
)
print(response.choices[0].message.content)
```

**SSH Tunnel for Local Access:**

```bash
# Create SSH tunnel (run locally)
ssh -L 8000:<NODE_IP>:8000 -N <USER>@204.12.169.136

# Then access locally
curl http://localhost:8000/v1/models
```

#### OpenHands Integration

IQuest-Coder integrates with [OpenHands](https://openhands.dev), an open-source AI-powered coding assistant. The `iquest-serve code` command sets up the SSH tunnel and launches OpenHands.

**Prerequisites:**

```bash
# Install OpenHands (recommended)
uv tool install openhands --python 3.12

# or with pip
pip install openhands
```

**Usage:**

```bash
# Start OpenHands in CLI/TUI mode (default)
iquest-serve code

# Start OpenHands in GUI mode (accessible at http://localhost:3000)
iquest-serve code --mode serve

# Start with current directory mounted
iquest-serve code --mode serve --mount-cwd
```

**Manual LLM Configuration:**

When OpenHands starts, you'll need to configure the LLM settings manually:

1. Click "see advanced settings" in the Settings dialog
2. Enable the "Advanced" toggle
3. Set the following values:
   - **Custom Model**: `openai/IQuestLab/IQuest-Coder-V1-40B-Instruct`
   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: `none`
4. Click "Save Settings"

The command automatically:
1. Retrieves the vLLM endpoint from the running SLURM job
2. Sets up an SSH tunnel to the cluster (local port 8000)
3. Displays the configuration values to use in OpenHands
4. Launches OpenHands

---

## Running Multiple Claude Code Instances

This repository is configured to support multiple Claude Code instances running in parallel using git worktrees. This allows you to work on different features or branches simultaneously without conflicts.

### Setup

The `worktrees/` directory is already configured and added to `.gitignore`. To create a new worktree:

```bash
# Create a new worktree with a new feature branch
cd /Users/timshi/iquest-coder
git worktree add worktrees/new-feature -b new-feature
cd worktrees/new-feature
claude
```

### Managing Worktrees

```bash
# List all worktrees
git worktree list

# Remove a worktree when done
git worktree remove worktrees/feature-name

# Clean up stale worktree data
git worktree prune
```

Each worktree is a separate working directory with its own branch, allowing you to run independent Claude Code sessions for different tasks.

---

## Limitations

- **Reasoning vs. Efficiency Trade-off**: Thinking models provide superior reasoning but generate longer responses; Instruct models are more efficient for straightforward tasks.

- **Code Execution**: Models generate code but do not execute it; always validate outputs in sandboxed environments.

- **Domain Specificity**: While trained on diverse codebases, performance may vary on highly specialized or proprietary frameworks.

- **Factuality**: Models may generate plausible but incorrect code; verify critical implementations thoroughly.

---

## Citation

If you find our work helpful, please cite:

```bibtex
@article{iquest-coder-v1-2025,
    title={IQuest-Coder-V1 Technical Report},
    author={IQuest Coder Team},
    url={https://github.com/IQuestLab/IQuest-Coder-V1/blob/main/papers/IQuest_Coder_Technical_Report.pdf},
    year={2025}
}

@article{codescaling,
    title={Scaling Laws for Code: Every Programming Language Matters},
    author={Yang, Jian and Guo, Shawn and Jing, Lin and Zhang, Wei and Liu, Aishan and Hao, Chuan and Li, Zhoujun and Zhao, Wayne Xin and Liu, Xianglong and Lv, Weifeng and others},
    journal={arXiv preprint arXiv:2512.13472},
    year={2025}
}

@article{close_the_loop,
    title={Close the Loop: Synthesizing Infinite Tool-Use Data via Multi-Agent Role-Playing},
    author={Yuwen Li, Wei Zhang, Zelong Huang, Mason Yang, Jiajun Wu, Shawn Guo, Huahao Hu, Lingyi Sun, Jian Yang, Mingjie Tang, Byran Dai},
    journal={arXiv preprint arXiv:2512.23611},
    year={2025}
}

@article{loopcoder,
    title={LoopCoder: Scaling Code Intelligence via Looped Language Models},
    author={Jian Yang, Wei Zhang, Shawn Guo, Yizhi Li, Lin Jing, Zhengmao Ye, Shark Liu, Yuyang Song, Jiajun Wu, Che Liu, T. Zheng, Siwei Wu, L. Liao, X. Ma, Chuan Hao, Ran Tao, Yan Xing, Jianzhou Wang, Mingjie Tang, Aishan Liu, Zhoujun Li, Xianglong Liu, Weifeng Lv1, Bryan Dai},
    year={2025}
}

@article{swe_compress,
    title={Context as a Tool: Context Management for Long-Horizon SWE-Agents},
    author={hukai Liu, Jian Yang, Bo Jiang, Yizhi Li, Jinyang Guo, Xianglong Liu, Bryan Dai},
    journal={arXiv preprint arXiv:2512.22087},
    year={2025}
}
```

---

## License

Please refer to the [official repository](https://github.com/IQuestLab/IQuest-Coder-V1) for license information.
