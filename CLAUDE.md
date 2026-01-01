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

## Running Multiple Claude Code Instances

This repository supports multiple Claude Code instances running in parallel using git worktrees. This allows working on different features or branches simultaneously without conflicts.

### Creating a Worktree

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

The `worktrees/` directory is in `.gitignore`. Each worktree is a separate working directory with its own branch, allowing independent Claude Code sessions for different tasks.

## Resources

- [GitHub Repository](https://github.com/IQuestLab/IQuest-Coder-V1)
- [Hugging Face Models](https://huggingface.co/IQuestLab)
- [Technical Report](https://github.com/IQuestLab/IQuest-Coder-V1/blob/main/papers/IQuest_Coder_Technical_Report.pdf)
