# SWE-bench Quick Start Guide

Get started with SWE-bench evaluation in 5 minutes.

## Step 1: Install Dependencies

```bash
cd swe-bench
pip install -r requirements.txt
```

This installs:
- `sweagent` - The SWE-bench evaluation framework
- `pyyaml` - For parsing configuration files
- `openai` - For API compatibility

## Step 2: Start vLLM Server

Make sure your IQuest-Coder model is running:

```bash
# Start the server (from repository root)
iquest-serve start

# Check it's running
iquest-serve status
```

You should see output like:
```
🌐 OpenAI-Compatible API Endpoint
============================================

Base URL: http://10.0.0.1:8000/v1
```

## Step 3: Run Your First Evaluation

### Option A: Single Issue (Fastest)

Test on a single GitHub issue to verify everything works:

```bash
python run_swe_bench.py --issue "django/django#12345"
```

This will:
1. Download the repository
2. Create an isolated Docker environment
3. Run the agent to generate a fix
4. Validate the fix against test cases

### Option B: SWE-bench Lite

Run on the curated 300-instance subset:

```bash
python run_swe_bench.py --dataset lite
```

**Note**: This will take several hours to complete.

### Option C: SWE-bench Verified

Run on the 500-instance verified subset:

```bash
python run_swe_bench.py --dataset verified
```

**Note**: This will take longer (potentially 10+ hours depending on your setup).

## Step 4: View Results

Results are saved to `./swe_bench_results` by default.

```bash
ls -la swe_bench_results/

# View a specific trajectory
cat swe_bench_results/trajectories/django__django-12345.json
```

## Common Options

### Use Thinking Model

For better reasoning on complex issues:

```bash
# First, restart vLLM with Thinking model
iquest-serve stop
iquest-serve start --model IQuestLab/IQuest-Coder-V1-40B-Thinking --thinking

# Then run evaluation
python run_swe_bench.py --dataset lite --model thinking
```

### Custom Endpoint

If using a remote server or SSH tunnel:

```bash
python run_swe_bench.py --dataset lite --api-base http://10.0.0.1:8000/v1
```

### Custom Output Directory

```bash
python run_swe_bench.py --dataset lite --output-dir ./my_results
```

### Parallel Evaluation

Run multiple instances in parallel (requires more memory):

```bash
python run_swe_bench.py --dataset lite --max-workers 4
```

## Troubleshooting

### "SWE-agent is not installed"

```bash
pip install sweagent
```

### "Connection refused"

Make sure vLLM is running:
```bash
iquest-serve status
```

### "Docker daemon not running"

Start Docker Desktop or the Docker daemon.

### "Out of memory"

Reduce parallelism:
```bash
python run_swe_bench.py --dataset lite --max-workers 1
```

## Next Steps

1. Review the full documentation in `README.md`
2. Customize model configurations in `configs/`
3. Analyze results and trajectories
4. Share your findings!

## Expected Performance

Based on IQuest-Coder technical report:
- **SWE-bench Verified**: ~81.4% pass rate
- **SWE-bench Lite**: Strong performance expected
- **Full SWE-bench**: Comprehensive benchmark

Actual results may vary based on configuration and infrastructure.
