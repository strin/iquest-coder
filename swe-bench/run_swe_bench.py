#!/usr/bin/env python3
"""
SWE-bench evaluation script for IQuest-Coder models.

This script provides a convenient interface for running SWE-bench evaluations
using IQuest-Coder models served via vLLM's OpenAI-compatible endpoint.

Prerequisites:
    1. Install SWE-agent: pip install sweagent
    2. Start vLLM server: iquest-serve start
    3. Get endpoint URL: iquest-serve status

Usage:
    # Run on a single GitHub issue
    python run_swe_bench.py --issue "scikit-learn/scikit-learn#12345"

    # Run on SWE-bench Lite dataset
    python run_swe_bench.py --dataset lite

    # Run on SWE-bench Verified dataset
    python run_swe_bench.py --dataset verified

    # Use Thinking model
    python run_swe_bench.py --dataset lite --model thinking

    # Custom endpoint
    python run_swe_bench.py --dataset lite --api-base http://10.0.0.1:8000/v1

    # Specify output directory
    python run_swe_bench.py --dataset lite --output-dir ./results
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_sweagent_installed():
    """Check if SWE-agent is installed."""
    try:
        import sweagent
        return True
    except ImportError:
        return False


def get_config_path(model_type: str) -> Path:
    """Get the path to the model configuration file."""
    script_dir = Path(__file__).parent
    config_dir = script_dir / "configs"

    if model_type == "instruct":
        config_file = config_dir / "iquest_coder_instruct.yaml"
    elif model_type == "thinking":
        config_file = config_dir / "iquest_coder_thinking.yaml"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    return config_file


def update_config_api_base(config_path: Path, api_base: str) -> Path:
    """Update the api_base in the config file and return path to temp config."""
    import yaml
    import tempfile

    # Read the config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update api_base
    if 'model' not in config:
        config['model'] = {}
    if 'model_kwargs' not in config['model']:
        config['model']['model_kwargs'] = {}

    config['model']['model_kwargs']['api_base'] = api_base

    # Write to temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='iquest_coder_config_')
    with os.fdopen(temp_fd, 'w') as f:
        yaml.dump(config, f)

    return Path(temp_path)


def run_sweagent(
    config_path: Path,
    dataset: str = None,
    issue: str = None,
    output_dir: str = "./swe_bench_results",
    max_workers: int = 1,
):
    """Run SWE-agent with the specified configuration."""

    cmd = ["sweagent", "run"]

    # Add config
    cmd.extend(["--config", str(config_path)])

    # Add dataset or issue
    if dataset:
        if dataset == "lite":
            cmd.extend(["--dataset", "princeton-nlp/SWE-bench_Lite"])
        elif dataset == "verified":
            cmd.extend(["--dataset", "princeton-nlp/SWE-bench_Verified"])
        elif dataset == "full":
            cmd.extend(["--dataset", "princeton-nlp/SWE-bench"])
        else:
            # Assume it's a custom dataset name
            cmd.extend(["--dataset", dataset])
    elif issue:
        cmd.extend(["--issue", issue])
    else:
        raise ValueError("Either --dataset or --issue must be specified")

    # Add output directory
    cmd.extend(["--output-dir", output_dir])

    # Add parallelization
    if max_workers > 1:
        cmd.extend(["--num-workers", str(max_workers)])

    print("Running command:")
    print(" ".join(cmd))
    print()

    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running SWE-agent: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation using IQuest-Coder models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on a single issue
  python run_swe_bench.py --issue "scikit-learn/scikit-learn#12345"

  # Run on SWE-bench Lite
  python run_swe_bench.py --dataset lite

  # Run on SWE-bench Verified with Thinking model
  python run_swe_bench.py --dataset verified --model thinking

  # Use custom endpoint
  python run_swe_bench.py --dataset lite --api-base http://10.0.0.1:8000/v1
"""
    )

    parser.add_argument(
        "--model", "-m",
        choices=["instruct", "thinking"],
        default="instruct",
        help="Model variant to use (default: instruct)"
    )

    parser.add_argument(
        "--api-base", "-b",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1)"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset", "-d",
        choices=["lite", "verified", "full"],
        help="SWE-bench dataset to evaluate on"
    )
    group.add_argument(
        "--issue", "-i",
        help="Single GitHub issue to evaluate (format: owner/repo#number)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default="./swe_bench_results",
        help="Output directory for results (default: ./swe_bench_results)"
    )

    parser.add_argument(
        "--max-workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )

    args = parser.parse_args()

    # Check if SWE-agent is installed
    if not check_sweagent_installed():
        print("❌ SWE-agent is not installed.")
        print()
        print("Install it with one of the following commands:")
        print("  pip install sweagent")
        print("  uv pip install sweagent")
        print()
        sys.exit(1)

    # Get config path
    try:
        config_path = get_config_path(args.model)
        print(f"Using config: {config_path}")
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Update config with api_base
    print(f"API endpoint: {args.api_base}")

    try:
        # Check if PyYAML is installed
        import yaml
        temp_config_path = update_config_api_base(config_path, args.api_base)
        print(f"Created temporary config: {temp_config_path}")
        config_to_use = temp_config_path
    except ImportError:
        print("⚠️  PyYAML not installed, using config as-is.")
        print(f"   Make sure to update api_base in {config_path}")
        config_to_use = config_path

    print()

    # Run SWE-agent
    run_sweagent(
        config_path=config_to_use,
        dataset=args.dataset,
        issue=args.issue,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )

    # Clean up temp config
    if config_to_use != config_path:
        try:
            config_to_use.unlink()
        except:
            pass

    print()
    print("✅ Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
