#!/usr/bin/env python3
"""
IQuest-Coder SLURM Serving CLI

A command-line tool to start, stop, and monitor the IQuest-Coder model
served via vLLM on a SLURM cluster with OpenAI-compatible API.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Configuration
SLURM_LOGIN_NODE = "204.12.169.136"
DEFAULT_USER = os.environ.get("SLURM_USER", os.environ.get("USER", "timshi"))
DEFAULT_MODEL = "IQuestLab/IQuest-Coder-V1-40B-Instruct"
DEFAULT_TENSOR_PARALLEL = 8
DEFAULT_PORT = 8000
JOB_NAME = "iquest-coder-serve"
WORK_DIR = Path("/home") / DEFAULT_USER / ".iquest-serve"
STATE_FILE = Path.home() / ".iquest-serve-state.json"
DEFAULT_PARTITION = "main"


def run_ssh_command(cmd: str, user: str = DEFAULT_USER, capture: bool = True) -> subprocess.CompletedProcess:
    """Execute a command on the SLURM login node via SSH."""
    ssh_cmd = ["ssh", f"{user}@{SLURM_LOGIN_NODE}", cmd]
    if capture:
        return subprocess.run(ssh_cmd, capture_output=True, text=True)
    else:
        return subprocess.run(ssh_cmd)


def save_state(state: dict):
    """Save the current serving state to a local file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state() -> dict:
    """Load the serving state from the local file."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def clear_state():
    """Clear the serving state."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def generate_slurm_script(
    model: str,
    tensor_parallel: int,
    port: int,
    partition: str,
    num_gpus: int,
    thinking_mode: bool = False,
) -> str:
    """Generate the SLURM batch script for vLLM serving."""
    
    # Determine if this is a thinking model
    reasoning_parser = ""
    if thinking_mode or "Thinking" in model:
        reasoning_parser = "--reasoning-parser qwen3"
    
    script = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus={num_gpus}
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output={WORK_DIR}/logs/serve_%j.out
#SBATCH --error={WORK_DIR}/logs/serve_%j.err

# Print job info
echo "============================================"
echo "IQuest-Coder vLLM Serving"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Model: {model}"
echo "Tensor Parallel Size: {tensor_parallel}"
echo "Port: {port}"
echo "Start Time: $(date)"
echo "============================================"

# Get the node's IP address
NODE_IP=$(hostname -I | awk '{{print $1}}')
echo "Node IP: $NODE_IP"
echo "API Endpoint: http://$NODE_IP:{port}/v1"

# Save endpoint info for status command
echo "$NODE_IP" > {WORK_DIR}/current_endpoint.txt
echo "{port}" >> {WORK_DIR}/current_endpoint.txt

# Set up environment
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS - 1)))
export HF_HOME=/mnt/data/.cache/huggingface

# Create cache directory if needed
mkdir -p $HF_HOME

# Activate virtual environment if exists
if [ -f "{WORK_DIR}/venv/bin/activate" ]; then
    source {WORK_DIR}/venv/bin/activate
fi

# Start vLLM server
echo "Starting vLLM server..."
vllm serve {model} \\
    --tensor-parallel-size {tensor_parallel} \\
    --port {port} \\
    --host 0.0.0.0 \\
    --trust-remote-code \\
    --dtype auto \\
    {reasoning_parser}

echo "Server stopped at $(date)"
"""
    return script


def cmd_start(args):
    """Start the vLLM serving job on SLURM."""
    print(f"🚀 Starting IQuest-Coder serving on SLURM cluster...")
    print(f"   Model: {args.model}")
    print(f"   Tensor Parallel Size: {args.tensor_parallel}")
    print(f"   Port: {args.port}")
    print(f"   Partition: {args.partition}")
    print(f"   GPUs: {args.gpus}")
    print()

    # Check for existing job
    state = load_state()
    if state.get("job_id"):
        print(f"⚠️  Found existing job (ID: {state['job_id']}). Checking status...")
        result = run_ssh_command(f"squeue -j {state['job_id']} -h -o '%T' 2>/dev/null", args.user)
        if result.stdout.strip():
            print(f"❌ A serving job is already running (Job ID: {state['job_id']}, Status: {result.stdout.strip()})")
            print("   Use 'iquest-serve stop' to stop it first, or 'iquest-serve status' for details.")
            return 1
        else:
            print("   Previous job no longer exists. Proceeding with new job...")
            clear_state()

    # Create work directories on remote
    print("📁 Creating work directories...")
    run_ssh_command(f"mkdir -p {WORK_DIR}/logs {WORK_DIR}/scripts", args.user)

    # Generate and upload SLURM script
    script_content = generate_slurm_script(
        model=args.model,
        tensor_parallel=args.tensor_parallel,
        port=args.port,
        partition=args.partition,
        num_gpus=args.gpus,
        thinking_mode=args.thinking,
    )

    script_path = f"{WORK_DIR}/scripts/serve.sbatch"
    
    # Write script via SSH
    print("📝 Uploading SLURM script...")
    escaped_content = script_content.replace("'", "'\\''")
    result = run_ssh_command(f"cat > {script_path} << 'SLURM_SCRIPT_EOF'\n{script_content}\nSLURM_SCRIPT_EOF", args.user)
    
    if result.returncode != 0:
        print(f"❌ Failed to upload script: {result.stderr}")
        return 1

    # Submit job
    print("📤 Submitting SLURM job...")
    result = run_ssh_command(f"sbatch {script_path}", args.user)
    
    if result.returncode != 0:
        print(f"❌ Failed to submit job: {result.stderr}")
        return 1

    # Parse job ID
    output = result.stdout.strip()
    if "Submitted batch job" in output:
        job_id = output.split()[-1]
        print(f"✅ Job submitted successfully! Job ID: {job_id}")
        
        # Save state
        save_state({
            "job_id": job_id,
            "model": args.model,
            "port": args.port,
            "tensor_parallel": args.tensor_parallel,
            "user": args.user,
            "submitted_at": datetime.now().isoformat(),
        })
        
        print()
        print("📊 To check status and get API endpoint:")
        print("   iquest-serve status")
        print()
        print("🛑 To stop the server:")
        print("   iquest-serve stop")
        
        return 0
    else:
        print(f"❌ Unexpected output: {output}")
        return 1


def cmd_stop(args):
    """Stop the running vLLM serving job."""
    print("🛑 Stopping IQuest-Coder serving...")
    
    state = load_state()
    if not state.get("job_id"):
        print("❌ No active serving job found.")
        return 1

    job_id = state["job_id"]
    print(f"   Cancelling job {job_id}...")

    result = run_ssh_command(f"scancel {job_id}", state.get("user", args.user))
    
    if result.returncode != 0:
        print(f"⚠️  Warning: {result.stderr}")
    
    # Verify cancellation
    time.sleep(1)
    result = run_ssh_command(f"squeue -j {job_id} -h", state.get("user", args.user))
    
    if not result.stdout.strip():
        print(f"✅ Job {job_id} has been cancelled.")
        clear_state()
        return 0
    else:
        print(f"⚠️  Job may still be running. Current status:")
        print(f"   {result.stdout.strip()}")
        return 1


def cmd_status(args):
    """Show status and API endpoint information."""
    print("=" * 60)
    print("  IQuest-Coder vLLM Serving Status")
    print("=" * 60)
    print()
    
    state = load_state()
    if not state.get("job_id"):
        print("❌ No active serving job found.")
        print()
        print("💡 Start a new serving job with:")
        print("   iquest-serve start")
        return 1

    job_id = state["job_id"]
    user = state.get("user", args.user)
    
    # Get detailed job status
    result = run_ssh_command(
        f"squeue -j {job_id} -o '%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R' 2>/dev/null",
        user
    )
    
    if not result.stdout.strip() or "JOBID" not in result.stdout:
        # Job not in queue, check if completed
        result = run_ssh_command(
            f"sacct -j {job_id} --format=JobID,State,ExitCode,Start,End -n 2>/dev/null",
            user
        )
        if result.stdout.strip():
            print(f"📋 Job History (Job ID: {job_id}):")
            print(result.stdout)
            print()
            print("⚠️  Job is no longer running.")
            clear_state()
        else:
            print(f"❌ Job {job_id} not found.")
            clear_state()
        return 1
    
    # Parse job status
    lines = result.stdout.strip().split("\n")
    print("📋 SLURM Job Status:")
    for line in lines:
        print(f"   {line}")
    print()
    
    # Show saved job info
    print("📝 Job Information:")
    print(f"   Model: {state.get('model', 'Unknown')}")
    print(f"   Port: {state.get('port', 'Unknown')}")
    print(f"   Tensor Parallel: {state.get('tensor_parallel', 'Unknown')}")
    print(f"   Submitted: {state.get('submitted_at', 'Unknown')}")
    print()

    # Try to get endpoint information
    result = run_ssh_command(f"cat {WORK_DIR}/current_endpoint.txt 2>/dev/null", user)
    
    if result.returncode == 0 and result.stdout.strip():
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            node_ip = lines[0]
            port = lines[1]
            
            print("=" * 60)
            print("  🌐 OpenAI-Compatible API Endpoint")
            print("=" * 60)
            print()
            print(f"  Base URL: http://{node_ip}:{port}/v1")
            print()
            print("  📖 Usage Examples:")
            print()
            print("  # List available models")
            print(f"  curl http://{node_ip}:{port}/v1/models")
            print()
            print("  # Chat completion")
            print(f"  curl http://{node_ip}:{port}/v1/chat/completions \\")
            print("    -H 'Content-Type: application/json' \\")
            print("    -d '{")
            print(f'      "model": "{state.get("model", DEFAULT_MODEL)}",')
            print('      "messages": [{"role": "user", "content": "Hello!"}],')
            print('      "temperature": 0.6,')
            print('      "top_p": 0.85')
            print("    }'")
            print()
            print("  # Python with OpenAI SDK:")
            print("  from openai import OpenAI")
            print()
            print(f'  client = OpenAI(base_url="http://{node_ip}:{port}/v1", api_key="none")')
            print("  response = client.chat.completions.create(")
            print(f'      model="{state.get("model", DEFAULT_MODEL)}",')
            print('      messages=[{"role": "user", "content": "Write a hello world in Python"}],')
            print('      temperature=0.6,')
            print('      top_p=0.85')
            print("  )")
            print("  print(response.choices[0].message.content)")
            print()
            
            # Check if endpoint is accessible (via SSH tunnel suggestion)
            print("=" * 60)
            print("  🔗 SSH Tunnel (for local access)")
            print("=" * 60)
            print()
            print("  Run this command locally to create an SSH tunnel:")
            print(f"  ssh -L {port}:{node_ip}:{port} -N {user}@{SLURM_LOGIN_NODE}")
            print()
            print(f"  Then access: http://localhost:{port}/v1")
            print()
        else:
            print("⏳ Endpoint information not yet available.")
            print("   The server may still be starting up. Please wait and try again.")
    else:
        print("⏳ Endpoint information not yet available.")
        print("   The server may still be starting up. Please wait and try again.")
    
    return 0


def cmd_logs(args):
    """Show logs from the serving job."""
    state = load_state()
    if not state.get("job_id"):
        print("❌ No active serving job found.")
        return 1

    job_id = state["job_id"]
    user = state.get("user", args.user)
    
    log_file = f"{WORK_DIR}/logs/serve_{job_id}.out"
    
    if args.follow:
        print(f"📜 Following logs for job {job_id}... (Ctrl+C to stop)")
        run_ssh_command(f"tail -f {log_file}", user, capture=False)
    else:
        print(f"📜 Recent logs for job {job_id}:")
        print("-" * 60)
        result = run_ssh_command(f"tail -n {args.lines} {log_file}", user)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="IQuest-Coder SLURM Serving CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start serving with default settings
  iquest-serve start
  
  # Start with specific model and settings
  iquest-serve start --model IQuestLab/IQuest-Coder-V1-40B-Thinking --thinking
  
  # Check status and get API endpoint
  iquest-serve status
  
  # View logs
  iquest-serve logs
  iquest-serve logs -f  # Follow logs
  
  # Stop the server
  iquest-serve stop
"""
    )
    
    parser.add_argument(
        "--user", "-u",
        default=DEFAULT_USER,
        help=f"SLURM cluster username (default: {DEFAULT_USER})"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start vLLM serving on SLURM")
    start_parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model to serve (default: {DEFAULT_MODEL})"
    )
    start_parser.add_argument(
        "--tensor-parallel", "-tp",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL,
        help=f"Tensor parallel size (default: {DEFAULT_TENSOR_PARALLEL})"
    )
    start_parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for the API server (default: {DEFAULT_PORT})"
    )
    start_parser.add_argument(
        "--partition",
        default=DEFAULT_PARTITION,
        help=f"SLURM partition (default: {DEFAULT_PARTITION})"
    )
    start_parser.add_argument(
        "--gpus", "-g",
        type=int,
        default=DEFAULT_TENSOR_PARALLEL,
        help=f"Number of GPUs (default: {DEFAULT_TENSOR_PARALLEL})"
    )
    start_parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable thinking/reasoning mode (adds --reasoning-parser qwen3)"
    )
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the vLLM serving job")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show status and API endpoint")
    
    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show logs from the serving job")
    logs_parser.add_argument(
        "-f", "--follow",
        action="store_true",
        help="Follow log output"
    )
    logs_parser.add_argument(
        "-n", "--lines",
        type=int,
        default=50,
        help="Number of lines to show (default: 50)"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == "start":
        return cmd_start(args)
    elif args.command == "stop":
        return cmd_stop(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "logs":
        return cmd_logs(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
