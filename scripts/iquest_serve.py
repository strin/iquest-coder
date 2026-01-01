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

# Set Hugging Face cache to user-specific directory to avoid permission errors
export HF_HOME=/mnt/data/$USER/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/data/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/data/$USER/.cache/transformers

# Set up CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Load CUDA module if available (common on SLURM clusters)
if command -v module &> /dev/null; then
    module load cuda 2>/dev/null || true
fi

# Create cache directories if needed
mkdir -p $HF_HOME
mkdir -p $HUGGINGFACE_HUB_CACHE
mkdir -p $TRANSFORMERS_CACHE

# Clear any stale model cache for this specific model (to ensure fresh config.json)
MODEL_CACHE_DIR="models--$(echo '{model}' | sed 's/\\/--/g')"
echo "Clearing stale cache: $HUGGINGFACE_HUB_CACHE/$MODEL_CACHE_DIR"
rm -rf "$HUGGINGFACE_HUB_CACHE/$MODEL_CACHE_DIR" 2>/dev/null || true

# Activate virtual environment if exists
if [ -f "{WORK_DIR}/venv/bin/activate" ]; then
    source {WORK_DIR}/venv/bin/activate
fi

# Print vLLM version
echo "vLLM version: $(vllm --version 2>/dev/null || python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'unknown')"

# Verify CUDA is accessible
echo "Checking CUDA environment..."
nvidia-smi || echo "Warning: nvidia-smi not found"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Start vLLM server
echo "Starting vLLM server..."
vllm serve {model} \\
    --tensor-parallel-size {tensor_parallel} \\
    --port {port} \\
    --host 0.0.0.0 \\
    --trust-remote-code \\
    --dtype auto \\
    --model-impl transformers \\
    {reasoning_parser}

echo "Server stopped at $(date)"
"""
    return script


def cmd_setup(args):
    """Set up the vLLM environment on the SLURM cluster."""
    print("🔧 Setting up vLLM environment on SLURM cluster...")
    print()

    user = args.user

    # Create work directories
    print("📁 Creating work directories...")
    result = run_ssh_command(f"mkdir -p {WORK_DIR}/logs {WORK_DIR}/scripts", user)
    if result.returncode != 0:
        print(f"❌ Failed to create directories: {result.stderr}")
        return 1
    print("   ✅ Directories created")
    print()

    # Check if venv already exists
    print("🔍 Checking for existing virtual environment...")
    result = run_ssh_command(f"test -d {WORK_DIR}/venv && echo 'exists' || echo 'missing'", user)
    venv_exists = result.stdout.strip() == "exists"

    if venv_exists and not args.reinstall:
        print(f"   ⚠️  Virtual environment already exists at {WORK_DIR}/venv")
        print("   Use --reinstall to recreate it")
        print()
    else:
        if venv_exists:
            print(f"   🗑️  Removing existing virtual environment...")
            run_ssh_command(f"rm -rf {WORK_DIR}/venv", user)

        print("🐍 Creating virtual environment...")
        result = run_ssh_command(f"python3 -m venv {WORK_DIR}/venv", user)
        if result.returncode != 0:
            print(f"❌ Failed to create virtual environment: {result.stderr}")
            return 1
        print("   ✅ Virtual environment created")
        print()

    # Install vLLM
    print("📦 Installing vLLM (this may take a few minutes)...")
    install_cmd = f"""
    source {WORK_DIR}/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --upgrade vllm
    """

    result = run_ssh_command(install_cmd, user)
    if result.returncode != 0:
        print(f"❌ Failed to install vLLM: {result.stderr}")
        return 1

    print("   ✅ vLLM installed successfully")
    print()

    # Verify installation
    print("✅ Verifying installation...")
    result = run_ssh_command(f"source {WORK_DIR}/venv/bin/activate && vllm --version", user)
    if result.returncode == 0:
        version = result.stdout.strip()
        print(f"   ✅ vLLM version: {version}")
        print()
        print("=" * 60)
        print("  Setup completed successfully!")
        print("=" * 60)
        print()
        print("💡 You can now start serving with:")
        print("   iquest-serve start")
        return 0
    else:
        print(f"   ⚠️  Could not verify vLLM installation: {result.stderr}")
        print()
        print("   You can still try to start serving with:")
        print("   iquest-serve start")
        return 1


def cmd_start(args):
    """Start the vLLM serving job on SLURM."""
    print(f"🚀 Starting IQuest-Coder serving on SLURM cluster...")
    print(f"   Model: {args.model}")
    print(f"   Tensor Parallel Size: {args.tensor_parallel}")
    print(f"   Port: {args.port}")
    print(f"   Partition: {args.partition}")
    print(f"   GPUs: {args.gpus}")
    print()

    # Check if vLLM is installed
    print("🔍 Checking vLLM installation...")
    result = run_ssh_command(f"test -f {WORK_DIR}/venv/bin/vllm && echo 'installed' || echo 'missing'", args.user)
    vllm_installed = result.stdout.strip() == "installed"

    if not vllm_installed:
        print("   ⚠️  vLLM is not installed on the SLURM cluster")
        print()
        print("🔧 Running automatic setup...")
        print()

        # Run setup automatically
        setup_result = cmd_setup(args)
        if setup_result != 0:
            print()
            print("❌ Setup failed. Please run 'iquest-serve setup' manually and fix any issues.")
            return 1

        print()
        print("✅ Setup completed! Continuing with server start...")
        print()
    else:
        print("   ✅ vLLM is installed")
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
    """Show logs from the serving job, including vLLM request logs."""
    state = load_state()
    if not state.get("job_id"):
        print("❌ No active serving job found.")
        return 1

    job_id = state["job_id"]
    user = state.get("user", args.user)
    
    # Determine which log file to use
    if args.stderr:
        log_file = f"{WORK_DIR}/logs/serve_{job_id}.err"
        log_type = "stderr"
    else:
        log_file = f"{WORK_DIR}/logs/serve_{job_id}.out"
        log_type = "stdout"
    
    # Build the command based on options
    if args.requests:
        # Filter for request-related logs (vLLM logs requests with INFO level)
        # Common patterns: "POST", "Received request", "completion", "generate"
        filter_pattern = "POST\\|completion\\|Received\\|generate\\|request\\|INFO"
        if args.follow:
            print(f"📜 Streaming request logs for job {job_id}... (Ctrl+C to stop)")
            print(f"   Filtering for: API requests and completions")
            print("-" * 60)
            cmd = f"tail -f {log_file} | grep --line-buffered -i '{filter_pattern}'"
        else:
            print(f"📜 Recent request logs for job {job_id} ({log_type}):")
            print("-" * 60)
            cmd = f"grep -i '{filter_pattern}' {log_file} | tail -n {args.lines}"
    else:
        if args.follow:
            print(f"📜 Streaming {log_type} logs for job {job_id}... (Ctrl+C to stop)")
            print("-" * 60)
            cmd = f"tail -f {log_file}"
        else:
            print(f"📜 Recent {log_type} logs for job {job_id}:")
            print("-" * 60)
            cmd = f"tail -n {args.lines} {log_file}"
    
    if args.follow:
        # For streaming, we need to run without capture
        try:
            run_ssh_command(cmd, user, capture=False)
        except KeyboardInterrupt:
            print("\n👋 Stopped log streaming.")
    else:
        result = run_ssh_command(cmd, user)
        if result.stdout:
            print(result.stdout)
        else:
            print("(No logs matching criteria)")
        if result.stderr and "No such file" in result.stderr:
            print(f"⚠️  Log file not found. The job may still be starting.")
        elif result.stderr:
            print("Errors:", result.stderr)
    
    return 0


def get_endpoint_info(user: str) -> tuple[str, str, str] | None:
    """Get the current endpoint info (node_ip, port, model) from the serving state.
    
    Returns:
        Tuple of (node_ip, port, model) or None if not available.
    """
    state = load_state()
    if not state.get("job_id"):
        return None
    
    # Try to get endpoint information
    result = run_ssh_command(f"cat {WORK_DIR}/current_endpoint.txt 2>/dev/null", user)
    
    if result.returncode == 0 and result.stdout.strip():
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            node_ip = lines[0]
            port = lines[1]
            model = state.get("model", DEFAULT_MODEL)
            return (node_ip, port, model)
    
    return None


def cmd_code(args):
    """Start OpenHands with the custom vLLM endpoint from the SLURM cluster."""
    print("🤖 Starting OpenHands with IQuest-Coder...")
    print()
    
    state = load_state()
    if not state.get("job_id"):
        print("❌ No active serving job found.")
        print()
        print("💡 Start the vLLM server first with:")
        print("   iquest-serve start")
        return 1
    
    # Get endpoint info
    user = state.get("user", args.user)
    endpoint_info = get_endpoint_info(user)
    
    if not endpoint_info:
        print("❌ Endpoint information not available.")
        print("   The server may still be starting up. Please wait and try again.")
        print()
        print("💡 Check status with:")
        print("   iquest-serve status")
        return 1
    
    node_ip, port, model = endpoint_info
    local_port = args.local_port
    
    print(f"📡 Endpoint Information:")
    print(f"   Node IP: {node_ip}")
    print(f"   Port: {port}")
    print(f"   Model: {model}")
    print()
    
    # Set up SSH tunnel
    print(f"🔗 Setting up SSH tunnel (local:{local_port} -> {node_ip}:{port})...")
    
    # Check if tunnel is already running
    check_tunnel = subprocess.run(
        ["lsof", "-i", f":{local_port}"],
        capture_output=True,
        text=True
    )
    
    tunnel_process = None
    if check_tunnel.returncode != 0:
        # Start SSH tunnel in background
        tunnel_cmd = [
            "ssh",
            "-L", f"{local_port}:{node_ip}:{port}",
            "-N",
            "-f",  # Go to background
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=60",
            f"{user}@{SLURM_LOGIN_NODE}"
        ]
        
        result = subprocess.run(tunnel_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to create SSH tunnel: {result.stderr}")
            return 1
        
        print(f"   ✅ SSH tunnel established")
    else:
        print(f"   ✅ SSH tunnel already active on port {local_port}")
    
    print()
    
    # Build base URL for OpenHands
    base_url = f"http://localhost:{local_port}/v1"
    
    # OpenHands model format: for vLLM/SGLang endpoints, use "openai/<model-name>"
    openhands_model = f"openai/{model}"
    
    print(f"🚀 Starting OpenHands...")
    print(f"   Base URL: {base_url}")
    print(f"   Model: {openhands_model}")
    print()
    
    # Check if openhands is installed
    check_openhands = subprocess.run(
        ["which", "openhands"],
        capture_output=True,
        text=True
    )
    
    if check_openhands.returncode != 0:
        print("❌ OpenHands is not installed.")
        print()
        print("💡 Install OpenHands with:")
        print("   uv tool install openhands --python 3.12  # Recommended")
        print("   # or")
        print("   pip install openhands")
        return 1
    
    # Print configuration instructions for manual setup
    print("=" * 60)
    print("  📋 OpenHands Manual Configuration")
    print("=" * 60)
    print()
    print("When OpenHands starts, configure the LLM settings:")
    print()
    print("  1. Click 'see advanced settings' in the Settings dialog")
    print("  2. Enable the 'Advanced' toggle")
    print("  3. Set the following values:")
    print()
    print(f"     Custom Model: {openhands_model}")
    print(f"     Base URL: {base_url}")
    print("     API Key: none")
    print()
    print("=" * 60)
    print()
    
    if args.mode == "serve":
        print("🌐 Starting OpenHands GUI server...")
        print("   Access at: http://localhost:3000")
        print()
        
        openhands_cmd = ["openhands", "serve"]
        if args.mount_cwd:
            openhands_cmd.append("--mount-cwd")
    else:
        # CLI/TUI mode
        print("💻 Starting OpenHands in CLI mode...")
        print()
        
        openhands_cmd = ["openhands"]
        if args.task:
            openhands_cmd.extend(["-t", args.task])
    
    # Run OpenHands
    try:
        subprocess.run(openhands_cmd)
    except KeyboardInterrupt:
        print()
        print("👋 OpenHands session ended.")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="IQuest-Coder SLURM Serving CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First-time setup: install vLLM on SLURM cluster
  iquest-serve setup

  # Start serving with default settings
  iquest-serve start

  # Start with specific model and settings
  iquest-serve start --model IQuestLab/IQuest-Coder-V1-40B-Thinking --thinking

  # Check status and get API endpoint
  iquest-serve status

  # View logs
  iquest-serve logs              # Show recent logs
  iquest-serve logs -f           # Stream logs in real-time
  iquest-serve logs -f -r        # Stream only API request logs
  iquest-serve logs -e           # Show stderr logs
  
  # Stop the server
  iquest-serve stop

  # Start OpenHands with IQuest-Coder (CLI mode - default)
  iquest-serve code

  # Start OpenHands with a specific task
  iquest-serve code -t "Create a Python CLI tool"

  # Start OpenHands in GUI mode
  iquest-serve code --mode serve
"""
    )
    
    parser.add_argument(
        "--user", "-u",
        default=DEFAULT_USER,
        help=f"SLURM cluster username (default: {DEFAULT_USER})"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up vLLM environment on SLURM cluster")
    setup_parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Remove and recreate the virtual environment if it exists"
    )

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
    logs_parser = subparsers.add_parser("logs", help="Stream vLLM logs from the serving job")
    logs_parser.add_argument(
        "-f", "--follow",
        action="store_true",
        help="Stream log output in real-time (like tail -f)"
    )
    logs_parser.add_argument(
        "-n", "--lines",
        type=int,
        default=50,
        help="Number of lines to show (default: 50)"
    )
    logs_parser.add_argument(
        "-e", "--stderr",
        action="store_true",
        help="Show stderr logs instead of stdout"
    )
    logs_parser.add_argument(
        "-r", "--requests",
        action="store_true",
        help="Filter for API request logs (POST, completions, etc.)"
    )
    
    # Code command - Start OpenHands with IQuest-Coder
    code_parser = subparsers.add_parser(
        "code",
        help="Start OpenHands with IQuest-Coder endpoint"
    )
    code_parser.add_argument(
        "--mode",
        choices=["cli", "serve"],
        default="cli",
        help="OpenHands mode: 'cli' for terminal TUI, 'serve' for GUI (default: cli)"
    )
    code_parser.add_argument(
        "--local-port",
        type=int,
        default=8000,
        help="Local port for SSH tunnel (default: 8000)"
    )
    code_parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for generation (default: 0.6, recommended for IQuest-Coder)"
    )
    code_parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        dest="top_p",
        help="Top-p for generation (default: 0.85, recommended for IQuest-Coder)"
    )
    code_parser.add_argument(
        "-t", "--task",
        help="Initial task to give to OpenHands (CLI mode only)"
    )
    code_parser.add_argument(
        "--mount-cwd",
        action="store_true",
        help="Mount current working directory in OpenHands (serve mode only)"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "setup":
        return cmd_setup(args)
    elif args.command == "start":
        return cmd_start(args)
    elif args.command == "stop":
        return cmd_stop(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "logs":
        return cmd_logs(args)
    elif args.command == "code":
        return cmd_code(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
