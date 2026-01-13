:hi:Hello  @Yuandong Tian and Recursive Team -- Your Soperator B200 POC Cluster is ready! :rocket::rocket::rocket:
This is a 4 node, 32 B200 GPU, Nebius Managed Slurm on Kubernetes (Soperator v1.22.4) cluster. It's running in our  us-central1 region in Kansas USA.

Connection Instructions
Connect to one of two Slurm login nodes:
ssh <USER>@204.12.169.136
Replace <USER> with your username: [yuandong, timshi]
Mountpoints
Shared across all nodes
/  is the Shared Root filesystem. Applications installed in one node are available on all nodes. (5 TB)
/home is NFS and can be used for code/dev environment. Please do not use home for job runs or logs. (3.6 TB)
/mnt/data is the Shared Filesystem that can be used for checkpoints  (53 TB)

Node local (worker nodes)
/mnt/image-storage is used automatically for storage container images (1 TB)
/mnt/memory is a RAM disk (tmpfs) providing very fast, temporary in-memory storage (1.6 TB)
/scratch is fast block storage (non replicated) that could be used to access downloaded files (1 TB)

SLURM Configuration
Default Partition: `main` (use `-p main` or `--partition=main`)
Available Nodes: worker-[0-3] (4 nodes total, 8 B200 GPUs each, 32 GPUs total)
To check partition info: `sinfo -s`
To check job queue: `squeue`

Installing software packages
sudo apt install cowsay
You only need to run this command once on the login node as the shared filesystem is available on every node.

Hugging Face Cache Configuration
Due to multiple users on SLURM, configure Hugging Face to use a user-specific cache directory to avoid permission errors:

# Add to your ~/.bashrc or export before running commands
export HF_HOME=/mnt/data/timshi/.cache
export HUGGINGFACE_HUB_CACHE=/mnt/data/timshi/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/data/timshi/.cache/transformers

This ensures each user has their own cache directory on the shared data filesystem, preventing permission conflicts.

Cluster Monitoring
To preview Metrics and Logs: On the left navigation menu, go to Observability tile on web UI.
Metrics
To open Grafana monitoring dashboards in your browser:
  1. Execute this command on your local computer:
     ssh -L 3000:metrics-grafana.monitoring-system.svc:80 -N <USER>@204.12.169.136
  2. Open localhost:3000 in your browser
Logging
To open logs explorer in your browser:
  1. Execute this command on your local computer:
     ssh -L 9428:vm-logs-victoria-logs-single-server.logs-system.svc:9428 -N <USER>@204.12.169.136
  2. Open localhost:9428/select/vmui in your browser

Support
To open a support ticket, you can call @nebius_support , and include tenant tenant-e00j3xyex3349e6jf8 in the Description. The Slack thread will act as direct communication with Nebius Support.