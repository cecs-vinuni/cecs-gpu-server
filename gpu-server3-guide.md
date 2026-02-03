# VinUni - GPU Server 3 (RTX 5090) User Guide

## 1) Server Overview

**Hostname:** `server3-cecs-vinuni`

**Type:** Single-node GPU server managed by **Slurm**

**Operating System:** Ubuntu 22.04 LTS

**Hardware**

- **GPUs:** 4 × NVIDIA GeForce RTX 5090 (each ~32 GB VRAM)
- **CPU:** 48 logical CPU cores
- **RAM:** ~255 GB

**Slurm**

- **Scheduler:** Slurm (slurm-wlm 21.08.5)
- **Default partition:** `main`
- **GPU resource type:** `gres=gpu` (you request GPUs via `-gres=gpu:<N>`)

## 2) Storage Rule (Important)

To ensure performance and avoid home directory issues:

✅ **All large data (datasets, checkpoints, logs, outputs) must be stored under `/mnt/`**

Recommended structure:

- `/mnt/data/<project>/...` (datasets)
- `/mnt/experiments/<project>/...` (training outputs/checkpoints)
- `/mnt/logs/` (Slurm stdout/stderr)

Example:

```bash
DATA_DIR=/mnt/data/my_project
OUT_DIR=/mnt/experiments/my_project/${SLURM_JOB_ID}
mkdir -p "$OUT_DIR" /mnt/logs

```

## 3) Accessing the Server

### 3.1 SSH Login

```bash
ssh <your_username>@server3-cecs-vinuni

```

### 3.2 Quick Health Checks

```bash
hostname
nvidia-smi
sinfo

```

## 4) Slurm Basics (What Users Should Know)

Slurm is the job scheduler that allocates CPU, RAM, and GPU resources fairly.

### Common commands

- `sinfo` — cluster/partition status
- `squeue -u $USER` — show your jobs
- `srun` — run a job step (interactive or one-off)
- `salloc` — reserve resources for an interactive session
- `sbatch` — submit a batch job script
- `scancel <jobid>` — cancel a job

## 5) Running Jobs Without a GPU (CPU-only)

Run a simple command under Slurm:

```bash
srun -p main -n1 -c1 --mem=256M hostname

```

## 6) Requesting a GPU for a Single Command

### 6.1 Basic GPU request

```bash
srun -p main --gres=gpu:1 -n1 -c1 --mem=1G nvidia-smi -L

```

### 6.2 Confirm which GPU Slurm assigned

```bash
srun -p main --gres=gpu:1 -n1 -c1 --mem=1G bash -lc \
'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; nvidia-smi -L'

```

If `CUDA_VISIBLE_DEVICES=0`, your job is restricted to GPU index 0 inside the job.

## 7) Requesting a GPU “Session” (Interactive Work)

If you want to run multiple commands, notebooks, or long experiments, use an allocation.

### 7.1 Allocate 1 GPU + CPU/RAM for an interactive session

```bash
salloc -p main --gres=gpu:1 -c 8 --mem=48G -t 04:00:00

```

Once granted, you are holding the GPU.

### 7.2 Start an interactive shell inside the allocation

```bash
srun --pty bash

```

### 7.3 Verify the GPU assignment inside the session

```bash
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

```

### 7.4 Exit and release resources

```bash
exit   # exit the srun shell
exit   # exit the salloc allocation (releases the GPU)

```

## 8) Running Multiple Jobs on One GPU You Already Requested

**Goal:** Reserve **one GPU once**, then run multiple workloads on that same GPU reservation.

✅ Best practice:

1. `salloc --gres=gpu:1 ...` (reserve one GPU)
2. launch multiple `srun` steps inside that allocation

### Example: two steps running on the same assigned GPU

```bash
# 1) Allocate one GPU (one reservation)
salloc -p main --gres=gpu:1 -c 8 --mem=48G -t 00:30:00

# 2) Start two job steps inside the allocation
srun -n1 -c4 --mem=2G bash -lc 'echo STEP1 CUDA=$CUDA_VISIBLE_DEVICES; python3 -c "import time; time.sleep(20)"' &
srun -n1 -c4 --mem=2G bash -lc 'echo STEP2 CUDA=$CUDA_VISIBLE_DEVICES; python3 -c "import time; time.sleep(20)"' &
wait

# 3) Release
exit

```

**Expected behavior:** both steps show the same `CUDA_VISIBLE_DEVICES` value, meaning they share the same reserved GPU.

**Note:** If you run too many processes in parallel, you can exceed GPU memory and crash jobs. If you are unsure, run steps sequentially.

## 9) Standard PyTorch `sbatch` Template (Conda / venv)

Create: `train_pytorch.sbatch`

```bash
#!/bin/bash
#SBATCH -p main
#SBATCH --job-name=pt-train
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o /mnt/logs/%x-%j.out
#SBATCH -e /mnt/logs/%x-%j.err

set -euo pipefail

mkdir -p /mnt/logs

# IMPORTANT: keep data and outputs under /mnt
DATA_DIR="/mnt/data/your_project"     # <-- edit
OUT_DIR="/mnt/experiments/your_project/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "=== Job Info ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $USER"
echo "JobID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Data: $DATA_DIR"
echo "Out : $OUT_DIR"
echo "================"

nvidia-smi || true

# --------- Conda option ---------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate <YOUR_CONDA_ENV_NAME>   # <-- edit
fi

# --------- venv option (use instead of conda) ---------
# source "$HOME/venvs/yourenv/bin/activate"

which python || true
python -V || true

python -u train.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/train.log"

```

Submit:

```bash
sbatch train_pytorch.sbatch

```

Monitor:

```bash
squeue -u $USER
tail -f /mnt/logs/pt-train-<jobid>.out

```

Cancel:

```bash
scancel <jobid>

```

---

## 10) Policy: Default Limits

### Regular users (default QoS: `normal_1g`)

- **Maximum total concurrent usage per user:**
    - GPU: up to **1 GPU**
    - CPU: up to **8 cores**
    - RAM: up to **48 GB**
- **Maximum per job:**
    - GPU: up to **1 GPU**
    - CPU: up to **8 cores**
    - RAM: up to **48 GB**
- **Concurrency:** may run multiple jobs, but cannot exceed the per-user GPU/CPU/RAM caps.

## 11) Monitoring and Stopping Jobs

### Check your running/pending jobs

```bash
squeue -u $USER

```

### Cancel a job

```bash
scancel <jobid>

```

### If you used `salloc` (interactive allocation)

- Exit the allocation to release resources:

```bash
exit

```
