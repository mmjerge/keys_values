# RL worker fleet (Terraform, optional)

Optional infrastructure-as-code for the keys_values RL experiment fleet. If
you're happy running on your own machines, you never need this directory --
the training code has no dependency on it.

What it deploys (all in your own AWS account):

- **An autoscaling group of single-GPU workers** (L40S 48GB types by
  default) from the AWS Deep Learning Base AMI, provisioned at boot by
  `scripts/provision_worker.sh` (repo, pinned Python stack, canonical HELMET
  splits from S3, model checkpoint). The ASG spans **every AZ** and a
  **prioritized list of instance types**, so EC2 searches the whole
  (type x AZ) grid and backfills toward `worker_count` automatically as
  capacity appears -- no manual retrying on InsufficientInstanceCapacity.
  Workers are cattle: at boot they start `scripts/worker_loop.sh`, pull jobs
  from the S3 queue, and stop themselves when the queue drains.
- **IAM role** for the workers: read the canonical-splits bucket, read/write
  the results bucket, and stop *worker-tagged* instances only (for the
  self-stop-when-queue-empty behavior of `scripts/worker_loop.sh`).
- **Results bucket** holding the S3 job queue (`queue/pending/*.sh`) and run
  artifacts (`runs/<job>/`), with optional read-only grants for
  collaborators' AWS accounts.
- **SSH key pair + security group** (port 22 only; restrict
  `ssh_ingress_cidr` to your IP if you can).

Why a fleet of single-GPU boxes instead of one 8-GPU node: the GRPO/RLOO
training loop is single-GPU, so the experiment matrix (seeds x cache configs
x tasks) parallelizes across independent workers -- and single-GPU instance
types reliably have on-demand capacity where p4d/g6e.48xlarge frequently do
not (all of us-east-1/2 was out of 8-GPU capacity the day this was written).

## Usage

```bash
cd terraform
terraform init

# Derive a public key from your .pem once:
ssh-keygen -y -f ~/.ssh/my-key.pem > ~/.ssh/kv-worker.pub

terraform plan \
  -var ssh_public_key_path=~/.ssh/kv-worker.pub \
  -var worker_count=4 \
  -var 'reader_account_ids=["719355911555"]'

terraform apply ...same vars...
```

Interesting variables (see `variables.tf` for all):

| variable | default | notes |
|---|---|---|
| `worker_count` | 1 | ASG desired capacity; 0 = shared infra only |
| `instance_types` | `[g6e.2xlarge, g6e.4xlarge, g6e.8xlarge]` | priority order; all L40S 48GB. ASG falls back across types and AZs automatically |
| `use_spot` | `false` | spot = ~60-70% cheaper but interruptible; enable once jobs checkpoint/resume |
| `repo_branch` | `experiments` | what the workers check out |
| `model` | `Qwen/Qwen2.5-7B-Instruct` | downloaded at provision time |
| `start_worker_loop` | `true` | boot, drain the S3 job queue, self-stop |
| `results_bucket` | `keys-values-rl-results` | bucket names are global -- override for your own deployment |
| `create_results_bucket` | `true` | `false` to attach to an existing bucket |

Scaling with the queue: `terraform apply -var worker_count=N` when you
submit N jobs; workers self-stop as the queue drains (a self-stopped
instance still counts against the ASG until terminated -- scale
`worker_count` back down, or terminate stopped workers, when a campaign
ends).

## Job queue

Submit work by dropping shell snippets into the queue; idle workers claim,
run, and upload:

```bash
cat > job.sh <<'EOF'
cd ~/keys_values
python examples/grpo_helmet.py --device cuda --dataset-key nq \
    --kv-cache-name h2o-torch-quantized8 --cache-length 4096 \
    --disable-flashinfer --seed 1 --out-dir $OUT
EOF
aws s3 cp job.sh s3://<results-bucket>/queue/pending/nq_h2o_seed1.sh
```

Logs and artifacts land in `s3://<results-bucket>/runs/<job>/`.

## Costs and hygiene

- A `g6e.2xlarge` is ~$2.2/hr; four of them ~$9/hr. **Stopped instances cost
  only EBS.** With `start_worker_loop=true`, workers stop themselves when the
  queue drains.
- `terraform destroy` removes everything this module created (the canonical
  splits bucket is external and untouched).
- The original hand-built infrastructure (grpo-bench, the kv-worker IAM role,
  the buckets) is NOT imported into this state; the module uses distinct
  names (`kv-worker-tf`) so the two can coexist.
