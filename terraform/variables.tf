# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

variable "region" {
  description = "AWS region for the worker fleet."
  type        = string
  default     = "us-east-1"
}

variable "worker_count" {
  description = "Number of single-GPU worker instances. Set to 0 to manage only the shared infra (buckets, IAM, security group)."
  type        = number
  default     = 1
}

variable "instance_types" {
  description = "Worker instance types in priority order. The autoscaling group tries every (type x AZ) combination and backfills automatically as capacity appears, so list every type whose per-GPU memory fits your job (all L40S 48GB here; 7B GRPO needs ~36GB). Larger sizes cost more but often have capacity when the small one is dry."
  type        = list(string)
  default     = ["g6e.2xlarge", "g6e.4xlarge", "g6e.8xlarge"]
}

variable "use_spot" {
  description = "Use spot capacity (capacity-optimized allocation, ~60-70% cheaper, but instances can be interrupted mid-run -- only sensible once jobs checkpoint/resume). Default on-demand."
  type        = bool
  default     = false
}

variable "root_volume_gb" {
  description = "Root EBS volume size. Model checkpoints + HELMET data need headroom; 96GB filled up in practice."
  type        = number
  default     = 200
}

variable "ssh_public_key_path" {
  description = "Path to the SSH public key imported for the workers (e.g. derive from a .pem with: ssh-keygen -y -f key.pem > key.pub)."
  type        = string
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to SSH to workers. Default is open; restrict to your IP (e.g. 1.2.3.4/32) where possible."
  type        = string
  default     = "0.0.0.0/0"
}

variable "results_bucket" {
  description = "S3 bucket for the job queue and run artifacts (created and managed by this module). Bucket names are global; override to something unique for your deployment."
  type        = string
  default     = "keys-values-rl-results"
}

variable "create_results_bucket" {
  description = "Create the results bucket. Set false if it already exists (e.g. the original deployment) and you only want workers."
  type        = bool
  default     = true
}

variable "canonical_splits_bucket" {
  description = "Existing S3 bucket holding the canonical HELMET split caches (read-only; NOT managed by this module). Workers get read access and sync it at provision time."
  type        = string
  default     = "keys-values-helmet-canonical"
}

variable "reader_account_ids" {
  description = "AWS account IDs granted read-only access to the results bucket (e.g. collaborators)."
  type        = list(string)
  default     = []
}

variable "repo_url" {
  description = "Git remote the workers clone."
  type        = string
  default     = "https://github.com/mmjerge/keys_values.git"
}

variable "repo_branch" {
  description = "Branch the workers check out and provision from."
  type        = string
  default     = "experiments"
}

variable "model" {
  description = "Model checkpoint the provision script downloads (litgpt name)."
  type        = string
  default     = "Qwen/Qwen2.5-7B-Instruct"
}

variable "start_worker_loop" {
  description = "Start the S3 job-queue worker loop at boot. Default true: ASG workers have no stable identity, so pulling jobs from the queue is how they receive work (set false only for interactive/debug fleets you plan to SSH into)."
  type        = bool
  default     = true
}
