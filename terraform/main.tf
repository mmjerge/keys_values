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

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# --- AMI: Deep Learning Base (Ubuntu 24.04), same family as the manual boxes.

data "aws_ami" "dl_base" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["Deep Learning Base AMI with Single CUDA (Ubuntu 24.04) *"]
  }
}

data "aws_vpc" "default" {
  default = true
}

# --- SSH key + security group ------------------------------------------------

resource "aws_key_pair" "worker" {
  key_name   = "kv-worker"
  public_key = file(var.ssh_public_key_path)
}

resource "aws_security_group" "ssh" {
  name        = "kv-worker-ssh"
  description = "SSH for keys_values RL workers"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_ingress_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- Results bucket (job queue + run artifacts) --------------------------------

resource "aws_s3_bucket" "results" {
  count  = var.create_results_bucket ? 1 : 0
  bucket = var.results_bucket
}

data "aws_iam_policy_document" "results_readers" {
  count = var.create_results_bucket && length(var.reader_account_ids) > 0 ? 1 : 0
  statement {
    sid    = "CollaboratorReadOnly"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = [for id in var.reader_account_ids : "arn:aws:iam::${id}:root"]
    }
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      "arn:aws:s3:::${var.results_bucket}",
      "arn:aws:s3:::${var.results_bucket}/*",
    ]
  }
}

resource "aws_s3_bucket_policy" "results_readers" {
  count  = var.create_results_bucket && length(var.reader_account_ids) > 0 ? 1 : 0
  bucket = aws_s3_bucket.results[0].id
  policy = data.aws_iam_policy_document.results_readers[0].json
}

# --- Worker IAM role -----------------------------------------------------------

data "aws_iam_policy_document" "assume_ec2" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "worker" {
  name               = "kv-worker-tf"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

data "aws_iam_policy_document" "worker_permissions" {
  statement {
    sid     = "ReadCanonicalSplits"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      "arn:aws:s3:::${var.canonical_splits_bucket}",
      "arn:aws:s3:::${var.canonical_splits_bucket}/*",
    ]
  }
  statement {
    sid     = "ReadWriteResults"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [
      "arn:aws:s3:::${var.results_bucket}",
      "arn:aws:s3:::${var.results_bucket}/*",
    ]
  }
  statement {
    sid       = "SelfStopTaggedWorkers"
    effect    = "Allow"
    actions   = ["ec2:StopInstances"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "aws:ResourceTag/kv-worker"
      values   = ["true"]
    }
  }
}

resource "aws_iam_role_policy" "worker" {
  name   = "kv-worker-permissions"
  role   = aws_iam_role.worker.id
  policy = data.aws_iam_policy_document.worker_permissions.json
}

resource "aws_iam_instance_profile" "worker" {
  name = "kv-worker-tf"
  role = aws_iam_role.worker.name
}

# --- Workers -------------------------------------------------------------------

locals {
  user_data = <<-EOT
    #!/bin/bash
    # Provision as the ubuntu user at first boot.
    sudo -u ubuntu -i bash -c '
      curl -sL ${replace(var.repo_url, ".git", "")}/raw/${var.repo_branch}/scripts/provision_worker.sh \
        | KV_MODEL=${var.model} bash -s -- ${var.repo_branch} > ~/provision.log 2>&1
      %{if var.start_worker_loop}
      KV_STOP_WHEN_DONE=1 nohup ~/keys_values/scripts/worker_loop.sh > ~/worker_loop.log 2>&1 &
      %{endif}
    '
  EOT
}

resource "aws_instance" "worker" {
  count                  = var.worker_count
  ami                    = data.aws_ami.dl_base.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.worker.key_name
  vpc_security_group_ids = [aws_security_group.ssh.id]
  iam_instance_profile   = aws_iam_instance_profile.worker.name
  user_data              = local.user_data

  root_block_device {
    volume_size = var.root_volume_gb
    volume_type = "gp3"
  }

  tags = {
    Name      = "kv-rl-worker-${count.index}"
    kv-worker = "true"
  }
}
