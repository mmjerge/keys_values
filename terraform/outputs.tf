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

output "worker_public_ips" {
  description = "Public IPs of the worker instances."
  value       = aws_instance.worker[*].public_ip
}

output "worker_instance_ids" {
  value = aws_instance.worker[*].id
}

output "ami_used" {
  value = data.aws_ami.dl_base.name
}

output "ssh_config_snippet" {
  description = "Paste into ~/.ssh/config (adjust IdentityFile)."
  value = join("\n", [
    for i, ip in aws_instance.worker[*].public_ip : <<-EOT
      Host rl-${i}
          HostName ${ip}
          User ubuntu
          IdentityFile ~/.ssh/kv-worker.pem
          StrictHostKeyChecking accept-new
    EOT
  ])
}

output "results_bucket_uri" {
  value = "s3://${var.results_bucket}"
}
