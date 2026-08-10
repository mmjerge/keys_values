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

output "autoscaling_group" {
  description = "ASG name. List current workers with: aws ec2 describe-instances --filters Name=tag:kv-worker,Values=true Name=instance-state-name,Values=running"
  value       = aws_autoscaling_group.workers.name
}

output "ami_used" {
  value = data.aws_ami.dl_base.name
}

output "results_bucket_uri" {
  value = "s3://${var.results_bucket}"
}
