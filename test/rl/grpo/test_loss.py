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
import torch

from keys_values.config import Config
from keys_values.rl.grpo.loss import GRPOLossHeadModel


def test_loss_head_temperature_matches_rollout_logprobs():
    """The rollout records log-probs under softmax(logits / T). Given explicit
    old log-probs from that distribution and the same parameters, the loss
    head must reproduce them, so the ratio is exactly 1 for any T. Before the
    fix the head ignored T and the ratio was off whenever T != 1."""
    torch.manual_seed(0)
    bs, num, vocab, temp = 2, 5, 16, 0.7
    config = Config()
    config.padded_vocab_size = vocab
    logits = torch.randn(bs, num, vocab)
    targets = torch.randint(0, vocab, (bs, num))
    # What the rollout records (rollout.sampled_token_logprobs semantics)
    rollout_logps = (
        torch.log_softmax(logits / temp, dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    advantages = torch.tensor([1.0, -1.0])

    head = GRPOLossHeadModel(config, temperature=temp)
    head.set_batch(advantages=advantages, old_logps=rollout_logps)
    loss = head(logits, targets, input_pos=0)
    # ratio == 1 -> per-token loss == -advantage, summed over `num` tokens
    torch.testing.assert_close(loss, -advantages * num, rtol=1e-5, atol=1e-5)

    head_wrong = GRPOLossHeadModel(config, temperature=1.0)
    head_wrong.set_batch(advantages=advantages, old_logps=rollout_logps)
    loss_wrong = head_wrong(logits, targets, input_pos=0)
    assert not torch.allclose(loss_wrong, -advantages * num, atol=1e-3)


def test_loss_head_bf16_logits_give_fp32_loss():
    """Loss-side softmax runs in fp32 even for bf16 logits."""
    config = Config()
    config.padded_vocab_size = 8
    head = GRPOLossHeadModel(config)
    head.set_batch(advantages=torch.tensor([0.5]))
    logits = torch.randn(1, 3, 8).to(torch.bfloat16)
    targets = torch.randint(0, 8, (1, 3))
    loss = head(logits, targets, input_pos=0)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, torch.tensor([-1.5]))
