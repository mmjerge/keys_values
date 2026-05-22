# Qwen3 LoRA to LitGPT Workflow

This document describes how to convert a Hugging Face PEFT LoRA adapter trained on Qwen3 4B into a LitGPT checkpoint.

The recommended workflow is:

1. Merge the LoRA adapter into the original Hugging Face base model
2. Save the merged Hugging Face full model
3. Convert the merged model to LitGPT format

This workflow is used because Hugging Face PEFT LoRA adapters and LitGPT LoRA adapters use different parameter names and module layouts. Direct adapter conversion is not a simple file format conversion. It may require model-specific key mapping and QKV layout handling. Merging first avoids these issues.

## Files

Recommended directory layout:

```text
project
  lora_adapter
    adapter_config.json
    adapter_model.safetensors
  merged
  litgpt_merged
```

The adapter directory should contain `adapter_config.json` and either `adapter_model.safetensors` or `adapter_model.bin`.

## Step 1 Create Environment

Create a separate environment to avoid package conflicts.

```bash
conda create -n qwen3_lora_litgpt
conda activate qwen3_lora_litgpt
```

Install dependencies.

```bash
pip install -U torch transformers accelerate peft safetensors litgpt requests
```

Use a CUDA-enabled PyTorch build if you want GPU support.

## Step 2 Merge LoRA Adapter

Run the merge script.

NOTE: Run this script in a common base directory. Otherwise, the Hugging Face
checkpoint is downloaded again every time.

```bash
python merge_qwen3_lora.py \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-dir <data-dir>/lora_adapter \
  --output-dir <data-dir>/merged
```

The base model must be exactly the same model used during LoRA training.

## Step 3 Check Merged Model

The merged model directory should look similar to this:

```text
merged
  config.json
  generation_config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

Depending on the save settings and model size, the weight files may also be sharded, for example:

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
```

## Step 4 Convert to LitGPT

Run:

DOES NOT WORK!

```bash
litgpt convert_from_hf <data-dir>/merged <data-dir>/litgpt_merged
```

## Step 5 Test LitGPT Checkpoint

Run:

```bash
litgpt chat --checkpoint_dir ./litgpt_merged
```

Or:

```bash
litgpt generate --checkpoint_dir ./litgpt_merged --prompt "Hello"
```

## Common Issues

If the merged model gives poor answers, the most likely reason is that the adapter was merged into the wrong base model. Use the exact base model used during LoRA training.

If LitGPT conversion fails, upgrade LitGPT and check that the merged Hugging Face model directory contains `config.json`, tokenizer files, and model weight files.

## Recommended Commands

```bash
conda create -n qwen3_lora_litgpt python=3.10 -y
conda activate qwen3_lora_litgpt

pip install -U torch transformers accelerate peft safetensors litgpt

python merge_qwen3_lora.py \
  --base-model Qwen/Qwen3-4B \
  --adapter-dir ./lora_adapter \
  --output-dir ./merged_qwen3_4b

litgpt convert_from_hf ./merged_qwen3_4b ./litgpt_qwen3_4b_merged

litgpt chat --checkpoint_dir ./litgpt_qwen3_4b_merged
```

## Notes

This workflow creates a full merged model. It does not preserve the LoRA adapter as a separate LitGPT adapter.

This is the most reliable path for inference, evaluation, and deployment in LitGPT.
