# GPT-2 Weight Downloader

A utility script to download pretrained GPT-2 weights from OpenAI's servers and load them from TensorFlow checkpoint format into a NumPy parameter dictionary — ready to plug into a custom GPT-2 implementation.

## File

| File | Description |
|------|-------------|
| `gpt_download3.py` | Downloads GPT-2 model files and parses TF checkpoint weights into a `params` dict |

## How it works

The script has three functions:

### `download_and_load_gpt2(model_size, models_dir)`
The main entry point. Validates the model size, downloads all required files into `models_dir/<model_size>/`, then returns `settings` (hyperparams) and `params` (weights).

### `download_file(url, destination)`
Downloads a single file with a `tqdm` progress bar. Skips the download if the file already exists locally with the same size. SSL verification is disabled (`verify=False`).

### `load_gpt2_params_from_tf_ckpt(ckpt_path, settings)`
Reads the TensorFlow checkpoint and builds a nested Python dictionary of NumPy arrays. Transformer block weights (prefixed `h0`, `h1`, ...) are placed under `params["blocks"][i]`, and all other weights go into the top-level `params` dict.

## Output structure

```
settings  →  dict from hparams.json  (n_layer, n_head, n_embd, etc.)

params    →  {
    "wte": ...,        # Token embedding weights
    "wpe": ...,        # Positional embedding weights
    "blocks": [
        {              # One dict per transformer layer
            "attn": {...},
            "mlp":  {...},
            "ln_1": {...},
            "ln_2": {...},
        },
        ...
    ],
    "ln_f": {...}      # Final layer norm
}
```

## Available model sizes

| Model | Parameters |
|-------|-----------|
| `124M` | 124 million |
| `355M` | 355 million |
| `774M` | 774 million ← default |
| `1558M` | 1.5 billion |

## Requirements

```bash
pip install requests numpy tensorflow tqdm
```

## Usage

```bash
python gpt_download3.py
```

Downloads to `./gpt2/774M/` by default. Change the last two lines to use a different size:

```python
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

## Files downloaded per model

| File | Description |
|------|-------------|
| `checkpoint` | TF checkpoint pointer |
| `encoder.json` | BPE token-to-id mapping |
| `vocab.bpe` | BPE merge rules |
| `hparams.json` | Model hyperparameters |
| `model.ckpt.data-00000-of-00001` | Actual weight tensors (largest file) |
| `model.ckpt.index` | Checkpoint index |
| `model.ckpt.meta` | Checkpoint metadata |

## ⚠️ Git — don't commit the weights

Add a `.gitignore` to avoid staging the large binary files:

```
gpt2/
*.ckpt
*.ckpt.index
*.ckpt.meta
*.ckpt.data-*
```

The 774M weights alone are ~3GB and will exceed GitHub's file size limit.

## Context

Part of a hands-on GPT-2 reimplementation project, following *Build a Large Language Model From Scratch* by Sebastian Raschka.
