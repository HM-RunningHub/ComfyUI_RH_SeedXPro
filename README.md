# ComfyUI SeedXPro Translation Node

## Features

- **Simple and Easy to Use**: Ready to use out of the box, no complex configuration required
- **No Special Dependencies**: No additional special dependency packages required besides standard dependencies
- **Plug and Play**: Simply place in ComfyUI's custom_nodes directory to use
- Multi-language translation powered by ByteDance-Seed/Seed-X-PPO-7B model
- **Automatic Model Download**: Automatically downloads the model from Hugging Face to `models/Seed-X-PPO-7B` directory on first use
- No manual model download required

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place this plugin in the `custom_nodes` directory of ComfyUI
2. On first run, the system will automatically check and download the `ByteDance-Seed/Seed-X-PPO-7B` model
3. The model will be downloaded to ComfyUI's `models/Seed-X-PPO-7B` directory
4. After download completion, you can use the translation feature normally

## Notes

- The model file is large (~13GB), first download requires time and network bandwidth
- Ensure sufficient disk space for storing model files
- CUDA environment required for running the model
