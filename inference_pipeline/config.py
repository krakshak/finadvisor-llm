"""
Configuration settings for the FinAdvisor LLM inference pipeline.
"""

import torch

# Model Configuration
BASE_MODEL = "baffo32/decapoda-research-llama-7B-hf"
LORA_WEIGHTS = "kunchum/capstone-llama-finetuned"

# Device Configuration
def get_device():
    """Determine the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"

DEVICE = get_device()

# Generation Configuration
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 128,
    "repetition_penalty": 1.15,
    "do_sample": True,
}

# Model Loading Configuration
MODEL_CONFIG = {
    "load_in_8bit": True,
    "device_map": "auto",
}
