"""
Configuration settings for the FinAdvisor LLM inference pipeline.
"""

import torch

# Available Models Configuration
MODELS = {
    "LLaMa 7B": {
        "base_model": "baffo32/decapoda-research-llama-7B-hf",
        "lora_weights": "kunchum/capstone-llama-finetuned",
        "model_type": "llama"
    },
    "Mistral 7B": {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "lora_weights": "kunchum/capstone-mistral-finetuned",
        "model_type": "mistral"
    }
}

# Default Model
DEFAULT_MODEL = "LLaMa 7B"

# Backward compatibility
BASE_MODEL = MODELS[DEFAULT_MODEL]["base_model"]
LORA_WEIGHTS = MODELS[DEFAULT_MODEL]["lora_weights"]

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
