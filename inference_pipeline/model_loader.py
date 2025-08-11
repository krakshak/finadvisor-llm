"""
Model loading utilities for the FinAdvisor LLM inference pipeline.
"""

import torch
from transformers import (
    LlamaTokenizer, LlamaForCausalLM,
    AutoTokenizer, AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel
import logging

from .config import MODELS, DEFAULT_MODEL, DEVICE, MODEL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and initialization of the FinAdvisor LLM model."""
    
    def __init__(self, model_name=None):
        self.model_name = model_name or DEFAULT_MODEL
        self.model_config = MODELS[self.model_name]
        self.tokenizer = None
        self.model = None
        self.device = DEVICE
        
    def load_tokenizer(self):
        """Load the tokenizer for the model."""
        base_model = self.model_config["base_model"]
        model_type = self.model_config["model_type"]
        
        logger.info(f"Loading {model_type} tokenizer from {base_model}")
        
        if model_type == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        elif model_type == "mistral":
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            # Fallback to AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return self.tokenizer
    
    def load_model(self):
        """Load the base model and apply LoRA weights."""
        base_model = self.model_config["base_model"]
        lora_weights = self.model_config["lora_weights"]
        model_type = self.model_config["model_type"]
        
        logger.info(f"Loading {model_type} base model from {base_model}")
        
        if self.device == "cuda":
            # Load base model with 8-bit quantization for CUDA
            if model_type == "llama":
                self.model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    **MODEL_CONFIG
                )
            elif model_type == "mistral":
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    **MODEL_CONFIG
                )
            else:
                # Fallback to AutoModel
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    **MODEL_CONFIG
                )
            
            # Apply LoRA weights
            logger.info(f"Applying LoRA weights from {lora_weights}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                lora_weights, 
                force_download=True
            )
        else:
            # For CPU/MPS, load without quantization
            if model_type == "llama":
                self.model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                    device_map=None
                )
            elif model_type == "mistral":
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                    device_map=None
                )
            else:
                # Fallback to AutoModel
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                    device_map=None
                )
            
            # Apply LoRA weights
            logger.info(f"Applying LoRA weights from {lora_weights}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                lora_weights, 
                force_download=True
            )
            
            # Move to device
            self.model = self.model.to(self.device)
        
        # Optimize model
        self.model.half()
        self.model.eval()
        
        # Compile model if PyTorch 2.0+
        if torch.__version__ >= "2":
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)
            
        logger.info("Model loaded and optimized successfully")
        return self.model
    
    def initialize(self):
        """Initialize both tokenizer and model."""
        self.load_tokenizer()
        self.load_model()
        return self.tokenizer, self.model


# Global model loader instances
_model_loaders = {}

def get_model_loader(model_name=None):
    """Get the model loader instance for the specified model."""
    global _model_loaders
    model_name = model_name or DEFAULT_MODEL
    
    if model_name not in _model_loaders:
        _model_loaders[model_name] = ModelLoader(model_name)
    
    return _model_loaders[model_name]

def get_available_models():
    """Get list of available models."""
    return list(MODELS.keys())
