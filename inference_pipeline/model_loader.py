"""
Model loading utilities for the FinAdvisor LLM inference pipeline.
"""

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from peft import PeftModel
import logging

from .config import BASE_MODEL, LORA_WEIGHTS, DEVICE, MODEL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and initialization of the FinAdvisor LLM model."""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = DEVICE
        
    def load_tokenizer(self):
        """Load the tokenizer for the model."""
        logger.info(f"Loading tokenizer from {BASE_MODEL}")
        self.tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        return self.tokenizer
    
    def load_model(self):
        """Load the base model and apply LoRA weights."""
        logger.info(f"Loading base model from {BASE_MODEL}")
        
        if self.device == "cuda":
            # Load base model with 8-bit quantization for CUDA
            self.model = LlamaForCausalLM.from_pretrained(
                BASE_MODEL,
                **MODEL_CONFIG
            )
            
            # Apply LoRA weights
            logger.info(f"Applying LoRA weights from {LORA_WEIGHTS}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                LORA_WEIGHTS, 
                force_download=True
            )
        else:
            # For CPU/MPS, load without quantization
            self.model = LlamaForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                device_map=None
            )
            
            # Apply LoRA weights
            logger.info(f"Applying LoRA weights from {LORA_WEIGHTS}")
            self.model = PeftModel.from_pretrained(
                self.model, 
                LORA_WEIGHTS, 
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


# Global model loader instance
_model_loader = None

def get_model_loader():
    """Get the global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
