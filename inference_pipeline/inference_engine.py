"""
Inference engine for the FinAdvisor LLM pipeline.
"""

import torch
from transformers import GenerationConfig
import logging

from .config import DEFAULT_GENERATION_CONFIG, DEVICE
from .prompt_utils import generate_prompt, extract_response
from .model_loader import get_model_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceEngine:
    """Handles LLM inference for financial advisory tasks."""
    
    def __init__(self):
        self.model_loader = get_model_loader()
        self.tokenizer = None
        self.model = None
        self.device = DEVICE
        self._initialized = False
    
    def initialize(self):
        """Initialize the inference engine with model and tokenizer."""
        if not self._initialized:
            logger.info("Initializing inference engine...")
            self.tokenizer, self.model = self.model_loader.initialize()
            self._initialized = True
            logger.info("Inference engine initialized successfully")
    
    def generate_response(
        self,
        instruction,
        input_text=None,
        temperature=None,
        top_p=None,
        max_new_tokens=None,
        repetition_penalty=None,
        **kwargs
    ):
        """
        Generate a response using the FinAdvisor LLM.
        
        Args:
            instruction (str): The main instruction/question
            input_text (str, optional): Additional context
            temperature (float, optional): Sampling temperature
            top_p (float, optional): Top-p sampling parameter
            max_new_tokens (int, optional): Maximum new tokens to generate
            repetition_penalty (float, optional): Repetition penalty
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response text
        """
        if not self._initialized:
            self.initialize()
        
        # Use default values if not provided
        generation_params = DEFAULT_GENERATION_CONFIG.copy()
        if temperature is not None:
            generation_params["temperature"] = temperature
        if top_p is not None:
            generation_params["top_p"] = top_p
        if max_new_tokens is not None:
            generation_params["max_new_tokens"] = max_new_tokens
        if repetition_penalty is not None:
            generation_params["repetition_penalty"] = repetition_penalty
        
        # Add any additional parameters
        generation_params.update(kwargs)
        
        # Generate prompt
        prompt = generate_prompt(instruction, input_text)
        logger.info(f"Generated prompt for instruction: {instruction[:50]}...")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Create generation config
        generation_config = GenerationConfig(**generation_params)
        
        try:
            # Generate response
            with torch.autocast(self.device if self.device == "cuda" else "cpu"):
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode output
            generated_sequence = generation_output.sequences[0]
            generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
            
            # Extract response
            response = extract_response(generated_text)
            logger.info("Response generated successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def evaluate(
        self,
        instruction,
        input_text=None,
        temperature=0.1,
        top_p=0.75,
        max_new_tokens=128,
        repetition_penalty=1.15,
        **kwargs
    ):
        """
        Evaluate function compatible with the original notebook interface.
        
        Args:
            instruction (str): The main instruction/question
            input_text (str, optional): Additional context
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            max_new_tokens (int): Maximum new tokens to generate
            repetition_penalty (float): Repetition penalty
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response text
        """
        return self.generate_response(
            instruction=instruction,
            input_text=input_text,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            **kwargs
        )


# Global inference engine instance
_inference_engine = None

def get_inference_engine():
    """Get the global inference engine instance."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine
