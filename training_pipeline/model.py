import torch
from transformers import (
    LlamaForCausalLM, LlamaTokenizer,
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from .config import MODEL_CONFIGS

class ModelModule:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_tokenizer(self):
        model_config = MODEL_CONFIGS[self.config.MODEL_NAME]
        model_type = model_config["model_type"]
        
        if model_type == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.config.BASE_MODEL, add_eos_token=True
            )
        elif model_type == "mistral":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.BASE_MODEL, add_eos_token=True
            )
        else:
            # Fallback to AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.BASE_MODEL, add_eos_token=True
            )
        
        self.tokenizer.pad_token_id = 0  # unknown token
        return self.tokenizer

    def load_model(self):
        model_config = MODEL_CONFIGS[self.config.MODEL_NAME]
        model_type = model_config["model_type"]
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        if model_type == "llama":
            self.model = LlamaForCausalLM.from_pretrained(
                self.config.BASE_MODEL,
                quantization_config=quantization_config,
                device_map=self.config.DEVICE_MAP,
            )
        elif model_type == "mistral":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.BASE_MODEL,
                quantization_config=quantization_config,
                device_map=self.config.DEVICE_MAP,
            )
        else:
            # Fallback to AutoModel
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.BASE_MODEL,
                quantization_config=quantization_config,
                device_map=self.config.DEVICE_MAP,
            )
        
        self.model = prepare_model_for_kbit_training(self.model)
        return self.model

    def apply_lora(self):
        lora_config = LoraConfig(
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            target_modules=self.config.TARGET_MODULES,
            lora_dropout=self.config.LORA_DROPOUT,
            bias='none',
            task_type='CAUSAL_LM',
        )
        self.model = get_peft_model(self.model, lora_config)
        return self.model

    def setup(self):
        self.load_tokenizer()
        self.load_model()
        self.apply_lora()
        return self.model, self.tokenizer
