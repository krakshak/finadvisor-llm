import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

class ModelModule:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_tokenizer(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.config.MODEL_NAME_OR_PATH, add_eos_token=True
        )
        self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        return self.tokenizer

    def load_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        self.model = LlamaForCausalLM.from_pretrained(
            self.config.MODEL_NAME_OR_PATH,
            quantization_config=quantization_config,
            device_map=self.config.device_map,
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
