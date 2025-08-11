from dataclasses import dataclass, field
import os

# Model configurations for training
MODEL_CONFIGS = {
    "LLaMa 7B": {
        "base_model": "baffo32/decapoda-research-llama-7B-hf",
        "model_type": "llama",
        "target_modules": ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        "output_dir": "/kaggle/working/llama_7b_tuned"
    },
    "Mistral 7B": {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "model_type": "mistral",
        "target_modules": ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        "output_dir": "/kaggle/working/mistral_7b_tuned"
    }
}

@dataclass
class Config:
    RANDOM_SEED: int = 1234
    MICRO_BATCH_SIZE: int = 4
    BATCH_SIZE: int = 128
    GRADIENT_ACCUMULATION_STEPS: int = field(init=False)
    EPOCHS: int = 1
    LEARNING_RATE: float = 2e-5
    CUTOFF_LEN: int = 256
    LORA_R: int = 8
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 0.05
    VAL_SET_SIZE: int = 0
    MODEL_NAME: str = "LLaMa 7B"  # Default model
    BASE_MODEL: str = field(init=False)
    TARGET_MODULES: list = field(init=False)
    OUTPUT_DIR: str = field(init=False)
    DATA_PATH: str = '../datasets/final_dataset.json'
    DATASET_NAME: str = 'kunchum/capstone_1'
    SAMPLE_SIZE: int = 20000
    DEVICE_MAP: str = field(init=False)
    WORLD_SIZE: int = field(default_factory=lambda: int(os.environ.get('WORLD_SIZE', 1)))
    DDP: bool = field(init=False)

    def __post_init__(self):
        # Set model-specific configurations
        if self.MODEL_NAME in MODEL_CONFIGS:
            model_config = MODEL_CONFIGS[self.MODEL_NAME]
            self.BASE_MODEL = model_config["base_model"]
            self.TARGET_MODULES = model_config["target_modules"]
            self.OUTPUT_DIR = model_config["output_dir"]
        else:
            raise ValueError(f"Model {self.MODEL_NAME} not supported. Available models: {list(MODEL_CONFIGS.keys())}")
        
        # Standard configurations
        self.GRADIENT_ACCUMULATION_STEPS = self.BATCH_SIZE // self.MICRO_BATCH_SIZE
        self.DEVICE_MAP = 'auto'
        self.DDP = (self.WORLD_SIZE != 1)
        if self.DDP:
            self.DEVICE_MAP = {'': int(os.environ.get('LOCAL_RANK') or 0)}
            self.GRADIENT_ACCUMULATION_STEPS = self.GRADIENT_ACCUMULATION_STEPS // self.WORLD_SIZE
