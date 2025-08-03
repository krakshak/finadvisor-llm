from dataclasses import dataclass, field
import os

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
    TARGET_MODULES: list = field(default_factory=lambda: ['q_proj', 'v_prol'])
    # DATA_PATH: str = 'final_dataset.json'  # Uncomment and set as needed
    OUTPUT_DIR: str = '/kaggle/working/llama_7b_tuned_1'
    DATASET_NAME: str = 'kunchum/capstone_1'
    SAMPLE_SIZE: int = 20000
    DEVICE_MAP: str = field(init=False)
    WORLD_SIZE: int = field(default_factory=lambda: int(os.environ.get('WORLD_SIZE', 1)))
    DDP: bool = field(init=False)

    def __post_init__(self):
        self.GRADIENT_ACCUMULATION_STEPS = self.BATCH_SIZE // self.MICRO_BATCH_SIZE
        self.DEVICE_MAP = 'auto'
        self.DDP = (self.WORLD_SIZE != 1)
        if self.DDP:
            self.DEVICE_MAP = {'': int(os.environ.get('LOCAL_RANK') or 0)}
            self.GRADIENT_ACCUMULATION_STEPS = self.GRADIENT_ACCUMULATION_STEPS // self.WORLD_SIZE
