from datasets import load_dataset, DatasetDict
from typing import Optional

class DataModule:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.data = None
        self.sampled_data_dict = None
        self.train_data = None
        self.val_data = None

    def load_and_sample(self, dataset_name: str = 'kunchum/capstone_1', sample_size: int = 20000):
        data = load_dataset(dataset_name)
        data = data.shuffle(seed=self.config.RANDOM_SEED)
        data_sample = data['train'].select(range(sample_size))
        self.sampled_data_dict = DatasetDict({'train': data_sample})
        return self.sampled_data_dict

    def generate_prompt(self, data_point):
        if data_point['context_cleaned'] != "":
            text = (
                'Below is an instruction that describes a task, paired with an input that provides '
                'further context. Write a response that appropriately completes the request.\n\n'
            )
            text += f'### Instruction:\n{data_point["instruction"]}\n\n'
            text += f'### Input:\n{data_point["context_cleaned"]}\n\n'
            text += f'### Response:\n{data_point["response_cleaned"]}'
            return text
        else:
            text = (
                'Below is an instruction that describes a task. Write a response that '
                'appropriately completes the request.\n\n'
            )
            text += f'### Instruction:\n{data_point["instruction"]}\n\n'
            text += f'### Response:\n{data_point["response_cleaned"]}'
            return text

    def tokenize(self, prompt):
        result = self.tokenizer(prompt, truncation=True, max_length=self.config.CUTOFF_LEN + 1, padding='max_length')
        return {
            'input_ids': result['input_ids'][:-1],
            'attention_mask': result['attention_mask'][:-1],
        }

    def generate_and_tokenize_prompt(self, data_point):
        if data_point['context_cleaned'] != "":
            user_prompt = (
                'Below is an instruction that describes a task, paired with an input that '
                'provides further context. Write a response that appropriately completes the request.\n\n'
            )
            user_prompt += f'### Instruction:\n{data_point["instruction"]}\n\n'
            user_prompt += f'### Input:\n{data_point["context_cleaned"]}\n\n'
            user_prompt += f'### Response:\n'
        else:
            user_prompt = (
                'Below is an instruction that describes a task. Write a response that '
                'appropriately completes the request.'
            )
            user_prompt += f'### Instruction:\n{data_point["instruction"]}\n\n'
            user_prompt += f'### Response:\n'
        len_user_prompt_tokens = len(self.tokenizer(
            user_prompt,
            truncation=True,
            max_length=self.config.CUTOFF_LEN + 1,
            padding='max_length'
        )['input_ids']) - 1
        full_tokens = self.tokenizer(
            user_prompt + data_point['response_cleaned'],
            truncation=True,
            max_length=self.config.CUTOFF_LEN + 1,
            padding='max_length',
        )['input_ids'][:-1]
        return {
            'input_ids': full_tokens,
            'labels': [-100] * len_user_prompt_tokens + full_tokens[len_user_prompt_tokens:],
            'attention_mask': [1] * len(full_tokens),
        }

    def prepare_datasets(self):
        VAL_SET_SIZE = self.config.VAL_SET_SIZE
        RANDOM_SEED = self.config.RANDOM_SEED
        if VAL_SET_SIZE > 0:
            train_val = self.sampled_data_dict['train'].train_test_split(
                test_size=VAL_SET_SIZE, shuffle=False, seed=RANDOM_SEED
            )
            self.train_data = train_val['train'].map(self.generate_and_tokenize_prompt)
            self.val_data = train_val['test'].map(self.generate_and_tokenize_prompt)
        else:
            self.train_data = self.sampled_data_dict['train'].map(self.generate_and_tokenize_prompt)
            self.val_data = None
        return self.train_data, self.val_data
