import math
import mlflow
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback

class PerplexityCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            perplexity = math.exp(logs["loss"]) if logs["loss"] < 100 else float("inf")
            logs["perplexity"] = perplexity
            print(f"Step {state.global_step} - Training Loss: {logs['loss']:.4f} - Perplexity: {perplexity:.4f}")
            # Log metrics to MLflow
            mlflow.log_metric("loss", logs["loss"], step=state.global_step)
            mlflow.log_metric("perplexity", perplexity, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            for k, v in metrics.items():
                mlflow.log_metric(k, v, step=state.epoch if state.epoch is not None else state.global_step)

    def on_save(self, args, state, control, **kwargs):
        # Log model checkpoint directory as artifact
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        if os.path.exists(checkpoint_dir):
            mlflow.log_artifacts(checkpoint_dir, artifact_path=f"checkpoints/checkpoint-{state.global_step}")

class TrainerModule:
    def __init__(self, config, model, tokenizer, train_data, val_data=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.trainer = None

    def setup_trainer(self):
        training_args = TrainingArguments(
            seed=self.config.RANDOM_SEED,
            data_seed=self.config.RANDOM_SEED,
            per_device_train_batch_size=self.config.MICRO_BATCH_SIZE,
            gradient_accumulation_steps=self.config.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=self.config.EPOCHS,
            learning_rate=self.config.LEARNING_RATE,
            fp16=True,
            logging_steps=20,
            save_strategy='steps',
            save_steps=50,
            output_dir=self.config.OUTPUT_DIR,
            save_total_limit=3,
            load_best_model_at_end=True if self.config.VAL_SET_SIZE > 0 else False,
            ddp_find_unused_parameters=False if getattr(self.config, 'ddp', False) else None,
        )
        self.trainer = Trainer(
            model=self.model,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[PerplexityCallback()]
        )
        self.model.config.use_cache = False
        return self.trainer

    def train(self):
        if self.trainer is None:
            self.setup_trainer()
        return self.trainer.train()
