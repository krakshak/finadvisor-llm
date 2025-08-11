"""
End-to-end LLaMA training using modular pipeline.
"""
import os
from training_pipeline.config import Config
from training_pipeline.utils import set_seed, mlflow_set_experiment, mlflow_start_run
from training_pipeline.model import ModelModule
from training_pipeline.data import DataModule
from training_pipeline.trainer import TrainerModule
from training_pipeline.evaluator import EvaluatorModule


def main():
    # 1. Config and Environment
    config = Config()
    set_seed(config.RANDOM_SEED)
    mlflow_set_experiment("llama-training")
    mlflow_start_run()
    # Optionally log config params
    import mlflow
    mlflow.log_params({k: v for k, v in config.__dict__.items() if not k.startswith('__') and not callable(v)})

    # 2. Model & Tokenizer
    model_module = ModelModule(config)
    model, tokenizer = model_module.setup()

    # 3. Data
    data_module = DataModule(config, tokenizer)
    data_module.load_and_sample(dataset_name=config.DATASET_NAME, sample_size=config.SAMPLE_SIZE)
    train_data, val_data = data_module.prepare_datasets()

    # 4. Evaluator/metrics
    evaluator = EvaluatorModule()

    # 5. Trainer with metrics
    trainer_module = TrainerModule(
        config, model, tokenizer, train_data, val_data
    )
    trainer = trainer_module.setup_trainer()
    trainer.compute_metrics = evaluator.compute_metrics

    # 6. Train
    trainer.train()

    # 7. Save final model and tokenizer, and log as MLflow artifact
    output_dir = os.path.join(config.OUTPUT_DIR, "final_model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    import mlflow
    mlflow.log_artifacts(output_dir, artifact_path="final_model")

if __name__ == "__main__":
    main()
