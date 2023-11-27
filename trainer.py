from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from metrics import compute_metrics

def train(model, train_dataset, valid_dataset, data_collator=None):
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        report_to='wandb' if cfg.use_wandb else None,
        num_train_epochs=cfg.trainer_epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        save_strategy='epoch',
        seed=cfg.seed,
        data_seed=cfg.seed,
        optim='adamw_torch',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        remove_unused_columns=False,
        fp16=cfg.fp16,
        save_total_limit=3,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)]
    )

    if cfg.use_wandb:
        wandb.finish()

    trainer.train()
    return trainer