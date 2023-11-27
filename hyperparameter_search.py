import wandb
import torch
from functools import partial
from datasets import load_dataset
from transformers import TraniningArguments, AutoTokenizer, AutoModel
from model_zoo import *
from dataset_zoo import *
from main import cfg

def initialize_wandb_sweep():
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'eval/loss', 'goal': 'minimize'},
    }
    parameters_dict = {
        'batch_size': {'values': [2, 4]},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,
        },
        'gradient_accumulation_steps': {'values': [1, 8, 16, 32]},
        'weight_decay': {'values': [0.0, 0.01, 0.05, 0.1, 0.15]},
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=cfg.project_name)
    return sweep_id


def train_param_search(
    model, training_dataset, validation_dataset, data_collator, config=None
):
    with wandb.init(config=config):
        config = wandb.config

        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            report_to='wandb',
            num_train_epochs=cfg.hyper_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size*2,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            save_strategy='epoch',
            seed=seed,
            data_seed=seed,
            optim='adamw_torch',
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            load_best_model_at_end=True,
            remove_unused_columns=False,
            fp16=True,
            fp16_opt_level='02',
            save_total_limit=3,
            logging_steps=200,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        trainer.train()

def get_best_sweep_params(sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"{cfg.project_name}/{sweep_id}")
    best_run = sorted(sweep.runs, key=lambda r: r.summary.get("eval/loss", float("inf")))[0]
    return best_run.config


def hyper_param_search():
    dataset = load_dataset(cfg.data_path)
    train_set = dataset['train']
    valid_set = dataset['valid']
    try:
        num_labels = len(train_set['labels'][0])
    except:
        num_labels = len(np.unique(train_set['labels']))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.device(device):
        base_model = AutoModel.from_pretrained(cfg.model_path, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        # In case not enough GPU for finetuning backbone
        base_model.gradient_checkpointing_enable()

    model = ConvBertWithBackbone(base_model, cfg.task_type, num_labels)

    training_dataset = FineTuneDatasetCollator(train_set)
    validation_dataset = FineTuneDatasetCollator(valid_set)
    data_collator = collate_fn(tokenizer)

    # `wandb.agent` expects a function that takes `config`,
    # so we create a new partial function with fixed arguments.
    train_fn = partial(
        train_param_search, model, training_dataset, validation_dataset, data_collator
    )

    sweep_id = initialize_wandb_sweep()
    wandb.agent(sweep_id, train_fn, count=cfg.count)
    return sweep_id