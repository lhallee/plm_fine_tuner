import argparse
import os
import csv
import random
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from trainer import train
from dataset_zoo import *
from model_zoo import *


def log_results(results):
    keys_to_exclude = {'test_runtime', 'test_samples_per_second', 'test_steps_per_second'}
    filtered_results = {k: v for k, v in results.items() if k not in keys_to_exclude}
    for key, value in filtered_results.items():
        if isinstance(value, (float, int)):
            filtered_results[key] = round(value, 5)
    data = {
        'model_path': cfg.model_path,
        'data_path': cfg.data_path,
        'task_type': cfg.task_type,
        **filtered_results
    }
    csv_file = cfg.log_dir
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


class cfg:
    project_name = 'test' # wandb
    wandb_api_key = None
    use_wandb = False
    model_path = 'facebook/esm2_t6_8M_UR50D' # paths
    T5 = False
    weight_path = None
    data_path = 'lhallee/dl_binary_reg'
    output_dir = './out'
    log_dir = './log.csv'
    task_type = 'binary' # settings
    # binary, mutliclass, multilabel, regression
    model_type = 'convbert'
    # linear, linear_backbone, convbert, convbert_backbone
    peft = False
    cls = False     # embed type cls, average, full
    average = False
    full = True # if full batch_size must be 1
    dropout = 0.1
    num_layers = 1
    nhead = 4
    kernel = 7
    pooling='max'
    input_dim = None    # to get updated by code
    hidden_dim = None
    num_labels = None
    training_labels_mean = None
    lr = 1e-4
    batch_size = 1
    grad_accum = 16
    weight_decay = 0.01
    fp16 = False
    trainer_epochs = 200
    trim_len = False # trim length of seqs when loading datasets
    patience = 10
    max_length = None
    r = 64 # Lora
    lora_alpha = 128
    target_modules = ['query', 'key', 'value', 'pooler/dense']
    lora_dropout = 0.0
    bias = 'none'
    count = 1 # number of hyperparameter runs
    hyper_epochs = 1 # number of hyperparameter epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 7
    test = False


def parse_args():
    parser = argparse.ArgumentParser(description='Configurations for the model.')
    parser.add_argument('--project_name', default=cfg.project_name, type=str)
    parser.add_argument('--wandb_api_key', default=cfg.wandb_api_key, type=str)
    parser.add_argument('--use_wandb', default=cfg.use_wandb, type=bool)
    parser.add_argument('--model_path', default=cfg.model_path, type=str)
    parser.add_argument('--T5', default=cfg.T5, type=bool)
    parser.add_argument('--weight_path', default=cfg.weight_path, type=str)
    parser.add_argument('--data_path', default=cfg.data_path, type=str)
    parser.add_argument('--output_dir', default=cfg.output_dir, type=str)
    parser.add_argument('--log_dir', default=cfg.log_dir, type=str)
    parser.add_argument('--task_type', default=cfg.task_type, type=str)
    parser.add_argument('--model_type', default=cfg.model_type, type=str)
    parser.add_argument('--peft', default=cfg.peft, type=bool)
    parser.add_argument('--cls', default=cfg.cls, type=bool)
    parser.add_argument('--average', default=cfg.average, type=bool)
    parser.add_argument('--full', default=cfg.full, type=bool)
    parser.add_argument('--hidden_dim', default=cfg.hidden_dim, type=int)
    parser.add_argument('--dropout', default=cfg.dropout, type=float)
    parser.add_argument('--num_layers', default=cfg.num_layers, type=int)
    parser.add_argument('--nhead', default=cfg.nhead, type=int)
    parser.add_argument('--pooling', default=cfg.pooling, type=str)
    parser.add_argument('--lr', default=cfg.lr, type=float)
    parser.add_argument('--batch_size', default=cfg.batch_size, type=int)
    parser.add_argument('--grad_accum', default=cfg.grad_accum, type=int)
    parser.add_argument('--weight_decay', default=cfg.weight_decay, type=float)
    parser.add_argument('--fp16', default=cfg.fp16, type=bool)
    parser.add_argument('--trainer_epochs', default=cfg.trainer_epochs, type=int)
    parser.add_argument('--trim_len', default=cfg.trim_len, type=bool)
    parser.add_argument('--patience', default=cfg.patience, type=int)
    parser.add_argument('--max_length', default=cfg.max_length, type=int)
    parser.add_argument('--r', default=cfg.r, type=int)
    parser.add_argument('--lora_alpha', default=cfg.lora_alpha, type=int)
    parser.add_argument('--target_modules', nargs='+', default=cfg.target_modules, type=str)
    parser.add_argument('--lora_dropout', default=cfg.lora_dropout, type=float)
    parser.add_argument('--bias', default=cfg.bias, type=str)
    parser.add_argument('--count', default=cfg.count, type=int)
    parser.add_argument('--hyper_epochs', default=cfg.hyper_epochs, type=int)
    parser.add_argument('--seed', default=cfg.seed, type=int)
    parser.add_argument('--kernel', default=cfg.kernel, type=int)
    parser.add_argument('--test', default=cfg.test, type=bool)
    return parser.parse_args()


def main():
    embed = False
    datacollator = None
    train_data, valid_data, test_data = get_data(cfg)
    if cfg.peft:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForSequenceClassification
        backbone = AutoModelForSequenceClassification.from_pretrained(cfg.model_path)
        lora_config = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.bias,
            task_type=TaskType.SEQ_CLS,
            inference_mode=True
        )
        model = get_peft_model(backbone, lora_config)
        model.print_trainable_parameters()
    elif cfg.T5:
        from transformers import T5EncoderModel
        backbone = T5EncoderModel.from_pretrained(cfg.model_path, output_hidden_states=True)
        backbone.to(cfg.device)
    else:
        backbone = AutoModel.from_pretrained(cfg.model_path, output_hidden_states=True)
        backbone.to(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    if cfg.weight_path is not None:
        backbone.load_state_dict(torch.load(cfg.weight_path))
    cfg.input_dim = backbone.config.hidden_size
    cfg.hidden_dim = backbone.config.hidden_size
    print(cfg.__dict__)
    if cfg.peft == False:
        if cfg.model_type == 'linear':
            embed = True
            model = LinearClassifier(cfg)
        elif cfg.model_type == 'linear_backbone':
            model = LinearHeadWithBackbone(backbone=backbone, cfg=cfg)
        elif cfg.model_type == 'convbert':
            embed = True
            model = ConvBert(cfg)
        elif cfg.model_type == 'convbert_backbone':
            model = ConvBertWithBackbone(backbone=backbone, cfg=cfg)
        else:
            print('Incorrect model type')
    if embed:
        train_seqs, train_labels = get_seqs(train_data)
        valid_seqs, valid_labels = get_seqs(valid_data)
        test_seqs, test_labels = get_seqs(test_data)
        if cfg.test:
            train_embds = embed_dataset(backbone, tokenizer, train_seqs[:2], cfg)
            valid_embds = embed_dataset(backbone, tokenizer, valid_seqs[:2], cfg)
            test_embds = embed_dataset(backbone, tokenizer, test_seqs[:2], cfg)
            train_dataset = FineTuneDatasetEmbeds(train_embds, train_labels[:2], cfg)
            valid_dataset = FineTuneDatasetEmbeds(valid_embds, valid_labels[:2], cfg)
            test_dataset = FineTuneDatasetEmbeds(test_embds, test_labels[:2], cfg)
        else:
            train_embds = embed_dataset(backbone, tokenizer, train_seqs, cfg)
            valid_embds = embed_dataset(backbone, tokenizer, valid_seqs, cfg)
            test_embds = embed_dataset(backbone, tokenizer, test_seqs, cfg)
            train_dataset = FineTuneDatasetEmbeds(train_embds, train_labels, cfg)
            valid_dataset = FineTuneDatasetEmbeds(valid_embds, valid_labels, cfg)
            test_dataset = FineTuneDatasetEmbeds(test_embds, test_labels, cfg)
    else:
        train_dataset = FineTuneDatasetCollator(train_data, cfg)
        valid_dataset = FineTuneDatasetCollator(valid_data, cfg)
        test_dataset = FineTuneDatasetCollator(test_data, cfg)
        datacollator = collate_fn(tokenizer)
        backbone.train()

    trainer = train(model, train_dataset, valid_dataset, cfg=cfg, data_collator=datacollator)
    predictions, labels, metrics_output = trainer.predict(test_dataset)
    log_results(metrics_output)


if __name__ == "__main__":
    args = parse_args()
    cfg.project_name = args.project_name
    cfg.wandb_api_key = args.wandb_api_key
    cfg.use_wandb = args.use_wandb
    cfg.model_path = args.model_path
    cfg.T5 = args.T5
    cfg.weight_path = args.weight_path
    cfg.data_path = args.data_path
    cfg.output_dir = args.output_dir
    cfg.log_dir = args.log_dir
    cfg.task_type = args.task_type
    cfg.model_type = args.model_type
    cfg.peft = args.peft
    cfg.cls = args.cls
    cfg.average = args.average
    cfg.full = args.full
    cfg.hidden_dim = args.hidden_dim
    cfg.dropout = args.dropout
    cfg.num_layers = args.num_layers
    cfg.nhead = args.nhead
    cfg.kernel = args.kernel
    cfg.pooling = args.pooling
    cfg.lr = args.lr
    cfg.batch_size = args.batch_size
    cfg.grad_accum = args.grad_accum
    cfg.weight_decay = args.weight_decay
    cfg.fp16 = args.fp16
    cfg.trainer_epochs = args.trainer_epochs
    cfg.trim_len = args.trim_len
    cfg.patience = args.patience
    cfg.max_length = args.max_length
    cfg.r = args
    cfg.seed = args.seed
    cfg.test = args.test
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if cfg.use_wandb:
        import wandb
        os.environ['WANDB_PROJECT'] = cfg.project_name
        os.environ['WANDB_API_KEY'] = cfg.wandb_api_key
        os.environ['WANDB_NAME'] = cfg.model_path.split('/')[-1] + '_' + cfg.task_type + '_' + cfg.data_path.split('/')[
            -1] + '_' + cfg.model_type
        wandb.login()
        wandb.init()
    main()