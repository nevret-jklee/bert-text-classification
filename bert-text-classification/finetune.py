import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import wandb
import pandas as pd
import argparse
import pyarrow.parquet as pq

import torch
from datetime import datetime

from transformers import TrainingArguments, Trainer, ElectraTokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
# from konlpy.tag import Mecab

from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
                  
seed_everything(42)


def get_config():
    p = argparse.ArgumentParser(description="Set arguments.")

    p.add_argument("--file_name", default="acc_final_preproc", type=str)
    p.add_argument("--plm", default="klue/roberta-base", type=str, help="Pre-trained Language Model")
    p.add_argument("--dir_path", default="/data/nevret/bert-finetuning-custom/bert-text-classification", type=str)
    
    p.add_argument("--seed", default="42", type=int)
    p.add_argument("--learning_rate", default=2e-5, type=float) 
    p.add_argument("--epochs", default=5, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    
    config = p.parse_args()

    return config

def train_model(config):
    current = datetime.now()
    LOGGER = get_logger(os.path.join(config.dir_path + f'/logs/train_{config.file_name}'))
    LOGGER.info(current.strftime('%Y-%m-%d %H:%M:%S\n'))
    
    # setproctitle(config.process_name)
    target = 'target'

    train = pq.read_table(os.path.join(config.dir_path + f'/data/train_{config.file_name}.parquet')).to_pandas()
    valid = pq.read_table(os.path.join(config.dir_path + f'/data/valid_{config.file_name}.parquet')).to_pandas()
    
    # TOKENIZER
    if config.plm == 'kykim/electra-kor-base':
        tokenizer = ElectraTokenizerFast.from_pretrained(config.plm)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.plm)

    # MODEL
    model_config = AutoConfig.from_pretrained(config.plm)
    model_config.num_labels = len(train['label'].unique())
    model = AutoModelForSequenceClassification.from_pretrained(config.plm, config=model_config).to(device)
    
    LOGGER.info(f'PLM: \n{config.plm}\n')
    LOGGER.info(f'Model: \n{model}\n')
    LOGGER.info(f'Config: \n{config}\n')
    
    tokenized_train = tokenizer(
        # [' '.join(train[target][i]) for i in range(len(train[target]))],    
        list(train[target]),
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
    )

    tokenized_valid = tokenizer(
        # [' '.join(valid[target][i]) for i in range(len(valid[target]))],
        list(valid[target]),
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
    )

    # TOKENIZED DATASET
    train_dataset = BERTDataset(tokenized_train, train['label'].values)
    valid_dataset = BERTDataset(tokenized_valid, valid['label'].values)

    # WANDB
    WANDB_AUTH_KEY = os.getenv('WANDB_AUTH_KEY')
    wandb.login(key=WANDB_AUTH_KEY)

    wandb_name = f'Section-MD:{config.plm[5:]}_EP:{config.epochs}_BS:{config.batch_size}_FN:{config.file_name}'
    # wandb.init(...)
    wandb.init(
        entity='nevret',
        project='kistep_result',
        name=wandb_name,
        group=wandb_name,
        job_type='train',
    )

    args = TrainingArguments(
        output_dir=f"{config.dir_path}/{wandb_name}",
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        save_strategy='epoch',
        evaluation_strategy='epoch',   
        fp16=True,                     # 16 bit 저장
        fp16_opt_level='01',           # 최적화 수준 level
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,                          # 학습할 모델
        args=args,                            # TrainingArguments
        train_dataset=train_dataset,          # 학습 데이터셋
        eval_dataset=valid_dataset,           # 평가 데이터셋
        tokenizer=tokenizer,                  # 토크나이저
        compute_metrics=compute_metrics,      # 평가 지표 계산 함수
        #data_collator=default_data_collator  # Collator
    )

    # Training
    trainer.train()
    
    model.save_pretrained(os.path.join(config.dir_path+f'/result_{wandb_name}'))



if __name__ == '__main__':
    config = get_config()
    print(config)
    
    train_model(config)
    
