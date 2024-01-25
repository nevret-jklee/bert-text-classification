import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pickle
import pandas as pd
import numpy as np
import argparse
import pyarrow.parquet as pq
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.utils import BERTDataset, get_logger, seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)


def get_config():
    p = argparse.ArgumentParser(description="Set arguments.")

    p.add_argument("--epochs", default=5, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--file_name", default="acc_final_preproc", type=str)
    p.add_argument("--dir_path", default="/data/nevret/bert-finetuning-custom/bert-text-classification", type=str)
    p.add_argument("--plm", default="klue/roberta-base", type=str, help="Pre-trained Language Model")

    config = p.parse_args()

    return config

def predict(config):
    target = 'target'

    LOGGER = get_logger(os.path.join(config.dir_path + f'/logs/predict_{config.file_name}'))
    LOGGER.info(f"========== doc_section ==========")

    # PREDICT DATASET
    test = pq.read_table(os.path.join(config.dir_path + f'/data/valid_{config.file_name}.parquet')).to_pandas()

    # LABEL ENCODING
    doc_section = open(config.dir_path+f'/data/doc_section_encoder.pkl', 'rb')
    le = pickle.load(doc_section)
    doc_section.close()

    tokenizer = AutoTokenizer.from_pretrained(config.plm)
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(config.dir_path + f'/model/')).to(device)
    
    tokenized_test = tokenizer(
        list(test[target]),
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
    )

    test_dataset = BERTDataset(tokenized_test, test['label'].values)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # model.eval()
    output_prob, output_pred = [], []
    for i, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device),
            )

        logits = outputs[0]    # tensor
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()    # array
        logits = logits.detach().cpu().numpy()    # array

        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    pred_answer, output_prob = np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()
    # |output_prob|: (test_size/batch_size, batch_size, label_size)
    # |output_prob|: (2306, 32, 33)

    LOGGER.info(f"Accuracy: {accuracy_score(test['label'].values, pred_answer)}")
                 
    label_list = [i for i in list(le.classes_) if i in list(test['doc_section'].unique())]
    label_list = [i if i != 'NA_' else 'NA' for i in label_list]

    test['pred_label'] = pred_answer
    test['pred_doc_section'] = le.inverse_transform(pred_answer)

    LOGGER.info(classification_report(test['label'].values, test['pred_label'].values, target_names=label_list))
    
    

if __name__ == '__main__':
    config = get_config()
    print(config)
    
    predict(config)
