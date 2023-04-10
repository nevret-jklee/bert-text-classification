import os
from ssl import ALERT_DESCRIPTION_NO_RENEGOTIATION
import pandas as pd
import pickle
import numpy as np

import pyarrow.parquet as pq
from setproctitle import setproctitle
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, ElectraForSequenceClassification, AdamW

from db import fetch_target_raw_data
from utils import get_logger, seed_everything

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PATH = '/data/nevret/bert_ftn'
LOGGER = get_logger()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_everything(42)

class ELECTRADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, label=None):
        self.encodings = encodings
        self.label = label

    def __getitem__(self, idx=0):
        item = {k:v[idx].clone().detach() for k, v in self.encodings.items()}
        item['label'] = torch.tensor(0)

        return item

    def __len__(self):
        return len([0])
    

def encoder_save():
    test = pq.read_table(os.path.join(PATH + '/data/test_final.parquet')).to_pandas()

    le = LabelEncoder()
    le.fit(test['doc_section'].unique().tolist())
    output = open('doc_section_encoder.pkl', 'wb')
    pickle.dump(le, output)
    output.close()


def predict(input):
    test = pq.read_table(os.path.join(PATH + '/data/test_final.parquet')).to_pandas()

    tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(PATH+'/result/koelectra_nm_kywd_abst_gabst')).to(device)

    tokenized_test = tokenizer(
        input, # list(test[target]),
        # list(test['target']),
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
    )

    input_dataset = ELECTRADataset(tokenized_test, 0)
    input_loader = DataLoader(input_dataset, batch_size=1, shuffle=False)

    output_pred, output_prob = [], []

    for i, data in enumerate(tqdm(input_loader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device),
            )

        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()

        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    pred_answer, output_prob = np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()
    print('')

    doc_section = open('doc_section_encoder.pkl', 'rb')
    le = pickle.load(doc_section)
    doc_section.close()

    pred_section = le.inverse_transform(np.array(pred_answer).reshape(-1, 1))
    
    print('')


if __name__ == '__main__':
    predict('축산물 수산물')
    print('')