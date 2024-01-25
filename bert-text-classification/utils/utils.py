import os, re
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(self, encodings, label):
        self.encodings = encodings
        self.label = label

    def __getitem__(self, idx):
        item = {k:v[idx].clone().detach() for k, v in self.encodings.items()}
        item['label'] = torch.tensor(self.label[idx])
        return item

    def __len__(self):
        return len(self.label)

def clean_data(raw_text):
    def load_stopwords():
        with open('/data/nevret/bert-finetuning-custom/stopwords.txt', 'r') as f:
            stopwords = f.readlines()
        return stopwords[0].split(',')

    stopwords = load_stopwords()
    # words = ''.join(text.tolist())

    # 한글&숫자만 남기기
    hangul = re.compile("[^a-zA-Z0-9가-힣]")
    text = hangul.sub(' ', raw_text)

    # 숫자+문자 제거하기 (ex. '2009년도', '7월', ...)
    text = re.sub('\\d+?[A-Za-z가-힣]*', '', text)

    # 숫자로만 이루어진 단어 제거
    text = re.sub(r'\b\d+\b', '', text)
    
    # 오직 영어로 이루어진 문장 제거
    text = re.sub(r'^[a-zA-Z\s]+$', '', text)

    # 영어와 숫자가 붙어있는 단어 제거
    text = re.sub(r'\b[a-zA-Z]+\d+\b|\b\d+[a-zA-Z]+\b', '', text)

    # 문자열 
    text = re.sub('[ +]', ' ', text)

    # 불용어 제거
    text = [x for x in text.split(' ') if x not in stopwords and len(x) > 1]

    # join 후 공백 제거
    return ' '.join(text).strip()

def get_logger(filename='./train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    
    # Check handler exists
    if len(logger.handlers) > 0:
        return logger # Logger already exists
    
    logger.setLevel(INFO)

    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger

def seed_everything(seed:int = 1004):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  
