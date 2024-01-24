import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pickle
import pandas as pd
import argparse
import pyarrow.parquet as pq
from tqdm import tqdm

import torch

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer

from utils.db_info import *
from utils.utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

seed_everything(42)


def get_config():
    p = argparse.ArgumentParser(description="Set arguments.")
    
    p.add_argument("--file_name", default="acc_final_preproc", type=str)
    p.add_argument("--dir_path", default="/data/nevret/bert_text_classification", type=str)
    p.add_argument("--plm", default="klue/roberta-base", type=str, help="Pre-trained Language Model")
    
    config = p.parse_args()
    
    return config

def preprocess_data(config):
    current = datetime.now()
    LOGGER = get_logger(os.path.join(config.dir_path + f'/logs/preproc_{config.file_name}'))
    LOGGER.info(current.strftime('%Y-%m-%d %H:%M:%S\n'))
    LOGGER.info(config)
    
    # PREPROC_RAW_DATA
    # raw_data = get_raw_data()
    raw_data = pq.read_table(os.path.join(config.dir_path + '/total_raw_data.parquet')).to_pandas()
    raw_data = raw_data[~raw_data['doc_class'].str.endswith('99')]
    raw_data = raw_data[['doc_section', 'doc_class', 'doc_subclass', 'target']]
    
    # CHATGPT AUGMENTATION DATA
    chatgpt_data = fetch_chatgpt_aug_data()
    chatgpt_data['c_top_id'].fillna('NA_', inplace=True)
    chatgpt_data = chatgpt_data[~chatgpt_data['aug_text'].isna()].reset_index(drop=True)
    chatgpt_data = chatgpt_data[['c_top_id', 'c_mid_id', 'c_sm_id', 'aug_text']].rename(columns={'c_top_id': 'doc_section', 
                                                                                                 'c_mid_id': 'doc_class',
                                                                                                 'c_sm_id':  'doc_subclass',
                                                                                                 'aug_text': 'target'}).reset_index(drop=True)
    
    chatgpt_data['target'] = chatgpt_data['target'].apply(lambda text: clean_data(text))
    chatgpt_data.to_parquet(f'/data/nevret/kistep/chatgpt_preproc_data.parquet', engine='pyarrow', compression='gzip')

    preproc_df = pd.concat([raw_data, chatgpt_data]).sample(frac=1).reset_index(drop=True) 
    preproc_df = preproc_df[preproc_df['target']!=''].drop_duplicates('target').dropna().reset_index(drop=True)

    # PREVENT 'NA' LABEL RECOGNIZED AS 'NAN'
    preproc_df.loc[preproc_df['doc_section']=='NA', 'doc_section'] = 'NA_'

    target = 'target'    # target
    preproc_df = preproc_df.drop_duplicates(target).reset_index(drop=True)

    # LABEL ENCODING
    le = LabelEncoder()
    le.fit(preproc_df['doc_section'].unique().tolist())
    preproc_df['label'] = le.transform(preproc_df['doc_section'])
    preproc_df['label'] = preproc_df['label'].astype(int)
    LOGGER.info(f"Label: {preproc_df['label'].unique()}\n")
    
    output = open(os.path.join(config.dir_path + f'/section/data/doc_section_encoder.pkl'), 'wb')
    pickle.dump(le, output)
    output.close()

    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(config.plm)

    lengths = []
    tk0 = tqdm(preproc_df[target].fillna('').values, total=len(preproc_df))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)

    max_len = max(lengths) + 2
    LOGGER.info(f"Token max_len: {max_len}\n")

    preproc_df['token_len'] = lengths
    preproc_df['token'] = preproc_df[target].apply(tokenizer.tokenize)
    preproc_df = preproc_df[preproc_df['token_len'] > 5].reset_index(drop=True)

    # MINIMUM SUBCLASS COUNTING
    tmp = preproc_df['doc_subclass'].value_counts().to_frame()
    tmp_list = tmp[tmp['count']<=2].index.to_list()
    preproc_df = preproc_df[~preproc_df['doc_subclass'].isin(tmp_list)]

    # DIVIDE DATASET
    train, valid, _, _ = train_test_split(
        preproc_df,
        preproc_df['label'],
        random_state=42,
        test_size=0.1, 
        stratify=preproc_df['doc_subclass'],    # label
    )

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    # test = test.reset_index(drop=True)
    
    LOGGER.info(f'Train data shape: {train.shape}')
    LOGGER.info(f'Valid data shape: {valid.shape}')

    # GET TRAIN, VALID, TEST
    train.to_parquet(os.path.join(config.dir_path + f'/section/data/train_{config.file_name}.parquet'))
    valid.to_parquet(os.path.join(config.dir_path + f'/section/data/valid_{config.file_name}.parquet'))

    print('Process finish!')



if __name__ == '__main__':
    config = get_config()
    print(config)
    
    preprocess_data(config)