import pandas as pd
import pyarrow.parquet as pq
import re

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base

db_id = r'ID'
db_pw = r'PW'
db_url = r'IP'
db_port = 'PORT'
db_nm = r'DB_NM'

tmp_str = fr'{db_id}:{db_pw}@{db_url}:{db_port}/{db_nm}'
engine = create_engine(f'mysql+pymysql://{tmp_str}')

FN = '_acc_final_preproc' 

# @log_time
def read_rsch_df() -> pd.DataFrame:
    """
    Fetch target raw data from database
    :return: raw dataframe

    """
    _q = """
        SELECT * 
            FROM limenet.sci_standard_index
    """
    # aug_df = pd.read_sql(sql=_q, con=engine)
    aug_df = pd.read_sql(sql=text(_q), con=engine.connect())

    return aug_df

# @log_time
def read_rsch_chatgpt_df() -> pd.DataFrame:
    """
    Fetch target raw data from database
    :return: raw dataframe

    """
    _q = """
        SELECT * 
            FROM limenet.sci_standard_index_rsch_appl
    """
    # aug_df = pd.read_sql(sql=_q, con=engine)
    aug_df = pd.read_sql(sql=text(_q), con=engine.connect())

    return aug_df

# @log_time
def fetch_ntis_document_2021() -> pd.DataFrame:
    """
    Fetch target raw data from database
    :return: raw dataframe

    """
    _q = """
        SELECT * 
            FROM jkdb.document_2021;
    """
    raw_data = pd.read_sql(sql=text(_q), con=engine.connect())

    return raw_data

def fetch_aug_data() -> pd.DataFrame:
    _q = """
        SELECT * 
            FROM jkdb.augmentation_process
    """
    raw_data = pd.read_sql(sql=text(_q), con=engine.connect())

    return raw_data

def fetch_chatgpt_aug_data() -> pd.DataFrame:
    _q = """
        SELECT * 
            FROM jkdb.sci_cls_chatgpt_aug_data
    """
    raw_data = pd.read_sql(sql=text(_q), con=engine.connect())

    return raw_data

# @log_time
def fetch_target_player() -> pd.DataFrame:
    """
    Fetch target raw data from database
    :return: raw dataframe

    """
    _q = """
        SELECT * 
            FROM kistep_dm.player;
    """
    raw_data = pd.read_sql(sql=text(_q), con=engine.connect())

    return raw_data

def get_labeled_data(raw_data):
    data_idx = raw_data.index

    unlabeled_data_idx = raw_data[raw_data['doc_section']=='UK'].index.tolist()
    unlabeled_data = raw_data.iloc[unlabeled_data_idx]

    labeled_data_idx = set(data_idx).difference(unlabeled_data_idx)
    labeled_data_idx = data_idx.isin(labeled_data_idx)
    labeled_data = raw_data.iloc[labeled_data_idx]
    labeled_data = labeled_data.sample(frac=1).reset_index(drop=True)

    return labeled_data, unlabeled_data
