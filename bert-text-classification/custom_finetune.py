import os
import argparse
import random
import regex as re
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score

from tqdm import tqdm
from setproctitle import setproctitle
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from transformers import TrainingArguments, AutoTokenizer, EarlyStoppingCallback, AutoModel, AutoConfig

from utils.custom import CustomDataset, CustomTrainer, CustomModel
from utils.utils import *

PATH = '/data/nevret/bert_ftn'
LOGGER = get_logger()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_everything(42)


def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

    labels = pred.label_ids
    preds = pred.predictions    # pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    acc = accuracy_score(labels, preds)
    # auc = roc_auc_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        # 'auroc': auc
    }

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]

    return new_words

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)

    return new_words

def text_aug(sentence, alpha_rs = 0.1, num_aug=3):
    words = sentence.split(' ')
    words = [word for word in words if word != ""]
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = num_aug

    n_rs = max(1, int(alpha_rs*num_words))

    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(" ".join(a_words))

    augmented_sentences = [sentence for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    return augmented_sentences

def encoding(X_train, X_valid):
    le = LabelEncoder()
    le.fit(X_train['doc_section'])
    
    X_train['doc_section'] = le.transform(X_train['doc_section'])
    X_valid['doc_section'] = le.transform(X_valid['doc_section'])

    train_labels = {
        'doc_section': X_train['doc_section'].values
    }

    valid_labels = {
        'doc_section': X_valid['doc_section'].values
    }

    return train_labels, valid_labels

def preprocess_data(args):
    preproc_data = pq.read_table(os.path.join(PATH + '/data/final_raw_data.parquet')).to_pandas()

    preproc_data.loc[preproc_data['doc_section']=='NA', 'doc_section'] = 'NA_'

    target = 'target'
    preproc_data = preproc_data.drop_duplicates(target).reset_index(drop=True)
    preproc_data = preproc_data[['doc_id', 'kor_pjt_nm', 'kor_kywd', 'rsch_area_cls_nm', 
                                 'rsch_goal_abstract', 'rsch_abstract', 'exp_efct_abstract',
                                 'doc_section', 'doc_class', 'doc_subclass', target]]

    # Tokenizing
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    # tokenizer = AutoTokenizer.from_pretrained('./nevret/korpatbert/model/electra')

    lengths = []
    tk0 = tqdm(preproc_data[target].fillna('').values, total=len(preproc_data))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
        
    max_len = max(lengths) + 2
    LOGGER.info(f'max_len: {max_len}') 

    preproc_data['token_len'] = lengths
    preproc_data['token'] = preproc_data[target].apply(tokenizer.tokenize)

    preproc_data = preproc_data[preproc_data['token_len'] > 5].reset_index(drop=True)

    # Divide train & valid & test
    train, valid, _, _ = train_test_split(
        preproc_data,
        preproc_data['label'],
        random_state=42,
        test_size=0.1,
        stratify=preproc_data['label'],
    )                    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    train, test, _, _ = train_test_split(
        train,
        train['label'],
        random_state=42,
        test_size=0.1,
        stratify=train['label'],
    )

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # GET TRAIN, VALID, TEST
    train.to_parquet(os.path.join(PATH + '/data/train_final.parquet'))
    valid.to_parquet(os.path.join(PATH + '/data/valid_final.parquet'))
    test.to_parquet(os.path.join(PATH + '/data/test_final.parquet'))

    preprocess_train_data(train)
    print('Process finish!')
    
def preprocess_train_data(train):
    target = 'target'

    top = [15, 16, 0, 4, 14, 3]
    mid = [1, 7, 5, 19, 8, 2, 18, 6, 20, 17, 13, 23]
    bot = [26, 21, 31, 25, 9, 12, 10, 32, 11, 28, 27, 24, 29, 22, 30]

    mid_goal_data = train[train['label'].isin(mid)]
    mid_efct_data = train[train['label'].isin(mid)]
    bot_goal_data = train[train['label'].isin(bot)]
    bot_efct_data = train[train['label'].isin(bot)]

    # mid data는 rsch_goal_abstract만 추가
    mid_goal_data[target] = mid_goal_data['rsch_goal_abstract'].copy()
    # mid_efct_data[target] = mid_efct_data['exp_efct_abstract'].copy()

    # bot data는 rsch_goal_abstract, exp_efct_abstract 추가
    bot_goal_data[target] = bot_goal_data['rsch_goal_abstract'].copy()
    bot_efct_data[target] = bot_efct_data['exp_efct_abstract'].copy()

    train = pd.concat([train, mid_goal_data, bot_goal_data, bot_efct_data], axis=0).reset_index(drop=True)

    #### TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)

    lengths = []
    tk0 = tqdm(train[target].fillna('').values, total=len(train))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
        
    max_len = max(lengths) + 2
    LOGGER.info(f'max_len: {max_len}') 

    train['token_len'] = lengths
    train['token'] = train[target].apply(tokenizer.tokenize)

    train = train[train['token_len'] > 5].reset_index(drop=True)
    train.to_parquet(os.path.join(PATH + '/data/preproc_train_final.parquet'))

    #### 
    org_train = pq.read_table(os.path.join(PATH + '/data/train_final.parquet')).to_pandas()
    train = pq.read_table(os.path.join(PATH + '/data/preproc_train_final.parquet')).to_pandas()

    bot_data = train[train['label'].isin(bot)].reset_index(drop=True)
    bot_data['target_len'] = bot_data[target].apply(lambda x: len(x))
    
    split_target_df = bot_data[bot_data['target_len'] > 1400]
    split_target_df[target] = split_target_df[target].apply(lambda x: x[len(x)//2:])
    split_target_df['target_len'] = split_target_df[target].apply(lambda x: len(x))

    concat_train = pd.concat([train, split_target_df], axis=0).reset_index(drop=True)
    
    downsampling_label = [15, 16, 0, 4, 14, 
                          3, 1, 7, 5, 8, 
                          19, 2]
    down_df = pd.DataFrame()
    for label in downsampling_label:
        label_df = train[train['label']==label].reset_index(drop=True)
        label_df = label_df.sample(frac=.1, random_state=42)

        down_df = down_df.append(label_df).reset_index(drop=True)

    tmp = train[~train['label'].isin(downsampling_label)]

    result = pd.concat([tmp, down_df], axis=0).reset_index(drop=True)

    test_label = [18, 6, 20, 17, 13, 23]
    test = result[result['label'].isin(test_label)]
    result = result[~result['label'].isin(test_label)]

    tmp_df = pd.DataFrame()  
    for label in test_label:
        label_df = test[test['label']==label].reset_index(drop=True)
        label_df = label_df.sample(frac=.5, random_state=42)

        tmp_df = tmp_df.append(label_df).reset_index(drop=True)

    result = pd.concat([result, tmp_df], axis=0).reset_index(drop=True)
    result.to_parquet(os.path.join(PATH + '/data/preproc_train_split.parquet'))

def train_model(args):
    seed_everything(42)
    setproctitle(args.process_name)
    target = 'target'
    
    train = pq.read_table(os.path.join(PATH + '/data/train_final.parquet')).to_pandas()
    valid = pq.read_table(os.path.join(PATH + '/data/valid_final.parquet')).to_pandas()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    # tokenizer = AutoTokenizer.from_pretrained('./nevret/korpatbert/model/electra')

    # train = pq.read_table(os.path.join(PATH+'/data/preproc_train_final.parquet')).to_pandas()

    
    config = AutoConfig.from_pretrained(args.checkpoint_path)
    config.num_labels = 33
    # model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path, config=config)
    model = CustomModel().to(device)

    train_labels, valid_labels = encoding(train, valid)

    ### test
    # top = [15, 16, 0, 4, 14, 3]
    # train = train[train['label'].isin(top)].reset_index(drop=True)


    # Tokenized dataset
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

    # print(tokenized_train['input_ids'][0])
    # print(tokenizer.decode(tokenized_train['input_ids'][0]))

    train_dataset = CustomDataset(tokenized_train, train_labels)
    valid_dataset = CustomDataset(tokenized_valid, valid_labels)

    # print(train_dataset.__len__())
    # print(train_dataset.__getitem__(40000))
    # print(train_dataset.__getitem__(40000)['input_ids'])

    args = TrainingArguments(
        output_dir=PATH + '/result/koelectra_focal',    # 모델 저장 경로
        num_train_epochs=7,                             # epoch
        per_device_train_batch_size=32,           
        per_device_eval_batch_size=32,

        do_train=True,
        do_eval=True,
        save_strategy='epoch',
        evaluation_strategy='epoch',                    # 모델의 평가를 언제 진행할지

        fp16=True,
        fp16_opt_level='01',
        load_best_model_at_end=True
    ) 

    # Custom Trainer
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    trainer.save_model(os.path.join(PATH+'/result/koelectra_focal_loss'))
    

def inference_model(args):
    target = 'target'
    # LABEL = 'H'
    test = pq.read_table(os.path.join(PATH + '/data/test_final.parquet')).to_pandas()
    
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    model = AutoModel.from_pretrained(os.path.join(PATH+'/result/koelectra_focal_loss')).to(device)

    tokenized_test = tokenizer(
        list(test[target]),
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True,
        add_special_tokens=True,
    )

    test_dataset = CustomDataset(tokenized_test, test['label'].values)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # model.eval()
    output_pred = []
    output_prob = []

    output_pred_2 = []
    output_pred_3 = []

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

        ####
        result_2 = []
        for i in logits:
            i[np.argmax(i)]=-999
            result_2.append(i)
        
        result_2 = np.argmax(np.array(result_2), axis=-1)

        output_pred_2.append(result_2)
        ####

        ####
        result_3 = []
        for i in logits:
            i[np.argmax(i)] = -999
            result_3.append(i)
        
        result_3 = np.argmax(np.array(result_3), axis=-1)

        output_pred_3.append(result_3)
        ####

        output_pred.append(result)
        output_prob.append(prob)

    pred_answer, output_prob = np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()
    
    pred_answer_2 = np.concatenate(output_pred_2).tolist()
    pred_answer_3 = np.concatenate(output_pred_3).tolist()

    data = {'label': test['label'].values,
            'pred1': pred_answer,
            'pred2': pred_answer_2,
            'pred3': pred_answer_3,
            'prediction': [0 for i in range(len(pred_answer))]
            }

    data = pd.DataFrame(data)
    for _, row in data.iterrows():
        if row['label'] == row['pred1']:
            row['prediction'] = row['pred1']
        elif row['label'] == row['pred2']:
            row['prediction'] = row['pred2']
        elif row['label'] == row['pred3']:
            row['prediction'] = row['pred3']
        else:
            row['prediction'] = row['pred1']

    print('test dataset accuracy: ', accuracy_score(test['label'].values, data['prediction'].tolist()))

    # data.to_excel('result.xlsx', index=False)

    # print('test dataset accuracy: ', accuracy_score(test['label'].values, pred_answer))
    # print('test dataset accuracy: ', accuracy_score(test['label'].values, pred_answer_2))
    # print('test dataset accuracy: ', accuracy_score(test['label'].values, pred_answer_3))

    # Label inverse transformer
    le = LabelEncoder()
    le.fit(test['doc_section'].unique().tolist())

    test['pred_label'] = pred_answer
    test['pred_doc_section'] = le.inverse_transform(pred_answer)

    test['pred_label_3'] = data['prediction'].tolist()
    test['pred_doc_section_3'] = le.inverse_transform(data['prediction'].tolist())

    test.to_csv(os.path.join(PATH + '/result/koelectra_fl_result.csv'), index=False)
    

def inference_test(args):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    result = pd.read_csv(os.path.join(PATH + '/result/koelectra_fl_result.csv'))

    # 'NA' label이 NaN으로 인식되는 것을 막기 위해
    result.fillna('NA_', inplace=True)

    label_list = ['EA', 'EB', 'EC', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI',
                  'HA', 'HB', 'HC', 'HD', 'HE',
                  'LA', 'LB', 'LC',
                  'NA_', 'NB', 'NC', 'ND', 
                  'OA', 'OB', 'OC', 
                  'SA', 'SB', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH', 'SI']

    # print(classification_report(result['label'].values, result['pred_label'].values))
    # print(classification_report(result['label'].values, result['pred_label'].values, target_names=label_list))

    print(classification_report(result['label'].values, result['pred_label_3'].values))
    print(classification_report(result['label'].values, result['pred_label_3'].values, target_names=label_list))
    print('')
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Set arguments.")

    parser.add_argument("--seed", default="42", type=int, help="Random seed for initialization")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--eps", default=1e-5, type=float, help="The initial eps.")
    parser.add_argument("--epochs", default=3, type=int, help="Total number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--test_batch_size", type=int, default=32, help="test_batch_size")

    parser.add_argument("--no_cuda", default=False, type=bool, help="Say True if you don't want to use cuda.")
    parser.add_argument("--ensemble", default=False, type=bool, help="Ensemble.")
    parser.add_argument("--save_tensor", default=True, type=str, help="Save tensor.")
    parser.add_argument("--mode", default="train", type=str, help="When you train the model.")
    parser.add_argument("--dir_path", default="kistep_koelectra", type=str, help="Save model path.")
    parser.add_argument("--model_name", default="kistep_koelectra", type=str, help="Model name.")
    parser.add_argument("--process_name", default="kistep_koelectra", type=str, help="process_name.")
    parser.add_argument("--checkpoint_path", default="monologg/koelectra-base-v3-discriminator", type=str, help="Pre-trained Language Model.")

    args = parser.parse_args()

    if args.mode == 'train':
        # preprocess_data(args)
        train_model(args)
    
    else:
        # inference_model(args)
        inference_test(args)