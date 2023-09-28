# text_translate
# preprocessing

# 라이브러리
import os
import sys
import pickle
import argparse
import bs4
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoConfig

# 대규모 데이터 셋 라이브러리
from datasets import load_dataset
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return lines

def load_data(args: argparse.Namespace):

    name = args.task_dataset.lower()

    train_data = {
        'EN_text': load_text_file('./data/train.en'),
        'DE_text': load_text_file('./data/train.de')
    }
    valid_data = {
        'EN_text': load_text_file('./data/val.en'),
        'DE_text': load_text_file('./data/val.de')
    }
    test_data = {
        'EN_text': load_text_file('./data/test.en'),
        'DE_text': load_text_file('./data/test.de')
    }
    # print(train_data['DE_text'])
    return train_data, valid_data, test_data

def preprocessing(args: argparse.Namespace) -> None:

    # Load data
    train_data, valid_data, test_data = load_data(args)

    model = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)


    # Preprocessing - data_dict 정의
    # 딕셔너리 형태로 선언하면 key와 value형태로 불러오기가 쉬움
    data_dict = {
        'train': {
            'EN_text_ids': [],
            'DE_text_ids': [],
            'src_attention_mask': [],
            'tgt_attention_mask': [],
            'src_vocab_size': config.vocab_size,
            'tgt_vocab_size': config.vocab_size
        },
        'valid': {
            'EN_text_ids': [],
            'DE_text_ids': [],
            'src_attention_mask': [],
            'tgt_attention_mask': [],
            'src_vocab_size': config.vocab_size,
            'tgt_vocab_size': config.vocab_size
        },
        'test': {
            'EN_text_ids': [],
            'DE_text_ids': [],
            'src_attention_mask': [],
            'tgt_attention_mask': [],
            'src_vocab_size': config.vocab_size,
            'tgt_vocab_size': config.vocab_size
        }
    }

    # pickle 파일로 데이터 저장
    # pickle은 list dictionary 같은 객체를 그 형태 그대로 저장하고 불러올 수 있음
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in tqdm(range(len(split_data['EN_text'])), desc= 'Preprocessing', position=0, leave=True):
            # prefix = "translate English to French: "
            src = split_data['EN_text'][idx]
            tgt = split_data['DE_text'][idx]
            src_tokenized = tokenizer(src, padding='max_length', truncation=True,
                               max_length=args.max_seq_len, return_tensors='pt')
            with tokenizer.as_target_tokenizer():
                tgt_tokenized = tokenizer(tgt, padding='max_length', truncation=True,
                                max_length=args.max_seq_len, return_tensors='pt')
                
            data_dict[split]['EN_text_ids'].append(src_tokenized['input_ids'].squeeze())
            data_dict[split]['DE_text_ids'].append(tgt_tokenized['input_ids'].squeeze())
            data_dict[split]['src_attention_mask'].append(src_tokenized['attention_mask'].squeeze())
            data_dict[split]['tgt_attention_mask'].append(tgt_tokenized['attention_mask'].squeeze())

        # pickle 파일로 데이터 저장
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)