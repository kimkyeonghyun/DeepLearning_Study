# text_classification
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
from utils import check_path, get_huggingface_model_name

def load_data(args: argparse.Namespace):

    name = args.task_dataset.lower()
    train_valid_split = args.train_valid_split

    train_data = {
        'image': [],
        'label': []
    }
    valid_data = {
        'image': [],
        'labe': []
    }
    test_data = {
        'image': [],
        'label': []
    }

    if name == 'imdb':
        dataset = load_dataset('imdb')

        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        # train-valid split
        train_df = train_df.sample(frac = 1).reset_index(drop = True) 
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['text'] = train_df['text'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['text'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['text'].tolist()
        test_data['label'] = test_df['label'].tolist()

    return train_data, valid_data, test_data, num_classes

def preprocessing(args: argparse.Namespace) -> None:

    # Load data
    train_data, valid_data, test_data, num_classes = load_data(args)

    model = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)


    # Preprocessing - data_dict 정의
    # 딕셔너리 형태로 선언하면 key와 value형태로 불러오기가 쉬움
    data_dict = {
        'train': {
            'texts': [],
            'labels': [],
            'attention_mask': [],
            'token_type_ids': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size
        },
        'valid': {
            'texts': [],
            'labels': [],
            'attention_mask': [],
            'token_type_ids': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size
        },
        'test': {
            'texts': [],
            'labels': [],
            'attention_mask': [],
            'token_type_ids': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size
        }
    }

    # pickle 파일로 데이터 저장
    # pickle은 list dictionary 같은 객체를 그 형태 그대로 저장하고 불러올 수 있음
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in tqdm(range(len(split_data['text'])), desc= 'Preprocessing', position=0, leave=True):
            text = split_data['text'][idx]
            label = split_data['label'][idx]

            token = tokenizer(text, padding='max_length', truncation=True,
                               max_length=args.max_seq_len, return_tensors='pt')
            # print("token['attention_mask']", token['attention_mask'])
            data_dict[split]['texts'].append(token['input_ids'].squeeze())
            # print(token["input_ids"].squeeze(), token["input_ids"].squeeze().shape)
            data_dict[split]['labels'].append(torch.tensor(label, dtype=torch.long))
            
            # print("data_dict[split]['attention_mask']", data_dict[split]['attention_mask'])
            # print(token['attention_mask'].squeeze()
            if args.model_type == 'bert':
                data_dict[split]['attention_mask'].append(token['attention_mask'].squeeze())
                data_dict[split]['token_type_ids'].append(token['token_type_ids'].squeeze())

        # pickle 파일로 데이터 저장
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)