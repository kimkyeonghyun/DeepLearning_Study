# text_translate
# model

# 라이브러리
import numpy as np
import os
import math
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, AutoModel
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # 실행되고 있는 파일이 속한 디렉토리의 부모 디렉토리를 sys.path에 추가
from utils.utils import get_huggingface_model_name

class TranslationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None: # :뒤, -> 같은 경우는 함수의 주석 역할을 함
        super(TranslationModel, self).__init__()
        self.args = args

        if args.model_type == 'helsinki':
            model_name = get_huggingface_model_name(args.model_type)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        elif args.model_type == 'helsinki_hand':
            model_name = get_huggingface_model_name(args.model_type)
            config = AutoConfig.from_pretrained(model_name)
            config.encoder_layers = args.num_transformer_layers
            config.decoder_layers = args.num_transformer_layers
            config.encoder_attention_heads = args.num_transformer_heads
            config.decoder_attention_heads = args.num_transformer_heads
            self.model = AutoModelForSeq2SeqLM.from_config(config)

    def forward(self, src, src_attention_mask, tgt, tgt_attention_mask):

        if self.args.model_type == 'helsinki':
            # print(inspect.signature(self.model.forward))

            output = self.model(input_ids = src, attention_mask = src_attention_mask, decoder_input_ids = tgt, decoder_attention_mask = tgt_attention_mask)
            logits = output.logits
            return logits
        
        elif self.args.model_type == 'helsinki_hand':

            output = self.model(input_ids = src, attention_mask = src_attention_mask, decoder_input_ids = tgt, decoder_attention_mask = tgt_attention_mask)
            logits = output.logits
            return logits
    
    def generator(self, src, max_length):
        gen = self.model.generate(src, max_length = max_length)
        return gen

def postprocess(args, predictions, labels):
    model = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model)
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels