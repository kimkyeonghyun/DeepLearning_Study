# text classification
# model

# 라이브러리
import os
import math
import sys
import argparse
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # 실행되고 있는 파일이 속한 디렉토리의 부모 디렉토리를 sys.path에 추가
from utils.utils import get_huggingface_model_name

class ClassificationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None: # :뒤, -> 같은 경우는 함수의 주석 역할을 함
        super(ClassificationModel, self).__init__()
        self.args = args

        if args.model_type == 'lstm':
            self.embedding = nn.Embedding(num_embeddings = args.vocab_size, embedding_dim = args.embedding_dims)
            
            self.model = nn.LSTM(input_size = args.embedding_dims, hidden_size = args.hidden_size, num_layers = args.num_layers, batch_first=True, bidirectional = True)

            # 마지막 분류 layer 정의
            self.classifier = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.Dropout(self.args.dropout_rate),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.num_classes),
            )
        
        elif args.model_type == 'transformer_encoder':
            self.embedding = nn.Embedding(num_embeddings = args.vocab_size, embedding_dim = args.embedding_dims)
            # self.pos_encoder = PositionalEncoding(args.embedding_dims, args.max_seq_len, args.device)
            self.pos_encoder = PositionalEncoding(args.embedding_dims, args.dropout_rate, args.max_seq_len)
            encoder_layer = nn.TransformerEncoderLayer(args.embedding_dims, args.num_transformer_heads, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, args.num_transformer_layers)
            self.classifier = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.Dropout(self.args.dropout_rate),
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.num_classes),
            )
        
        elif args.model_type == 'bert':
            model_name = get_huggingface_model_name(args.model_type)
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels = args.num_classes)
            self.hidden_size = self.model.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(self.args.dropout_rate),
                nn.ReLU(),
                nn.Linear(self.hidden_size, args.num_classes),
            )

    def forward(self, texts, attention_mask, token_type_ids):
        if self.args.model_type == 'lstm':
            texts = self.embedding(texts)  # Embed the input: [batch_size, seq_length] -> [batch_size ,seq_length ,embed_dim]
            output,(hidden_state,_)=self.model(texts)  # Pass through LSTM: output shape -> [batch_size ,seq_length ,hidden_dim]
            final_output=torch.cat((hidden_state[-2,:], hidden_state[-1,:]), dim = 1)

            logits=self.classifier(final_output)  # Pass through classifier: [batch_size ,hidden_dim] -> [batch_size,num_classes]
        
        elif self.args.model_type == 'transformer_encoder':
            # print('texts', texts.shape)
            emb_texts = self.embedding(texts) # Embed the input: [batch_size, seq_length] -> [batch_size ,seq_length ,embed_dim]
            # print('emb_texts', emb_texts.shape)
            pos_texts = self.pos_encoder(emb_texts* math.sqrt(self.args.embedding_dims)) # [batch_size ,seq_length ,embed_dim]
            # print('pos_texts', pos_texts.shape)
            # pos_texts = pos_texts.unsqueeze(0).repeat(texts.size(0), 1, 1) # [batch_size, seq_len, embed_size]
            # print('pos_texts', pos_texts.shape)
            # trasformer expects [seq_len, batch_size, embed_size]
            input_ = pos_texts.permute(1, 0, 2) # [seq_len, batch_size, embed_size]
            output=self.encoder(input_) # [seq_len, batch_size, embed_size]
            # print('output', output.shape)
            output=output.permute(1, 0, 2) # [batch_size, seq_len, embed_size]
            # print('output', output.shape)
            final_output=torch.mean(output,dim=1)  # [batch_size, embed_size] - Take average across sequence length dimension 
            # print('final_output', final_output.shape)
            logits=self.classifier(final_output)   
        
        elif self.args.model_type == 'bert':
            output = self.model(input_ids = texts, attention_mask = attention_mask, token_type_ids = token_type_ids, return_dict=True)
            logits = output.logits
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # position: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self,x):
         x=x+ self.pe[:x.size(0), :] # [batch_size ,seq_length ,embed_dim]
         return self.dropout(x)