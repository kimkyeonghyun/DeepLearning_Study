# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:48:35 2023

@author: MasterUser
"""
# VGG 모델
import torch.nn as nn

def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
        )
    return model

# 컨볼루션 연산이 3번 속하는 경우
# 컨볼루션 - 활성화함수 - 컨볼루션 - 활성화함수 - 컨볼루션 - 활성화함수 - 풀링
def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
        )
    return model

# 위에서 정의한 블록들을 이용해 VGG네트워크를 구성
# 필터의 개수가 2의 n승의 값을 가지기 대문에 base_dim이랑 변수를 추가하여 단순화
class Vgg(nn.Module):
    def __init__(self, base_dim, num_classes):
        super(Vgg, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),
            conv_2_block(base_dim, 2*base_dim),
            conv_3_block(2*base_dim, 4*base_dim),
            conv_3_block(4*base_dim, 8*base_dim),
            conv_3_block(8*base_dim, 8*base_dim)
            )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim * 1 * 1, 100),
            nn.ReLU(True),
            nn.Linear(100, 20),
            nn.ReLU(True),
            nn.Linear(20, num_classes),
            )
        
        
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x