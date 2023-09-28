# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:47:50 2023

@author: MasterUser
"""

# 모델 ResNet
import torch.nn as nn


# Basic Block
# 컨볼루션 연산과 활성화함수는 항상 붙어 있기 때문에 이를 함수로 만듦
def conv_block_1(in_dim, out_dim, act_fn, stride = 1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 1, stride = stride),
        act_fn
        )
    return model
    
def conv_block_3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size = 3, stride = 1, padding = 1),
        act_fn,
        )
    return model
    
# Bottle Neck Module
# - bottle neck 모듈은 1x1 컨볼류션 -> 3x3 컨볼루션 -> 1x1 컨볼루션으로 이루어짐
# - down이라는 변수로 크기 감소 여부를 표시하고 조건문으로 경우의 수를 나눠 구현
# - ResNet의 skip-connection은 다순 더하기로 정의되어 있기 대문에 특성지도의 크기를 일치시켜야함
# - 차원을 맞춰주는 역할로 dim_equalizer라는 것을 정의
class BottleNeck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, act_fn, down = False):
        super(BottleNeck, self).__init__()
        self.down = down
        
        # 특성지도의 크기가 감소하는 경우
        if self.down:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, act_fn, 2),
                conv_block_3(mid_dim, mid_dim, act_fn),
                conv_block_1(mid_dim, out_dim, act_fn),
                )
            self.downsample = nn.Conv2d(in_dim, out_dim, 1, 2)
            
        # 특성지도의 크기가 그대로인 경우
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, act_fn),
                conv_block_3(mid_dim, mid_dim, act_fn),
                conv_block_1(mid_dim, out_dim, act_fn),
                )
        
        # 더하기를 위해 차원을 맞춰주는 부분
        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size = 1)
        
    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
            
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            
            out = out + x
        
        return out
    
# ResNet Model
class ResNet(nn.Module):
    def __init__(self, base_dim, num_classes):
        super(ResNet, self).__init__()
        self.act_fn = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
            ) 
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim * 4, self.act_fn),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.act_fn),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.act_fn, down = True),
            )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim * 4, base_dim * 2, base_dim * 8, self.act_fn),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.act_fn),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.act_fn),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.act_fn, down = True),
            )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim * 8, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.act_fn, down = True),
            )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim * 16, base_dim * 8, base_dim * 32, self.act_fn),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.act_fn),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.act_fn),
            )
        self.avgpool = nn.AvgPool2d(7, 1)
        self.fc_layer = nn.Linear(base_dim * 32, num_classes)
        
        
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        
        return out
    
    
    
    
    
    
    
    
    
    