# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:12:59 2023

@author: MasterUser
"""

# 파이토치 분류 모델

# 필요한 라이브러리
import time
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset # datasets에 다양한 데이터를 사용하기 용이하게 정리되어있음
import torchvision.transforms as transforms # 이미지 데이터를 자르거나 확대 및 다양하게 변형시키는 함수들이 구현
from torch.utils.data import DataLoader # 전처리가 끝난 데이터들을 지정한 배치 크기에 맞게 모아서 전달해주는 역할
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameter
batch_size = 100
learning_rate = 0.0002
num_epoch = 1000

# 데이터
trainset = dset.ImageFolder('C:/Users/MasterUser/Desktop/중앙대학교 인턴/파이토치 구조 이해(+데이터셋 로드)/cifar10/train', transform = transforms.Compose([transforms.ToTensor()]), target_transform = None)
testset = dset.ImageFolder('C:/Users/MasterUser/Desktop/중앙대학교 인턴/파이토치 구조 이해(+데이터셋 로드)/cifar10/test', transform = transforms.Compose([transforms.ToTensor()]), target_transform = None)

# 데이터셋 확인
print(trainset.__getitem__(0)[0].size(), trainset.__len__())
print(testset.__getitem__(0)[0].size(), testset.__len__())
print(len(trainset), len(testset))

# dataloader 설정
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 2)
test_loader = DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)

# Model
# Basic blocks
# - 모델에 반복되는 부분이 많기 때문에 이를 함수로 만들어 단순화함
# - 맨 위에 이미지를 보면 컨볼루션 연산이 2번 연속하는 경우와 3번 연속하는 경우가 있는데 이를 각각 만들어줌
# - 아래의 코드는 최적의 방법이라기 보다는 그림의 구조를 모방한 코드

# 컨볼류션 연산이 2번 연속하는 경우
# 컨볼류션 - 활성화함수 - 컨볼루션 - 활성화함수 - 풀링
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
    def __init__(self, base_dim, num_classes = 10):
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

# Optimizer & Loss
# gpu가 사용 가능한 경우에는 device를 0번 gpu로 설정하고 불가능하면 cpu로 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 앞서 정의한대로 Vgg 클래스를 인스턴스화 하고 지정한 장치에 올림
model = Vgg(base_dim = 16).to(device)

# 손실함수 및 최적화 함수를설정
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# 모델 자녀 노드이 이름과 모듈을 출력
for i in model.named_children():
    print(i)
        
# 학습
loss_arr = []
for i in tqdm(range(num_epoch)):
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device)
        y_ = label.to(device)
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()
        
        if j % 1000 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())
        
# 학습시 손실 시각화
plt.plot(loss_arr)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 테스트 데이터 정확도 측정
correct = 0 # 맞은 개수
total = 0 # 전체 개수
        
# 인퍼런스 모드를 위해 no_grad해줌
with torch.no_grad():
    # 테스트로더에서 이미지와 정답을 불러옴
    for image, label in test_loader:
        x = image.to(device)
        y_ = label.to(device)
        
        output = model.forward(x)
        
        # torch.max를 이용해 최대 값 및 최대값 인덱스를 뽑아냄, 최대값은 필요없기 때문에 인데스만 사용
        _, output_index = torch.max(output, 1)
        
        # 전체 개수는 라벨의 개수로 더해줌
        total += label.size(0) # 전체 개수를 알고 있음에도 batch_size, drop_last의 영향으로 몇몇 데이터가 잘릴수 있기 때문
        
        # 모델의 결과의 최대값 인덱스와 라벨이 일치하는 개수를 correct에 더해줌
        correct += (output_index == y_).sum().float()
    
    # 테스트 데이터 전체에 대해 위의 작업을 시행한 후 정확돌르 구해줌
    print(f'Accuracy of Test Data: {100*correct/total}%')
        
        
        