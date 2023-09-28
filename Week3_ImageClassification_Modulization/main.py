# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 18:08:21 2023

@author: MasterUser
"""

import argparse
import time
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import ResNet_Cutmix as RN
import VGG_model as VGG
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description = 'PyTorch CIFAR-10 Training')
parser.add_argument('--net_type', default = 'ResNet', type = str,
                    help = 'networktype: ResNet, and VGG')
parser.add_argument('-j', '--workers', default = 4, type = int, metavar = 'N',
                    help = 'number of data loading workers (default: 4)')
parser.add_argument('--epochs', default = 90, type = int, metavar = 'N',
                    help = 'number of total epochs to run')
parser.add_argument('-b', '--batch_size', default = 256, type = int, metavar = 'N',
                    help = 'mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default = 0.01, type = float, metavar = 'LR',
                    help = 'initial learning rate')
parser.add_argument('--momentum', default = 0.9, type = float, metavar = 'M',
                    help = 'momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--base_dim', default = 16, type = int,
                    help = 'base_dim (default: 16)')
parser.add_argument('--no-bottleneck', dest = 'bottleneck', action = 'store_false',
                    help = 'to use basicblock for CIFAR datasets (default: bottleneck)') # bottleneck: 1x1 convolution으로 연산량을 줄임
parser.add_argument('--dataset', dest = 'dataset', default = 'cifar10', type = str,
                    help = 'dataset (options: cifar10, cifar100 and imagenet)')
parser.add_argument('--no-verbose', dest = 'verbose', action = 'store_false',
                    help = 'to print the status at every iteration')
parser.add_argument('--alpha', default = 300, type = float,
                    help = 'number of new channel increases per depth (default: 300)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')

parser.set_defaults(bottleneck = True)
parser.set_defaults(verbose = True)

best_err1 = 100
best_err5 = 100

def main():
    global args, best_err1, best_err5
    args = parser.parse_args()
    
    normalize = transforms.Normalize(mean = [x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std = [x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_train = transforms.Compose([
        # transforms.Resize(256),                   
        # transforms.RandomResizedCrop(224),
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])

    transform_test = transforms.Compose([
        # transforms.Resize(224),                   
        # # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalize
        ])
    
    train_data = datasets.CIFAR10(root = '../', train = True, download = True, transform = transform_train)
    val_data = datasets.CIFAR10(root = '../', train = False, download = True, transform = transform_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size = args.batch_size, shuffle = True, num_workers = args.workers, pin_memory = True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size = args.batch_size, shuffle = True, num_workers = args.workers, pin_memory = True)
    numberofclass = 10
    
    print("=> creating nodel '{}'".format(args.net_type))
    if args.net_type == 'ResNet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)
    elif args.net_type =='VGG':
        model = VGG.Vgg(args.base_dim, num_classes = numberofclass)
        
    model = torch.nn.DataParallel(model).cuda()
    
    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    criterion = nn.CrossEntropyLoss().cuda()
    
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = True)
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay = args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer,
                                                  lr_lambda = lambda epoch: 0.95 ** epoch,
                                                  last_epoch = -1,
                                                  verbose = False)
    
    cudnn.benchmark = True
    
    train_loss_list = []
    val_loss_list = []
    err1_list = []
    err5_list = []
    
    for epoch in range(0, args.epochs):
        
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        
        train_loss_list.append(train_loss)
        
        scheduler.step()
        
        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)
        
        val_loss_list.append(val_loss)
        
        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5
        
        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        err1_list.append(best_err1)
        err5_list.append(best_err5)
    
    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
    print(train_loss_list)
    print(val_loss_list)
    print(err1_list)
    print(err5_list)
    plot(args.epochs, train_loss_list, val_loss_list, err1_list, err5_list)
    
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to train mode
    model.train()
    
    end = time.time()
    current_LR = optimizer.param_groups[0]['lr']
    for i, [input, target] in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        # print(input.shape)
        target = target.cuda()
        
        output = model(input)

        loss = criterion(output, target)
        
        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk = (1, 5))
        
        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time. update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    
    return losses.avg

def validate(val_loader, model, criterion, epoch):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
    
        # switch to evaluate mode
        model.eval()
    
        end = time.time()
        for i, [input, target] in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
    
            output = model(input)
            loss = criterion(output, target)
    
            # measure accuracy and record loss
            err1, err5 = accuracy(output.data, target, topk=(1, 5))
    
            losses.update(loss.item(), input.size(0))
    
            top1.update(err1.item(), input.size(0))
            top5.update(err5.item(), input.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % args.print_freq == 0 and args.verbose == True:
                print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                      'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                    epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
    
        print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
            epoch, args.epochs, top1=top1, top5=top5, loss=losses))
        return top1.avg, top5.avg, losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))


#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))
    # print(res)
    return res

def plot(epoch, train_loss, val_loss, err1, err5):
    plt.plot(train_loss, label = 'train_loss')
    plt.plot(val_loss, label = 'val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    plt.plot(err1, label = 'err1')
    plt.plot(err5, label = 'err5')
    plt.xlabel('epoch')
    plt.ylabel('err')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()


