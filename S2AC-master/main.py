# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle
from data import office31_loader, office_caltech_10_loader,imagedataloader
from params import param
import time
import numpy as np
from models.model import CORAL,Deep_coral,AlexNet,LOG_CORAL,LOG_CORALAttn,LOG_CORALAttn3,LOG_CORALAttn4,CORALAttn,CORALAttn3,Deep_coralAttn, Deep_coralAttn1,Deep_coralAttnVisualization,Deep_coralAttn2,Deep_coralAttn3,Deep_coralAttn4,Deep_coralwithposition
#from models.model import *
from utils import save
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models.alexnet as ALEXNET
from sklearn.metrics import f1_score,average_precision_score,precision_recall_curve,recall_score,precision_score,plot_precision_recall_curve
import cv2 as cv

args = param()
source='Dlsr'
target='Webcam'
name='DW'

tgt_loader = office31_loader(target,batch_size=args.train_batch)
src_loader = office31_loader(source,batch_size=args.test_batch)

criterion = nn.CrossEntropyLoss()

def trainAttnet5(model,optimizer,epoch,lambda_):
    result = []
    train_steps = min(len(src_loader),len(tgt_loader))
    iter_target = iter(tgt_loader)
    iter_source = iter(src_loader)
    for i in range(train_steps):
        src_data,src_label = iter_source.next()
        x_90 = src_data.transpose(2, 3)
        x_180 = src_data.flip(2, 3)
        x_270 = src_data.transpose(2, 3).flip(2, 3)
        src_data = torch.cat((src_data, x_90, x_180, x_270), 0)
        # add auxiliary rotation loss
        rot_labels = torch.zeros(4 * args.train_batch, ).cuda()
        for j in range(4 * args.train_batch):
            if j < args.train_batch:
                rot_labels[j] = 0
            elif j < 2 * args.train_batch:
                rot_labels[j] = 1
            elif j < 3 * args.train_batch:
                rot_labels[j] = 2
            else:
                rot_labels[j] = 3
        rot_labels = F.one_hot(rot_labels.to(torch.int64), 4).float()

        if i % len(tgt_loader) == 0:
            iter_target = iter(tgt_loader)
        tgt_data, _ = iter_target.next()
        if torch.cuda.is_available():
            src_data = src_data.cuda()
            tgt_data = tgt_data.cuda()
            src_label = src_label.cuda()
        optimizer.zero_grad()
        src_out,tgt_out,src_logits = model(src_data,tgt_data)
        loss_classifier = criterion(src_out[:8],src_label)
        loss_classifier1 =torch.sum(F.binary_cross_entropy_with_logits(src_logits,rot_labels))
        if coral_type == 'CORAL':
            loss_coral = CORAL(src_out,tgt_out) if CORAL(src_out,tgt_out)>0 else -CORAL(src_out,tgt_out)
        else:
            loss_coral = LOG_CORAL(src_out,tgt_out)
        sum_loss = lambda_*loss_coral+loss_classifier
        sum_loss +=loss_classifier1
        sum_loss.backward()
        optimizer.step()
        result.append({
            'epoch': epoch,
            'step': i + 1,
            'total_steps': train_steps,
            'lambda': lambda_,
            'coral_loss': loss_coral.item(),
            'classification_loss': loss_classifier.item(),
            'selfsupervised_loss':loss_classifier1.item(),
            'total_loss': sum_loss.item()
        })
        print('Train Epoch: {:2d} [{:2d}/{:2d}]\t'
              'Lambda: {:.4f}, Class: {:.6f}, CORAL: {:.6f}, Total_Loss: {:.6f}'.format(
            epoch,
            i + 1,
            train_steps,
            lambda_,
            loss_classifier.item(),
            loss_coral.item(),
            loss_classifier1.item(),
            sum_loss.item()
        ))
    return result

def test(model,dataset_loader,every_epoch):
    model.eval()
    test_loss = 0
    corrcet = 0
    for tgt_data,tgt_label in dataset_loader:
        if torch.cuda.is_available():
            tgt_data = tgt_data.cuda()
            tgt_label = tgt_label.cuda()
        tgt_out,_,_= model(tgt_data,tgt_data)
        test_loss = criterion(tgt_out,tgt_label).item()
        pred = tgt_out.data.max(1,keepdim=True)[1]
        corrcet += pred.eq(tgt_label.data.view_as(pred)).cpu().sum()
    test_loss /= len(dataset_loader)
    return {
        'epoch': every_epoch,
        'average_loss': test_loss,
        'correct': corrcet,
        # 'precision':precision,
        # 'recall':recall,
        # 'f1':f1,
        'total': len(dataset_loader.dataset),
        'accuracy': 100. * float(corrcet) / len(dataset_loader.dataset)
    }

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    coral_type = 'CORAL'
    f1 = {'1s': 0, '1t': 0, '2s': 0, '2t': 0, '3s': 0, '3t': 0, '4s': 0, '4t': 0}
    precision=[]
    recall=[]
    modelAttnet5=Deep_coralAttn1(num_classes=31)
    optimizerAttnet5 = torch.optim.SGD([{'params': modelAttnet5.feature.parameters()},
                                 {'params':modelAttnet5.fc.parameters(),'lr':10*args.lr}],
                                lr= args.lr,momentum=args.momentum,weight_decay=args.weight_clay)
    if torch.cuda.is_available():
        modelAttnet5 = modelAttnet5.cuda()
    training_staAttnet5 = []
    test_s_staAttnet5 = []
    test_t_staAttnet5 = []
    result_lossAttnet5 = []
    for e in range(args.epochs):
        lambda_ = 10.0
        res = trainAttnet5(modelAttnet5,optimizerAttnet5,e+1,lambda_)
        print('###EPOCH {}: Class: {:.6f}, CORAL: {:.6f},Self:{:.6f}, Total_Loss: {:.6f}'.format(
            e + 1,
            sum(row['classification_loss'] / row['total_steps'] for row in res),
            sum(row['coral_loss'] / row['total_steps'] for row in res),
            sum(row['selfsupervised_loss'] / row['total_steps'] for row in res),
            sum(row['total_loss'] / row['total_steps'] for row in res),
        ))
        result_lossAttnet5.append({'epoch': e + 1,
                                 'classification_loss': sum(row['classification_loss'] / row['total_steps'] for row in res),
                                 'coral_loss': sum(row['coral_loss'] / row['total_steps'] for row in res),
                                 'selfsupervised_loss': sum(row['selfsupervised_loss'] / row['total_steps'] for row in res),
                                 'total_loss': sum(row['total_loss'] / row['total_steps'] for row in res),
                                 })
        training_staAttnet5.append(res)
        test_source5 = test(modelAttnet5, src_loader, e)
        test_target5 = test(modelAttnet5, tgt_loader, e)
        test_s_staAttnet5.append(test_source5)
        test_t_staAttnet5.append(test_target5)
        print('###Test Source: Epoch: {},avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            e + 1,
            #test_source5['f1'],
            test_source5['average_loss'],
            test_source5['correct'],
            test_source5['total'],
            test_source5['accuracy'],
        ))
        print('###Test Target: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            e + 1,
            #test_target5['f1'],
            test_target5['average_loss'],
            test_target5['correct'],
            test_target5['total'],
            test_target5['accuracy'],
        ))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
