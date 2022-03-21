#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:46:06 2022

@author: alex
"""
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy

from Average import AverageMeter
from Tripletloss import batch_all_triplet_loss1
from Model import AD_3DCNN

########################################################################
def momentum_update(model_q, model_k, beta = 0.999):
    param_k = model_k.state_dict()
    param_q = model_q.named_parameters()
    for n, q in param_q:
        if n in param_k:
            param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
    model_k.load_state_dict(param_k)

def queue_data(data, k):
    return torch.cat([data, k], dim=0)

def dequeue_data(data, K=8):
    if len(data) > K:
        return data[-K:]
    else:
        return data

def initialize_queue(model_k, device, train_loader, batch_size, queue_size):
    #queue = torch.zeros((0, 16), dtype=torch.float) 
    queue = torch.zeros((0, 2), dtype=torch.float) 
    label_queue = torch.zeros((0), dtype=torch.float) 
    queue = queue.to(device)
    label_queue = label_queue.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        x_k = data
        x_k = x_k.to(device)
        k = model_k(x_k)
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K = queue_size)

        label_k = target
        #label_k = label_k.squeeze(1)
        label_k = label_k.to(device)
        label_k = label_k.float()
        label_queue = queue_data(label_queue, label_k)
        label_queue = dequeue_data(label_queue, K = queue_size)
        break
    return queue, label_queue

def train(model_q, model_k, device, train_loader, queue, label_queue, optimizer, epoch, temp=0.07):
    model_q.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    #loss1 = []

    for batch_idx, (data, target) in enumerate(train_loader):
        x_q = data
        #pdb.set_trace()
        x_q = x_q.to(device)
        output_q = model_q(x_q)
        k = model_k(x_q)
        k = k.detach()
        
        concat_q = torch.cat([queue, output_q], dim = 0)
        N = x_q.shape[0]
        K = concat_q.shape[0]
        


        #logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)        
        logits = torch.mm(output_q.view(N, -1), concat_q.T.view(-1, K))
        
        labels = torch.zeros(N, dtype = torch.long)
        labels = labels.to(device)
        #target = target.squeeze(1)
        target = target.to(device)
        target = target.float()
        ##pdb.set_trace()
        concat_labels = torch.cat([label_queue, target], dim = 0)
        concat_labels = concat_labels.long()
        CrossEnt_loss = loss_function(concat_q, concat_labels)
        triplet_loss, fraction_positive_triplets = batch_all_triplet_loss1(concat_labels,concat_q,0.2, squared=False)
        #print('Triplet loss:{}'.format(triplet_loss))
        #CrossEnt_loss = loss_function(logits, target)
        loss = triplet_loss + CrossEnt_loss
        #loss1.append(triplet_loss + CrossEnt_loss)
        #print ('Cross Entropy loss:{}'.format(loss))
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        m_outputs = model_k(x_q)
        m_outputs = m_outputs.detach()
        #m_outputs2 = copy.deepcopy(m_outputs)
        queue = queue_data(queue, m_outputs)
        queue = dequeue_data(queue)
        label_queue = queue_data(label_queue, target)
        label_queue = dequeue_data(label_queue)
        
        momentum_update(model_q, model_k)

        #train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())

        #loss1.append(train_loss.avg)
    return train_loss.avg
if __name__ == '__main__':   

    net = AD_3DCNN(code_len = 8, dropout=0.5)
    criterion = nn.CrossEntropyLoss()
    loss_function = nn.CrossEntropyLoss()
    trainloader_PIOP1_hand = pd.readcsv('Your path')
    #print(net)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    
    print(device)
    net.to(device)
    m_net = copy.deepcopy(net)
    queue, label_queue = initialize_queue(m_net, device, trainloader_PIOP1_hand, batch_size = 3, queue_size = 10)
    #queue = initialize_queue(m_net, device, trainloader, batch_size = 4, queue_size = 10)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    for epoch in range(0,30):  # loop over the dataset multiple times
        #print('Epoch:{}'.format(epoch))
        loss = train(net, m_net, device, trainloader_PIOP1_hand, queue, label_queue, optimizer, epoch)
        print('--------------------------------------------------------------------')
        print("Epoch:{}/{} Total Loss:{:.3f}".format(epoch + 1, 30,loss))  
        print('--------------------------------------------------------------------')