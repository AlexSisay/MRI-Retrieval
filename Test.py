#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:39:17 2022

@author: alex
"""
import torch
from Average import AverageMeter

def test(test_loader,device,  model, criterion):
    model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()
    teY_pred,label = [],[]
    teF = [] 
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data
            N = images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            #print(outputs.data)
            #prediction = outputs.max(1, keepdim=True)
            hashs, pred_lab = model.extract_codes_and_logits(images)
            #print('hashs',hashs.data)
            #print('Labels',pred_lab.data)
            #predic_Labels = abs(pred_lab).max(1, keepdim=True)
            #print(predic_Labels)
            #print('label Classified', outputs)
            test_loss.update(criterion(outputs, labels).item())
            hashs_cpu = hashs.cpu()
            teF.extend(hashs_cpu.data)
            #x_type = F.log_softmax(pred_lab,dim=1) 
            pred = pred_lab.max(1,keepdim=True)[1]
            teY_pred.extend(pred.cpu().data.numpy().tolist())
            label.extend(labels.cpu())
            #print(hashs_cpu.data.numpy().tolist())
    print('------------------------------------------------------------')
    print('[epoch %d], [test loss %.5f]' % (1, test_loss.avg))
    print('------------------------------------------------------------')
    #print(trF)
    return test_loss.avg, teF,teY_pred,label

