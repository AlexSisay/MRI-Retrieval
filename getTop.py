#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:47:44 2022

@author: alex
"""
from tqdm import tqdm
#Feature1 = torch.squeeze(Feature1)
#print(Hash.shape)
#print(Feature1)
def getTop(label, path, dis, number=9):
    #print(path)
    #print(dis)
    top = []
    length = len(path)
    for index in tqdm(range(length)):
        if len(top) < number:
            top.append((dis[index], path[index], label[index]))
            top.sort(key=lambda d: d[0])
        else:
            if dis[index] > top[-1][0]:
                continue
            else:
                top = top[:-1]
                top.append((dis[index], path[index], label[index]))
                top.sort(key=lambda d: d[0])
    print(top)
    return top
