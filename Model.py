#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:39:38 2022

@author: alex
"""

import torch.nn as nn




##############################################################

class AD_3DCNN(nn.Module):
    """The model we use in the paper."""
    
    def __init__(self, code_len = 16, label_len = 2,  dropout=0):
        nn.Module.__init__(self)

        self.code_length = code_len
        self.label_length = label_len
        
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 8, 3), #conv1
            nn.BatchNorm3d(8),
            nn.ReLU(),
            
            nn.Conv3d(8, 8, 3), #conv2
            nn.BatchNorm3d(8),
            nn.MaxPool3d(2,stride =2),
            nn.ReLU(),
            ################################
            nn.Conv3d(8, 16, 3), #conv3
            nn.BatchNorm3d(16),
            nn.ReLU(),
            
            nn.Conv3d(16, 16, 3), #conv4
            nn.BatchNorm3d(16),
            nn.MaxPool3d(2,stride =2),
            nn.ReLU(),
            ####################################
            nn.Conv3d(16, 32, 3), #conv5
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 32, 3), #conv6
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2,stride =2),
            nn.ReLU(),
            ###################################
            nn.Conv3d(32, 64, 3), #conv7
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Conv3d(64, 64, 3), #conv8
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2,stride =2),
            nn.ReLU(),

            #####################################

            nn.Conv3d(64, 128, 3), #conv9
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            nn.Conv3d(128, 128, 3), #conv10
            nn.BatchNorm3d(128),
            nn.MaxPool3d(2,stride =2),
            nn.ReLU(),
                 
        )    
            
        
        self.classifier3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, self.label_length),
        )
        self.codes3 = nn.Sequential(
            nn.Linear(256, self.code_length),
             nn.Tanh()
        )

        self.codes_classifier3 = nn.Sequential(
            nn.Linear(self.code_length, self.label_length),
            nn.Tanh()
        )
        

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier3(features)
        codes = self.codes3(features)
        return logits #,codes

    def extract_codes_and_logits(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        codes = self.codes3(features)
        logits = self.codes_classifier3(codes)
        return  codes, logits
