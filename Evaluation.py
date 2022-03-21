#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:49:09 2022

@author: alex
"""
import numpy as np

def calc_hamming_dist(b1, b2):
    """Compute the hamming distance between every pair of data 
    points represented in each row of b1 and b2"""
    p1 = np.sign(b1).astype(np.int8)
    p2 = np.sign(b2).astype(np.int8)

    r = p1.shape[1]
    d = (r - np.matmul(p1, np.transpose(p2))) // 2
    return d

#@timer
def calc_hamming_rank(b1, b2, force_slow=False):
    """Return rank of pairs. Takes vector of hashes b1 and b2 and 
    returns correspondence rank of b1 to b2
    """
    #print("Warning. Using slow \"calc_hamming_dist\"")
    dist_h = calc_hamming_dist(b2, b1)
    return np.argsort(dist_h, 1, kind='mergesort')

def compute_map(hashes_train, hashes_test, labels_train, labels_test, top_n=0, and_mode=False, force_slow=False,weighted_mode = False):
    """Compute MAP for given set of hashes and labels"""
    """
    Calculates MAP at k ranking metric.
    Args:
        hashes_train: Generated Hash code for test set
        hashes_test: Generated Hash code for test set
        labels_train:Train lebels
        labels_test: Test lebels
    Returns:
        MAP, recall@topn and topN-precision, 
    """
    order = calc_hamming_rank(hashes_train, hashes_test)

    #print("Warning. Using slow \"compute_map\"")
    s = __compute_s(labels_train, labels_test, and_mode)
    return __calc_map(order, np.transpose(s), top_n)

#@timer
def __compute_s(train_l, test_l, and_mode):
    """Return similarity matrix between two label vectors
    The output is binary matrix of size n_train x n_test
    """
    if and_mode:
        return np.bitwise_and(train_l, np.transpose(test_l)).astype(dtype=np.bool)
    else:
        return np.equal(train_l, np.transpose(test_l))


#@timer
def __calc_map(order, s, top_n):
    """compute mean average precision (MAP), Average precision and recall@k"""
    Q, N = s.shape
    if top_n == 0:
        top_n = N
    pos = np.asarray(range(1, top_n + 1), dtype=np.float32)
    map = 0
    av_precision = np.zeros(top_n)
    av_recall = np.zeros(top_n)
    for q in range(Q):
        total_number_of_relevant_documents = np.sum(s[q].astype(np.float32))
        relevance = s[q, order[q, :top_n]].astype(np.float32)
        cumulative = np.cumsum(relevance)
        number_of_relative_docs = cumulative[-1:]
        if number_of_relative_docs != 0:
            precision = cumulative / pos
            recall = cumulative / total_number_of_relevant_documents
            av_precision += precision
            av_recall += recall
            ap = np.dot(precision, relevance) / number_of_relative_docs
            map += ap
    map /= Q
    av_precision /= Q
    av_recall /= Q

    curve = np.zeros([top_n, 2])

    curve[:, 0] = av_precision #TopN precision
    curve[:, 1] = av_recall #Recall@N

    return float(map), curve