#-*- coding: utf-8 -*-
import random
import numpy as np
from pybloom_live import BloomFilter
import math
from tqdm import tqdm, trange
import random
from Utils_mine.Tools2 import *

def build_hash(data,  capacity=None):
    if capacity is None:
        capacity = len(data) * 500
        capacity = int(math.ceil(capacity)) + 2000
        print("total_capacity", capacity)
    dict_pos = BloomFilter(capacity, 1e-3)
    
    
    print(len(dict_pos))
    for datum in tqdm(data):
        dict_pos.add(tuple(datum))
    print(len(dict_pos))
    # np.save("../data/SPRITE/length.npy", length_list)
    return dict_pos

def neg_data_generate2(adj_data_all,seed):
    args = parameters_set()
    neg_num = args.train_neg_num
    random.seed(seed)
    neg_ls = []
    exist_sample = build_hash(adj_data_all)
    print("The number of exist_samples are:",len(exist_sample))
    adj_data_all = np.array(adj_data_all)
    for i in adj_data_all:
        ctn = 0
        while ctn < neg_num:
            col1 = adj_data_all[:, 0]
            col2 = adj_data_all[:, 1]
            col3 = adj_data_all[:, 2]
            m1 = random.sample(range(int(np.max(col1 + 1))), 1)[0]
            m2 = random.sample(range(int(np.max(col2 + 1))), 1)[0]
            m3 = random.sample(range(int(np.max(col3 + 1))), 1)[0]
            temp = [m1, m2, m3]
            if tuple(temp) not in exist_sample:
                neg_ls.append(temp)
                # print(temp + [0])
                ctn += 1
                exist_sample.add(tuple(temp))
    neg_labels = [0]*len(neg_ls)
    return neg_ls, neg_labels

def neg_data_generate3(adj_data_all,seed):
    args = parameters_set()
    neg_num = args.train_neg_num
    random.seed(seed)
    neg_ls = []
    exist_sample = build_hash(adj_data_all)
    print("The number of exist_samples are:",len(exist_sample))
    adj_data_all = np.array(adj_data_all)
    for i in adj_data_all:
        ctn = 0
        while ctn < neg_num:
            col1 = adj_data_all[:, 0]
            col2 = adj_data_all[:, 1]
            col3 = adj_data_all[:, 2]
            m1 = random.sample(range(int(np.max(col1 + 1))), 1)[0]
            m2 = random.sample(range(int(np.max(col2 + 1))), 1)[0]
            m3 = random.sample(range(int(np.max(col3 + 1))), 1)[0]
            temp = [m1, m2, m3]
            if tuple(temp) not in exist_sample:
                neg_ls.append(temp)
                # print(temp + [0])
                ctn += 1
                exist_sample.add(tuple(temp))
    return neg_ls
def trainvaltest_neg_data_generate(adj_data_all,neg_num,seed):

    random.seed(seed)
    neg_ls = []
    exist_sample = build_hash(adj_data_all)
    print("The number of exist_samples are:",len(exist_sample))
    adj_data_all = np.array(adj_data_all)
    positive_samples = adj_data_all.tolist()
    for i in adj_data_all:
        ctn = 0
        while ctn < neg_num:
            col1 = adj_data_all[:, 0]
            col2 = adj_data_all[:, 1]
            col3 = adj_data_all[:, 2]
            m1 = random.sample(range(int(np.max(col1 + 1))), 1)[0]
            m2 = random.sample(range(int(np.max(col2 + 1))), 1)[0]
            m3 = random.sample(range(int(np.max(col3 + 1))), 1)[0]
            temp = [m1, m2, m3]
            if tuple(temp) not in exist_sample:
                neg_ls.append(temp+[0])
                print(temp + [0])
                ctn += 1
                exist_sample.add(tuple(temp))
    # neg_labels = [0]*len(neg_ls)
    all_samples = positive_samples + neg_ls
    return neg_ls,all_samples

def trainvaltest_neg_data_generate2(adj_data_all,neg_num,seed):
    random.seed(seed)
    neg_ls = []
    exist_sample = build_hash(adj_data_all)
    print("The number of exist_samples are:",len(exist_sample))
    adj_data_all = np.array(adj_data_all)
    positive_samples = adj_data_all.tolist()
    for i in adj_data_all:
        ctn = 0
        drug = i[0]
        while ctn < neg_num:
            col1 = adj_data_all[:, 0]
            col2 = adj_data_all[:, 1]
            col3 = adj_data_all[:, 2]
            # m1 = random.sample(range(int(np.max(col1 + 1))), 1)[0]
            m2 = random.sample(range(int(np.max(col2 + 1))), 1)[0]
            m3 = random.sample(range(int(np.max(col3 + 1))), 1)[0]
            temp = [drug, m2, m3]
            if tuple(temp) not in exist_sample:
                neg_ls.append(temp)
                # print(temp + [0])
                ctn += 1
                exist_sample.add(tuple(temp))
    # neg_labels = [0]*len(neg_ls)
    positive_labels = [1] * len(positive_samples)
    negative_labels = [0] * len(neg_ls)
    all_samples = positive_samples + neg_ls
    all_labels = positive_labels + negative_labels  # 包含所有样本的标签
    return neg_ls,all_samples,all_labels, negative_labels

def test_neg_data_generate(adj_data_all,seed):
    args = parameters_set()
    neg_num = args.test_neg_num
    random.seed(seed)
    neg_ls = []
    exist_sample = build_hash(adj_data_all)
    print("The number of exist_samples are:",len(exist_sample))
    adj_data_all = np.array(adj_data_all)
    positive_samples = adj_data_all.tolist()
    for i in adj_data_all:
        ctn = 0
        while ctn < neg_num:
            col1 = adj_data_all[:, 0]
            col2 = adj_data_all[:, 1]
            col3 = adj_data_all[:, 2]
            m1 = random.sample(range(int(np.max(col1 + 1))), 1)[0]
            m2 = random.sample(range(int(np.max(col2 + 1))), 1)[0]
            m3 = random.sample(range(int(np.max(col3 + 1))), 1)[0]
            temp = [m1, m2, m3]
            if tuple(temp) not in exist_sample:
                neg_ls.append(temp)
                # print(temp + [0])
                ctn += 1
                exist_sample.add(tuple(temp))
    all_samples = positive_samples + neg_ls
    return neg_ls, all_samples

def neg_data_generate_mchnn(adj_data_all,seed):
    args = parameters_set()
    neg_num = args.test_neg_num
    random.seed(seed)
    neg_ls_1 = []
    neg_ls_2 = []
    neg_ls_3 = []
    neg_ls_4 = []
    exist_sample = build_hash(adj_data_all)
    print("The number of exist_samples are:",len(exist_sample))
    adj_data_all = np.array(adj_data_all)
    for i in adj_data_all:
        ctn_1 = 0
        ctn_2 = 0
        ctn_3 = 0
        ctn_4 = 0
        col1 = adj_data_all[:, 0]
        col2 = adj_data_all[:, 1]
        col3 = adj_data_all[:, 2]
        while ctn_1 < neg_num:
            m1 = random.sample(range(int(np.max(col1 + 1))), 1)[0]
            m2 = random.sample(range(int(np.max(col2 + 1))), 1)[0]
            m3 = random.sample(range(int(np.max(col3 + 1))), 1)[0]
            temp = [m1, m2, m3]
            if tuple(temp) not in exist_sample:
                neg_ls_1.append(temp)
                # print(temp + [0])
                ctn_1 += 1
                exist_sample.add(tuple(temp))
        while ctn_2 < neg_num:
            m1 = int(i[0])
            m2 = int(i[1])
            m3 = random.sample(range(int(np.max(col3 + 1))), 1)[0]
            temp = [m1, m2, m3]
            if tuple(temp) not in exist_sample:
                neg_ls_2.append(temp)
                # print(temp + [0])
                ctn_2 += 1
                exist_sample.add(tuple(temp))
        while ctn_3 < neg_num:
            m1 = random.sample(range(int(np.max(col1 + 1))), 1)[0]
            m2 = int(i[1])
            m3 = int(i[2])
            temp = [m1, m2, m3]
            if tuple(temp) not in exist_sample:
                neg_ls_3.append(temp)
                # print(temp + [0])
                ctn_3 += 1
                exist_sample.add(tuple(temp))
        while ctn_4 < neg_num:
            m1 = int(i[0])
            m2 = random.sample(range(int(np.max(col2 + 1))), 1)[0]
            m3 = int(i[2])
            temp = [m1, m2, m3]
            if tuple(temp) not in exist_sample:
                neg_ls_4.append(temp)
                # print(temp + [0])
                ctn_4 += 1
                exist_sample.add(tuple(temp))

    neg_labels_1 = [0]*len(neg_ls_1)
    neg_labels_2 = [0]*len(neg_ls_2)
    neg_labels_3 = [0]*len(neg_ls_3)
    neg_labels_4 = [0]*len(neg_ls_4)

    return neg_ls_1, neg_labels_1,neg_ls_2,neg_labels_2,neg_ls_3,neg_labels_3,neg_ls_4,neg_labels_4