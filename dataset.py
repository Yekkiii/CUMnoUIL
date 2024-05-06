import numpy as np
import pandas as pd
import torch
import scipy
import scipy.io
from utils import rand_train_test_idx, class_rand_splits
import os



class NCDataset(object):
    def __init__(self, name):
       
        self.name = name
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        if split_type == 'random':
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=True)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname):
    if dataname in ['crosssite','weibo','douban']:
        dataset=load_crosssite(data_dir,dataname)
    elif dataname == 'crosssite_en':
        dataset=load_crosssite_en(data_dir,dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset


def load_crosssite(data_dir,dataname):
    """
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_post': user's posting history,
                         'node_part': user's participation history,
                         'node_feat': user's profile features,
                         'topic': target topic,
                         'topic_post': tweets of target topic
                         'num_nodes': num of nodes}
    """
    if os.path.exists(data_dir + dataname + '.npy'):
        data = np.load(data_dir + dataname + '.npy', allow_pickle=True).item()
    dataset = NCDataset('crosssite')

    flag_add_nomap_samples = False
    add_weibo = True
    add_douban = True

    data = pd.DataFrame.from_dict(data).T
    data = data.sample(20000,random_state=7)

    # Load cross site datasets
    if dataname=="crosssite":
        num_nodes = len(data)*2
        post_history = data['tweets_wb'].append(data['tweets_db'])
        part_history = data['topic_wb'].append(data['topic_db'])
        profile_feature = data['node_feature_wb'].append(data['node_feature_db'])
        target_topic = data['topic'].append(data['topic'])
        topic_tweet = data['topic_tweet'].append(data['topic_tweet'])
        y = data['y'].append(data['y'])
    
    # Appending unmapped single-platform data
    if flag_add_nomap_samples:
        if add_weibo:
            singlePlatformData_wb = np.load(data_dir + 'crosssite/wb_with_feature_ML_balance.npy', allow_pickle=True).item() # unmapped single-platform data
            singlePlatformData_wb = pd.DataFrame.from_dict(singlePlatformData_wb).T
            num_nodes = num_nodes + len(singlePlatformData_wb)
            post_history = post_history.append(singlePlatformData_wb['tweets_wb'])
            part_history = part_history.append(singlePlatformData_wb['topic_wb'])
            profile_feature = profile_feature.append(singlePlatformData_wb['node_feature_wb'])
            target_topic = target_topic.append(singlePlatformData_wb['topic'])
            topic_tweet = topic_tweet.append(singlePlatformData_wb['topic_tweet'])
            y = y.append(singlePlatformData_wb['y'])
        if add_douban:
            singlePlatformData_db = np.load(data_dir + 'crosssite/db_with_feature_ML_balance.npy', allow_pickle=True).item() 
            singlePlatformData_db = pd.DataFrame.from_dict(singlePlatformData_db).T
            num_nodes = num_nodes + len(singlePlatformData_db)
            post_history = post_history.append(singlePlatformData_db['tweets_db'])
            part_history = part_history.append(singlePlatformData_db['topic_db'])
            profile_feature = profile_feature.append(singlePlatformData_db['node_feature_db'])
            target_topic = target_topic.append(singlePlatformData_db['topic'])
            topic_tweet = topic_tweet.append(singlePlatformData_db['topic_tweet'])
            y = y.append(singlePlatformData_db['y'])

    # Load weibo data ONLY
    elif dataname=="weibo":
        num_nodes = len(data)
        post_history = data['tweets_wb']
        part_history = data['topic_wb']
        profile_feature = data['node_feature_wb']
        target_topic = data['topic']
        topic_tweet = data['topic_tweet']
        y = data['y']

    # Load douban data ONLY
    elif dataname=="douban":
        num_nodes = len(data)
        post_history = data['tweets_db']
        part_history = data['topic_db']
        profile_feature = data['node_feature_db']
        target_topic = data['topic']
        topic_tweet = data['topic_tweet']
        y = data['y']

    post_history = torch.Tensor(post_history).squeeze(2)
    part_history = torch.Tensor(part_history)
    target_topic = torch.Tensor(target_topic).squeeze(1)
    topic_tweet = torch.Tensor(topic_tweet).squeeze(2)
    y = torch.LongTensor(y)


    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_post': post_history,
                     'node_part': part_history,
                     'node_feat': profile_feature,
                     'topic': target_topic,
                     'topic_post': topic_tweet,
                     'num_nodes': num_nodes}
    dataset.label = y

    return dataset

def load_crosssite_en(data_dir,dataname):
    """
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_post': user's posting history,
                         'node_part': user's participation history,
                         'node_feat': user's profile features,
                         'topic': target topic,
                         'topic_post': tweets of target topic
                         'num_nodes': num of nodes}
    """
    if os.path.exists(data_dir + dataname +'.npy'):
        data = np.load(data_dir + dataname +'.npy', allow_pickle=True).item()
    dataset = NCDataset('crosssite_en')

    data = pd.DataFrame.from_dict(data).T

    # Load English cross site datasets
    num_nodes = len(data)*2
    post_history = data['tweets_ins'].append(data['tweets_tw'])
    part_history = data['topic_ins'].append(data['topic_tw'])
    profile_feature = data['node_feature_ins'].append(data['node_feature_tw'])
    target_topic = data['topic'].append(data['topic'])
    topic_tweet = data['topic_tweet'].append(data['topic_tweet'])
    y = data['y'].append(data['y'])

    post_history = torch.Tensor(post_history).squeeze(2)
    part_history = torch.Tensor(part_history)
    target_topic = torch.Tensor(target_topic).squeeze(1)
    topic_tweet = torch.Tensor(topic_tweet).squeeze(2)
    y = torch.LongTensor(y)


    dataset.graph = {'edge_index': None,
                     'edge_feat': None,
                     'node_post': post_history,
                     'node_part': part_history,
                     'node_feat': profile_feature,
                     'topic': target_topic,
                     'topic_post': topic_tweet,
                     'num_nodes': num_nodes}
    dataset.label = y

    return dataset

