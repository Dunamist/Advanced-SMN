import numpy as np
from torch.utils.data import Dataset as DS
import pickle


class TrainDataset(DS):
    '''
       utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
       response:(self.batch_size, self.max_sentence_len)
       label:(self.batch_size)
       root 中包含三个文件路径
            utterance_path:应返回list(50W,max_num_utterance,max_sentence_len)
            correct_response_path：应返回list(50W,max_sentence_len)
            error_response_path:应返回list(50W,max_sentence_len)

        注意：经过dataloader操作之后 数据全部会自动变成tensor 根本不需要自己显式的操作
    '''

    def __init__(self, root, transform=None):
        utterance_path, correct_response_path, error_response_path = root
        with open(utterance_path, 'rb') as f:
            self.utterance = pickle.load(f, encoding='bytes')
        with open(correct_response_path, 'rb') as f:
            self.correct_response = pickle.load(f, encoding='bytes')
        with open(error_response_path, 'rb') as f:
            self.error_response = pickle.load(f, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.utterance) * 2

    '''
        label = 1 表示属于第一类 第一类是正类
        label = 0 表示属于第〇类 第〇类是负类
        在预测时，输出则包含分别属于第一类和第〇类的概率
    '''

    def __getitem__(self, idx):
        utterance = self.utterance[idx / 2]
        # idx 为奇数传正样本 为偶数传负样本
        if idx % 2 == 0:
            response = self.correct_response[idx / 2]
            label = 1
        else:
            response = self.error_response[idx / 2]
            label = 0
        if self.transform is not None:
            utterance = self.transform(utterance)
            response = self.transform(response)
        return utterance, response, label


class ValidDataset(DS):
    '''
       utterance:(self.batch_size, self.max_num_utterance, self.max_sentence_len)
       responses:(self.batch_size, 10, self.max_sentence_len)
       correct_index:(self.batch_size)

       root 中包含三个文件路径
            utterance_path:应返回list(1W,max_num_utterance,max_sentence_len)
            responses_path：应返回list(1W,10,max_sentence_len)
            index_path:应返回list(1W)
    '''

    def __init__(self, root, transform=None):
        utterance_path, response_path, index_path = root
        with open(utterance_path, 'rb') as f:
            self.utterance = pickle.load(f, encoding='bytes')
        with open(response_path, 'rb') as f:
            self.responses = pickle.load(f, encoding='bytes')
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f, encoding='bytes')
        self.transform = transform

    def __len__(self):
        return len(self.utterance)

    def __getitem__(self, idx):
        utterance = self.utterance[idx]
        responses = self.responses[idx]
        correct_index = self.index[idx]
        if self.transform is not None:
            utterance = self.transform(utterance)
            responses = self.transform(responses)
        return utterance, responses, correct_index
