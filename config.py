import os
import argparse
import time
import torch
import random
import numpy as np
from MIND_corpus import MIND_Corpus


class Config:
    def parse_argument(self):
        parser = argparse.ArgumentParser(description='DIGAT Experiments')
        # General config
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'dev', 'test'], help='Mode')
        parser.add_argument('--news_encoder', type=str, default='MyOwn', choices=['MSA', 'MyOwn'], help='News encoder')
        parser.add_argument('--graph_encoder', type=str, default='CandiGraphNewsRec',choices=['DIGAT', 'wo_interaction', 'CandiGraphNewsRec', 'User_graph_wo_inter'], help='Graph encoder')
        parser.add_argument('--dev_model_path', type=str, default='best_model/MIND-small/MSA-CandiGraphNewsRec/#1/MSA-CandiGraphNewsRec',
                            help='Dev model path')
        parser.add_argument('--test_model_path', type=str, default='best_model/MIND-small/MSA-CandiGraphNewsRec/#1/MSA-CandiGraphNewsRec',
                            help='Test model path')
        parser.add_argument('--test_output_file', type=str, default='', help='Test output file')
        parser.add_argument('--device_id', type=int, default=0, help='Device ID of GPU')
        parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
        parser.add_argument('--local_rank', type=int, default=-1,
                            help='Local GPU rank for distributed training (-1 for single GPU)')
        # Dataset config
        parser.add_argument('--dataset', type=str, default='MIND-small', choices=['MIND-small', 'MIND-large'],
                            help='Directory root of dataset')
        parser.add_argument('--word_threshold', type=int, default=0, help='Word threshold')
        parser.add_argument('--max_title_length', type=int, default=32, help='Sentence truncate length for title')
        parser.add_argument('--max_abstract_length', type=int, default=68, help='Sentence truncate length for title')
        # Training config
        parser.add_argument('--negative_sample_num', type=int, default=4,
                            help='Negative sample number of each positive sample')
        parser.add_argument('--max_history_num', type=int, default=50,
                            help='Maximum number of history news for each user')
        parser.add_argument('--epoch', type=int, default=16, help='Training epoch')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
        parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
        parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
        parser.add_argument('--gradient_clip_norm', type=float, default=1,
                            help='Gradient clip norm (non-positive value for no gradient clipping)')
        # Dev config
        parser.add_argument('--dev_criterion', type=str, default='avg',
                            choices=['auc', 'mrr', 'ndcg5', 'ndcg10', 'avg'], help='Dev criterion to select model')
        parser.add_argument('--early_stopping_epoch', type=int, default=5,
                            help='Epoch of stop training after the dev result does not improve')
        # Model config
        parser.add_argument('--word_embedding_dim', type=int, default=300, choices=[50, 100, 200, 300],
                            help='Word embedding dimension')
        parser.add_argument('--cnn_method', type=str, default='naive', choices=['naive', 'group3', 'group4', 'group5'],
                            help='CNN group')
        parser.add_argument('--cnn_kernel_num', type=int, default=400, help='Number of CNN kernel')
        parser.add_argument('--cnn_window_size', type=int, default=3, help='Window size of CNN kernel')
        parser.add_argument('--MSA_head_num', type=int, default=16, help='Head number of multihead self-attention')
        parser.add_argument('--MSA_head_dim', type=int, default=25, help='Head dimension of multihead self-attention')
        parser.add_argument('--attention_dim', type=int, default=256, help="Attention dimension")
        parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--graph_depth', type=int, default=3, help='Number of dual-graph modeling layers')
        # SAG config
        parser.add_argument('--SAG_hops', type=int, default=1, help='click-candidate graph')
        parser.add_argument('--SAG_neighbors', type=int, default=5, help='click-candidate graph neighbors')

        self.attribute_dict = dict(vars(parser.parse_args()))
        for attribute in self.attribute_dict:
            setattr(self, attribute, self.attribute_dict[attribute])
        self.seed = self.seed if self.seed >= 0 else (int)(time.time())
        self.train_root = 'data/Recommend/MIND/MINDsmall_train'
        self.dev_root = 'data/Recommend/MIND/MINDsmall_dev'
        self.test_root = 'data/Recommend/MIND/MINDsmall_dev'
        self.dropout_rate = 0.2
        self.epoch = 12
        self.news_graph_size = 1
        neighbors = 1
        for i in range(self.SAG_hops):
            if i == 0:
                neighbors *= self.SAG_neighbors
            else:
                neighbors *= self.SAG_neighbors - 1
            self.news_graph_size += neighbors
        if self.local_rank in [-1, 0]:
            for attribute in self.attribute_dict:
                print(attribute + ' : ' + str(getattr(self, attribute)))

    def set_cuda(self):
        assert torch.cuda.is_available(), 'GPU is not available'
        if self.local_rank == -1:
            torch.cuda.set_device(self.device_id)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # For reproducibility
    #label construct
    def preliminary_setup(self):
        if not os.path.exists('data/label/truth.txt'):
            with open(os.path.join(self.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_f:
                with open('data/label/truth.txt', 'w', encoding='utf-8') as truth_f:
                    for dev_ID, line in enumerate(dev_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')
                        labels = [int(impression[-1]) for impression in impressions.strip().split(' ')]
                        truth_f.write(
                            ('' if dev_ID == 0 else '\n') + str(dev_ID + 1) + ' ' + str(labels).replace(' ', ''))
        MIND_Corpus.preprocess(self)

    def __init__(self):
        self.parse_argument()
        self.set_cuda()
        self.preliminary_setup()


if __name__ == '__main__':
    config = Config()