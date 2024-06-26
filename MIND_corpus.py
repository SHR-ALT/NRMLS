import os
import torch
import numpy as np
import json
import pickle
import collections
import re
from torchtext.vocab import GloVe

"""
    Processing the MIND dataset
"""


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


pat = re.compile(r"[\w]+|[.,!?;|]")


class MIND_Corpus:
    @staticmethod
    def preprocess(config):
        user_ID_file = 'data/dict/user_ID-%s.json' % config.dataset
        news_ID_file = 'data/dict/news_ID-%s.json' % config.dataset
        category_file = 'data/dict/category-%s.json' % config.dataset
        subCategory_file = 'data/dict/subCategory-%s.json' % config.dataset
        vocabulary_file = 'data/dict/vocabulary-' + str(config.word_threshold) + '-' + config.dataset + '.json'
        word_embedding_file = 'data/dict/word_embedding-' + str(config.word_threshold) + '-' + str(
            config.word_embedding_dim) + '-' + config.dataset + '.pkl'
        news_graph_file = 'data/dict/news_graph-' + str(config.SAG_hops) + '-' + str(config.SAG_neighbors) + '-' + str(
            config.dataset) + '.pkl'
        user_history_graph_file = 'data/dict/user_history_graph-' + str(config.max_history_num) + '-' + str(
            config.dataset) + '.pkl'
        preprocessed_data_files = [user_ID_file, news_ID_file, category_file, subCategory_file, vocabulary_file,
                                   word_embedding_file]

        if not all(list(map(os.path.exists, preprocessed_data_files))):
            user_ID_dict = {'<UNK>': 0}
            news_ID_dict = {'<PAD>': 0}
            category_dict = {}
            subCategory_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1}
            word_counter = collections.Counter()

            # 1. user ID dictionay
            with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
                for line in train_behaviors_f:
                    impression_ID, user_ID, time, history, impressions = line.split('\t')
                    if user_ID not in user_ID_dict:
                        user_ID_dict[user_ID] = len(user_ID_dict)
                with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
                    json.dump(user_ID_dict, user_ID_f)

            # 2. news ID dictionay & news category dictionay & news subCategory dictionay
            for i, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                with open(os.path.join(prefix, 'news.tsv'), 'r', encoding='utf-8') as news_f:
                    for line in news_f:
                        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split(
                            '\t')
                        if news_ID not in news_ID_dict:
                            news_ID_dict[news_ID] = len(news_ID_dict)
                            if category not in category_dict:
                                category_dict[category] = len(category_dict)
                            if subCategory not in subCategory_dict:
                                subCategory_dict[subCategory] = len(subCategory_dict)
                            words = pat.findall(title.lower().replace('é', 'e')) + pat.findall(abstract.lower().replace('é', 'e'))
                            # occurrences of words
                            for word in words:
                                if is_number(word):
                                    word_counter['<NUM>'] += 1
                                else:
                                    if i == 0:
                                        word_counter[word] += 1
                                    else:
                                        if word in word_counter:
                                            word_counter[word] += 1
            with open(news_ID_file, 'w', encoding='utf-8') as news_ID_f:
                json.dump(news_ID_dict, news_ID_f)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f)
            with open(subCategory_file, 'w', encoding='utf-8') as subCategory_f:
                json.dump(subCategory_dict, subCategory_f)

            # 3. word dictionay
            word_counter_list = [[word, word_counter[word]] for word in word_counter]
            word_counter_list.sort(key=lambda x: x[1], reverse=True)  # sort by word frequency
            filtered_word_counter_list = list(filter(lambda x: x[1], word_counter_list))
            for i, word in enumerate(filtered_word_counter_list):
                word_dict[word[0]] = i + 2
            with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                json.dump(word_dict, vocabulary_f)

            # 4. Glove word embedding
            if config.word_embedding_dim == 300:
                glove = GloVe(name='840B', dim=300, cache='glove', max_vectors=10000000000)
            else:
                glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='glove', max_vectors=10000000000)
            glove_stoi = glove.stoi
            glove_vectors = glove.vectors
            glove_mean = torch.mean(glove_vectors, dim=0, keepdim=False)
            glove_std = torch.std(glove_vectors, dim=0, keepdim=False, unbiased=True)
            word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
            word_embedding_vectors[0] = glove_mean
            for word in word_dict:
                index = word_dict[word]
                if index != 0:
                    if word in glove_stoi:
                        word_embedding_vectors[index] = glove_vectors[glove_stoi[word]]
                    else:
                        word_embedding_vectors[index] = torch.normal(mean=glove_mean, std=glove_std)
            with open(word_embedding_file, 'wb') as word_embedding_f:
                pickle.dump(word_embedding_vectors, word_embedding_f)

        if not os.path.exists(news_graph_file):
            # 5. construct news graph (SAG)
            with open('data/dict/news_ID-' + str(config.dataset) + '.json', 'r', encoding='utf-8') as news_ID_f:
                news_ID_dict = json.load(news_ID_f)
            news_node_ID, news_graph, news_graph_mask = construct_SAG(config.dataset, config.train_root,
                                                                      config.dev_root, config.test_root,
                                                                      config.SAG_neighbors, config.SAG_hops,
                                                                      config.news_graph_size, news_ID_dict)
            news_num = len(news_ID_dict)
            assert news_num == news_graph.shape[0]
            for i in range(news_num):
                news_graph[i] += np.identity(config.news_graph_size, dtype=bool)
            with open(news_graph_file, 'wb') as news_graph_f:
                pickle.dump({
                    'news_node_ID': news_node_ID,
                    'news_graph': news_graph,
                    'news_graph_mask': news_graph_mask
                }, news_graph_f, protocol=4)

        if not os.path.exists(user_history_graph_file):
            # 6. construct user graph
            with open(category_file, 'r', encoding='utf-8') as category_f:
                category_dict = json.load(category_f)
            # newsID --> int(categoryID)
            news_category_dict = {}
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                with open(os.path.join(prefix, 'news.tsv'), 'r', encoding='utf-8') as news_f:
                    for line in news_f:
                        news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split(
                            '\t')
                        news_category_dict[news_ID] = category_dict[category]
            category_num = len(category_dict)
            graph_size = config.max_history_num + category_num
            user_history_graph_data = {}
            prefix_mode = ['train', 'dev', 'test']
            for prefix_index, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                mode = prefix_mode[prefix_index]
                user_history_num = 0
                with open(os.path.join(prefix, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
                    for line in behaviors_f:
                        user_history_num += 1
                user_history_graph = np.zeros([user_history_num, graph_size, graph_size], dtype=bool)
                user_history_graph_mask = np.zeros([user_history_num, graph_size], dtype=bool)
                user_history_category_mask = np.zeros([user_history_num, category_num + 1], dtype=bool)
                user_history_category_indices = np.zeros([user_history_num, config.max_history_num], dtype=np.int64)
                with open(os.path.join(prefix, 'behaviors.tsv'), 'r', encoding='utf-8') as behaviors_f:
                    for line_index, line in enumerate(behaviors_f):
                        impression_ID, user_ID, time, history, impressions = line.split('\t')

                        history_graph = np.identity(graph_size, dtype=bool)
                        history_graph_mask = np.zeros(graph_size, dtype=bool)
                        # extra one category index for padding news
                        history_category_mask = np.zeros(category_num + 1, dtype=bool)
                        history_category_indices = np.full([config.max_history_num], category_num, dtype=np.int64)
                        if len(history.strip()) > 0:
                            history_news_ID = history.split(' ')

                            offset = max(0, len(history_news_ID) - config.max_history_num)
                            offset_1 = max(0, config.max_history_num - len(history_news_ID))
                            history_news_num = min(len(history_news_ID), config.max_history_num)
                            for i in range(history_news_num):
                                # 类别对应的引
                                category_index = news_category_dict[history_news_ID[i + offset]]
                                history_category_mask[category_index] = 1
                                history_category_indices[offset_1 + i] = category_index
                                history_graph[offset_1 + i, config.max_history_num + category_index] = 1  # News-Topic Edge
                                history_graph[config.max_history_num + category_index, offset_1 + i] = 1  # News-Topic Edge
                                history_graph_mask[offset_1 + i] = 1
                                history_graph_mask[config.max_history_num + category_index] = 1
                                # 遍历当前新闻后面的新闻？
                                for j in range(i + 1, history_news_num):
                                    _category_index = news_category_dict[history_news_ID[j + offset]]
                                    if category_index == _category_index:
                                        history_graph[offset_1 + i, offset_1 + j] = 1  # News-News Edge
                                        history_graph[offset_1 + j, offset_1 + i] = 1  # News-News Edge
                                    else:
                                        history_graph[
                                            config.max_history_num + category_index, config.max_history_num + _category_index] = 1  # Topic-Topic Edge
                                        history_graph[
                                            config.max_history_num + _category_index, config.max_history_num + category_index] = 1  # Topic-Topic Edge
                        user_history_graph[line_index] = history_graph
                        user_history_graph_mask[line_index] = history_graph_mask
                        user_history_category_mask[line_index] = history_category_mask
                        user_history_category_indices[line_index] = history_category_indices
                    user_history_graph_data[mode + '_user_history_graph'] = user_history_graph
                    user_history_graph_data[mode + '_user_history_graph_mask'] = user_history_graph_mask
                    user_history_graph_data[mode + '_user_history_category_mask'] = user_history_category_mask
                    user_history_graph_data[mode + '_user_history_category_indices'] = user_history_category_indices
            with open(user_history_graph_file, 'wb') as user_history_graph_f:
                pickle.dump(user_history_graph_data, user_history_graph_f, protocol=4)

    def __init__(self, config):
        with open('data/dict/user_ID-' + str(config.dataset) + '.json', 'r', encoding='utf-8') as user_ID_f:
            self.user_ID_dict = json.load(user_ID_f)
            config.user_num = len(self.user_ID_dict)
        with open('data/dict/news_ID-' + str(config.dataset) + '.json', 'r', encoding='utf-8') as news_ID_f:
            self.news_ID_dict = json.load(news_ID_f)
            self.news_num = len(self.news_ID_dict)
        with open('data/dict/category-' + str(config.dataset) + '.json', 'r', encoding='utf-8') as category_f:
            self.category_dict = json.load(category_f)
            config.category_num = len(self.category_dict)
        with open('data/dict/subCategory-' + str(config.dataset) + '.json', 'r', encoding='utf-8') as subCategory_f:
            self.subCategory_dict = json.load(subCategory_f)
            config.subCategory_num = len(self.subCategory_dict)
        with open('data/dict/vocabulary-' + str(config.word_threshold) + '-' + str(
                config.dataset) + '.json', 'r', encoding='utf-8') as vocabulary_f:
            self.word_dict = json.load(vocabulary_f)
            config.vocabulary_size = len(self.word_dict)
        with open('data/dict/news_graph-' + str(config.SAG_hops) + '-' + str(config.SAG_neighbors) + '-' + str(
                config.dataset) + '.pkl', 'rb') as news_graph_f:
            news_graph_data = pickle.load(news_graph_f)
            self.news_node_ID = news_graph_data['news_node_ID']
            self.news_graph = news_graph_data['news_graph']
            self.news_graph_mask = news_graph_data['news_graph_mask']
            self.news_graph_mask[:, 0] = 0
        with open('data/dict/user_history_graph-' + str(config.max_history_num) + '-' + str(config.dataset) + '.pkl',
                  'rb') as user_history_graph_f:
            user_history_graph_data = pickle.load(user_history_graph_f)
            self.train_user_history_graph = user_history_graph_data['train_user_history_graph']
            self.train_user_history_graph_mask = user_history_graph_data['train_user_history_graph_mask']
            self.train_user_history_category_mask = user_history_graph_data['train_user_history_category_mask']
            self.train_user_history_category_indices = user_history_graph_data['train_user_history_category_indices']
            self.dev_user_history_graph = user_history_graph_data['dev_user_history_graph']
            self.dev_user_history_graph_mask = user_history_graph_data['dev_user_history_graph_mask']
            self.dev_user_history_category_mask = user_history_graph_data['dev_user_history_category_mask']
            self.dev_user_history_category_indices = user_history_graph_data['dev_user_history_category_indices']
            self.test_user_history_graph = user_history_graph_data['test_user_history_graph']
            self.test_user_history_graph_mask = user_history_graph_data['test_user_history_graph_mask']
            self.test_user_history_category_mask = user_history_graph_data['test_user_history_category_mask']
            self.test_user_history_category_indices = user_history_graph_data['test_user_history_category_indices']

        # meta data
        self.dataset_type = config.dataset
        assert self.dataset_type in ['MIND-small',
                                     'MIND-large'], 'Dataset is chosen from \'MIND-small\' and \'MIND-large\''
        self.negative_sample_num = config.negative_sample_num  # negative sample number for training
        self.max_history_num = config.max_history_num  # max history number for each training user
        self.max_title_length = config.max_title_length  # max title length for each news
        self.news_title_text = np.zeros([self.news_num, self.max_title_length],
                                        dtype=np.int32)  # [news_num, max_title_length]
        self.news_title_mask = np.zeros([self.news_num, self.max_title_length],
                                        dtype=bool)  # [news_num, max_title_length]

        self.max_abstract_length = config.max_abstract_length
        self.news_abstract_text = np.zeros([self.news_num, self.max_abstract_length],
                                        dtype=np.int32)
        self.news_abstract_mask = np.zeros([self.news_num, self.max_abstract_length],
                                        dtype=bool)

        self.train_behaviors = []  # [[history], click impression, [non-click impressions], behavior_index]
        self.dev_behaviors = []  # [[history], candidate_news_ID, behavior_index]
        self.dev_indices = []  # index for dev
        self.test_behaviors = []  # [[history], candidate_news_ID, behavior_index]
        self.test_indices = []  # index for test

        self.title_word_num = 0
        self.abstract_word_num = 0

        # generate news meta data
        news_ID_set = set()
        news_lines = []
        for prefix in [config.train_root, config.dev_root, config.test_root]:
            with open(os.path.join(prefix, 'news.tsv'), 'r', encoding='utf-8') as news_f:
                for line in news_f:
                    news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split(
                        '\t')
                    if news_ID not in news_ID_set:
                        news_lines.append(line)
                        news_ID_set.add(news_ID)
        assert self.news_num == len(news_ID_set) + 1, 'news num mismatch %d v.s. %d' % (self.news_num, len(news_ID_set))
        for line in news_lines:
            news_ID, category, subCategory, title, abstract, _, title_entities, abstract_entities = line.split('\t')
            index = self.news_ID_dict[news_ID]
            title_words = pat.findall(title.lower().replace('é', 'e'))
            # deal with title words
            for i, word in enumerate(title_words):
                if i == self.max_title_length:
                    break
                if is_number(word):
                    self.news_title_text[index][i] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_title_text[index][i] = self.word_dict[word]
                else:
                    self.news_title_text[index][i] = self.word_dict['<UNK>']
                self.news_title_mask[index][i] = 1
            self.title_word_num += len(title_words)

            abstract_words = pat.findall(abstract.lower().replace('é', 'e'))
            # deal with abstract words
            for j, word in enumerate(abstract_words):
                if j == self.max_abstract_length:
                    break
                if is_number(word):
                    self.news_abstract_text[index][j] = self.word_dict['<NUM>']
                elif word in self.word_dict:
                    self.news_abstract_text[index][j] = self.word_dict[word]
                else:
                    self.news_abstract_text[index][j] = self.word_dict['<UNK>']
                self.news_abstract_mask[index][j] = 1
            self.abstract_word_num += len(abstract_words)
        # generate behavior meta data
        with open(os.path.join(config.train_root, 'behaviors.tsv'), 'r', encoding='utf-8') as train_behaviors_f:
            for behavior_index, line in enumerate(train_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                click_impressions = []
                non_click_impressions = []
                for impression in impressions.strip().split(' '):
                    if impression[-2:] == '-1':
                        click_impressions.append(self.news_ID_dict[impression[:-2]])
                    else:
                        non_click_impressions.append(self.news_ID_dict[impression[:-2]])
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    # user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history = [0] * padding_num + history[-self.max_history_num:]
                    for click_impression in click_impressions:
                        self.train_behaviors.append(
                            [user_history, click_impression, non_click_impressions, behavior_index])
                else:
                    for click_impression in click_impressions:
                        self.train_behaviors.append(
                            [[0 for _ in range(self.max_history_num)], click_impression, non_click_impressions,
                             behavior_index])
        with open(os.path.join(config.dev_root, 'behaviors.tsv'), 'r', encoding='utf-8') as dev_behaviors_f:
            for dev_ID, line in enumerate(dev_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    # user_history = history[-self.max_history_num:] + [0] * padding_num
                    user_history = [0] * padding_num + history[-self.max_history_num:]
                    for impression in impressions.strip().split(' '):
                        self.dev_indices.append(dev_ID)
                        self.dev_behaviors.append([user_history, self.news_ID_dict[impression[:-2]], dev_ID])
                else:
                    for impression in impressions.strip().split(' '):
                        self.dev_indices.append(dev_ID)
                        self.dev_behaviors.append(
                            [[0 for _ in range(self.max_history_num)], self.news_ID_dict[impression[:-2]], dev_ID])
        with open(os.path.join(config.test_root, 'behaviors.tsv'), 'r', encoding='utf-8') as test_behaviors_f:
            for test_ID, line in enumerate(test_behaviors_f):
                impression_ID, user_ID, time, history, impressions = line.split('\t')
                if len(history) != 0:
                    history = list(map(lambda x: self.news_ID_dict[x], history.strip().split(' ')))
                    padding_num = max(0, self.max_history_num - len(history))
                    user_history = [0] * padding_num + history[-self.max_history_num:]
                    for impression in impressions.strip().split(' '):
                        self.test_indices.append(test_ID)
                        self.test_behaviors.append([user_history, self.news_ID_dict[impression[:-2]], test_ID])
                else:
                    for impression in impressions.strip().split(' '):
                        self.test_indices.append(test_ID)
                        self.test_behaviors.append(
                            [[0 for _ in range(self.max_history_num)], self.news_ID_dict[impression[:-2]], test_ID])
