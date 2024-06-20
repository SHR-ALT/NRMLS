from config import Config
import torch
from MIND_corpus import MIND_Corpus


if __name__ == '__main__':
    print(torch.cuda.device_count())
    config = Config()
    mind_corpus = MIND_Corpus(config)


