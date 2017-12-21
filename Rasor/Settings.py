import torch

import logging

metadata_path = "../data/preprocessed_glove_with_unks.split.metadata.pkl"
emb_path = "../data/preprocessed_glove_with_unks.split.emb.npy"
tokenized_trn_json_path = '../data/train-v1.1.tokenized.split.json'
tokenized_dev_json_path = '../data/dev-v1.1.tokenized.split.json'


can_use_gpu = torch.cuda.is_available()
max_ans_len = 30
emb_dim = 300
learn_single_unk = False
init_scale = 5e-3
learning_rate = 1e-3
batch_size = 30
max_num_epochs = 20
num_bilstm_layers = 2
hidden_dim = 100

log_file_name = 'testingLog'

logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
logger = logging.getLogger(__name__)


