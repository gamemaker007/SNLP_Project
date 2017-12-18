import numpy as np
import torch


metadata_path = "data/preprocessed_glove_with_unks.split.metadata.pkl"
emb_path = "data/preprocessed_glove_with_unks.split.emb.npy"
tokenized_trn_json_path = 'data/train-v1.1.tokenized.split.json'
tokenized_dev_json_path = 'data/dev-v1.1.tokenized.split.json'

can_use_gpu = torch.cuda.is_available()
seed = np.random.random_integers(1e6, 1e9)
max_ans_len = 30
emb_dim = 300
learn_single_unk = False
init_scale = 5e-3
learning_rate = 1e-3
max_grad_norm = 10
ff_drop_x = 0.2
batch_size = 30
max_num_epochs = 150
num_bilstm_layers = 2
hidden_dim = 100
lstm_drop_h = 0.1
lstm_drop_x = 0.4
log_file_name = 'testingLog'