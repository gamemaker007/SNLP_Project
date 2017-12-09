import pickle as pkl
import numpy as np
import random
from torch.utils.data import Dataset
from collections import namedtuple
from collections import Counter
import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import logging

metadata_path = "data/preprocessed_glove.metadata.pkl"
emb_path = "data/preprocessed_glove.emb.npy"
tokenized_trn_json_path = 'data/train-v1.1.tokenized.split.json'
tokenized_dev_json_path = 'data/dev-v1.1.tokenized.split.json'

can_use_gpu = torch.cuda.is_available()
device = None  # 'cpu' / 'gpu<index>'
save_freq = None  # how often to save model (in epochs); None for only after best EM/F1 epochs
test_json_path = None  # path of test set JSON
pred_json_path = None  # path of test predictions JSON
tst_load_model_path = None  # path of trained model data, used for producing test set predictions
tst_split = True  # whether to split hyphenated unknown words of test set, see setup.py
seed = np.random.random_integers(1e6, 1e9)
max_ans_len = 30  # maximal answer length, answers of longer length are discarded
emb_dim = 300  # dimension of word embeddings
learn_single_unk = True  # whether to have a single tunable word embedding for all unknown words # (or multiple fixed random ones)
init_scale = 5e-3  # uniformly random weights are initialized in [-init_scale, +init_scale]
learning_rate = 1e-3
lr_decay = 0.95
lr_decay_freq = 5000  # frequency with which to decay learning rate, measured in updates
max_grad_norm = 10  # gradient clipping
ff_dims = [100]  # dimensions of hidden FF layers
ff_drop_x = 0.2  # dropout rate of FF layers
batch_size = 32
max_num_epochs = 150  # max number of epochs to train for
num_bilstm_layers = 2  # number of BiLSTM layers, where BiLSTM is applied
hidden_dim = 100  # dimension of hidden state of each uni-directional LSTM
lstm_drop_h = 0.1  # dropout rate for recurrent hidden state of LSTM
lstm_drop_x = 0.4  # dropout rate for inputs of LSTM
lstm_couple_i_and_f = True  # customizable LSTM configuration, see base/model.py
lstm_learn_initial_state = False
lstm_tie_x_dropout = True
lstm_sep_x_dropout = False
lstm_sep_h_dropout = False
lstm_w_init = 'uniform'
lstm_u_init = 'uniform'
lstm_forget_bias_init = 'uniform'
default_bias_init = 'uniform'
extra_drop_x = 0  # dropout rate at an extra possible place
q_aln_ff_tie = True  # whether to tie the weights of the FF over question and the FF over passage
sep_stt_end_drop = True  # whether to have separate dropout masks for span start and # span end representations
adam_beta1 = 0.9  # see base/optimizer.py
adam_beta2 = 0.999
adam_eps = 1e-8
objective = 'span_multinomial'  # 'span_multinomial': multinomial distribution over all spans
# 'span_binary':      logistic distribution per span
# 'span_endpoints':   two multinomial distributions, over span start and end
ablation = None
log_file_name = 'testingLog'


logging.basicConfig(filename=log_file_name,level=logging.DEBUG)
logger = logging.getLogger(__name__)

SquadDatasetVectorized = namedtuple('SquadDatasetVectorized', [
    'qtn_ctx',
    'qtn_ctx_lens',
    'qtns',
    'qtn_lens',
    'anss'
])

WordEmbData = namedtuple('WordEmbData', [
  'word_emb',                 # float32 (num words, emb dim)
  'str_to_word',              # map word string to word index
  'first_known_word',         # words found in GloVe are at positions [first_known_word, first_unknown_word)
  'first_unknown_word',       # words not found in GloVe are at positions [first_unknown_word, first_unallocated_word)
  'first_unallocated_word'    # extra random embeddings
])

TokenizedText = namedtuple('TokenizedText', [
    'text',  # original text string
    'tokens',  # list of parsed tokens
    'originals',  # list of original tokens (may differ from parsed ones)
    'whitespace_afters',  # list of whitespace strings, each appears after corresponding original token in original text
])

SquadArticle = namedtuple('SquadArticle', [
    'art_title_str'
])

SquadContext = namedtuple('SquadContext', [
    'art_idx',
    'tokenized'  # TokenizedText of context's text
])

SquadQuestion = namedtuple('SquadQuestion', [
    'ctx_idx',
    'qtn_id',
    'tokenized',  # TokenizedText of question's text
    'ans_texts',  # list of (possibly multiple) answer text strings
    'ans_word_idxs'  # list where each entry is either a (answer start word index, answer end word index) tuple
    # or None for answers that we failed to parse
])


class SquadDatasetTabular(object):
    def __init__(self):
        self.arts = []  # SquadArticle objects
        self.ctxs = []  # SquadContext objects
        self.qtns = []  # SquadQuestion objects

    def new_article(self, art_title_str):
        self.arts.append(SquadArticle(art_title_str))
        return len(self.arts) - 1

    def new_context(self, art_idx, ctx_tokenized):
        self.ctxs.append(SquadContext(art_idx, ctx_tokenized))
        return len(self.ctxs) - 1

    def new_question(self, ctx_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs):
        self.qtns.append(
            SquadQuestion(ctx_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs))

def read_word_emb_data():
    with open(metadata_path, 'rb') as f:
        str_to_word, first_known_word, first_unknown_word, first_unallocated_word = pkl.load(f)
    with open(emb_path, 'rb') as f:
        word_emb = np.load(f)
    word_emb_data = WordEmbData(word_emb, str_to_word, first_known_word, first_unknown_word, first_unallocated_word)
    return word_emb_data


def _make_tabular_dataset(tokenized_json_path, word_strs, has_answers, max_ans_len=None):
    tabular = SquadDatasetTabular()

    num_questions = 0
    num_answers = 0
    num_invalid_answers = 0
    num_long_answers = 0
    num_invalid_questions = 0

    answers_per_question_counter = Counter()
    with io.open(tokenized_json_path, 'r', encoding='utf-8') as f:
        j = json.load(f)
        data = j['data']
        for article in data:
            art_title_str = article['title']
            art_idx = tabular.new_article(art_title_str)

            paragraphs = article['paragraphs']
            for paragraph in paragraphs:
                ctx_str = paragraph['context']
                ctx_tokens = paragraph['tokens']
                word_strs.update(ctx_tokens)
                ctx_originals = paragraph['originals']
                ctx_whitespace_afters = paragraph['whitespace_afters']
                ctx_tokenized = TokenizedText(ctx_str, ctx_tokens, ctx_originals, ctx_whitespace_afters)
                ctx_idx = tabular.new_context(art_idx, ctx_tokenized)

                qas = paragraph['qas']
                for qa in qas:
                    num_questions += 1
                    qtn_id = qa['id']

                    qtn_str = qa['question']
                    qtn_tokens = qa['tokens']
                    word_strs.update(qtn_tokens)
                    qtn_originals = qa['originals']
                    qtn_whitespace_afters = qa['whitespace_afters']
                    qtn_tokenized = TokenizedText(qtn_str, qtn_tokens, qtn_originals, qtn_whitespace_afters)

                    ans_texts = []
                    ans_word_idxs = []
                    if has_answers:
                        answers = qa['answers']

                        for answer in answers:
                            num_answers += 1
                            ans_text = answer['text']
                            ans_texts.append(ans_text)
                            if not answer['valid']:
                                ans_word_idxs.append(None)
                                num_invalid_answers += 1
                                continue
                            ans_start_word_idx = answer['start_token_idx']
                            ans_end_word_idx = answer['end_token_idx']
                            if max_ans_len and ans_end_word_idx - ans_start_word_idx + 1 > max_ans_len:
                                ans_word_idxs.append(None)
                                num_long_answers += 1
                            else:
                                ans_word_idxs.append((ans_start_word_idx, ans_end_word_idx))
                        answers_per_question_counter[len(ans_texts)] += 1  # this counts also invalid answers
                        num_invalid_questions += 1 if all(ans is None for ans in ans_word_idxs) else 0

                    tabular.new_question(ctx_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs)
    return tabular

def _contract_word_emb_data(old_word_emb_data, word_strs, is_single_unk):
    old_word_emb, old_str_to_word, old_first_known_word, old_first_unknown_word, old_first_unallocated_word = \
        old_word_emb_data

    known_word_strs = []
    unknown_word_strs = []
    for word_str in word_strs:
        if word_str in old_str_to_word and old_str_to_word[word_str] < old_first_unknown_word:
            known_word_strs.append(word_str)
        else:
            unknown_word_strs.append(word_str)

    str_to_word = {}
    emb_size = old_first_known_word + (len(known_word_strs) + 1 if is_single_unk else len(word_strs))
    word_emb = np.zeros((emb_size, old_word_emb.shape[1]), dtype=np.float32)

    for i, word_str in enumerate(known_word_strs):
        word = old_first_known_word + i
        str_to_word[word_str] = word
        word_emb[word, :] = old_word_emb[old_str_to_word[word_str]]

    first_unknown_word = old_first_known_word + len(known_word_strs)

    if is_single_unk:
        for word_str in unknown_word_strs:
            str_to_word[word_str] = first_unknown_word
    else:
        num_new_unks = 0
        for i, word_str in enumerate(unknown_word_strs):
            word = first_unknown_word + i
            str_to_word[word_str] = word
            if word_str in old_str_to_word:
                word_emb[word, :] = old_word_emb[old_str_to_word[word_str]]
            else:
                if old_first_unallocated_word + num_new_unks >= len(old_word_emb):
                    sys.exit(1)
                word_emb[word, :] = old_word_emb[old_first_unallocated_word + num_new_unks]
                num_new_unks += 1
    return WordEmbData(word_emb, str_to_word, old_first_known_word, first_unknown_word, None)


def _make_vectorized_dataset(tabular, word_emb_data):
    num_ctxs = len(tabular.ctxs)
    num_qtns = len(tabular.qtns)
    max_ctx_len = max(len(ctx.tokenized.tokens) for ctx in tabular.ctxs)
    max_qtn_len = max(len(qtn.tokenized.tokens) for qtn in tabular.qtns)

    all_ctxs = np.zeros((num_ctxs, max_ctx_len), dtype=np.int32)
    all_ctx_lens = np.zeros(num_ctxs, dtype=np.int32)
    qtns = np.zeros((num_qtns, max_qtn_len), dtype=np.int32)
    qtn_lens = np.zeros(num_qtns, dtype=np.int32)
    qtn_ctx_idxs = np.zeros(num_qtns, dtype=np.int32)

    qtn_ctx = np.zeros((num_qtns, max_ctx_len), dtype=np.int32)
    qtn_ctx_lens = np.zeros(num_qtns, dtype=np.int32)
    qtn_ans_inds = np.zeros(num_qtns, dtype=np.int32)
    anss = np.zeros((num_qtns, 2), dtype=np.int32)

    for ctx_idx, ctx in enumerate(tabular.ctxs):
        ctx_words = [word_emb_data.str_to_word[word_str] for word_str in ctx.tokenized.tokens]
        all_ctxs[ctx_idx, :len(ctx_words)] = ctx_words
        all_ctx_lens[ctx_idx] = len(ctx_words)

    for qtn_idx, qtn in enumerate(tabular.qtns):
        qtn_words = [word_emb_data.str_to_word[word_str] for word_str in qtn.tokenized.tokens]
        qtns[qtn_idx, :len(qtn_words)] = qtn_words
        qtn_lens[qtn_idx] = len(qtn_words)
        qtn_ctx[qtn_idx] = all_ctxs[qtn.ctx_idx]
        qtn_ctx_lens[qtn_idx] = all_ctx_lens[qtn.ctx_idx]
        ans = next((ans for ans in qtn.ans_word_idxs if ans), None) if qtn.ans_word_idxs else None
        if ans:
            ans_start_word_idx, ans_end_word_idx = ans
            anss[qtn_idx] = [ans_start_word_idx, ans_end_word_idx]
            qtn_ans_inds[qtn_idx] = 1
        else:
            qtn_ans_inds[qtn_idx] = 0

    return SquadDatasetVectorized(qtn_ctx, qtn_ctx_lens, qtns, qtn_lens, anss)

class DatasetLoader(Dataset):
    def __init__(self, ctx, ctx_lens, qtns, qtn_lens, anss):
        self.ctx = ctx
        self.ctx_lens = ctx_lens
        self.qtns = qtns
        self.qtn_lens = qtn_lens
        self.anss = anss

    def __len__(self):
        return self.qtns.shape[0]

    def __getitem__(self, key):
        return self.ctx[key], self.ctx_lens[key], self.qtns[key], self.qtn_lens[key],self.anss[key]

def _np_ans_word_idxs_to_ans_idx(ans_start_word_idx, ans_end_word_idx, max_ans_len):
    # all arguments are concrete ints
    assert ans_end_word_idx - ans_start_word_idx + 1 <= max_ans_len
    return ans_start_word_idx * max_ans_len + (ans_end_word_idx - ans_start_word_idx)


def prepare_data(batch):
    ctxs = []
    ctx_lens = []
    qtns = []
    qtn_lens = []
    anss = []

    for datum in batch:
        ctxs.append(datum[0])
        ctx_lens.append(datum[1])
        qtns.append(datum[2])
        qtn_lens.append(datum[3])
        anss.append(datum[4])

    max_question_length = int(max(qtn_lens))
    max_ctx_length = int(max(ctx_lens))
    n_samples = len(qtns)

    contexts = torch.zeros(max_ctx_length, n_samples).long()
    contexts_mask = torch.zeros(max_ctx_length, n_samples)
    contexts_lens = torch.zeros(n_samples, ).long()

    questions = torch.zeros(max_question_length, n_samples).long()
    questions_mask = torch.zeros(max_question_length, n_samples)
    final_answers = torch.zeros(n_samples, 2).long()
    final_y = torch.zeros(n_samples, ).long()

    if can_use_gpu:
        contexts = contexts.cuda()
        contexts_mask = contexts_mask.cuda()
        contexts_lens = contexts_lens.cuda()
        questions = questions.cuda()
        questions_mask = questions_mask.cuda()
        final_answers = final_answers.cuda()
        final_y = final_y.cuda()

    for idx, [ctx, ctx_len, qtn, qtn_len, ans] in enumerate(zip(ctxs, ctx_lens, qtns, qtn_lens, anss)):
        if can_use_gpu:
            contexts[:ctx_len, idx] = torch.from_numpy(ctx[:ctx_len]).long().cuda()
            questions[:qtn_len, idx] = torch.from_numpy(qtn[:qtn_len]).long().cuda()
            contexts_mask[:ctx_len, idx] = 1.
            questions_mask[:qtn_len, idx] = 1.
            final_answers[idx] = torch.from_numpy(ans).cuda()
            contexts_lens[idx] = int(ctx_len)
            final_y[idx] = int(_np_ans_word_idxs_to_ans_idx(ans[0], ans[1], max_ans_len))

        else:
            contexts[:ctx_len, idx] = torch.from_numpy(ctx[:ctx_len]).long()
            questions[:qtn_len, idx] = torch.from_numpy(qtn[:qtn_len]).long()
            contexts_mask[:ctx_len, idx] = 1.
            questions_mask[:qtn_len, idx] = 1.
            final_answers[idx] = torch.from_numpy(ans)
            contexts_lens[idx] = int(ctx_len)
            final_y[idx] = int(_np_ans_word_idxs_to_ans_idx(ans[0], ans[1], max_ans_len))

    return contexts, contexts_mask, questions, questions_mask, final_answers, contexts_lens, final_y


def load_data(train_loc, dev_loc, batch_size):
    word_emb_data = read_word_emb_data()
    word_strs = set()

    trn_tab_ds = _make_tabular_dataset(train_loc, word_strs, has_answers=True, max_ans_len=max_ans_len)

    dev_tab_ds = _make_tabular_dataset(dev_loc, word_strs, has_answers=True, max_ans_len=max_ans_len)

    word_emb_data = _contract_word_emb_data(word_emb_data, word_strs, learn_single_unk)

    trn_vec_ds = _make_vectorized_dataset(trn_tab_ds, word_emb_data)

    dev_vec_ds = _make_vectorized_dataset(dev_tab_ds, word_emb_data)

    train_dataset = DatasetLoader(trn_vec_ds.qtn_ctx, trn_vec_ds.qtn_ctx_lens, trn_vec_ds.qtns, trn_vec_ds.qtn_lens,
                                  trn_vec_ds.anss)
    dev_dataset = DatasetLoader(dev_vec_ds.qtn_ctx, dev_vec_ds.qtn_ctx_lens, dev_vec_ds.qtns, dev_vec_ds.qtn_lens,
                                dev_vec_ds.anss)

    train_loader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              collate_fn=prepare_data
                              )
    dev_loader = DataLoader(dataset=dev_dataset,
                            shuffle=False,
                            batch_size=batch_size)
    # collate_fn=prepare_data)
    return train_loader, dev_loader, word_emb_data

train_data, dev_data, word_emb_data = load_data(tokenized_trn_json_path, tokenized_dev_json_path, 32)

word_emb_tensor = torch.from_numpy(word_emb_data.word_emb)

if can_use_gpu:
    word_emb_tensor = word_emb_tensor.cuda()

word_emb = Variable(word_emb_tensor)

class LSTM(nn.Module):
    def __init__(self, nin, hidden_size):
        super(LSTM, self).__init__()
        if torch.cuda.is_available():
            self.linear_f = nn.Linear(nin + hidden_size, hidden_size).cuda()
            self.linear_i = nn.Linear(nin + hidden_size, hidden_size).cuda()
            self.linear_ctilde = nn.Linear(nin + hidden_size, hidden_size).cuda()
            self.linear_o = nn.Linear(nin + hidden_size, hidden_size).cuda()

        else:
            self.linear_f = nn.Linear(nin + hidden_size, hidden_size)
            self.linear_i = nn.Linear(nin + hidden_size, hidden_size)
            self.linear_ctilde = nn.Linear(nin + hidden_size, hidden_size)
            self.linear_o = nn.Linear(nin + hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.init_weights()

    def forward(self, x, mask):
        hidden, c = self.init_hidden(x.size(1))

        def step(emb, hid, c_t_old, mask_cur):
            combined = torch.cat((hid, emb), 1)

            f = F.sigmoid(self.linear_f(combined))
            i = F.sigmoid(self.linear_i(combined))
            o = F.sigmoid(self.linear_o(combined))
            c_tilde = F.tanh(self.linear_ctilde(combined))

            c_t = f * c_t_old + i * c_tilde
            c_t = mask_cur[:, None] * c_t + (1. - mask_cur)[:, None] * c_t_old

            hid_new = o * F.tanh(c_t)
            hid_new = mask_cur[:, None] * hid_new + (1. - mask_cur)[:, None] * hid

            return hid_new, c_t, i

        h_hist = []
        i_hist = []
        for i in range(x.size(0)):
            hidden, c, i = step(x[i].squeeze(), hidden, c, mask[i])
            h_hist.append(hidden[None, :, :])
            i_hist.append(i[None, :, :])

        return torch.cat(h_hist), torch.cat(i_hist)

    def init_hidden(self, bat_size):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(bat_size, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(bat_size, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(bat_size, self.hidden_size))
            c0 = Variable(torch.zeros(bat_size, self.hidden_size))
        return h0, c0

    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear_f, self.linear_i, self.linear_ctilde, self.linear_o]

        for layer in lin_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.fill_(0)


class QAModel(nn.Module):
    def __init__(self, word_embeddings, emb_dim, hidden_dim):
        super(QAModel, self).__init__()
        self.LSTM1 = LSTM(emb_dim, hidden_dim)
        self.LSTM1_rev = LSTM(emb_dim, hidden_dim)

        self.LSTM2 = LSTM(2 * hidden_dim, hidden_dim)
        self.LSTM2_rev = LSTM(2 * hidden_dim, hidden_dim)

        self.ff_dims = 100
        self.p_start_dim = emb_dim + 2 * hidden_dim + emb_dim

        self.p_LSTM1 = LSTM(self.p_start_dim, hidden_dim)
        self.p_LSTM1_rev = LSTM(self.p_start_dim, hidden_dim)

        self.p_LSTM2 = LSTM(2 * hidden_dim, hidden_dim)
        self.p_LSTM2_rev = LSTM(2 * hidden_dim, hidden_dim)

        self.word_emb = word_emb

        self.linear1 = nn.Linear(2 * hidden_dim, self.ff_dims)
        self.linear_w = nn.Linear(self.ff_dims, 1, bias=False)
        self.linear_q_aligned = nn.Linear(emb_dim, self.ff_dims, bias=False)
        self.linear_ans_start = nn.Linear(2 * hidden_dim, self.ff_dims)
        self.linear_ans_end = nn.Linear(2 * hidden_dim, self.ff_dims, bias=False)
        self.linear_span = nn.Linear(self.ff_dims, 1, bias=False)

        if can_use_gpu:
            self.linear1 = self.linear1.cuda()
            self.linear_w = self.linear_w.cuda()
            self.linear_q_aligned = self.linear_q_aligned.cuda()
            self.linear_ans_start = self.linear_ans_start.cuda()
            self.linear_ans_end = self.linear_ans_end.cuda()
            self.linear_span = self.linear_span.cuda()

        self.init_weights()

    def forward(self, contexts, contexts_mask, questions, questions_mask, anss, contexts_lens):
        n_timesteps_cntx = contexts.size(0)
        n_timesteps_quest = questions.size(0)
        n_samples = contexts.size(1)

        emb_cntx = self.word_emb[contexts.view(-1)].view(n_timesteps_cntx, n_samples, emb_dim)
        emb_quest = self.word_emb[questions.view(-1)].view(n_timesteps_quest, n_samples, emb_dim)

        q_indep = self.compute_q_indep(emb_quest, questions_mask, emb_cntx.size(0))
        q_align = self.compute_q_aligned(emb_cntx, emb_quest, contexts_mask, questions_mask)

        p_star = torch.cat((emb_cntx, q_indep, q_align), 2)

        passage_level = self.sequence_encoder(p_star, contexts_mask, self.p_LSTM1, self.p_LSTM1_rev, self.p_LSTM2,
                                              self.p_LSTM2_rev)

        loss, acc, sum_acc, sum_loss = self.compute_answer(passage_level[2], passage_level[2], contexts_lens,
                                                           batch_size, anss)
        return loss, acc, sum_acc, sum_loss

    def sequence_encoder(self, inp, mask, lstm1, lstm_rev1, lstm2, lstm_rev2):
        reverse_emb = self.reverseTensor(inp)
        reverse_mask = self.reverseTensor(mask)

        #  LSTM1
        seq1 = lstm1(inp, mask)
        seq_reverse1 = lstm_rev1(reverse_emb, reverse_mask)

        inp_seq2 = torch.cat((seq1[0], self.reverseTensor(seq_reverse1[0])), len(seq1[0].size()) - 1)
        reverse_inp_seq2 = self.reverseTensor(inp_seq2)

        #  LSTM2
        seq2 = lstm2(inp_seq2, mask)
        seq_reverse2 = lstm_rev2(reverse_inp_seq2, reverse_mask)

        cat_seq2 = torch.cat((seq2[0], self.reverseTensor(seq_reverse2[0])), len(seq2[0].size()) - 1)
        return seq2, seq_reverse2, cat_seq2

    def compute_q_indep(self, q_emb, q_mask, max_p_len):
        encoder_out = self.sequence_encoder(q_emb, q_mask, self.LSTM1, self.LSTM1_rev, self.LSTM2, self.LSTM2_rev)
        q_indep_h = encoder_out[2]
        q_indep_ff = self.linear1(q_indep_h)
        q_indep_scores = self.linear_w(q_indep_ff)

        q_indep_weights = self.softmax_columns_with_mask(q_indep_scores.squeeze(), q_mask)  # (max_q_len, batch_size)
        q_indep = torch.sum(q_indep_weights.unsqueeze(2) * q_indep_h, dim=0)  # (batch_size, 2*hidden_dim)

        q_indep_repeated = torch.cat([q_indep.unsqueeze(0)] * max_p_len)

        return q_indep_repeated

    def compute_q_aligned(self, p_emb, q_emb, p_mask, q_mask):
        q_align_ff_p = self.linear_q_aligned(p_emb)
        q_align_ff_q = self.linear_q_aligned(q_emb)

        q_align_ff_p_shuffled = q_align_ff_p.permute(1, 0, 2)  # (batch_size, max_p_len, ff_dim)
        q_align_ff_q_shuffled = q_align_ff_q.permute(1, 2, 0)
        q_align_scores = torch.bmm(q_align_ff_p_shuffled, q_align_ff_q_shuffled)

        p_mask_shuffled = p_mask.unsqueeze(2).permute(1, 0, 2)
        q_mask_shuffled = q_mask.unsqueeze(2).permute(1, 2, 0)
        pq_mask = torch.bmm(p_mask_shuffled, q_mask_shuffled)

        q_align_weights = self.softmax_depths_with_mask(q_align_scores, pq_mask)
        q_emb_shuffled = q_emb.permute(1, 0, 2)
        q_align = torch.bmm(q_align_weights, q_emb_shuffled)
        q_align_shuffled = q_align.permute(1, 0, 2)
        return q_align_shuffled

    def reverseTensor(self, tensor):
        idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
        if torch.cuda.is_available():
            idx = Variable(torch.LongTensor(idx).cuda())
        else:
            idx = Variable(torch.LongTensor(idx))
        inverted_tensor = tensor.index_select(0, idx)
        return inverted_tensor

    def compute_answer(self, p_level_h_for_stt, p_level_h_for_end, p_lens, batch_size, anss):
        max_p_len = p_level_h_for_stt.size(0)
        p_stt_lin = self.linear_ans_start(p_level_h_for_stt)
        p_end_lin = self.linear_ans_end(p_level_h_for_end)
        span_lin_reshaped, span_masks_reshaped = self._span_sums(p_stt_lin, p_end_lin, p_lens, max_p_len, batch_size,
                                                                 self.ff_dims, max_ans_len)
        span_ff_reshaped = F.relu(span_lin_reshaped)  # (batch_size, max_p_len*max_ans_len, ff_dim)
        span_scores_reshaped = self.linear_span(span_ff_reshaped).squeeze()
        xents, accs, a_hats = self._span_multinomial_classification(span_scores_reshaped, span_masks_reshaped, anss)
        loss = xents.mean()
        acc = accs.mean()
        sum_acc = accs.sum()
        sum_loss = loss.sum()
        return loss, acc, sum_acc, sum_loss

    def softmax_columns_with_mask(self, x, mask, allow_none=False):
        assert len(x.size()) == 2
        assert len(mask.size()) == 2
        # for numerical stability

        x = x * mask
        x = x - x.min(dim=0, keepdim=True)[0]
        x = x * mask
        x = x - x.max(dim=0, keepdim=True)[0]
        e_x = mask * torch.exp(x)
        sums = e_x.sum(dim=0, keepdim=True)
        if allow_none:
            sums += torch.eq(sums, 0)
        y = e_x / sums
        return y

    def _span_multinomial_classification(self, x, x_mask, y):
        # x       float32 (batch_size, num_classes)   scores i.e. logits
        # x_mask  int32   (batch_size, num_classes)   score masks (each sample has a variable number of classes)
        # y       int32   (batch_size,)               target classes i.e. ground truth answers (given as class indices)
        assert len(x.size()) == len(x_mask.size()) == 2
        assert len(y.size()) == 1

        # substracting min needed since all non masked-out elements of a row may be negative.
        x = x * x_mask
        x = x - x.min(dim=0, keepdim=True)[0]  # (batch_size, num_classes)
        x = x * x_mask  # (batch_size, num_classes)
        y_hats = x.max(dim=1)[1]  # (batch_size,)
        accs = torch.eq(y_hats.long(), y.long()).float()  # (batch_size,)

        x = x - x.max(dim=1, keepdim=True)[0]  # (batch_size, num_classes)
        x = x * x_mask  # (batch_size, num_classes)
        exp_x = torch.exp(x)  # (batch_size, num_classes)
        exp_x = exp_x * x_mask  # (batch_size, num_classes)

        sum_exp_x = exp_x.sum(dim=1)  # (batch_size,)
        log_sum_exp_x = torch.log(sum_exp_x)  # (batch_size,)
        index1 = torch.arange(0, x.size(0)).long()
        if can_use_gpu:
            index1 = index1.cuda()
        x_star = x[index1, y.data]  # (batch_size,)
        xents = log_sum_exp_x - x_star  # (batch_size,)
        return xents, accs, y_hats

    def _span_sums(self, stt, end, p_lens, max_p_len, batch_size, dim, max_ans_len):
        max_ans_len_range = torch.arange(0, max_ans_len).unsqueeze(0)  # (1, max_ans_len)
        offsets = torch.arange(0, max_p_len).unsqueeze(1)  # (max_p_len, 1)
        if can_use_gpu:
            max_ans_len_range = max_ans_len_range.cuda()
            offsets = offsets.cuda()

        end_idxs = max_ans_len_range + offsets  # (max_p_len, max_ans_len)
        end_idxs_flat = end_idxs.view(-1).long()  # (max_p_len*max_ans_len,)
        extra_zeros = torch.zeros(max_ans_len - 1, batch_size, dim)
        if can_use_gpu:
            extra_zeros = extra_zeros.cuda()
        end_padded = torch.cat([end, Variable(extra_zeros)], 0)  # (max_p_len+max_ans_len-1, batch_size, dim)

        end_structured = end_padded[end_idxs_flat]  # (max_p_len*max_ans_len, batch_size, dim)

        end_structured = end_structured.view(max_p_len, max_ans_len, batch_size,
                                             dim)  # (max_p_len, max_ans_len, batch_size, dim)
        stt_shuffled = stt.unsqueeze(3).permute(0, 3, 1, 2)  # (max_p_len, 1, batch_size, dim)

        span_sums = stt_shuffled + end_structured  # (max_p_len, max_ans_len, batch_size, dim)
        span_sums_reshaped = span_sums.permute(2, 0, 1, 3).contiguous().view(batch_size, max_p_len * max_ans_len,
                                                                             dim)  # (batch_size, max_p_len*max_ans_len, dim)

        p_lens_shuffled = p_lens.unsqueeze(1)  # (batch_size, 1)
        end_idxs_flat_shuffled = end_idxs_flat.unsqueeze(0)  # (1, max_p_len*max_ans_len)

        span_masks_reshaped = torch.lt(Variable(end_idxs_flat_shuffled), p_lens_shuffled)  # (batch_size, max_p_len*max_ans_len)
        span_masks_reshaped = span_masks_reshaped.float()

        # (batch_size, max_p_len*max_ans_len, dim), (batch_size, max_p_len*max_ans_len)
        return span_sums_reshaped, span_masks_reshaped

    def softmax_depths_with_mask(self, x, mask):
        assert len(x.size()) == 3
        assert len(mask.size()) == 3
        # for numerical stability
        x = x * mask
        x = x - x.min(dim=2, keepdim=True)[0]
        x = x * mask
        x = x - x.max(dim=2, keepdim=True)[0]
        e_x = mask * torch.exp(x)
        sums = e_x.sum(dim=2, keepdim=True)
        y = e_x / (sums + (torch.eq(sums, 0).float()))
        y = y * mask
        return y

    def init_weights(self):
        initrange = 0.1
        with_bias = [self.linear1, self.linear_ans_start]
        without_bias = [self.linear_w, self.linear_q_aligned, self.linear_ans_end, self.linear_span]

        for layer in with_bias:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.fill_(0)
        for layer in without_bias:
            layer.weight.data.uniform_(-initrange, initrange)


model = QAModel(word_emb,emb_dim, hidden_dim)
if can_use_gpu:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

x = 0
train_loss_hist = []
train_acc_hist = []
for epoch in range(max_num_epochs):
    loss_curr_epoch = 0.0
    acc_curr_epoch = 0.0
    n_done = 0
    uidx = 0
    for data in train_data:
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        contexts = Variable(data[0])
        n_done += contexts.size(1)
        contexts_mask = Variable(data[1])
        questions = Variable(data[2])
        questions_mask = Variable(data[3])
        anss = data[4]
        contexts_lens = Variable(data[5])
        y = Variable(data[6])

        loss, acc, sum_acc, sum_loss = model(contexts, contexts_mask, questions, questions_mask, y, contexts_lens)

        loss_curr_epoch += sum_loss
        acc_curr_epoch += sum_acc
        msg = 'uidx = ', uidx, 'Current batch Accuracy %= ', acc.data[0] * 100, ' --  Loss = ', loss.data[0]
        print(msg)
        logger.info(msg)
        uidx += 1
        loss.backward()

    current_epoch_train_loss = loss_curr_epoch / n_done
    current_epoch_train_acc = acc_curr_epoch / n_done
    epoch_msg = 'End of Epoch' + (
    epoch + 1) + '. Training Accuracy %= ', current_epoch_train_acc * 100, ' --  Loss = ', current_epoch_train_loss
    print(epoch_msg)
    logger.info(epoch_msg)
    train_loss_hist.append(current_epoch_train_loss)
    train_acc_hist.append(current_epoch_train_acc)

