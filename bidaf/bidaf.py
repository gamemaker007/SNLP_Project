import pickle as pkl
import numpy as np
import random
from torch.utils.data import Dataset
from collections import namedtuple
from collections import Counter
import io,sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable

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
batch_size = 2
max_num_epochs = 150  # max number of epochs to train for
num_bilstm_layers = 2  # number of BiLSTM layers, where BiLSTM is applied
hidden_dim = 100  # dimension of hidden state of each uni-directional LSTM
lstm_drop_h = 0.1  # dropout rate for recurrent hidden state of LSTM
lstm_drop_x = 0.4  # dropout rate for inps of LSTM
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

	contexts = torch.zeros(n_samples,max_ctx_length ).long()
	contexts_mask = torch.zeros(n_samples, max_ctx_length)
	contexts_lens = torch.zeros(n_samples, ).long()
	questions_lens = torch.zeros(n_samples, ).long()

	questions = torch.zeros(n_samples, max_question_length).long()
	questions_mask = torch.zeros(n_samples, max_question_length)
	final_answers = torch.zeros(n_samples, 2).long()
	span_start = torch.zeros(n_samples, ).long()
	span_end = torch.zeros(n_samples, ).long()

	if can_use_gpu:
		contexts = contexts.cuda()
		contexts_mask = contexts_mask.cuda()
		contexts_lens = contexts_lens.cuda()
		questions = questions.cuda()
		questions_mask = questions_mask.cuda()
		final_answers = final_answers.cuda()
		span_start = span_start.cuda()
		span_end = span_end.cuda()



	for idx, [ctx, ctx_len, qtn, qtn_len, ans] in enumerate(zip(ctxs, ctx_lens, qtns, qtn_lens, anss)):
		if can_use_gpu:
			contexts[idx,:ctx_len] = torch.from_numpy(ctx[:ctx_len]).long().cuda()
			questions[idx,:qtn_len] = torch.from_numpy(qtn[:qtn_len]).long().cuda()
			contexts_mask[idx,:ctx_len] = 1.
			questions_mask[idx,:qtn_len] = 1.
			final_answers[idx] = torch.from_numpy(ans).cuda()
			contexts_lens[idx] = int(ctx_len)
			questions_lens[idx] = int(qtx_len)
			span_start[idx] = int(ans[0])
			span_end[idx] = int(ans[1])

		else:
			contexts[idx, :ctx_len] = torch.from_numpy(ctx[:ctx_len]).long()
			questions[idx,:qtn_len] = torch.from_numpy(qtn[:qtn_len]).long()
			contexts_mask[idx,:ctx_len] = 1.
			questions_mask[idx,:qtn_len] = 1.
			final_answers[idx] = torch.from_numpy(ans)
			contexts_lens[idx] = int(ctx_len)
			questions_lens[idx] = int(qtn_len)
			span_start[idx] = int(ans[0])
			span_end[idx] = int(ans[1])

	return contexts, contexts_mask, questions, questions_mask, final_answers, contexts_lens, questions_lens, span_start, span_end


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
							batch_size=batch_size,
							collate_fn=prepare_data)
	# collate_fn=prepare_data)
	return train_loader, dev_loader, word_emb_data

train_data, dev_data, word_emb_data = load_data(tokenized_trn_json_path, tokenized_dev_json_path, batch_size)

word_emb_tensor = torch.from_numpy(word_emb_data.word_emb)

if can_use_gpu:
	word_emb_tensor = word_emb_tensor.cuda()

word_emb = Variable(word_emb_tensor)


class CharCNN():
	def __init__(self,emb_dim, num_filters=100, kernel_size=5, out_size=5):
		super(CharCNN,self).__init__()
		self.emb_dim = emb_dim
		self.num_filters = num_filters
		self.kernel_size = kernel_size

		self.conv_1d = nn.Conv1d(emb_dim, num_filters, kernel_size)
		self.linear = nn.Linear(num_filters,out_size)

	def forward(self,x):
		x = F.relu(self.conv_1d(x))
		max_out,_ = x.max(dim=2)
		out = self.linear(max_out)
		return out


class HighwayNetwork(nn.Module):
	def __init__(self, dim, num_layers):
		super(HighwayNetwork, self).__init__()
		self.num_layers = num_layers
		self.dim = dim
		self.H = nn.ModuleList([nn.Linear(dim, dim) for l in range(num_layers)])
		self.T = nn.ModuleList([nn.Linear(dim, dim) for t in range(num_layers)])
		for t_i in self.T:
			t_i.bias.data.fill_(-1)
	def forward(self, x):
		for i in range(self.num_layers):
			h_i = F.relu(self.H[i](x))
			t_i = F.sigmoid(self.T[i](x))
			x = t_i * h_i + (1 - t_i) * x
		return x
		

def sort(data, seq_len):                                                  
	""" Sort the data (B, T, D) and sequence lengths                            
	"""                                                                         
	sorted_seq_len, sorted_idx = seq_len.sort(0, descending=True)               
	sorted_data = data[sorted_idx]                                              
	return sorted_data, sorted_seq_len, sorted_idx

class BiLSTM(nn.Module):
	def __init__(self, inp_size, hidden_size=100, num_layers=1, dropout=0.2):
		super(BiLSTM, self).__init__()
		self.lstm = nn.LSTM(input_size=inp_size, hidden_size=hidden_size,
						   num_layers=num_layers,
						   batch_first=True,
						   dropout=dropout,
						   bidirectional=True)


	def forward(self, inp, lens):
		# batch_size = inp.size()[0]
		# print('bilstm')
		# print(inp.size())
		# sort_inp, sort_lens, sort_idx = sort(inp, lens)
		# print(sort_inp.size()) 
		# packed = nn.utils.rnn.pack_padded_sequence(sort_inp, list(sort_lens.data), batch_first=True)    
		
		output, _ = self.lstm(inp)
		
		# unpacked, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
		# # get the last time step for each sequence
		# print(output.size())
		# idx = (sort_lens - 1).view(-1, 1).expand(output.size(0), output.size(2)).unsqueeze(1)
		# decoded = output.gather(1, idx)
		# # restore the sorting
		# print(decoded.size())
		# _, original_idx = sort_idx.sort(0, descending=True)
		# unsorted_idx = original_idx.view(-1, 1, 1).expand_as(unpacked)
		# output = unpacked.gather(0, unsorted_idx.long())
		# print(output.size())
		# print('byelstm')
		# sys.exit()
		return output


class ContextualEmbeddingLayer(nn.Module):
	def __init__(self, emb_dim):
		super(ContextualEmbeddingLayer, self).__init__()
		self.bilstm = BiLSTM(emb_dim)

	def forward(self, inp, lens):
		return self.bilstm(inp, lens)
		

class ModelingLayer(nn.Module):
	def __init__(self, inp_size):
		super(ModelingLayer, self).__init__()
		self.bilstm = BiLSTM(8*inp_size, num_layers=2)

	def forward(self, inp, lens):
		return self.bilstm(inp, lens)
		

class Attention(nn.Module):	

	def forward(self, X, Y):
		X_ = torch.bmm(Y,X) 
		return X_

def masked_softmax(vector, mask):

	if mask is None:
		result = torch.nn.functional.softmax(vector, dim=-1)
	else:
		# To limit numerical errors from large vector elements outside mask, we zero these out
		result = torch.nn.functional.softmax(vector * mask, dim=-1)
		result = result * mask
		result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
	return result

def masked_log_softmax(vector, mask):
	if mask is not None:
		vector = vector + mask.log()
	return torch.nn.functional.log_softmax(vector, dim=1)

def _last_dim_sofmax(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None):
	tensor_shape = tensor.size()
	# print(tensor_shape)
	# print(mask.size())
	reshaped_tensor = tensor.view(-1, tensor.size()[-1])
	if mask is not None:
		while mask.dim() < tensor.dim():
			mask = mask.unsqueeze(1)
		mask = mask.expand_as(tensor).contiguous().float()
		mask = mask.view(-1, mask.size()[-1])
	reshaped_result = masked_softmax(reshaped_tensor, mask)
	return reshaped_result.view(*tensor_shape)

def replace_masked_values(tensor: Variable, mask: Variable, replace_with: float) -> Variable:
	if tensor.dim() != mask.dim():
		raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
	one_minus_mask = 1.0 - mask
	values_to_add = replace_with * one_minus_mask
	return tensor * mask + values_to_add


class C2Q(nn.Module):
	def __init__(self):
		super(C2Q, self).__init__()
		self.attention = Attention()
  
	def forward(self,U, S, mask):
		# print(S.size())	
		# print(mask)
		# sys.exit()
		S_t = _last_dim_sofmax(S,mask)	 
		# print(S_t.size())
		# sys.exit()
		U_ = self.attention(U, S_t)
		return U_


class Q2C(nn.Module):
	def __init__(self):
		super(Q2C, self).__init__()
		self.attention = Attention()

	def forward(self,H, S, mask):
		
		N,T,d2 = H.size()
		S_max = torch.max(S, -1)[0].unsqueeze(-2) # Nx1xT
		# print(S_max.size())
		# sys.exit()
		S_t = _last_dim_sofmax(S_max, mask)
		H_ = self.attention(H, S_t) # Nx1 x2d
		H_ = H_.expand(-1,T,-1) # NxTx2d
		return H_


class Similarity(nn.Module):
	def __init__(self, d2):
		super(Similarity, self).__init__()
		self.d2 = d2
		self.linear = nn.Linear(3*d2,1)

	def forward(self, H, U):
		h_size = H.size() #	NxTx2d
		u_size = U.size() # NxJx2d
		
		h_ = H.unsqueeze(2).expand(-1,-1,u_size[1],-1) # NxTxJx2d
		u_ = U.unsqueeze(1).expand(-1,h_size[1],-1,-1) # NxTxJx2d
		h_u = torch.mul(h_,u_) # NxTxJx2d
		H_U = torch.cat((h_,u_,h_u),3) # NxTxJx6d
		S = self.linear(H_U) # NxTxJx1
		S = S.squeeze() # NxTxJ
		return S

class BiAttentionLayer(nn.Module):
	def __init__(self):
		super(BiAttentionLayer,self).__init__()
		# self.H = H 
		# self.U = U
		
		self.sim = Similarity(2*100)
		self.c2q = C2Q()
		self.q2c = Q2C()


	def forward(self, H ,U, c_mask, q_mask):
		S = self.sim(H, U)		

		U_ = self.c2q(U, S, q_mask)
		S_ = replace_masked_values(S,q_mask.unsqueeze(1),-1e7)
		H_ = self.q2c(H, S_, c_mask)
		G = torch.cat([H, U_, torch.mul(H, U_), torch.mul(H, H_)], -1)
		return G

class OutputLayer(nn.Module):
	def __init__(self, size):
		super(OutputLayer, self).__init__()
		self.linear1 = nn.Linear(10*size,1)
		self.linear2 = nn.Linear(10*size,1)
		self.bilstm = BiLSTM(2*size)

	def forward(self, G, M, lens, mask):
		N = torch.cat([G, M], dim=-1)
		p1 = self.linear1(N).squeeze(-1)
		p1_ = masked_softmax(p1, mask)
		M2 = self.bilstm(M, lens)
		M = torch.cat([G, M2], dim=-1)
		p2 = self.linear2(M).squeeze(-1)
		p2_ = masked_softmax(p2, mask)
		return p1,p2,p1_,p2_


class Bidaf(nn.Module):
	def __init__(self, word_emb, emb_dim, hidden_size):
		super(Bidaf, self).__init__()
		# self.layer1 = EmbeddingLayer()
		self.highwayC = HighwayNetwork(emb_dim, 2)
		self.highwayQ = HighwayNetwork(emb_dim, 2)
		self.contextC = ContextualEmbeddingLayer(emb_dim)
		self.contextQ = ContextualEmbeddingLayer(emb_dim)
		self.biattention = BiAttentionLayer()
		self.model = ModelingLayer(hidden_size)
		self.output = OutputLayer(hidden_size)
		self.word_emb = word_emb

	def forward(self, contexts, contexts_mask, questions, questions_mask, contexts_lens, questions_lens):
		n_timesteps_cntx = contexts.size()[1]
		n_timesteps_quest = questions.size()[1]
		n_samples = contexts.size()[0]
		# print('cs')
		# print(contexts.size())

		word_emb_cntx = self.word_emb[contexts.view(-1)].view( n_samples,n_timesteps_cntx, emb_dim)
		word_emb_quest = self.word_emb[questions.view(-1)].view(n_samples,n_timesteps_quest,  emb_dim)
		# print('cemb')
		# print(word_emb_cntx.size())
		X = self.highwayC(word_emb_cntx)
		# print('hw')
		# print(X.size())
		Q = self.highwayQ(word_emb_quest)
		# print(Q.size())
		H = self.contextC(X, contexts_lens)
		# print('con')
		# print(H.size())
		U = self.contextQ(Q, questions_lens)
		# print(U.size())
		G = self.biattention(H,U, contexts_mask, questions_mask)
		# print('biatt')
		# print(G.size())
		
		M = self.model(G, contexts_lens)
		# print('modell')
		# print(M.size())
		 
		p1,p2,p1_,p2_ = self.output(G,M, contexts_lens, contexts_mask)
		
		return p1,p2,p1_,p2_



def _get_best_span(span_start_logits: Variable, span_end_logits: Variable) -> Variable:
	if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
		raise ValueError("Inp shapes must be (batch_size, passage_length)")
	batch_size, passage_length = span_start_logits.size()
	max_span_log_prob = [-1e20] * batch_size
	span_start_argmax = [0] * batch_size
	best_word_span = Variable(span_start_logits.data.new()
							  .resize_(batch_size, 2).fill_(0)).long()

	span_start_logits = span_start_logits.data.cpu().numpy()
	span_end_logits = span_end_logits.data.cpu().numpy()

	for b in range(batch_size):  # pylint: disable=invalid-name
		for j in range(passage_length):
			val1 = span_start_logits[b, span_start_argmax[b]]
			if val1 < span_start_logits[b, j]:
				span_start_argmax[b] = j
				val1 = span_start_logits[b, j]

			val2 = span_end_logits[b, j]

			if val1 + val2 > max_span_log_prob[b]:
				best_word_span[b, 0] = span_start_argmax[b]
				best_word_span[b, 1] = j
				max_span_log_prob[b] = val1 + val2
	return best_word_span
		

def get_score(best_span, span_start, span_end):
	batch_size= span_start.size()[0]
	f1_score = 0.0
	em_score = 0.0
	for i in range(batch_size):
		
		ground = set(list(range(span_start.data[i],span_end.data[i]+1)))
		pred = set(list(range(best_span.data[i][0], best_span.data[i][1]+1)))
		# print(ground)		
		# print(pred)
		inter = pred & ground
		# print('inter')
		# print(len(inter))
		# print(inter)
		prec =  len(inter)/len(pred)
		rec = len(inter)/len(ground)
		f1=0.0
		em = 0.0
		if prec!=0 and rec!=0:
			f1 = 2*prec*rec/(prec+rec)
		if pred == ground:
			em=1.0
		f1_score+=f1
		em_score+=em
	return f1_score, em_score

# all_letters = string.ascii_letters + " .,;'"
# n_letters = len(all_letters)

# # Find letter index from all_letters, e.g. "a" = 0
# def letterToIndex(letter):
# 	return all_letters.find(letter)


# # Turn a line into a <line_length x 1 x n_letters>,
# # or an array of one-hot letter vectors
# def lineToTensor(line):
# 	tensor = torch.zeros(len(line), 1, n_letters)
# 	for li, letter in enumerate(line):
# 		tensor[li][0][letterToIndex(letter)] = 1
# 	return tensor
model = Bidaf(word_emb, emb_dim, hidden_dim)

if can_use_gpu:
	model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(max_num_epochs):
	loss_curr_epoch = 0.0
	F1_curr_epoch = 0.0
	EM_curr_epoch = 0.0
	n_done = 0
	uidx = 0
	train_size = len(train_data)

	for data in train_data:
		
		model.train()
		model.zero_grad()
		optimizer.zero_grad()

		contexts = Variable(data[0])
		n_done += contexts.size()[0]
		contexts_mask = Variable(data[1])
		questions = Variable(data[2])
		questions_mask = Variable(data[3])
		anss = data[4]
		contexts_lens = Variable(data[5])
		questions_lens = Variable(data[6])
		span_start = Variable(data[7])
		span_end = Variable(data[8])

		p1, p2, p1_, p2_ = model(contexts, contexts_mask, questions, questions_mask, contexts_lens, questions_lens)

		log_p1 = masked_log_softmax(p1, contexts_mask)
		log_p2 = masked_log_softmax(p2, contexts_mask)

		loss = F.nll_loss(log_p1, span_start,size_average=False) + F.nll_loss(log_p2, span_end,size_average=False)
		
		loss.backward()
		optimizer.step()
		best_span = _get_best_span(p1,p2)

		f1,em = get_score(best_span, span_start, span_end)

		batch_size = questions.size()[0]
		loss_curr_epoch += loss
		F1_curr_epoch += f1
		EM_curr_epoch += em
		msg = 'uidx = '+str(uidx)+ ' -- Current batch F1 %= '+ str(100.0*f1/batch_size) + ' --  EM = '+ str(100*em/batch_size) + ' --  Loss = '+ str(loss.data[0]/batch_size) + ' (' +str(n_done)+'/' + str(train_size) + ')'
		# print(msg)
		logger.info(msg)
		

	epoch_msg = 'End of Epoch ' + str(epoch + 1) + '.-- Training F1 %= ' +str(F1_curr_epoch * 100.0/train_size) + ' --  EM = '+ str(100.0*EM_curr_epoch/train_size)+ ' --  Loss = ' + str(loss_curr_epoch.data[0] / train_size)
	# print(epoch_msg)
	logger.info(epoch_msg)

	valloss_curr_epoch = 0.0
	valF1_curr_epoch = 0.0
	valEM_curr_epoch = 0.0
	test_size = len(dev_data)
	for data in dev_data:
		model.eval()
		contexts = Variable(data[0])
		n_done += contexts.size()[0]
		contexts_mask = Variable(data[1])
		questions = Variable(data[2])
		questions_mask = Variable(data[3])
		anss = data[4]
		contexts_lens = Variable(data[5])
		questions_lens = Variable(data[6])
		span_start = Variable(data[7])
		span_end = Variable(data[8])

		p1, p2, p1_, p2_ = model(contexts, contexts_mask, questions, questions_mask, contexts_lens, questions_lens)
		log_p1 = masked_log_softmax(p1, contexts_mask)
		log_p2 = masked_log_softmax(p2, contexts_mask)

		loss = F.nll_loss(log_p1, span_start,size_average=False) + F.nll_loss(log_p2, span_end,size_average=False)
		
		f1,em = get_score(best_span, span_start, span_end)

		batch_size = questions.size()[0]
		valloss_curr_epoch += loss
		valF1_curr_epoch += f1
		valEM_curr_epoch += em
		break
		
	epoch_msg = 'End of Epoch ' + str(epoch + 1) + '.-- Validation F1 %= ' +str(valF1_curr_epoch * 100.0/test_size) + ' --  EM = '+ str(100.0*valEM_curr_epoch/test_size)+ ' --  Loss = ' + str(valloss_curr_epoch.data[0] / test_size)
	# print(epoch_msg)
	logger.info(epoch_msg)
	
	
	



