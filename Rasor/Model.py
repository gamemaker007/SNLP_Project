from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from Settings import *

WordEmbData = namedtuple('WordEmbData', [
    'word_emb',
    'str_to_word',
    'first_known_word',
    'first_unknown_word',
    'first_unallocated_word'
])

TokenizedText = namedtuple('TokenizedText', [
    'text',
    'tokens',
    'originals',
    'whitespace_afters',
])

SquadArticle = namedtuple('SquadArticle', [
    'art_title_str'
])

SquadContext = namedtuple('SquadContext', [
    'art_idx',
    'tokenized'
])

SquadQuestion = namedtuple('SquadQuestion', [
    'ctx_idx',
    'qtn_id',
    'tokenized',
    'ans_texts',
    'ans_word_idxs'
])


class SquadStorage(object):
    def __init__(self):
        self.arts = []
        self.ctxs = []
        self.qtns = []

    def new_article(self, art_title_str):
        self.arts.append(SquadArticle(art_title_str))
        return len(self.arts) - 1

    def new_context(self, art_idx, ctx_tokenized):
        self.ctxs.append(SquadContext(art_idx, ctx_tokenized))
        return len(self.ctxs) - 1

    def new_question(self, ctx_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs):
        self.qtns.append(SquadQuestion(ctx_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs))


class DatasetLoader(Dataset):
    def __init__(self, squaddataset):
        self.squaddataset = squaddataset

    def __len__(self):
        return len(self.squaddataset.qtns)

    def __getitem__(self, key):
        squadquestion = self.squaddataset.qtns[key]
        ctx_idx = squadquestion.ctx_idx
        context = self.squaddataset.ctxs[ctx_idx]
        ctx_tokens = context.tokenized.tokens
        qtn_tokens = squadquestion.tokenized.tokens
        len_ctx = len(ctx_tokens)
        len_qtn = len(qtn_tokens)
        cntx_originals = context.tokenized.originals
        cntx_whitespace = context.tokenized.whitespace_afters

        anss = [0, 0]
        ans = next((ans for ans in squadquestion.ans_word_idxs if ans), None) if squadquestion.ans_word_idxs else None
        if ans:
            ans_start_word_idx, ans_end_word_idx = ans
            anss = [ans_start_word_idx, ans_end_word_idx]

        return ctx_tokens, len_ctx, qtn_tokens, len_qtn, anss, squadquestion.ans_texts, cntx_originals, cntx_whitespace


class QAModel(nn.Module):
    def __init__(self, word_embeddings, emb_dim, hidden_dim):
        super(QAModel, self).__init__()

        self.hidden_dim = hidden_dim

        self.ff_dims = 100
        self.p_start_dim = emb_dim + 2 * hidden_dim + emb_dim

        self.word_emb = word_embeddings

        self.drop = nn.Dropout(p=0.4)

        self.passage_level_lstm = nn.LSTM(self.p_start_dim, hidden_dim, 2, bidirectional=True)
        self.q_indep_encoder = nn.LSTM(emb_dim, hidden_dim, 2, bidirectional=True)

        self.linear1 = nn.Linear(2 * hidden_dim, self.ff_dims)
        self.linear_w = nn.Linear(self.ff_dims, 1, bias=False)
        self.linear_q_aligned = nn.Linear(emb_dim, self.ff_dims)
        self.linear_ans_start = nn.Linear(2 * hidden_dim, self.ff_dims)
        self.linear_ans_end = nn.Linear(2 * hidden_dim, self.ff_dims)
        self.linear_stt = nn.Linear(self.ff_dims, 1, bias=False)
        self.linear_end = nn.Linear(self.ff_dims, 1, bias=False)
        self.init_weights()

    def forward(self, contexts, contexts_mask, questions, questions_mask, anss, contexts_lens, ans_start, ans_end):
        n_timesteps_cntx = contexts.size(0)
        n_timesteps_quest = questions.size(0)
        n_samples = contexts.size(1)

        context_embeddings = self.word_emb[contexts.view(-1)].view(n_timesteps_cntx, n_samples, emb_dim)
        question_embeddings = self.word_emb[questions.view(-1)].view(n_timesteps_quest, n_samples, emb_dim)

        passage_independent = self.compute_passage_independent(question_embeddings, questions_mask,
                                                               context_embeddings.size(0), n_samples)
        passage_aligned = self.compute_passage_aligned(context_embeddings, question_embeddings, contexts_mask,
                                                       questions_mask)
        p_star = torch.cat((context_embeddings, passage_independent, passage_aligned), 2)

        h_0 = Variable(torch.zeros(4, n_samples, self.hidden_dim))
        c_0 = Variable(torch.zeros(4, n_samples, self.hidden_dim))
        if torch.cuda.is_available:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        passage_level = self.passage_level_lstm(p_star, (h_0, c_0))

        loss, acc, sum_acc, sum_loss, a_hats = self.get_span_endpoints_rep(passage_level[0], contexts_lens,
                                                                           n_samples, ans_start, ans_end,
                                                                           contexts_mask, anss)

        computed_start_word_idx, computed_end_word_idxs = self.convert_idx_to_final_word_idxs(a_hats, max_ans_len)

        return loss, acc, sum_acc, sum_loss, computed_start_word_idx, computed_end_word_idxs

    def convert_idx_to_final_word_idxs(self, ans_idx, max_ans_len):
        ans_start_word_idx = torch.zeros(ans_idx.size(0), ).long()
        ans_end_word_idx = torch.zeros(ans_idx.size(0), ).long()
        for i in range(ans_idx.size(0)):
            ans_start_word_idx[i] = ans_idx.data[i] // max_ans_len
            ans_end_word_idx[i] = ans_start_word_idx[i] + ans_idx.data[i] % max_ans_len

        return ans_start_word_idx, ans_end_word_idx

    def compute_passage_independent(self, q_emb, q_mask, max_p_len, n_samples):
        h_0 = Variable(torch.zeros(4, n_samples, self.hidden_dim))
        c_0 = Variable(torch.zeros(4, n_samples, self.hidden_dim))
        if torch.cuda.is_available:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        encoder_out = self.q_indep_encoder(q_emb, (h_0, c_0))

        passage_independed_h = encoder_out[0]
        passage_independent_linear = F.relu(self.linear1(passage_independed_h))
        passage_independent_linear_scores = self.linear_w(passage_independent_linear)

        passage_independent_alignments_weights = self.softmax_columns_with_mask(
            passage_independent_linear_scores.squeeze(), q_mask)
        q_indep = torch.sum(passage_independent_alignments_weights.unsqueeze(2) * passage_independed_h,
                            dim=0)

        q_indep_repeated = torch.cat([q_indep.unsqueeze(0)] * max_p_len)

        return q_indep_repeated

    def compute_passage_aligned(self, p_emb, q_emb, p_mask, q_mask):
        q_align_ff_p = F.relu(self.linear_q_aligned(p_emb))
        q_align_ff_q = F.relu(self.linear_q_aligned(q_emb))

        _scores = torch.bmm(q_align_ff_p.permute(1, 0, 2), q_align_ff_q.permute(1, 2, 0))

        new_p_mask_ = p_mask.unsqueeze(2).permute(1, 0, 2)
        new_q_mask = q_mask.unsqueeze(2).permute(1, 2, 0)
        mulltiplied_mask = new_p_mask_ * new_q_mask
        attention_weights = self.softmax_depths_with_mask(_scores, mulltiplied_mask)
        new_q_emb = q_emb.permute(1, 0, 2)
        q_align = torch.bmm(attention_weights, new_q_emb)
        new_q_align = q_align.permute(1, 0, 2)
        return new_q_align

    def reverseTensor(self, tensor):
        idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
        if torch.cuda.is_available():
            idx = Variable(torch.LongTensor(idx).cuda())
        else:
            idx = Variable(torch.LongTensor(idx))
        inverted_tensor = tensor.index_select(0, idx)
        return inverted_tensor

    def get_anwser_values(self, p_level_h, p_lens, batch_size, anss):
        passage_level_h_for_start = self.drop(p_level_h)
        passage_level_h_for_end = self.drop(p_level_h)

        max_p_len = passage_level_h_for_start.size(0)
        p_stt_lin = self.linear_ans_start(passage_level_h_for_start)

        p_end_lin = self.linear_ans_end(passage_level_h_for_end)

        span_lin_reshaped, span_masks_reshaped = self._span_sums(p_stt_lin, p_end_lin, p_lens, max_p_len, batch_size,
                                                                 self.ff_dims, 1, max_ans_len)
        span_ff_reshaped = F.relu(span_lin_reshaped)
        span_scores_reshaped = self.linear_span(span_ff_reshaped).squeeze()
        entropies, accuracies, ans_hats = self.multi_classification(span_scores_reshaped, span_masks_reshaped, anss)
        loss = entropies.mean()
        acc = accuracies.mean()
        sum_acc = accuracies.sum()
        sum_loss = entropies.sum()
        return loss, acc, sum_acc, sum_loss, ans_hats

    def get_span_endpoints_rep(self, p_level_h, p_lens, batch_size, ans_start, ans_end, contexts_mask,
                               anss_y):

        p_level_h_for_stt = self.drop(p_level_h)
        p_level_h_for_end = self.drop(p_level_h)

        max_p_len = p_level_h_for_stt.size(0)
        p_stt_ff = F.relu(self.linear_ans_start(p_level_h_for_stt))

        p_end_ff = F.relu(self.linear_ans_end(p_level_h_for_end))

        word_stt_scores = self.linear_stt(p_stt_ff)
        word_end_scores = self.linear_end(p_end_ff)

        stt_log_probs, stt_xents = self._word_multinomial_classification(
            word_stt_scores.squeeze().t(), contexts_mask.t(), ans_start)
        end_log_probs, end_xents = self._word_multinomial_classification(
            word_end_scores.squeeze().t(), contexts_mask.t(), ans_end)

        xents = stt_xents + end_xents
        loss = xents.mean()

        end_log_probs = end_log_probs.unsqueeze(2).permute(1, 0, 2)
        stt_log_probs = stt_log_probs.unsqueeze(2).permute(1, 0, 2)
        span_log_probs_reshaped, span_masks_reshaped = self._span_sums(stt_log_probs, end_log_probs, p_lens, max_p_len,
                                                                       batch_size, 1, max_ans_len)

        span_log_probs_reshaped = span_log_probs_reshaped.view(batch_size, max_p_len * max_ans_len)

        a_hats = self.argmax_with_mask(span_log_probs_reshaped, span_masks_reshaped)

        accs = torch.eq(a_hats, anss_y).float()

        acc = accs.mean()
        sum_acc = accs.sum()
        sum_loss = loss.sum()
        return loss, acc, sum_acc, sum_loss, a_hats

    def softmax_columns_with_mask(self, inp, inp_mask, allow_none=False):
        inp = inp * inp_mask
        inp = inp - inp.min(dim=0, keepdim=True)[0]
        inp = inp * inp_mask
        inp = inp - inp.max(dim=0, keepdim=True)[0]
        exp_val = inp_mask * torch.exp(inp)
        totals = exp_val.sum(dim=0, keepdim=True)
        if allow_none:
            totals += torch.eq(totals, 0)
        return exp_val / totals

    def argmax_with_mask(self, inp, inp_mask):
        x_min = inp.min(dim=1, keepdim=True)[0]
        inp = inp_mask * inp + (1 - inp_mask) * x_min
        return inp.max(dim=1)[1]

    def _word_multinomial_classification(self, inp, inp_mask, out):
        inp = inp * inp_mask
        inp = inp - inp.min(dim=1, keepdim=True)[0]
        inp = inp * inp_mask
        inp = inp - inp.max(dim=1, keepdim=True)[0]
        inp = inp * inp_mask
        exp_x = torch.exp(inp)
        exp_x = exp_x * inp_mask

        sum_exp_x = exp_x.sum(dim=1, keepdim=True)
        log_sum_exp_x = torch.log(sum_exp_x)

        log_probs = inp - log_sum_exp_x
        log_probs = log_probs * inp_mask

        index1 = torch.arange(0, inp.size(0)).long()
        if can_use_gpu:
            index1 = index1.cuda()

        xents = -log_probs[index1, out.cuda()]

        return log_probs, xents

    def multi_classification(self, inp, inp_mask, output):
        assert len(inp.size()) == len(inp_mask.size()) == 2
        assert len(output.size()) == 1

        inp = inp * inp_mask
        inp = inp - inp.min(dim=1, keepdim=True)[0]
        inp = inp * inp_mask
        y_hats = inp.max(dim=1)[1]
        accs = torch.eq(y_hats, output).float()

        inp = inp - inp.max(dim=1, keepdim=True)[0]
        inp = inp * inp_mask
        exp_x = torch.exp(inp)
        exp_x = exp_x * inp_mask

        sum_exp_x = exp_x.sum(dim=1)
        log_sum_exp_x = torch.log(sum_exp_x)

        new_index = torch.arange(0, inp.size(0)).long()
        if can_use_gpu:
            new_index = new_index.cuda()
        x_star = inp[new_index, output.data]
        xents = log_sum_exp_x - x_star
        return xents, accs, y_hats

    def _span_sums(self, stt, end, p_lens, max_p_len, batch_size, dim, max_ans_len):
        max_ans_len_range = torch.arange(0, max_ans_len).unsqueeze(0)
        offsets = torch.arange(0, max_p_len).unsqueeze(1)
        if can_use_gpu:
            max_ans_len_range = max_ans_len_range.cuda()
            offsets = offsets.cuda()

        end_idxs = max_ans_len_range + offsets
        end_idxs_flat = end_idxs.view(-1).long()
        extra_zeros = torch.zeros(max_ans_len - 1, batch_size, dim)
        if can_use_gpu:
            extra_zeros = extra_zeros.cuda()

        end_padded = torch.cat([end, Variable(extra_zeros)], 0)

        end_structured = end_padded[end_idxs_flat]

        end_structured = end_structured.view(max_p_len, max_ans_len, batch_size,
                                             dim)
        stt_shuffled = stt.unsqueeze(3).permute(0, 3, 1, 2)

        span_sums = stt_shuffled + end_structured
        span_sums_reshaped = span_sums.permute(2, 0, 1, 3).contiguous().view(batch_size, max_p_len * max_ans_len,
                                                                             dim)

        p_lens_shuffled = p_lens.unsqueeze(1)
        end_idxs_flat_shuffled = end_idxs_flat.unsqueeze(0)

        span_masks_reshaped = torch.lt(Variable(end_idxs_flat_shuffled), p_lens_shuffled)
        span_masks_reshaped = span_masks_reshaped.float()

        return span_sums_reshaped, span_masks_reshaped

    def softmax_depths_with_mask(self, inp, inp_mask):
        assert len(inp.size()) == 3
        assert len(inp_mask.size()) == 3
        inp = inp * inp_mask
        inp = inp - inp.min(dim=2, keepdim=True)[0]
        inp = inp * inp_mask
        inp = inp - inp.max(dim=2, keepdim=True)[0]
        exp_val = inp_mask * torch.exp(inp)
        sums = exp_val.sum(dim=2, keepdim=True)
        output = exp_val / (sums + (torch.eq(sums, 0).float()))
        output = output * inp_mask
        return output

    def init_weights(self):
        with_bias = [self.linear1, self.linear_ans_start, self.linear_q_aligned, self.linear_ans_end]
        without_bias = [self.linear_w, self.linear_stt, self.linear_end]

        for layer in with_bias:
            layer.weight.data.uniform_(-init_scale, init_scale)
            layer.bias.data.fill_(0)
        for layer in without_bias:
            layer.weight.data.uniform_(-init_scale, init_scale)
