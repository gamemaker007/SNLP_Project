
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Settings import *

class QAModel(nn.Module):
    def __init__(self, word_embeddings, emb_dim, hidden_dim):
        super(QAModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.ff_dims = 100
        self.p_start_dim = emb_dim + 2 * hidden_dim + emb_dim

        self.ff_dropout = nn.Dropout(p=0.4)

        self.passage_level_lstm = nn.LSTM(self.p_start_dim, hidden_dim, 2, bidirectional=True)
        self.q_indep_lstm = nn.LSTM(emb_dim, hidden_dim, 2, bidirectional=True)

        self.linear1 = nn.Linear(2 * hidden_dim, self.ff_dims)
        self.linear_w = nn.Linear(self.ff_dims, 1, bias=False)
        self.linear_q_aligned = nn.Linear(emb_dim, self.ff_dims)
        self.linear_ans_start = nn.Linear(2 * hidden_dim, self.ff_dims)
        self.linear_ans_end = nn.Linear(2 * hidden_dim, self.ff_dims)
        self.linear_span = nn.Linear(self.ff_dims, 1, bias=False)

        self.init_weights()

    def forward(self, contexts, contexts_mask, questions, questions_mask, anss, contexts_lens):
        n_timesteps_cntx = contexts.size(0)
        n_timesteps_quest = questions.size(0)
        n_samples = contexts.size(1)

        emb_cntx = self.word_emb[contexts.view(-1)].view(n_timesteps_cntx, n_samples, emb_dim)
        emb_quest = self.word_emb[questions.view(-1)].view(n_timesteps_quest, n_samples, emb_dim)

        q_indep = self.compute_q_indep(emb_quest, questions_mask, emb_cntx.size(0), n_samples)
        q_align = self.compute_q_aligned(emb_cntx, emb_quest, contexts_mask, questions_mask)
        p_star = torch.cat((emb_cntx, q_indep, q_align), 2)


        h_0 = Variable(torch.zeros(4, n_samples, self.hidden_dim))
        c_0 = Variable(torch.zeros(4, n_samples, self.hidden_dim))
        if torch.cuda.is_available:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        passage_level = self.passage_level_lstm(p_star, (h_0, c_0))

        loss, acc, sum_acc, sum_loss, a_hats = self.compute_answer(passage_level[0], contexts_lens, n_samples, anss)

        ans_hat_start_word_idxs, ans_hat_end_word_idxs = self._tt_ans_idx_to_ans_word_idxs(a_hats, max_ans_len)

        return loss, acc, sum_acc, sum_loss, ans_hat_start_word_idxs, ans_hat_end_word_idxs

    def _tt_ans_idx_to_ans_word_idxs(self, ans_idx, max_ans_len):

        ans_start_word_idx = torch.zeros(ans_idx.size(0), ).long()
        ans_end_word_idx = torch.zeros(ans_idx.size(0), ).long()
        for i in range(ans_idx.size(0)):
            ans_start_word_idx[i] = ans_idx.data[i] // max_ans_len
            ans_end_word_idx[i] = ans_start_word_idx[i] + ans_idx.data[i] % max_ans_len

        return ans_start_word_idx, ans_end_word_idx

    def compute_q_indep(self, q_emb, q_mask, max_p_len, n_samples):
        # encoder_out = self.sequence_encoder(q_emb, q_mask, self.LSTM1, self.LSTM1_rev, self.LSTM2, self.LSTM2_rev)

        h_0 = Variable(torch.zeros(4, n_samples, self.hidden_dim))
        c_0 = Variable(torch.zeros(4, n_samples, self.hidden_dim))
        if torch.cuda.is_available:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        encoder_out = self.q_indep_lstm(q_emb, (h_0, c_0))

        q_indep_h = encoder_out[0]
        q_indep_ff = F.relu(self.linear1(q_indep_h))
        q_indep_scores = self.linear_w(q_indep_ff)

        q_indep_weights = self.softmax_columns_with_mask(q_indep_scores.squeeze(), q_mask)  # (max_q_len, batch_size)
        q_indep = torch.sum(q_indep_weights.unsqueeze(2) * q_indep_h, dim=0)  # (batch_size, 2*hidden_dim)

        q_indep_repeated = torch.cat([q_indep.unsqueeze(0)] * max_p_len)

        return q_indep_repeated

    def compute_q_aligned(self, p_emb, q_emb, p_mask, q_mask):
        q_align_ff_p = F.relu(self.linear_q_aligned(p_emb))
        q_align_ff_q = F.relu(self.linear_q_aligned(q_emb))

        q_align_scores = torch.bmm(q_align_ff_p.permute(1, 0, 2), q_align_ff_q.permute(1, 2, 0))

        p_mask_shuffled = p_mask.unsqueeze(2).permute(1, 0, 2)
        q_mask_shuffled = q_mask.unsqueeze(2).permute(1, 2, 0)
        pq_mask = p_mask_shuffled * q_mask_shuffled
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

    def compute_answer(self, p_level_h, p_lens, batch_size, anss):
        p_level_h_for_stt = self.ff_dropout(p_level_h)
        p_level_h_for_end = self.ff_dropout(p_level_h)

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
        sum_loss = xents.sum()
        return loss, acc, sum_acc, sum_loss, a_hats

    def softmax_columns_with_mask(self, x, mask, allow_none=False):
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
        x = x * x_mask
        x = x - x.min(dim=1, keepdim=True)[0]  # (batch_size, num_classes)
        x = x * x_mask  # (batch_size, num_classes)
        y_hats = x.max(dim=1)[1]  # (batch_size,)
        accs = torch.eq(y_hats, y).float()  # (batch_size,)

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
        # print(end.size(), extra_zeros.size())
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

        span_masks_reshaped = torch.lt(end_idxs_flat_shuffled, p_lens_shuffled)  # (batch_size, max_p_len*max_ans_len)
        span_masks_reshaped = span_masks_reshaped.float()

        return span_sums_reshaped, span_masks_reshaped

    def softmax_depths_with_mask(self, x, mask):

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
        with_bias = [self.linear1, self.linear_ans_start, self.linear_q_aligned, self.linear_ans_end]
        without_bias = [self.linear_w, self.linear_span]

        for layer in with_bias:
            layer.weight.data.uniform_(-init_scale, init_scale)
            layer.bias.data.fill_(0)
        for layer in without_bias:
            layer.weight.data.uniform_(-init_scale, init_scale)
