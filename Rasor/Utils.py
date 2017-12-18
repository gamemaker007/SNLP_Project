import pickle as pkl
from torch.autograd import Variable
from collections import namedtuple, Counter
import io
import json
import torch
from torch.utils.data import Dataset, DataLoader
import string
import re
from Settings import *

WordEmbData = namedtuple('WordEmbData', [
    'word_emb',  # float32 (num words, emb dim)
    'str_to_word',  # map word string to word index
    'first_known_word',  # words found in GloVe are at positions [first_known_word, first_unknown_word)
    'first_unknown_word',  # words not found in GloVe are at positions [first_unknown_word, first_unallocated_word)
    'first_unallocated_word'  # extra random embeddings
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
        self.qtns.append(SquadQuestion(ctx_idx, qtn_id, qtn_tokenized, ans_texts, ans_word_idxs))


def read_word_emb_data():
    with open(metadata_path, 'rb') as f:
        str_to_word, first_known_word, first_unknown_word, first_unallocated_word = pkl.load(f)
    with open(emb_path, 'rb') as f:
        word_emb = np.load(f)

    word_emb_data = WordEmbData(word_emb, str_to_word, first_known_word, first_unknown_word, first_unallocated_word)
    return word_emb_data


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def construct_answer_hat(ctx_originals, ctx_whitespace_afters, ans_hat_start_word_idx, ans_hat_end_word_idx):
    ans_hat_str = ''
    for word_idx in range(ans_hat_start_word_idx, ans_hat_end_word_idx + 1):
        ans_hat_str += ctx_originals[word_idx]
        if word_idx < ans_hat_end_word_idx:
            ans_hat_str += ctx_whitespace_afters[word_idx]
    return ans_hat_str


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
    old_word_emb, old_str_to_word, old_first_known_word, old_first_unknown_word, old_first_unallocated_word = old_word_emb_data

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
                print(old_first_unallocated_word)
                word_emb[word, :] = old_word_emb[old_first_unallocated_word + num_new_unks]
                num_new_unks += 1
    del old_word_emb, old_str_to_word, old_first_unknown_word, old_first_unallocated_word, old_word_emb_data
    return WordEmbData(word_emb, str_to_word, old_first_known_word, first_unknown_word, None)


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
    anss_text = []
    ctx_originals = []
    ctx_whitespace = []
    for datum in batch:
        ctx = datum[0]
        indexed_ctx = [word_emb_data.str_to_word[word_str] for word_str in ctx]
        ctxs.append(indexed_ctx)
        ctx_lens.append(datum[1])

        qtn = datum[2]
        indexed_qtx = [word_emb_data.str_to_word[word_str] for word_str in qtn]
        qtns.append(indexed_qtx)
        qtn_lens.append(datum[3])

        anss.append(datum[4])
        anss_text.append(datum[5])
        ctx_originals.append(datum[6])
        ctx_whitespace.append(datum[7])

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

    for idx, [ctx, ctx_len, qtn, qtn_len, ans] in enumerate(zip(ctxs, ctx_lens, qtns, qtn_lens, anss)):
        contexts[:ctx_len, idx] = torch.LongTensor(ctx[:ctx_len])
        questions[:qtn_len, idx] = torch.LongTensor(qtn[:qtn_len])
        contexts_mask[:ctx_len, idx] = 1.
        questions_mask[:qtn_len, idx] = 1.
        # final_answers[idx] = torch.LongTensor(ans)
        contexts_lens[idx] = int(ctx_len)
        final_y[idx] = int(_np_ans_word_idxs_to_ans_idx(ans[0], ans[1], max_ans_len))

    if can_use_gpu:
        contexts = contexts.cuda()
        contexts_mask = contexts_mask.cuda()
        contexts_lens = contexts_lens.cuda()

        questions = questions.cuda()
        questions_mask = questions_mask.cuda()
        final_answers = final_answers.cuda()
        final_y = final_y.cuda()

    return contexts, contexts_mask, questions, questions_mask, final_answers, contexts_lens, final_y, anss_text, ctx_originals, ctx_whitespace


def calculate_em_andf1(originals, whitespace, ans_hat_start_word_idx, ans_hat_end_word_idx, ans_text):
    ems = []
    f1s = []

    for i in range(len(originals)):
        ans_hat_str = construct_answer_hat(originals[i], whitespace[i], ans_hat_start_word_idx[i],
                                           ans_hat_end_word_idx[i])

        ems.append(metric_max_over_ground_truths(exact_match_score, ans_hat_str, ans_text[i]))
        f1s.append(metric_max_over_ground_truths(f1_score, ans_hat_str, ans_text[i]))
    return ems, f1s


def validate_data(model, dev_data):
    loss_curr_epoch = 0.0
    acc_curr_epoch = 0.0
    total_ems = []
    total_f1s = []

    for data in dev_data:
        model.eval()

        contexts = Variable(data[0], volatile=True)
        contexts_mask = Variable(data[1], volatile=True)
        contexts_lens = Variable(data[5], volatile=True)

        questions = Variable(data[2], volatile=True)
        questions_mask = Variable(data[3], volatile=True)

        anss = data[4]
        ans_text = data[7]

        y = Variable(data[6])
        originals = data[8]
        whitespace = data[9]

        loss, acc, sum_acc, sum_loss, ans_hat_start_word_idx, ans_hat_end_word_idx = model(contexts, contexts_mask,
                                                                                           questions, questions_mask, y,
                                                                                           contexts_lens)

        ems, f1s = calculate_em_andf1(originals, whitespace, ans_hat_start_word_idx, ans_hat_end_word_idx, ans_text)

        loss_curr_epoch += sum_loss.data[0]
        acc_curr_epoch += acc.data[0]

        total_ems += ems
        total_f1s += f1s

    return np.mean(total_ems), np.mean(total_f1s)


def load_data(train_loc, dev_loc, batch_size):
    word_emb_data = read_word_emb_data()

    word_strs = set()

    trn_tab_ds = _make_tabular_dataset(train_loc, word_strs, has_answers=True, max_ans_len=max_ans_len)

    dev_tab_ds = _make_tabular_dataset(dev_loc, word_strs, has_answers=True, max_ans_len=max_ans_len)

    word_emb_data = _contract_word_emb_data(word_emb_data, word_strs, learn_single_unk)

    train_dataset = DatasetLoader(trn_tab_ds)
    dev_dataset = DatasetLoader(dev_tab_ds)

    train_loader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              collate_fn=prepare_data)

    dev_loader = DataLoader(dataset=dev_dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            collate_fn=prepare_data)
    return train_loader, dev_loader, word_emb_data