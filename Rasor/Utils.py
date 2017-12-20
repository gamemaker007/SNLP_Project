import numpy as np
from collections import Counter
import string
import pickle as pkl
import json
import re
import io

from Model import *


def normalize_answer(s):
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


def normalize_answer(s):
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


def _contract_word_emb_data(old_word_emb_data, word_strs, is_single_unk):
    old_embeddings, old_word_idx, old_first_known_word, old_first_unknown_word, old_first_unallocated_word = old_word_emb_data

    known_words = []
    unknown_words = []
    for word_str in word_strs:
        if word_str in old_word_idx and old_word_idx[word_str] < old_first_unknown_word:
            known_words.append(word_str)
        else:
            unknown_words.append(word_str)

    word_idx = {}
    total_vocab_size = old_first_known_word + len(word_strs)
    embeddings = np.zeros((total_vocab_size, old_embeddings.shape[1]), dtype=np.float32)

    for i, word_str in enumerate(known_words):
        word = old_first_known_word + i
        word_idx[word_str] = word
        embeddings[word, :] = old_embeddings[old_word_idx[word_str]]

    first_unknown_word = old_first_known_word + len(known_words)

    num_new_unks = 0
    for i, word_str in enumerate(unknown_words):
        word = first_unknown_word + i
        word_idx[word_str] = word
        if word_str in old_word_idx:
            embeddings[word, :] = old_embeddings[old_word_idx[word_str]]
        else:
            embeddings[word, :] = old_embeddings[old_first_unallocated_word + num_new_unks]
            num_new_unks += 1
    del old_embeddings, old_word_idx, old_first_unknown_word, old_first_unallocated_word, old_word_emb_data
    return WordEmbData(embeddings, word_idx, old_first_known_word, first_unknown_word, None)


def convert_to_answer_idx(start_idx, end_idx, ans_len_max):
    # all arguments are concrete ints
    assert end_idx - start_idx + 1 <= ans_len_max
    return start_idx * ans_len_max + (end_idx - start_idx)

def read_word_emb_data():
    with open(metadata_path, 'rb') as f:
        str_to_word, first_known_word, first_unknown_word, first_unallocated_word = pkl.load(f)
    with open(emb_path, 'rb') as f:
        word_emb = np.load(f)

    word_emb_data = WordEmbData(word_emb, str_to_word, first_known_word, first_unknown_word, first_unallocated_word)
    return word_emb_data

def read_dataset(tokenized_json_path, word_strs, has_answers, max_ans_len=None):
    tabular = SquadStorage()
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
    ans_start = torch.zeros(n_samples, ).long()
    ans_end = torch.zeros(n_samples, ).long()

    for idx, [ctx, ctx_len, qtn, qtn_len, ans] in enumerate(zip(ctxs, ctx_lens, qtns, qtn_lens, anss)):
        contexts[:ctx_len, idx] = torch.LongTensor(ctx[:ctx_len])
        questions[:qtn_len, idx] = torch.LongTensor(qtn[:qtn_len])
        contexts_mask[:ctx_len, idx] = 1.
        questions_mask[:qtn_len, idx] = 1.
        # final_answers[idx] = torch.LongTensor(ans)
        contexts_lens[idx] = int(ctx_len)
        final_y[idx] = int(convert_to_answer_idx(ans[0], ans[1], max_ans_len))
        ans_start[idx] = ans[0]
        ans_end[idx] = ans[1]

    if can_use_gpu:
        contexts = contexts.cuda()
        contexts_mask = contexts_mask.cuda()
        contexts_lens = contexts_lens.cuda()

        questions = questions.cuda()
        questions_mask = questions_mask.cuda()
        final_answers = final_answers.cuda()
        final_y = final_y.cuda()

    return contexts, contexts_mask, questions, questions_mask, final_answers, contexts_lens, final_y, anss_text, ctx_originals, ctx_whitespace, ans_start, ans_end

def load_data(train_loc, dev_loc, batch_size):
    word_emb_data = read_word_emb_data()

    word_strs = set()

    trn_tab_ds = read_dataset(train_loc, word_strs, has_answers=True, max_ans_len=max_ans_len)

    dev_tab_ds = read_dataset(dev_loc, word_strs, has_answers=True, max_ans_len=max_ans_len)

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

train_data, dev_data, word_emb_data = load_data(tokenized_trn_json_path, tokenized_dev_json_path, batch_size)

word_emb_tensor = torch.from_numpy(word_emb_data.word_emb)

if can_use_gpu:
    word_emb_tensor = word_emb_tensor.cuda()

word_emb = Variable(word_emb_tensor)





