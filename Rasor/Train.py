import torch
from Model import *
from Utils import *
from Settings import  *
from torch.autograd import Variable
import logging

logging.basicConfig(filename=log_file_name,level=logging.DEBUG)
logger = logging.getLogger(__name__)

train_data, dev_data, word_emb_data = load_data(tokenized_trn_json_path, tokenized_dev_json_path, batch_size)
word_emb_tensor = torch.from_numpy(word_emb_data.word_emb)

if can_use_gpu:
    word_emb_tensor = word_emb_tensor.cuda()

word_emb = Variable(word_emb_tensor)


model = QAModel(word_emb, emb_dim, hidden_dim)
if can_use_gpu:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss_hist = []
train_acc_hist = []
train_em_hist = []
train_f1_hist = []

valid_em_hist = []
valid_f1_hist = []
valid_loss_hist = []

best_eval_f1 = 0.0
patience = 0
max_wait = 5
for epoch in range(max_num_epochs):
    loss_curr_epoch = 0.0
    acc_curr_epoch = 0.0
    n_done = 0
    uidx = 0
    num_batches = 0
    for data in train_data:

        model.train()
        optimizer.zero_grad()

        contexts = Variable(data[0])
        contexts_mask = Variable(data[1])
        contexts_lens = Variable(data[5])

        n_done += contexts.size(1)
        num_batches += 1

        questions = Variable(data[2])
        questions_mask = Variable(data[3])

        anss = data[4]
        ans_text = data[7]

        y = Variable(data[6])
        originals = data[8]
        whitespace = data[9]

        loss, acc, sum_acc, sum_loss, ans_hat_start_word_idx, ans_hat_end_word_idx = model(contexts, contexts_mask,
                                                                                           questions, questions_mask, y,
                                                                                           contexts_lens)

        # em,f1 = calculate_em_andf1(originals, whitespace, ans_hat_start_word_idx, ans_hat_end_word_idx, ans_text)
        del contexts, contexts_mask, contexts_lens, questions, questions_mask
        loss_curr_epoch += sum_loss.data[0]
        acc_curr_epoch += acc.data[0]
        if uidx % 100 == 0:
            msg = 'uidx = ', uidx, 'Current batch Accuracy %= ', acc.data[0] * 100, ' --  Loss = ', loss.data[0]
            print(msg)
            logger.info(msg)
        uidx += 1
        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
        # torch.cuda.empty_cache()

    current_epoch_train_loss = loss_curr_epoch / n_done
    current_epoch_train_acc = acc_curr_epoch / num_batches

    epoch_em, epoch_f1 = validate_data()

    if epoch_f1 > best_eval_f1:
        best_eval_f1 = epoch_f1
        model_file_name = 'best_model' + str(epoch)
        torch.save(model.state_dict(), model_file_name)
        msg = 'Saving model'
        print(msg)
        logger.info(msg)
        patience = 0
    else:
        patience += 1
        if patience >= max_wait:
            msg = 'Early Stopping'
            print(msg)
            logger.info(msg)
            break

    msg = str.format('End of Epoch {0}. Training Accuracy %= {1}, Training Loss = {2}, Valid EM = {3}, Valid F1 = {4}',
                     (epoch + 1), current_epoch_train_acc * 100, current_epoch_train_loss, epoch_em, epoch_f1)
    print(msg)
    logger.info(msg)
    train_loss_hist.append(current_epoch_train_loss)
    train_acc_hist.append(current_epoch_train_acc)
