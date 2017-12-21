from Utils import *


model = QAModel(word_emb, emb_dim, hidden_dim)
if can_use_gpu:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def calculate_em_andf1(originals, whitespace, ans_hat_start_word_idx, ans_hat_end_word_idx, ans_text):
    ems = []
    f1s = []
    for i in range(len(originals)):
        ans_hat_str = construct_answer_hat(originals[i], whitespace[i], ans_hat_start_word_idx[i],
                                           ans_hat_end_word_idx[i])
        ems.append(metric_max_over_ground_truths(exact_match_score, ans_hat_str, ans_text[i]))
        f1s.append(metric_max_over_ground_truths(f1_score, ans_hat_str, ans_text[i]))
    return ems, f1s


def validate_data():
    loss_curr_epoch = 0.0
    acc_curr_epoch = 0.0
    total_ems = 0.0
    total_f1s = 0.0
    n_done = 0
    for data in dev_data:
        model.eval()

        contexts = Variable(data[0], volatile=True)
        contexts_mask = Variable(data[1], volatile=True)
        contexts_lens = Variable(data[5], volatile=True)
        n_done += contexts_mask.size(1)
        questions = Variable(data[2], volatile=True)
        questions_mask = Variable(data[3], volatile=True)

        anss = data[4]
        ans_text = data[7]

        y = Variable(data[6])
        originals = data[8]
        whitespace = data[9]
        start_span = data[10]
        end_span = data[11]

        loss, acc, sum_acc, sum_loss, ans_hat_start_word_idx, ans_hat_end_word_idx = model(contexts, contexts_mask,
                                                                                           questions, questions_mask, y,
                                                                                           contexts_lens, start_span,
                                                                                           end_span)

        ems, f1s = calculate_em_andf1(originals, whitespace, ans_hat_start_word_idx, ans_hat_end_word_idx, ans_text)

        loss_curr_epoch += sum_loss.data[0]
        acc_curr_epoch += acc.data[0]
        total_ems += sum(ems)
        total_f1s += sum(f1s)

    return total_ems / n_done, total_f1s / n_done


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
        start_span = data[10]
        end_span = data[11]

        loss, acc, sum_acc, sum_loss, ans_hat_start_word_idx, ans_hat_end_word_idx = model(contexts, contexts_mask,
                                                                                           questions, questions_mask, y,
                                                                                           contexts_lens, start_span,
                                                                                           end_span)

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
