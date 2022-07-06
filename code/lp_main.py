import math
import os
import ast
import random
import time
import argparse
import gc

import tensorflow as tf
import numpy as np
import pandas as pd

from lp_model import LPModel
from kgs import StuKG4LP
from utils import set_seed, tf_config, tf_session, write2log_and_print, get_log_output_path, distribute_model, \
    get_multi_batch_size, print_results
from evaluate import compute_ranks, assess_performance

parser = argparse.ArgumentParser(description='Lifelong_KGE')
parser.add_argument('--gpu', type=str, default="0")
# "../dataset/fb15k-237/" "../dataset/wn18rr/" "../dataset/yago3-10/"
parser.add_argument('--data_path', type=str, default="../dataset/fb15k-237/")
parser.add_argument('--mode', type=str, default="stu", choices=["stu", "lifelong"])
parser.add_argument('--encoder', type=str, default="trans")
parser.add_argument('--max_length', type=int, default=3)
parser.add_argument('--path_repeat_times', type=int, default=2)
parser.add_argument('--stu_repeat_times', type=int, default=2)

parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--intermediate_size', type=int, default=1024)

parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=0.0002)

parser.add_argument('--input_keep_prob', type=float, default=0.3)
parser.add_argument('--transformer_keep_prob', type=float, default=0.9)
parser.add_argument('--intermediate_keep_prob', type=float, default=0.9)

parser.add_argument('--central_ent_weight', type=float, default=0.5)
parser.add_argument('--rel_weight', type=float, default=0.0)
parser.add_argument('--head_ent_weight', type=float, default=0.0)

parser.add_argument('--num_samples', type=int, default=2048)

parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--eval_batch_size', type=int, default=2048)
parser.add_argument('--max_epoch_num', type=int, default=1024)
parser.add_argument('--min_epoch_num', type=int, default=600)

parser.add_argument('--save', type=ast.literal_eval, default=False)
parser.add_argument('--vars_on_cpu', type=ast.literal_eval, default=False)
parser.add_argument('--opt_on_cpu', type=ast.literal_eval, default=False)

parser.add_argument('--win_size', type=int, default=3)

parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--second_num_layers', type=int, default=2)
parser.add_argument('--num_attention_heads', type=int, default=2)

parser.add_argument('--soft_label', type=float, default=0.2)
parser.add_argument('--decay', type=float, default=0.9999)

opts = parser.parse_args()
print(opts)

set_seed()
tf_config(opts.gpu)
stamp, log_file_path = get_log_output_path(opts)
write2log_and_print(log_file_path, opts.__dict__)


def get_stu_model(session):
    folder = "SavedModel/%s_%dl_%s_%s/" % (opts.data_path.split('/')[-2] + "_" + opts.mode, opts.max_length,
                                           opts.encoder, stamp)
    model_saved_path = os.path.join(opts.data_path, folder)
    if opts.save:
        write2log_and_print(log_file_path, "model saved path: " + model_saved_path)
    write2log_and_print(log_file_path, "timestamp:" + stamp)

    kg = StuKG4LP(opts.data_path, opts.mode)
    _, id_paths = kg.get_paths(opts.max_length, opts.path_repeat_times)
    lp_model = LPModel(opts, session, kg.entity_num, kg.relation_num, model_saved_path)

    train_paths = id_paths
    write2log_and_print(log_file_path, "# training relation paths: " + str(len(train_paths)))

    test_triples = kg.test_triples[['h_id', 'r_id', 't_id']].values
    test_matrix = kg.test_filter_mat

    valid_triples = kg.valid_triples[['h_id', 'r_id', 't_id']].values
    valid_matrix = kg.valid_filter_mat

    return lp_model, train_paths, valid_triples, test_triples, valid_matrix, test_matrix, kg.local_entities, \
           kg.relation_num


def train(lp_model: LPModel, hr_dict, training_data, batch_size, rel_num, win_size=3):
    start = time.time()

    num_batch = len(training_data) // batch_size
    indexes = np.random.choice(len(training_data), size=num_batch * batch_size, replace=False)
    fetches = {'loss': lp_model.loss, 'train_op': lp_model.train_op}
    losses = 0
    for i in range(0, num_batch):
        feed_dict = dict()

        one_batch_choices = indexes[i * batch_size: (i + 1) * batch_size]
        one_batch_data = training_data.iloc[one_batch_choices]
        seq = one_batch_data.values[:, :opts.max_length]
        feed_dict[lp_model.train_seq] = seq

        hrs_of_heads = list()
        for h, r, t in seq.tolist():
            hrs = sample_hrs(h, hr_dict, win_size)
            for hh, rr in hrs:
                # if hh == t and rr == get_rev_rel_id(r, rel_num):
                #     hh, rr = random.sample(hr_dict.get(h), 1)[0]
                hrs_of_heads.append((hh, rr, h))
        assert len(hrs_of_heads) == win_size * batch_size
        hrs_of_heads = np.array(hrs_of_heads)
        feed_dict[lp_model.train_hr_seq] = hrs_of_heads

        res = lp_model.session.run(fetches, feed_dict)
        loss = res['loss']
        losses += loss
        end_t = time.time()
        print('\rtraining: batch=%i/%i, loss=%.3f, time=%.3f s' % (i + 1, num_batch, loss, end_t - start), end='')
    print()
    return losses / num_batch, time.time() - start


def sample_hrs(h, hr_dict, window_size):
    hrs = hr_dict.get(h)
    if len(hrs) >= window_size:
        return random.sample(hrs, window_size)
    else:
        random_hts = list()
        for _ in range(window_size - len(hrs)):
            hh, rr = random.sample(hrs, 1)[0]
            random_hts.append((hh, rr))
        return hrs + random_hts


def eval_lp(sess, lp_model, triples, tgt_ents, filter_mat, hr_dict, win_size, method='min', return_ranks=False):
    # eval_batch_size = lp_model.args.eval_batch_size * lp_model.num_gpu
    eval_batch_size = get_multi_batch_size(lp_model.args.eval_batch_size, lp_model.num_gpu, lp_model.args.opt_on_cpu)
    label = triples[:, 2]
    triples, padding_num = lp_model.padding_data(triples, eval_batch_size)
    num_batch = len(triples) // eval_batch_size
    # eval_seq, fetch_entity_probs, targets = lp_model.eval_seq, lp_model.entity_probs, lp_model.local_ents
    ranks = []
    start = time.time()

    idx = np.array(range(lp_model.entity_num))
    for i, ent in enumerate(tgt_ents):
        idx[ent] = i

    for i in range(num_batch):
        seq = triples[i * eval_batch_size:(i + 1) * eval_batch_size]

        heads = seq[:, 0].tolist()
        hrs_of_heads = list()
        for h in heads:
            hrs = sample_hrs(h, hr_dict, win_size)
            for hh, rr in hrs:
                hrs_of_heads.append((hh, rr, h))
        assert len(hrs_of_heads) == win_size * eval_batch_size
        hrs_of_heads = np.array(hrs_of_heads)

        hrs_of_heads1 = list()
        for h in heads:
            hrs = sample_hrs(h, hr_dict, win_size)
            for hh, rr in hrs:
                hrs_of_heads1.append((hh, rr, h))
        assert len(hrs_of_heads1) == win_size * eval_batch_size
        hrs_of_heads1 = np.array(hrs_of_heads1)

        feed_dict = {lp_model.eval_seq: seq,
                     lp_model.eval_hr_seq: hrs_of_heads,
                     lp_model.eval_seq1: seq,
                     lp_model.eval_hr_seq1: hrs_of_heads1,
                     lp_model.local_ents: np.array(tgt_ents)}

        batch_probs = sess.run(lp_model.entity_probs, feed_dict)  # float32

        batch_labels = label[i * eval_batch_size:(i + 1) * eval_batch_size]  # <class 'numpy.ndarray'> (2048,)
        batch_labels = idx[batch_labels]

        batch_filter_mat = filter_mat[i * eval_batch_size:(i + 1) * eval_batch_size]
        if i == num_batch - 1:
            batch_probs = batch_probs[0: eval_batch_size - padding_num, :]
        filter_probs = batch_filter_mat.multiply(batch_probs).toarray()
        filter_probs[range(len(batch_labels)), batch_labels] = batch_probs[range(len(batch_labels)), batch_labels]
        filter_ranks = compute_ranks(filter_probs, method=method, label=batch_labels)
        ranks.append(filter_ranks)
        print('\revaluation: batch=%i/%i, time=%.3f s' % (i + 1, num_batch, time.time() - start), end='')
        del batch_filter_mat, batch_probs, filter_probs
        gc.collect()
    print()
    ranks = np.concatenate(ranks)
    if return_ranks:
        return ranks
    _, hits1, _ = assess_performance(ranks, top=1)
    _, hits3, _ = assess_performance(ranks, top=3)
    mr, hits10, mrr = assess_performance(ranks, top=10)
    return hits1, hits3, hits10, mrr, mr


def get_local_labels(labels, ent_list):
    labels = labels.tolist()
    new_labels = []
    for label in labels:
        new_labels.append(ent_list.index(label))
    return np.array(new_labels)


def get_hr_dict(triples, fun_rels=set()):
    hr_dict = dict()
    add_num = 0
    for h, r, t in triples:
        hrs = hr_dict.get(t, list())
        hrs.append((h, r))
        if r in fun_rels:
            for _ in range(1):
                hrs.append((h, r))
                add_num += 1
        hr_dict[t] = hrs
    print("add more hrs:", add_num)
    return hr_dict


def get_rev_rel_id(rel_id, rel_num):
    return rel_id + rel_num if rel_id < rel_num else rel_id - rel_num


def get_one2many_relations(triples):
    rels = set()
    one2many_rels = set()
    for h, r, t in triples:
        rels.add(r)
    print("total relations:", len(rels))
    hr_dict = dict()
    for h, r, t in triples:
        ts = hr_dict.get((h, r), set())
        ts.add(t)
        hr_dict[(h, r)] = ts
    for k, v in hr_dict.items():
        if len(v) > 2:
            one2many_rels.add(k[1])
    print("one2many_rels:", len(one2many_rels))
    return one2many_rels, rels - one2many_rels


if __name__ == '__main__':
    start_time = time.time()
    sess = tf_session()
    saved_model_vars_list = list()
    swa_num = 10

    local_train_data = None
    model, train_data, valid_data, test_data, valid_filter_mat, test_filter_mat, tgt_ents, rel_num = get_stu_model(sess)
    assert rel_num % 2 == 0
    rel_num = rel_num // 2

    distribute_model(model, len(tgt_ents))
    epoch, mean_loss, max_hits1, times, max_times = 0, 0, 0, 0, 10

    train_data_list = train_data.values.tolist()
    _, func_rels = get_one2many_relations(train_data_list)
    train_hr_dict = get_hr_dict(train_data_list)

    print("# entities in training data:", len(train_hr_dict))

    valid_data_list = valid_data.tolist()
    valid_hr_dict = get_hr_dict(valid_data_list)

    test_data_list = test_data.tolist()
    test_hr_dict = get_hr_dict(test_data_list)

    seen_hr_dict = get_hr_dict(train_data_list + valid_data_list, fun_rels=func_rels)

    for h in valid_hr_dict.keys():
        if h not in train_hr_dict.keys():
            train_hr_dict[h] = [(h, model.relation_num)]
    for h in test_hr_dict.keys():
        if h not in train_hr_dict.keys():
            train_hr_dict[h] = [(h, model.relation_num)]

    for h in valid_hr_dict.keys():
        if h not in seen_hr_dict.keys():
            seen_hr_dict[h] = [(h, model.relation_num)]
    for h in test_hr_dict.keys():
        if h not in seen_hr_dict.keys():
            seen_hr_dict[h] = [(h, model.relation_num)]

    for epoch in range(1, opts.max_epoch_num + 1):
        mean_loss, epoch_time = train(model, train_hr_dict, train_data, opts.batch_size, rel_num, win_size=opts.win_size)
        write2log_and_print(log_file_path, 'Epoch %i: loss=%.3f, time=%.1f s' % (epoch, mean_loss, epoch_time))
        if epoch % opts.eval_freq == 0 and valid_data is not None:
            h1, h3, h10, mrr, mr = eval_lp(sess, model, triples=valid_data,
                                           tgt_ents=tgt_ents, filter_mat=valid_filter_mat,
                                           hr_dict=train_hr_dict, win_size=opts.win_size)
            print_results(h1, h3, h10, mrr, mr)
            write2log_and_print(log_file_path, 'Valid. results: Hits@1=%.3f, Hits@3=%.3f, Hits@10=%.3f, MRR=%.3f, '
                                               'MR=%.1f' % (h1, h3, h10, mrr, mr), is_print=False)

            if h1 > max_hits1:
                max_hits1 = h1
                times = 0
            else:
                times += 1
            if times >= max_times and epoch > opts.min_epoch_num:
                break

            if epoch >= opts.min_epoch_num - opts.eval_freq * swa_num:
                train_vars = tf.trainable_variables()
                saved_model_vars_list.append(sess.run(train_vars))

            # current_model_num = len(saved_model_vars_list)
            # if len(saved_model_vars_list) >= math.ceil(swa_num * 1.5):
            #     print("swa:")
            #     all_assign = []
            #     for i, var in enumerate(train_vars):
            #         values = []
            #         for saved_model in saved_model_vars_list[current_model_num-swa_num: current_model_num]:
            #             values.append(saved_model[i])
            #         all_assign.append(tf.assign(var, np.mean(values, axis=0)))
            #     sess.run(all_assign)
            #     h1, h3, h10, mrr, mr = eval_lp(sess, model, triples=valid_data,
            #                                    tgt_ents=tgt_ents, filter_mat=valid_filter_mat,
            #                                    hr_dict=train_hr_dict, win_size=opts.win_size)
            #     print_results(h1, h3, h10, mrr, mr)
            #     write2log_and_print(log_file_path, 'Valid. results: Hits@1=%.3f, Hits@3=%.3f, Hits@10=%.3f, MRR=%.3f, '
            #                                        'MR=%.1f' % (h1, h3, h10, mrr, mr), is_print=False)
            #
            #     print("reset:")
            #     all_assign = []
            #     saved_model = saved_model_vars_list[-1]
            #     for var, new_var in zip(train_vars, saved_model):
            #         all_assign.append(tf.assign(var, new_var))
            #     sess.run(all_assign)

    if test_data is not None:
        for i in range(1):
            h1, h3, h10, mrr, mr = eval_lp(sess, model, triples=test_data, tgt_ents=tgt_ents,
                                           filter_mat=test_filter_mat,
                                           hr_dict=train_hr_dict, win_size=opts.win_size)
            msg = 'Final test: Hits@1=%.3f, Hits@3=%.3f, Hits@10=%.3f, MRR=%.3f, MR=%.1f, mean loss=%.3f' \
                  % (h1, h3, h10, mrr, mr, mean_loss)
            write2log_and_print(log_file_path, msg)
            print_results(h1, h3, h10, mrr, mr)

        train_vars = tf.trainable_variables()
        current_model_num = len(saved_model_vars_list)
        if len(saved_model_vars_list) >= swa_num:
            print("swa:")
            all_assign = []
            for i, var in enumerate(train_vars):
                values = []
                for saved_model in saved_model_vars_list[current_model_num - swa_num: current_model_num]:
                    values.append(saved_model[i])
                all_assign.append(tf.assign(var, np.mean(values, axis=0)))
            sess.run(all_assign)
            h1, h3, h10, mrr, mr = eval_lp(sess, model, triples=test_data, tgt_ents=tgt_ents,
                                           filter_mat=test_filter_mat,
                                           hr_dict=train_hr_dict, win_size=opts.win_size)
            msg = 'Final test: Hits@1=%.3f, Hits@3=%.3f, Hits@10=%.3f, MRR=%.3f, MR=%.1f, mean loss=%.3f' \
                  % (h1, h3, h10, mrr, mr, mean_loss)
            write2log_and_print(log_file_path, msg)
            print_results(h1, h3, h10, mrr, mr)

            all_assign = []
            for i, var in enumerate(train_vars):
                values = []
                for saved_model in saved_model_vars_list[current_model_num - int(1.5 * swa_num): current_model_num]:
                    values.append(saved_model[i])
                all_assign.append(tf.assign(var, np.mean(values, axis=0)))
            sess.run(all_assign)
            h1, h3, h10, mrr, mr = eval_lp(sess, model, triples=test_data, tgt_ents=tgt_ents,
                                           filter_mat=test_filter_mat,
                                           hr_dict=train_hr_dict, win_size=opts.win_size)
            msg = 'Final test: Hits@1=%.3f, Hits@3=%.3f, Hits@10=%.3f, MRR=%.3f, MR=%.1f, mean loss=%.3f' \
                  % (h1, h3, h10, mrr, mr, mean_loss)
            write2log_and_print(log_file_path, msg)
            print_results(h1, h3, h10, mrr, mr)

    write2log_and_print(log_file_path, "Total training time: {:.3f} s".format(time.time() - start_time))
    model.save() if opts.save else exit(0)
