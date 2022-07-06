import gc
import time

import numpy as np
import pandas as pd
from lp_model import LPModel
from utils import get_multi_batch_size, padding_data


def eval_teacher_lp(sess, lp_model: LPModel, triples, tgt_ents):
    start = time.time()
    print("compute teacher's logits...")

    eval_batch_size = lp_model.args.eval_batch_size
    triples, padding_num = padding_data(triples, eval_batch_size)
    num_batch = len(triples) // eval_batch_size
    eval_seq, fetch_entity_probs, targets = lp_model.eval_seq, lp_model.entity_probs, lp_model.local_ents

    probs = None

    for i in range(num_batch):
        feed_dict = {eval_seq: triples[i * eval_batch_size:(i + 1) * eval_batch_size], targets: np.array(tgt_ents)}
        batch_probs = sess.run(fetch_entity_probs, feed_dict)  # float32
        if i == num_batch - 1:
            batch_probs = batch_probs[0: eval_batch_size - padding_num, :]
        if probs is None:
            probs = batch_probs
        else:
            probs = np.row_stack([probs, batch_probs])
        print("\rlogits", probs.shape, end="")

    print('evaluation time=%.1f s' % (time.time() - start))
    return probs


def eval_lp(sess, lp_model, triples, tgt_ents, filter_mat, method='min', return_ranks=False):
    # eval_batch_size = lp_model.args.eval_batch_size * lp_model.num_gpu
    eval_batch_size = get_multi_batch_size(lp_model.args.eval_batch_size, lp_model.num_gpu, lp_model.args.opt_on_cpu)
    label = triples[:, 2]
    triples, padding_num = lp_model.padding_data(triples, eval_batch_size)
    num_batch = len(triples) // eval_batch_size
    eval_seq, fetch_entity_probs, targets = lp_model.eval_seq, lp_model.entity_probs, lp_model.local_ents
    ranks = []
    start = time.time()

    idx = np.array(range(lp_model.entity_num))
    for i, ent in enumerate(tgt_ents):
        idx[ent] = i

    for i in range(num_batch):
        feed_dict = {eval_seq: triples[i * eval_batch_size:(i + 1) * eval_batch_size],
                     targets: np.array(tgt_ents)}
        batch_probs = sess.run(fetch_entity_probs, feed_dict)  # float32

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


def compute_ranks(probs, method, label):
    if method == 'min':
        probs = probs - probs[range(len(label)), label].reshape(len(probs), 1)
        ranks = (probs > 0).sum(axis=1) + 1
    else:
        ranks = pd.DataFrame(probs).rank(axis=1, ascending=False, method=method)
        ranks = ranks.values[range(len(label)), label]
    return ranks


def assess_performance(ranks, top=10):
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_10 = sum(ranks <= top) * 1.0 / len(ranks)
    mrr = (1. / ranks).sum() / len(ranks)
    return m_r, h_10, mrr
