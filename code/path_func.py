import math
import random
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp


def sample_paths(triples, entity_num, max_length, repeat_times=2, fast=True):
    if fast:
        return sample_paths_fast(triples, entity_num, max_length, repeat_times=repeat_times)
    return sample_paths_rsn(triples, entity_num, max_length, repeat_times=repeat_times)


def random_walk(triple_list, rt_dic, max_length):
    c_length = 3
    paths = list()
    input_paths = triple_list
    while c_length < max_length:
        paths = list()
        for path in input_paths:
            tail = path[-1]
            rt = random.choice(rt_dic.get(tail))
            new_path = path + list(rt)
            paths.append(new_path)
        c_length += 2
        input_paths = paths
    print(len(paths), paths[0])
    assert len(paths[0]) == max_length
    return paths


def sample_paths_fast(triples, entity_num, max_length, repeat_times=2):
    print("fast sampling id paths...")

    start_time = time.time()
    kb = triples.copy()[['h_id', 'r_id', 't_id']]
    triples_list = kb.values.tolist()
    rt_dic = dict()

    if max_length == 3:
        data = kb.values
        data = pd.DataFrame(data)
        print("# sampled paths:", len(data))
        return data

    for h, r, t in triples_list:
        rt_list = rt_dic.get(h, list())
        rt_list.append((r, t))
        rt_dic[h] = rt_list
    print("# relation-tail dict:", len(rt_dic))

    final_paths = list()
    for i in range(repeat_times):
        paths = random_walk(triples_list, rt_dic, max_length)
        final_paths = final_paths + paths
    print("# paths:", len(final_paths))

    data = pd.DataFrame(np.array(final_paths))
    data = data.drop_duplicates()
    print("# paths after drop_duplicates:", len(data))
    print("# final sampled paths: %d, time: %.1f s" % (len(data), time.time() - start_time))
    return data


def sample_paths_rsn(triples, entity_num, max_length, repeat_times=2, alpha=0.7, beta=0.5):
    print("sampling id paths...")
    start_time = time.time()
    kb = triples.copy()[['h_id', 'r_id', 't_id']]

    if max_length == 3:
        data = kb.values
        data = pd.DataFrame(data)
        print("# sampled paths:", len(data))
        return data

    rel_tail_list = np.unique(kb[['r_id', 't_id']].values, axis=0)
    rel_tail_data = pd.DataFrame(rel_tail_list, columns=['r_id', 't_id'])
    rel_tail_data = rel_tail_data.reset_index().rename({'index': 'tail_id'}, axis='columns')

    rel_tail_kb = kb.merge(rel_tail_data, left_on=['r_id', 't_id'], right_on=['r_id', 't_id'])
    head_tail = np.unique(rel_tail_kb[['h_id', 'tail_id']].values, axis=0)

    head_tail_mat = sp.csr_matrix((np.ones(len(head_tail)), (head_tail[:, 0], head_tail[:, 1])),
                                  shape=(entity_num, rel_tail_list.shape[0]))

    em = np.ndarray(shape=(0,))
    rel_tail_kb['across'] = rel_tail_kb.t_id.isin(em)
    rel_tail_kb.loc[rel_tail_kb.across, 'across'] = beta
    rel_tail_kb.loc[rel_tail_kb.across == 0, 'across'] = 1 - beta
    rel_tail_kb = rel_tail_kb[['h_id', 't_id', 'tail_id', 'across']]

    def gen_tail_dict(x):
        return x.tail_id.values, x.across.values / x.across.sum()

    rel_tail_kb = rel_tail_kb.groupby('h_id').apply(gen_tail_dict)

    rel_tail_kb = pd.DataFrame({'tails': rel_tail_kb})

    # start sampling
    hrt = np.repeat(kb.values, repeat_times, axis=0)

    # for initial triples
    def perform_random(x):
        return np.random.choice(x.tails[0], 1, p=x.tails[1].astype(np.float))

    # else
    def perform_random2(x):
        # calculate depth bias
        pre_c = head_tail_mat[np.repeat(x.pre, x.tails[0].shape[0]), x.tails[0]]
        pre_c[pre_c == 0] = alpha
        pre_c[pre_c == 1] = 1 - alpha
        p = x.tails[1].astype(np.float).reshape([-1, ]) * pre_c.A.reshape([-1, ])
        p = p / p.sum()
        return np.random.choice(x.tails[0], 1, p=p)

    rt_x = rel_tail_kb.loc[hrt[:, 2]].apply(perform_random, axis=1)
    rt_x = rel_tail_list[np.concatenate(rt_x.values)]

    rts = [hrt, rt_x]
    c_length = 5
    while c_length < max_length:
        curr = rel_tail_kb.loc[rt_x[:, 1]]
        curr.loc[:, 'pre'] = hrt[:, 0]

        rt_x = curr.apply(perform_random2, axis=1)
        rt_x = rel_tail_list[np.concatenate(rt_x.values)]

        rts.append(rt_x)
        c_length += 2

    data = np.concatenate(rts, axis=1)
    data = pd.DataFrame(data)
    print("# paths:", len(data))
    data = data.drop_duplicates(keep='first')
    print("# paths after drop_duplicates:", len(data))
    print("# final sampled paths: %d, time: %.1f s" % (len(data), time.time() - start_time))
    return data


def augment_paths(paths1, paths2, entities1, entities2, ent_links, path_len, stu_repeat_times=1):

    print("teacher paths:", len(paths1), paths1.dtypes)
    print("student paths:", len(paths2), paths2.dtypes)

    augmented_paths1 = paths1.copy(deep=True)
    augmented_paths2 = paths2.copy(deep=True)

    ent1_list = []
    ent2_list = []
    for row in ent_links.itertuples():
        ent1_list.append(row.ent1)
        ent2_list.append(row.ent2)

    ent12_list = list(set(entities1) - set(ent1_list))
    ent21_list = list(set(entities2) - set(ent2_list))

    print(len(ent2_list), len(set(ent2_list)))
    print(len(entities2), len(set(entities2)))
    print(len(ent21_list), len(entities2) - len(ent_links))

    assert len(ent12_list) == len(entities1) - len(ent_links)
    assert len(ent21_list) == len(entities2) - len(ent_links)

    ent12_map = pd.Series(ent2_list+ent12_list, index=ent1_list+ent12_list)
    ent21_map = pd.Series(ent1_list+ent21_list, index=ent2_list+ent21_list)

    for i in range(path_len):
        if i % 2 == 0:
            col_name = "ent_" + str(math.ceil(i / 2))
            print("replacing column", col_name)
            augmented_paths1[col_name] = ent12_map[paths1[col_name]].values
            augmented_paths2[col_name] = ent21_map[paths2[col_name]].values

    cross_kg_paths = pd.concat(([augmented_paths1, augmented_paths2]), ignore_index=True, sort=False)
    print("raw augmented cross-KG paths:", cross_kg_paths.columns.values, len(cross_kg_paths))
    cross_kg_paths = pd.concat(([cross_kg_paths, paths1, paths1, paths2, paths2]), ignore_index=True, sort=False)
    cross_kg_paths = cross_kg_paths.drop_duplicates(keep=False)
    print("augmented cross-KG paths:", cross_kg_paths.columns.values, len(cross_kg_paths))
    augmented_paths = pd.concat(([cross_kg_paths, paths1, paths2]), ignore_index=True, sort=False)
    print("total augmented paths:", augmented_paths.columns.values, len(augmented_paths))
    augmented_paths = augmented_paths.drop_duplicates(keep="first")
    print("total augmented paths after drop_duplicates:", augmented_paths.columns.values, len(augmented_paths))
    for i in range(stu_repeat_times-1):
        augmented_paths = pd.concat(([augmented_paths, paths2]), ignore_index=True, sort=False)
        print("augmented paths plus student paths:", augmented_paths.columns.values, len(augmented_paths))
    print("final total paths of teacher and student KGs:", augmented_paths.columns.values, len(augmented_paths))
    return augmented_paths, cross_kg_paths


def path_id2uri(input_rel_paths, path_len, entity_id2uri, relation_id2uri):
    print("converting id paths into uri paths...")
    rel_paths = input_rel_paths.copy(deep=True)
    col_list = []
    for i in range(path_len):
        if i % 2 == 0:
            col_name = "ent_" + str(math.ceil(i / 2))
            col_list.append(col_name)
            rel_paths[col_name] = entity_id2uri[rel_paths[i]].values
        else:
            col_name = "rel_" + str(math.ceil(i / 2))
            col_list.append(col_name)
            rel_paths[col_name] = relation_id2uri[rel_paths[i]].values
    return rel_paths[col_list]


def path_uri2id(input_rel_paths, path_len, entity_uri2id, relation_uri2id):
    print("converting uri paths into id paths...")
    rel_paths = input_rel_paths.copy(deep=True)
    col_list = []
    for i in range(path_len):
        col_list.append(i)
        if i % 2 == 0:
            col_name = "ent_" + str(math.ceil(i / 2))
            rel_paths[i] = entity_uri2id[rel_paths[col_name]].values
        else:
            col_name = "rel_" + str(math.ceil(i / 2))
            rel_paths[i] = relation_uri2id[rel_paths[col_name]].values
    return rel_paths[col_list]


def ent_links_2id(ent_links, entity_uri2id):
    print("converting entity links into ids...")
    ent_links = ent_links.copy(deep=True)
    col_list = []
    for i in range(2):
        col_list.append(i)
        col_name = "ent" + str(i+1)
        ent_links[i] = entity_uri2id[ent_links[col_name]].values
    return ent_links[col_list]
